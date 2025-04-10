import argparse
import datetime
import json
import random
import os
import time
from pathlib import Path
import copy

import numpy as np
import wandb
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import util.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch, evaluate_v2
from models import build_model



def get_args_parser():
    parser = argparse.ArgumentParser('RoomFormer', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr_drop', default='400', type=str)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # new
    parser.add_argument('--input_channels', default=1, type=int)
    parser.add_argument('--start_from_checkpoint', default='', help='resume from checkpoint')
    parser.add_argument('--image_norm', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval_every_epoch', type=int, default=20)
    parser.add_argument('--ckpt_every_epoch', type=int, default=20)
    parser.add_argument('--label_smoothing', type=float, default=0.)
    parser.add_argument('--ignore_index', type=int, default=-1)
    parser.add_argument('--image_size', type=int, default=256)

    # poly2seq
    parser.add_argument('--poly2seq', action='store_true')
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--num_bins', type=int, default=64)
    parser.add_argument('--pre_decoder_pos_embed', action='store_true')
    parser.add_argument('--learnable_dec_pe', action='store_true')
    parser.add_argument('--dec_qkv_proj', action='store_true')
    parser.add_argument('--dec_attn_concat_src', action='store_true')
    parser.add_argument('--dec_layer_type', type=str, default='v1')

    # parser.add_argument('--use_room_attn_at_last_dec_layer', default=False, action='store_true', help="use room-wise attention in last decoder layer")

    # backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=800, type=int,
                        help="Number of query slots (num_polys * max. number of corner per poly)")
    parser.add_argument('--num_polys', default=20, type=int,
                        help="Number of maximum number of room polygons")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--query_pos_type', default='sine', type=str, choices=('static', 'sine', 'none'),
                        help="Type of query pos in decoder - \
                        1. static: same setting with DETR and Deformable-DETR, the query_pos is the same for all layers \
                        2. sine: since embedding from reference points (so if references points update, query_pos also \
                        3. none: remove query_pos")
    parser.add_argument('--with_poly_refine', default=True, action='store_true',
                        help="iteratively refine reference points (i.e. positional part of polygon queries)")
    parser.add_argument('--masked_attn', default=False, action='store_true',
                        help="if true, the query in one room will not be allowed to attend other room")
    parser.add_argument('--semantic_classes', default=-1, type=int,
                        help="Number of classes for semantically-rich floorplan:  \
                        1. default -1 means non-semantic floorplan \
                        2. 19 for Structured3D: 16 room types + 1 door + 1 window + 1 empty")
    parser.add_argument('--disable_poly_refine', action='store_true',
                        help="iteratively refine reference points (i.e. positional part of polygon queries)")

    # loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_true',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_coords', default=5, type=float,
                        help="L1 coords coefficient in the matching cost")

    # loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--room_cls_loss_coef', default=0.2, type=float)
    parser.add_argument('--coords_loss_coef', default=5, type=float)
    parser.add_argument('--raster_loss_coef', default=1, type=float)

    # dataset parameters
    parser.add_argument('--dataset_name', default='stru3d')
    parser.add_argument('--dataset_root', default='data/stru3d', type=str)

    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--job_name', default='train_stru3d', type=str)

    return parser


def main(args):

    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)
    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed * dist.get_world_size() + rank
    # fix the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # setup wandb for logging
    if rank == 0:
        utils.setup_wandb()
        wandb.init(project="RoomFormer")
        wandb.run.name = args.run_name


    # build dataset and dataloader
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    tokenizer = None
    if args.poly2seq:
        args.vocab_size = dataset_train.get_vocab_size()
        tokenizer = dataset_train.get_tokenizer()

    # sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # overfit one sample
    if args.debug:
        dataset_val = torch.utils.data.Subset(copy.deepcopy(dataset_train), [445])
        dataset_train = copy.deepcopy(dataset_val)  # torch.utils.data.Subset(copy.deepcopy(dataset_val), [10])

    sampler_train = DistributedSampler(dataset_train, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.seed)
    sampler_val = DistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=rank, shuffle=False, seed=args.seed)

    def trivial_batch_collator(batch):
        """
        A batch collator that does nothing.
        """
        if 'target_seq' in batch[0]:
            # Concatenate tensors for each key in the batch
            delta_x1 = torch.stack([item['delta_x1'] for item in batch], dim=0)
            delta_x2 = torch.stack([item['delta_x2'] for item in batch], dim=0)
            delta_y1 = torch.stack([item['delta_y1'] for item in batch], dim=0)
            delta_y2 = torch.stack([item['delta_y2'] for item in batch], dim=0)
            seq11 = torch.stack([item['seq11'] for item in batch], dim=0)
            seq21 = torch.stack([item['seq21'] for item in batch], dim=0)
            seq12 = torch.stack([item['seq12'] for item in batch], dim=0)
            seq22 = torch.stack([item['seq22'] for item in batch], dim=0)
            target_seq = torch.stack([item['target_seq'] for item in batch], dim=0)
            token_labels = torch.stack([item['token_labels'] for item in batch], dim=0)
            mask = torch.stack([item['mask'] for item in batch], dim=0)
            target_polygon_labels = torch.stack([item['target_polygon_labels'] for item in batch], dim=0)

            # Delete the keys from the batch
            for item in batch:
                del item['delta_x1']
                del item['delta_x2']
                del item['delta_y1']
                del item['delta_y2']
                del item['seq11']
                del item['seq21']
                del item['seq12']
                del item['seq22']
                del item['target_seq']
                del item['token_labels']
                del item['mask']
                del item['target_polygon_labels']

            # Return the concatenated batch
            return batch, {
                'delta_x1': delta_x1,
                'delta_x2': delta_x2,
                'delta_y1': delta_y1,
                'delta_y2': delta_y2,
                'seq11': seq11,
                'seq21': seq21,
                'seq12': seq12,
                'seq22': seq22,
                'target_seq': target_seq,
                'token_labels': token_labels,
                'mask': mask,
                'target_polygon_labels': target_polygon_labels,
            }
            
        return batch, None

    data_loader_train = DataLoader(dataset_train, 
                                   args.batch_size, 
                                   shuffle=False,
                                   sampler=sampler_train,
                                   num_workers=args.num_workers,
                                   collate_fn=trivial_batch_collator, 
                                   pin_memory=True,
                                   drop_last=True)
    data_loader_val = DataLoader(dataset_val, 
                                 args.batch_size, 
                                 shuffle=False,
                                 sampler=sampler_val,
                                 collate_fn=trivial_batch_collator, 
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=False, 
                                 )

    # build model
    model, criterion = build_model(args, tokenizer=tokenizer)
    model.to(device)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)

    if args.lr_drop:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)
    else:
        lr_scheduler = None


    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        # for key, value in checkpoint['model'].items():
        #     if key.startswith('module.'):
        #         checkpoint[key[7:]] = checkpoint['model'][key]
        #         del checkpoint[key]
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                if 'initial_lr' in pg_old:
                    pg['initial_lr'] = pg_old['initial_lr']
            # print(optimizer.param_groups)
            if lr_scheduler is not None and checkpoint['lr_scheduler'] is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = False
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                if lr_scheduler is not None:
                    lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))

            if lr_scheduler is not None:
                lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        if not args.poly2seq:
            test_stats = evaluate(
                model, criterion, args.dataset_name, data_loader_val, device
            )
        else:
            test_stats = evaluate_v2(
                model, criterion, args.dataset_name, data_loader_val, device, poly2seq=args.poly2seq
            )

    if args.start_from_checkpoint:
        checkpoint = torch.load(args.start_from_checkpoint, map_location='cpu')
        if checkpoint['model']['module.backbone.0.body.conv1.weight'].size(1) != args.input_channels:
            checkpoint['model']['module.backbone.0.body.conv1.weight'] = checkpoint['model']['module.backbone.0.body.conv1.weight'].repeat(1, args.input_channels, 1, 1)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
    
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, args.poly2seq)
        if lr_scheduler is not None:
            lr_scheduler.step()
        if (epoch + 1) in args.lr_drop or (epoch + 1) % args.ckpt_every_epoch == 0 or (epoch + 1) == args.epochs:
            if rank == 0:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 20 epochs
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': None if lr_scheduler is None else lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
            dist.barrier()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if rank == 0:
            wandb.log({"epoch": epoch})
            wandb.log({"lr_rate": train_stats['lr']})

        train_log_dict = {
                "train/loss": train_stats['loss'],
                "train/loss_ce": train_stats['loss_ce'],
                "train/loss_coords": train_stats['loss_coords'],
                "train/loss_coords_unscaled": train_stats['loss_coords_unscaled'],
                "train/cardinality_error": train_stats['cardinality_error_unscaled']
                }

        if args.semantic_classes > 0:
            # need to log additional metrics for semantically-rich floorplans
            train_log_dict["train/loss_ce_room"] = train_stats['loss_ce_room']
        else:
            if "loss_raster" in train_stats:
                # only apply the rasterization loss for non-semantic floorplans
                train_log_dict["train/loss_raster"] = train_stats['loss_raster']

        if rank == 0:
            wandb.log(train_log_dict)
    
        # eval every 20
        if (epoch + 1) % args.eval_every_epoch == 0:
            if not args.poly2seq:
                test_stats = evaluate(
                    model, criterion, args.dataset_name, data_loader_val, device, 
                    plot_density=True, output_dir=output_dir, epoch=epoch, poly2seq=args.poly2seq,
                )
            else:
                test_stats = evaluate_v2(
                    model, criterion, args.dataset_name, data_loader_val, device, 
                    plot_density=True, output_dir=output_dir, epoch=epoch, poly2seq=args.poly2seq,
                )
            log_stats.update(**{f'test_{k}': v for k, v in test_stats.items()})

            val_log_dict = {
                    "val/loss": test_stats['loss'],
                    "val/loss_ce": test_stats['loss_ce'],
                    "val/loss_coords": test_stats['loss_coords'],
                    "val/loss_coords_unscaled": test_stats['loss_coords_unscaled'],
                    "val/cardinality_error": test_stats['cardinality_error_unscaled'],
                    "val_metrics/room_prec": test_stats['room_prec'],
                    "val_metrics/room_rec": test_stats['room_rec'],
                    "val_metrics/corner_prec": test_stats['corner_prec'],
                    "val_metrics/corner_rec": test_stats['corner_rec'],
                    "val_metrics/angles_prec": test_stats['angles_prec'],
                    "val_metrics/angles_rec": test_stats['angles_rec']
                    }

            if args.semantic_classes > 0:
                # need to log additional metrics for semantically-rich floorplans
                val_log_dict["val/loss_ce_room"] = test_stats['loss_ce_room']
                val_log_dict["val_metrics/room_sem_prec"] = test_stats['room_sem_prec']
                val_log_dict["val_metrics/room_sem_rec"] = test_stats['room_sem_rec']
                if 'window_door_prec' in test_stats:
                    val_log_dict["val_metrics/window_door_prec"] = test_stats['window_door_prec']
                    val_log_dict["val_metrics/window_door_rec"] = test_stats['window_door_rec']

            else:
                if "loss_raster" in test_stats:
                    # only apply the rasterization loss for non-semantic floorplans
                    val_log_dict["val/loss_raster"] =  test_stats['loss_raster']

            if 'room_iou' in test_stats:
                val_log_dict["val_metrics/room_iou"] = test_stats['room_iou']
                    
            if rank == 0:
                wandb.log(val_log_dict)

        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RoomFormer training script', parents=[get_args_parser()])
    args = parser.parse_args()
    now = datetime.datetime.now()
    # run_id = now.strftime("%Y-%m-%d-%H-%M-%S")
    args.run_name = args.job_name # run_id+'_'+args.job_name 
    args.output_dir = os.path.join(args.output_dir, args.run_name)

    args.lr_drop = [] if len(args.lr_drop) == 0 else [int(x) for x in args.lr_drop.split(',')]
    if args.debug:
        args.batch_size = 1
    if args.disable_poly_refine:
        args.with_poly_refine = False

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
