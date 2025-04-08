import argparse
import datetime
import json
import random
import os
import time
from pathlib import Path

from collections import defaultdict

import plotly.graph_objects as go
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from datasets import build_dataset
from models import build_model
import cv2

from datasets import get_dataset_class_labels

from util.poly_ops import pad_gt_polys
from util.plot_utils import plot_room_map, plot_score_map, plot_floorplan_with_regions, plot_semantic_rich_floorplan, plot_semantic_rich_floorplan_tight, plot_semantic_rich_floorplan_nicely

def unnormalize_image(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (x * std + mean)


def plot_gt_floor(data_loader, device, output_dir, plot_gt=True, semantic_rich=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for batched_inputs, _ in data_loader:

        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"] for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]

        # draw GT map
        if plot_gt:
            for i, gt_inst in enumerate(gt_instances):
                if not semantic_rich:
                    # plot regular room floorplan
                    gt_polys = []
                    density_map = np.transpose((samples[i] * 255).cpu().numpy(), [1, 2, 0])
                    density_map = np.repeat(density_map, 3, axis=2)

                    gt_corner_map = np.zeros([256, 256, 3])
                    for j, poly in enumerate(gt_inst.gt_masks.polygons):
                        corners = poly[0].reshape(-1, 2)
                        gt_polys.append(corners)
                        
                    gt_room_polys = [np.array(r) for r in gt_polys]
                    gt_floorplan_map = plot_floorplan_with_regions(gt_room_polys, scale=1000)
                    cv2.imwrite(os.path.join(output_dir, '{}_gt.png'.format(scene_ids[i])), gt_floorplan_map)
                else:
                    # plot semantically-rich floorplan
                    gt_sem_rich = []
                    for j, poly in enumerate(gt_inst.gt_masks.polygons):
                        # if gt_inst.gt_classes.cpu().numpy()[j] not in [1, 9, 11]:
                        #     continue
                        corners = poly[0].reshape(-1, 2).astype(np.int32)
                        corners_flip_y = corners.copy()
                        corners_flip_y[:,1] = 255 - corners_flip_y[:,1]
                        corners = corners_flip_y
                        gt_sem_rich.append([corners, gt_inst.gt_classes.cpu().numpy()[j]])

                    gt_sem_rich_path = os.path.join(output_dir, '{}_floor.png'.format(str(scene_ids[i]).zfill(5)))
                    plot_semantic_rich_floorplan_nicely(gt_sem_rich, gt_sem_rich_path, prec=1, rec=1, plot_text=False, is_bw=False,
                                                       door_window_index=[10, 9], 
                                                       img_w=samples[i].shape[2], 
                                                       img_h=samples[i].shape[1],
                                                       semantics_label_mapping=get_dataset_class_labels(args.dataset_name))


def plot_polys(data_loader, device, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for batched_inputs, _ in data_loader:

        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"] for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]
        

        for i in range(len(samples)):
            density_map = np.transpose((samples[i]).cpu().numpy(), [1, 2, 0])
            if density_map.shape[2] == 3:
                density_map = density_map * 255
            else:
                density_map = np.repeat(density_map, 3, axis=2) * 255
            pred_room_map = np.zeros(density_map.shape).astype(np.uint8)

            room_polys = gt_instances[i].gt_masks.polygons
            room_ids = gt_instances[i].gt_classes.detach().cpu().numpy()
            for poly, poly_id in zip(room_polys, room_ids):
                poly = poly[0].reshape(-1,2).astype(np.int32)
                pred_room_map = plot_room_map(poly, pred_room_map, poly_id)

            # Blend the overlay with the density map using alpha blending
            alpha = 0.6  # Adjust for desired transparency
            pred_room_map = cv2.addWeighted(density_map.astype(np.uint8), alpha, pred_room_map.astype(np.uint8), 1-alpha, 0)

            # # plot predicted polygon overlaid on the density map
            # pred_room_map = np.clip(pred_room_map + density_map, 0, 255)
            cv2.imwrite(os.path.join(output_dir, '{}_pred_room_map.png'.format(scene_ids[i])), pred_room_map)


def plot_gt_image(data_loader, device, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for batched_inputs, _ in data_loader:

        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"] for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]
        

        for i in range(len(samples)):
            density_map = np.transpose((samples[i]).cpu().numpy(), [1, 2, 0])
            if density_map.shape[2] == 3:
                density_map = density_map * 255
            else:
                density_map = np.repeat(density_map, 3, axis=2) * 255

            # # plot predicted polygon overlaid on the density map
            # pred_room_map = np.clip(pred_room_map + density_map, 0, 255)
            cv2.imwrite(os.path.join(output_dir, '{}_gt_image.png'.format(scene_ids[i])), density_map)


def plot_histogram(count_dict, title, output_path):
    # Plot the histogram using Plotly
    keys = list(count_dict.keys())
    values = list(count_dict.values())

    # Determine the maximum value for the y-axis
    max_y = max(values)
    # Adjust y-axis ticks dynamically for large ranges
    tick_interval = max(1, max_y // 10)  # Divide the range into 10 intervals
    tickvals_y = list(range(0, max_y + tick_interval, tick_interval))

    # Determine tick values for x-axis dynamically
    tickvals_x = keys  # Use the keys (number of points in polygons) as tick values

    fig = go.Figure(data=[
        go.Bar(x=keys, y=values, 
               text=values, textposition='outside', marker=dict(color='blue'), 
               width=0.5)
    ])

    fig.update_layout(
        title={
            'text': f'Histogram of {title}',
            'font': {'size': 24},  # Increase title font size
            'x': 0.5,  # Center the title
        },
        xaxis_title={
            'text': f'Number of {title}',
            'font': {'size': 18}  # Increase x-axis label font size
        },
        yaxis_title={
            'text': 'Frequency',
            'font': {'size': 18}  # Increase y-axis label font size
        },
        xaxis=dict(
            tickmode='array',  # Use custom tick values
            tickvals=tickvals_x,  # Set custom tick values
            ticktext=[str(val) for val in tickvals_x],  # Set custom tick labels
            tickfont=dict(size=10),  # Increase x-axis tick font size
            tickangle=45,
        ),
        yaxis=dict(
            tickvals=tickvals_y,  # Set custom tick values
            ticktext=[str(val) for val in tickvals_y],  # Set custom tick labels
            tickfont=dict(size=14),  # Increase y-axis tick font size
        ),
        template='plotly_white',
        bargap=0.5,  # Add gap between bars (0.5 = 50% of bar width)
        # Increase figure width for a long x-axis
        width=max(600, 30 * len(keys)),  # Dynamic width based on number of bars
    )
    # Save the figure as an image
    fig.write_image(output_path, scale=3)
    print(f"Figure saved to {output_path}")

    # fig.show()


def loop_data(data_loader, eval_set, device, output_dir):
    max_num_points = -1
    max_num_polys = -1
    count_pts_dict = defaultdict(lambda: 0)
    count_room_dict = defaultdict(lambda: 0)
    count_length_dict = defaultdict(lambda: 0)
    for batched_inputs, batched_extras in data_loader:

        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"] for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]
        

        for i in range(len(samples)):
            if batched_extras is not None:
                t = batched_extras['mask'][i].sum().item()
                count_length_dict[t] += 1
            room_polys = gt_instances[i].gt_masks.polygons
            room_ids = gt_instances[i].gt_classes.detach().cpu().numpy()
            count_room_dict[len(room_ids)] += 1
            for poly, poly_id in zip(room_polys, room_ids):
                poly = poly[0].reshape(-1,2).astype(np.int32)
                count_pts_dict[len(poly)] += 1
                if len(poly) > max_num_points:
                    max_num_points = len(poly)
            if len(room_ids) > max_num_polys:
                max_num_polys = len(room_ids)
        
    print(f"Max pts: {max_num_points}, Max polys: {max_num_polys}")

    plot_histogram(count_pts_dict, "Points in Polygons", os.path.join(output_dir, f"{eval_set}_polygon_histogram.png"))
    plot_histogram(count_room_dict, "Rooms in Floorplan", os.path.join(output_dir, f"{eval_set}_room_histogram.png"))
    plot_histogram(count_length_dict, "Sequence Length", os.path.join(output_dir, f"{eval_set}_seqlen_histogram.png"))


def get_args_parser():
    parser = argparse.ArgumentParser('RoomFormer', add_help=False)
    parser.add_argument('--batch_size', default=10, type=int)

    # new
    parser.add_argument('--debug', action='store_true')

    # poly2seq
    parser.add_argument('--poly2seq', action='store_true')
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--num_bins', type=int, default=64)

    # backbone
    parser.add_argument('--input_channels', default=1, type=int)
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--lr_backbone', default=0, type=float)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    parser.add_argument('--image_norm', action='store_true')
    parser.add_argument('--disable_image_transform', action='store_true')

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
    parser.add_argument('--use_room_attn_at_last_dec_layer', default=False, action='store_true', help="use room-wise attention in last decoder layer")

    # aux
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_true',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    

    # dataset parameters
    parser.add_argument('--dataset_name', default='stru3d')
    parser.add_argument('--dataset_root', default='data/stru3d', type=str)
    parser.add_argument('--eval_set', default='test', type=str)

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--checkpoint', default='checkpoints/roomformer_scenecad.pth', help='resume from checkpoint')
    parser.add_argument('--output_dir', default='eval_stru3d',
                        help='path where to save result')

    # visualization options
    parser.add_argument('--plot_density', default=False, action='store_true', help="plot predicited room polygons overlaid on the density map")
    parser.add_argument('--plot_gt', default=False, action='store_true', help="plot ground truth floorplan")
    parser.add_argument('--plot_gt_image', default=False, action='store_true', help="plot ground truth image")


    return parser



def main(args):

    device = 'cpu' # torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # # build model
    # model = build_model(args, train=False)
    # model.to(device)

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)

    # build dataset and dataloader
    dataset_eval = build_dataset(image_set=args.eval_set, args=args)
    # for test
    if args.debug:
        dataset_eval = torch.utils.data.Subset(dataset_eval, list(range(0, args.batch_size, 1)))
    sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)

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
            }
            
        return batch, None

    data_loader_eval = DataLoader(dataset_eval, args.batch_size, sampler=sampler_eval,
                                 drop_last=False, collate_fn=trivial_batch_collator, num_workers=args.num_workers,
                                 pin_memory=True)

    # for n, p in model.named_parameters():
    #     print(n)

    output_dir = Path(args.output_dir)

    # checkpoint = torch.load(args.checkpoint, map_location='cpu')
    # missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    # unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    # if len(missing_keys) > 0:
    #     print('Missing Keys: {}'.format(missing_keys))
    # if len(unexpected_keys) > 0:
    #     print('Unexpected Keys: {}'.format(unexpected_keys))

    save_dir = output_dir # os.path.join(os.path.dirname(args.checkpoint), output_dir)
    os.makedirs(save_dir, exist_ok=True)

    if args.plot_gt:
        plot_gt_floor(
                    data_loader_eval, 
                    device, save_dir, 
                    plot_gt=args.plot_gt,
                    semantic_rich=args.semantic_classes>0
                    )
    
    if args.plot_density:
        plot_polys(data_loader_eval, device, save_dir)

    if args.plot_gt_image:
        plot_gt_image(data_loader_eval, device, save_dir)

    loop_data(data_loader_eval, args.eval_set, device, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RoomFormer evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
