#!/bin/bash
#SBATCH -J s3d              # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu,elor         # Request partition
#SBATCH --constraint="[a6000|a5000|a5500|3090|a100|6000ada]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                # Type/number of GPUs needed
#SBATCH --cpus-per-gpu=1              # Number of CPU cores per gpu
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# python main.py --dataset_name=stru3d \
#                --dataset_root=data/stru3d \
#                --num_queries=2800 \
#                --num_polys=70 \
#                --semantic_classes=19 \
#                --job_name=train_stru3d_sem_rich

MASTER_PORT=13535
WANDB_MODE=online python -m torch.distributed.run --nproc_per_node=1 --master_port=$MASTER_PORT main_ddp.py --dataset_name=stru3d \
               --dataset_root=data/coco_s3d_bw \
               --num_queries=1200 \
               --num_polys=30 \
               --semantic_classes=-1 \
               --job_name=s3d_bw_ddp_poly2seq_l512_nosem_bs32_coo10_cls1_nopolyrefine_predecPE_deccatsrc_v2 \
               --batch_size 32 \
               --input_channels=3 \
               --output_dir /share/elor/htp26/roomformer/output/ \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --ckpt_every_epoch 50 \
               --eval_every_epoch 20 \
               --lr 2e-4 \
               --lr_backbone 2e-5 \
               --label_smoothing 0.0 \
               --epochs 1500 \
               --lr_drop '' \
               --cls_loss_coef 1 \
               --coords_loss_coef 10 \
               --disable_poly_refine \
               --pre_decoder_pos_embed \
               --dec_attn_concat_src \
               # --dec_qkv_proj \
               # --resume /home/htp26/RoomFormerTest/output/s3d_bw_ddp_poly2seq_l512_nosem_bs32_coo10_cls1_nopolyrefine_predecPE_v1/checkpoint.pth \
               # --dec_layer_type='v2' \
            #    --set_cost_coords=10 \
            #    --coords_loss_coef=10