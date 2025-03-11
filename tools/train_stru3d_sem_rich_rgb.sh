#!/bin/bash
#SBATCH -J s3d_sem              # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu         # Request partition
#SBATCH --constraint="[a6000|a5000|3090|a100|6000ada]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a6000:1                # Type/number of GPUs needed
#SBATCH --cpus-per-gpu=2              # Number of CPU cores per gpu
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# python main.py --dataset_name=stru3d \
#                --dataset_root=data/stru3d \
#                --num_queries=2800 \
#                --num_polys=70 \
#                --semantic_classes=19 \
#                --job_name=train_stru3d_sem_rich

MASTER_PORT=13595
python -m torch.distributed.run --nproc_per_node=1 --master_port=$MASTER_PORT main_ddp.py --dataset_name=stru3d \
               --dataset_root=data/stru3d_sem_rich \
               --num_queries=2800 \
               --num_polys=70 \
               --semantic_classes=19 \
               --job_name=s3d_sem_rgb_ddp_fromckpt \
               --batch_size 6 \
               --input_channels=3 \
               --output_dir /share/kuleshov/htp26/roomformer/output/ \
               --resume /share/kuleshov/htp26/roomformer/output/s3d_sem_rgb_ddp_fromckpt/checkpoint.pth
            #    --start_from_checkpoint checkpoints/roomformer_stru3d_semantic_rich.pth
