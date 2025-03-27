#!/bin/bash
#SBATCH -J cubi              # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu,elor         # Request partition
#SBATCH --constraint="[a6000|a5000|3090|a100|6000ada]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                # Type/number of GPUs needed
#SBATCH --cpus-per-gpu=2              # Number of CPU cores per gpu
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

 
MASTER_PORT=13139
WANDB_MODE=offline python -m torch.distributed.run --nproc_per_node=1 --master_port=$MASTER_PORT main_ddp.py --dataset_name=cubicasa \
               --dataset_root=data/coco_cubicasa5k_nowalls_v2/ \
               --num_queries=1800 \
               --num_polys=40 \
               --semantic_classes=12 \
               --job_name=cubi_queries40x45_sem_debug_t1 \
               --batch_size 10 \
               --input_channels=3 \
               --output_dir /share/elor/htp26/roomformer/output \
               --eval_every_epoch=20 \
               --ckpt_every_epoch=40 \
               --epochs 500 \
               --debug \
               # --start_from_checkpoint output/s3d_sem_rgb_ddp_queries40x30/checkpoint0499.pth
               # --resume output/cubi_queries60x30_sem_debug_t3/checkpoint.pth