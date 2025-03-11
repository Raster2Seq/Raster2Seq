#!/bin/bash
#SBATCH -J s3d_org              # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov,gpu         # Request partition
#SBATCH --constraint="[a6000|a5000|3090|a100|6000ada]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                # Type/number of GPUs needed
#SBATCH --cpus-per-gpu=4              # Number of CPU cores per gpu
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

python main.py --dataset_name=stru3d \
               --dataset_root=data/stru3d \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 \
               --output_dir /share/kuleshov/htp26/roomformer/output/ \
               --job_name=stru3d_org_org \
               --resume=/share/kuleshov/htp26/roomformer/output/stru3d_org_org/checkpoint.pth

# MASTER_PORT=13518
# python -m torch.distributed.run --nproc_per_node=1 --master_port=$MASTER_PORT main_ddp.py --dataset_name=stru3d \
#             --dataset_root=data/stru3d \
#             --num_queries=800 \
#             --num_polys=20 \
#             --semantic_classes=-1 \
#             --job_name=stru3d_bs10_org_ddp_t2 \
#             --output_dir /share/kuleshov/htp26/roomformer/output/ \
#             # --resume=/share/kuleshov/htp26/roomformer/output/stru3d_bs10_org_ddp/checkpoint.pth