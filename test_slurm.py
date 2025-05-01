import os
import time
import subprocess

import numpy as np
import pandas as pd

slurm_template = """#!/bin/bash -e

export EPOCH_ID={epoch}
export EXP={exp}

echo "----------------------------"
echo $EPOCH_ID $EXP
echo "----------------------------"

CUDA_VISIBLE_DEVICES={device} python eval.py --dataset_name=stru3d \
               --dataset_root=data/coco_s3d_bw \
               --eval_set=test \
               --checkpoint=/home/htp26/RoomFormerTest/output/{exp}/checkpoint{epoch}.pth \
               --output_dir={slurm_output}/eval_s3d_sem_epoch{epoch} \
               --num_queries=1200 \
               --num_polys=30 \
               --semantic_classes=18 \
               --input_channels 3 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --disable_poly_refine \
               --dec_attn_concat_src \
               --ema4eval \
               --use_anchor \
               --per_token_sem_loss \
            #    --pre_decoder_pos_embed \

"""

###### ARGS
exp = "s3d_bw_ddp_poly2seq_l512_sem1_bs32_coo20_cls5_anchor_deccatsrc_correct_pts_finetune_t1"
device = "0"

config = pd.DataFrame({
    "epochs": [1649, 1699, 1749, 1799],
})
print(config)

###################################
slurm_file_path = f"/share/elor/htp26/roomformer/slurm_scripts/{exp}/run.sh"
slurm_output = f"/share/elor/htp26/roomformer/slurm_scripts/{exp}/"
os.makedirs(slurm_output, exist_ok=True)

for idx, row in config.iterrows():
    # device = str(idx % 2)
    # slurm_file_path = f"/lustre/scratch/client/vinai/users/haopt12/cnf_flow/slurm_scripts/{exp}/run{device}.sh"
    slurm_command = slurm_template.format(
        exp=exp,
        epoch=row.epochs,
        slurm_output=slurm_output,
        device=device,
    )
    mode = "w" if idx == 0 else "a"
    with open(slurm_file_path, mode) as f:
        f.write(slurm_command)
print("Slurm script is saved at", slurm_file_path)

# print(f"Summited {slurm_file_path}")
# subprocess.run(['sbatch', slurm_file_path])