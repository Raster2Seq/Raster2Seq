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

CUDA_VISIBLE_DEVICES={device} python eval.py --dataset_name=cubicasa \
               --dataset_root=data/coco_cubicasa5k_nowalls_v4-1_refined/ \
               --eval_set=test \
               --checkpoint=/home/htp26/RoomFormerTest/output/{exp}/checkpoint{epoch}.pth \
               --output_dir={slurm_output}/eval_cc5k_epoch{epoch} \
               --num_queries=2800 \
               --num_polys=50 \
               --semantic_classes=-1 \
               --input_channels 3 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --disable_poly_refine \
               --dec_attn_concat_src \
               --ema4eval \
               --use_anchor \
               # --pre_decoder_pos_embed \

"""

###### ARGS
exp = "cubi_v4-1refined_poly2seq_l512_bin32_nosem_coo20_cls1_anchor_deccatsrc_fromckpt2450_ignorewd_smoothing_clscoeffx5@6e-1_t1"
device = "0"

config = pd.DataFrame({
    "epochs": [1899],
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