#!/usr/bin/env bash

CKPT=checkpoints/s3dbw_sem_res256_ep0449.pth
python eval.py --dataset_name=stru3d \
               --dataset_root=data/coco_s3d_bw \
               --eval_set=test \
               --checkpoint=${CKPT} \
               --output_dir=eval_outputs/s3dbw_sem_results \
               --semantic_classes=19 \
               --input_channels 3 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --disable_poly_refine \
               --dec_attn_concat_src \
               --ema4eval \
               --use_anchor \
               --per_token_sem_loss \
               --save_pred
