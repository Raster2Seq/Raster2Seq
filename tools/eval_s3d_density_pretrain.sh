#!/usr/bin/env bash

CKPT=checkpoints/s3dd_res256_ep0499.pth
python eval.py --dataset_name=stru3d \
   --dataset_root=data/stru3d \
   --eval_set=test \
   --checkpoint=${CKPT} \
   --output_dir=eval_outputs/s3dd_results/ \
   --semantic_classes=-1 \
   --input_channels=1 \
   --poly2seq \
   --seq_len 512 \
   --num_bins 32 \
   --disable_poly_refine \
   --dec_attn_concat_src \
   --ema4eval \
   --use_anchor \
   --save_pred