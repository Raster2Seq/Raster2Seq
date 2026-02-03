#!/usr/bin/env bash

DATA=data/waffle/data/original_size_images/
FOLDER=00000
CKPT=cc5k_sem_res256_ep0499.pth

python predict.py \
    --dataset_root=${DATA}/${FOLDER} \
    --checkpoint=${CKPT} \
    --output_dir=pred_outputs/waffle_raster${FOLDER}_preds \
    --semantic_classes=12 \
    --input_channels 3 \
    --poly2seq \
    --seq_len 512 \
    --num_bins 32 \
    --disable_poly_refine \
    --dec_attn_concat_src \
    --use_anchor \
    --ema4eval \
    --per_token_sem_loss \
    --drop_wd \
    --save_pred \
    --one_color