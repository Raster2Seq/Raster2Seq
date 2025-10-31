#!/usr/bin/env bash

# python eval.py --dataset_name=cubicasa \
#                --dataset_root=data/coco_cubicasa5k_nowalls_v4-1_refined/ \
#                --eval_set=test \
#                --checkpoint=/home/htp26/RoomFormerTest/output/cubi_v4-1refined_queries56x50_sem_v1/checkpoint0499.pth \
#                --output_dir=eval_cubi \
#                --num_queries=2800 \
#                --num_polys=50 \
#                --semantic_classes=12 \
#                --input_channels 3 \
#                # --debug \
#             #    --ema4eval \
#             #    --save_pred \

DATA=data/waffle/data/original_size_images/
FOLDER=00000 # missing_images
# DATA=data/waffle_benchmark_processed/
# FOLDER=test

python predict.py \
               --dataset_root=${DATA}/${FOLDER} \
               --checkpoint=/home/htp26/RoomFormerTest/output/cubi_v4-1refined_queries56x50_sem_v1/checkpoint0499.pth \
               --output_dir=waffle_raster${FOLDER}_preds \
               --num_queries=2800 \
               --num_polys=50 \
               --semantic_classes=12 \
               --input_channels 3 \
               --drop_wd \
               --one_color \
               --image_scale 2 \
               --crop_white_space \
            #    --save_pred \

# CKPT=output/cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_convertv2_t1/checkpoint1899.pth
# CKPT=/home/htp26/RoomFormerTest/output/cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_convertv3_fromnoorder_t1/checkpoint0499.pth
# CKPT=output/cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_convertv3_fromnorder499_t1/checkpoint0499.pth

# python predict.py \
#                --dataset_root=${DATA}/${FOLDER} \
#                --checkpoint=${CKPT} \
#                --output_dir=waffle_raster${FOLDER}_preds2 \
#                --semantic_classes=12 \
#                --input_channels 3 \
#                --poly2seq \
#                --seq_len 512 \
#                --num_bins 32 \
#                --disable_poly_refine \
#                --dec_attn_concat_src \
#                --use_anchor \
#                --ema4eval \
#                --per_token_sem_loss \
#                --drop_wd \
#                --save_pred \
#                --one_color \
#                --image_scale 2 \
#                --crop_white_space \
#             #    --pre_decoder_pos_embed \
#                # --debug \