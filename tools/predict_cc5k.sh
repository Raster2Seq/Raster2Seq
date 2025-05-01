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

python predict.py \
               --dataset_root=data/waffle/benchmark/pngs/ \
               --checkpoint=/home/htp26/RoomFormerTest/output/cubi_v4-1refined_queries56x50_sem_v1/checkpoint0499.pth \
               --output_dir=/share/elor/htp26/roomformer/waffle_benchmark_preds \
               --num_queries=2800 \
               --num_polys=50 \
               --semantic_classes=12 \
               --input_channels 3 \

# python predict.py \
#                --dataset_root=data/waffle/benchmark/pngs/ \
#                --checkpoint=/home/htp26/RoomFormerTest/output/cubi_v4-1refined_poly2seq_l512_bin32_sem_coo20_cls5_nopolyrefine_predecPE_fromckpt_ignorewd_smoothing_t1/checkpoint1999.pth \
#                --output_dir=waffle_benchmark_preds \
#                --semantic_classes=11 \
#                --input_channels 3 \
#                --poly2seq \
#                --seq_len 512 \
#                --num_bins 32 \
#                --disable_poly_refine \
#                --pre_decoder_pos_embed \
#                --dec_attn_concat_src \
#                --ema4eval \
#                --per_token_sem_loss \
#                # --debug \
#                # --save_pred \