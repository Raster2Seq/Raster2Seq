#!/usr/bin/env bash

# python eval.py --dataset_name=stru3d \
#                --dataset_root=data/coco_s3d_bw \
#                --eval_set=test \
#                --checkpoint=output/s3d_bw_ddp_queries40x30/checkpoint0499.pth \
#                --output_dir=eval_stru3d_sem_rich \
#                --num_queries=1200 \
#                --num_polys=30 \
#                --semantic_classes=19 \
#                --input_channels 3 \
#             #    --debug \

python eval.py --dataset_name=stru3d \
               --dataset_root=data/coco_s3d_bw \
               --eval_set=test \
               --checkpoint=/home/htp26/RoomFormerTest/output/s3d_bw_ddp_poly2seq_l512_sem_bs32_coo20_cls1_nopolyrefine_predecPE_deccatsrc_v1/checkpoint0999.pth \
               --output_dir=eval_stru3d_sem_rich \
               --num_queries=1200 \
               --num_polys=30 \
               --semantic_classes=18 \
               --input_channels 3 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --disable_poly_refine \
               --pre_decoder_pos_embed \
               --dec_attn_concat_src \
               --ema4eval \
               --debug \
               # --per_token_sem_loss \
               # --add_cls_token \
               # --batch_size 1 \
               # --measure_time \
            #    --disable_sampling_cache
               # --dec_qkv_proj \

s3d_bw_ddp_poly2seq_l512_nosem_bs32_coo20_cls1_nopolyrefine_predecPE_deccatsrc_v1