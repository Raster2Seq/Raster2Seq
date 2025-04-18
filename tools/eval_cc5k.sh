#!/usr/bin/env bash

# python eval.py --dataset_name=cubicasa \
#                --dataset_root=data/coco_cubicasa5k_nowalls_v4-1_refined/ \
#                --eval_set=test \
#                --checkpoint=/home/htp26/RoomFormerTest/output/cubi_v4-1refined_poly2seq_l1024_nosem_coo20_cls1_nopolyrefine_predecPE_deccatsrc_t1/checkpoint0049.pth \
#                --output_dir=eval_cubi \
#                --num_queries=2800 \
#                --num_polys=50 \
#                --semantic_classes=12 \
#                --input_channels 3 \
#                --save_pred \
#             #    --debug \


python eval.py --dataset_name=cubicasa \
               --dataset_root=data/coco_cubicasa5k_nowalls_v4-1_refined/ \
               --eval_set=test \
               --checkpoint=/home/htp26/RoomFormerTest/output/cubi_v4-1refined_poly2seq_l1024_nosem_coo20_cls1_nopolyrefine_predecPE_deccatsrc_t1/checkpoint0049.pth \
               --output_dir=eval_cubi \
               --num_queries=2800 \
               --num_polys=50 \
               --semantic_classes=-1 \
               --input_channels 3 \
               --poly2seq \
               --seq_len 1024 \
               --num_bins 64 \
               --disable_poly_refine \
               --pre_decoder_pos_embed \
               --dec_attn_concat_src \
               # --save_pred \
            #    --debug \