#!/usr/bin/env bash
# EXP=cubi_v4-1refined_queries56x50_sem_v1
# python eval.py --dataset_name=cubicasa \
#                --dataset_root=data/coco_cubicasa5k_nowalls_v4-1_refined/ \
#                --eval_set=test \
#                --checkpoint=/home/htp26/RoomFormerTest/output/${EXP}/checkpoint0499.pth \
#                --output_dir=slurm_scripts/${EXP} \
#                --num_queries=2800 \
#                --num_polys=50 \
#                --semantic_classes=12 \
#                --input_channels 3 \
#                # --debug \
#             #    --ema4eval \
#             #    --save_pred \

EXP=cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_t1
python eval.py --dataset_name=cubicasa \
               --dataset_root=data/coco_cubicasa5k_nowalls_v4-1_refined/ \
               --eval_set=test \
               --checkpoint=/home/htp26/RoomFormerTest/output/${EXP}/checkpoint1899.pth \
               --output_dir=slurm_scripts/${EXP} \
               --num_queries=2800 \
               --num_polys=50 \
               --semantic_classes=12 \
               --input_channels 3 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --disable_poly_refine \
               --dec_attn_concat_src \
               --ema4eval \
               --per_token_sem_loss \
               --use_anchor \
               # --pre_decoder_pos_embed \
            #    --save_pred \
               # --debug \