#!/usr/bin/env bash

# python eval.py --dataset_name=stru3d \
#                --dataset_root=data/stru3d_sem_rich \
#                --eval_set=test \
#                --checkpoint=output/stru3d_org_org \
#                --output_dir=eval_stru3d \
#                --num_queries=800 \
#                --num_polys=20 \
#                --semantic_classes=-1 


python eval.py --dataset_name=stru3d \
               --dataset_root=data/coco_s3d_bw \
               --eval_set=test \
               --checkpoint=output/s3d_bw_ddp_poly2seq_l512_bin16_nosem_bs32_coo20_cls1_nopolyrefine_predecPE_deccatsrc_t1/checkpoint1349.pth \
               --output_dir=eval_s3d_nosem \
               --num_queries=1200 \
               --num_polys=30 \
               --semantic_classes=-1 \
               --input_channels=3 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --disable_poly_refine \
               --dec_attn_concat_src \
               --ema4eval \
               --pre_decoder_pos_embed \
               # --use_anchor \
               # --dec_layer_type='v5' \
            # #    --batch_size 1 \
               # --measure_time \
               # --batch_size 1 \
            #    --disable_sampling_cache