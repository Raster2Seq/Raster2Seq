#!/usr/bin/env bash

# python eval.py --dataset_name=stru3d \
#                --dataset_root=data/stru3d \
#                --eval_set=test \
#                --checkpoint=output/stru3d_bs10_org_ddp_t2/checkpoint.pth \
#                --output_dir=eval_stru3d \
#                --num_queries=800 \
#                --num_polys=20 \
#                --semantic_classes=-1 

EXP=s3d_projection_ddp_poly2seq_l512_bin32_nosem_bs32_coo20_cls2_anchor_deccatsrc_converterv3_t1/
python eval.py --dataset_name=stru3d \
               --dataset_root=data/stru3d \
               --eval_set=test \
               --checkpoint=output/${EXP}/checkpoint0499.pth \
               --output_dir=slurm_scripts/${EXP} \
               --num_queries=1200 \
               --num_polys=30 \
               --semantic_classes=-1 \
               --input_channels 1 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --disable_poly_refine \
               --dec_attn_concat_src \
               --ema4eval \
               --use_anchor \
            #    --pre_decoder_pos_embed \


