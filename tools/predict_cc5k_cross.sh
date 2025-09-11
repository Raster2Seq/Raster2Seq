EXP=cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_convertv3_fromnorder499_t1/
CKPT=output/${EXP}/checkpoint0499.pth

DATA=data/R2G_hr_dataset_processed_v1
SPLIT=test 
NAME=cc5kmodel_r2g_${SPLIT}_preds
SAVE_DIR=cross_eval_out/${NAME}

python predict.py \
               --dataset_root=${DATA}/${SPLIT} \
               --checkpoint=${CKPT} \
               --output_dir=${SAVE_DIR} \
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
               --one_color \
            #    --crop_white_space \

python eval_from_json.py --dataset_name=r2g \
               --dataset_root=${DATA} \
               --eval_set=${SPLIT} \
               --output_dir=${SAVE_DIR}/eval \
               --semantic_classes=-1 \
               --input_channels 3 \
               --input_json_dir ${SAVE_DIR}/${EXP}/jsons/ \
               --num_workers 0 \
               --device cpu \
               --image_size 256 \
