#!/usr/bin/env bash

EXP=r2g_orgin_ckpt
python eval_from_json.py --dataset_name=r2g \
               --dataset_root=data/R2G_hr_dataset_processed_v1 \
               --eval_set=test \
               --output_dir=slurm_scripts4/${EXP}/eval \
               --semantic_classes=12 \
               --input_channels 3 \
               --input_json_dir /home/htp26/Raster-to-Graph/output/r2g_org_test_2/jsons/ \
               --num_workers 0 \
               --device cpu \
               --image_size 512 \
               # --debug \
            #    --save_pred \
            #    --debug \