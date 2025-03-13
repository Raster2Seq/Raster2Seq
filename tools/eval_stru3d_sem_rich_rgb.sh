#!/usr/bin/env bash

python eval.py --dataset_name=stru3d \
               --dataset_root=data/stru3d_sem_rich \
               --eval_set=test \
               --checkpoint=output/s3d_sem_rgb_ddp_queries40x30/checkpoint.pth \
               --output_dir=eval_stru3d_sem_rich \
               --num_queries=1200 \
               --num_polys=30 \
               --semantic_classes=19 \
               --input_channels 3 \
            #    --debug