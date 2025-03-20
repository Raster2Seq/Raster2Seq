#!/usr/bin/env bash

python eval.py --dataset_name=cubicasa \
               --dataset_root=data/coco_cubicasa5k \
               --eval_set=test \
               --checkpoint=output/s3d_bw_ddp_queries40x30/checkpoint0499.pth \
               --output_dir=eval_stru3d_sem_rich \
               --num_queries=1200 \
               --num_polys=30 \
               --semantic_classes=19 \
               --input_channels 3 \
               # --debug