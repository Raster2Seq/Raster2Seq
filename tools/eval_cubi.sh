#!/usr/bin/env bash

python eval.py --dataset_name=cubicasa \
               --dataset_root=data/coco_cubicasa5k_nowalls_v4-1_refined/ \
               --eval_set=test \
               --checkpoint=/home/htp26/RoomFormerTest/output/cubi_v4-1refined_queries56x50_sem_v1/checkpoint0499.pth \
               --output_dir=eval_cubi \
               --num_queries=2800 \
               --num_polys=50 \
               --semantic_classes=12 \
               --input_channels 3 \