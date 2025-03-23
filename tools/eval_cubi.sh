#!/usr/bin/env bash

python eval.py --dataset_name=cubicasa \
               --dataset_root=data/coco_cubicasa5k \
               --eval_set=test \
               --checkpoint=output/cubi_queries50x30_sem_t1/checkpoint0439.pth \
               --output_dir=eval_cubi \
               --num_queries=1500 \
               --num_polys=30 \
               --semantic_classes=19 \
               --input_channels 3 \
            #    --debug