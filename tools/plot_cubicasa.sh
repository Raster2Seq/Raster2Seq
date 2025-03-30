#!/usr/bin/env bash

python plot_floor.py --dataset_name=cubicasa \
               --dataset_root=data/coco_cubicasa5k_nowalls_v3/ \
               --eval_set=train \
               --output_dir=output_gt_cc5k \
               --semantic_classes=12 \
               --input_channels 3 \
            #    --debug
            #    --plot_density \
            #    --plot_gt \
            #    --plot_gt_image \
               # --disable_image_transform \