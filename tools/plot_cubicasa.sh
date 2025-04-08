#!/usr/bin/env bash

python plot_floor.py --dataset_name=cubicasa \
               --dataset_root=data/coco_cubicasa5k_nowalls_v4-1_refined \
               --eval_set=val \
               --output_dir=output_gt_cc5k_refined \
               --semantic_classes=12 \
               --input_channels 3 \
               --poly2seq \
               --seq_len 1024 \
               --num_bins 64
               # --plot_density \
               # --plot_gt \
            #    --plot_gt_image \
               # --disable_image_transform \