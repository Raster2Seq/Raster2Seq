#!/usr/bin/env bash

# python plot_floor.py --dataset_name=stru3d \
#                --dataset_root=data/stru3d \
#                --eval_set=test \
#                --output_dir=/share/kuleshov/htp26/floorplan_datasets/stru3d_sem_rich/test \
#                --semantic_classes=19 \
#                --input_channels 1 \
#                --disable_image_transform

python plot_floor.py --dataset_name=stru3d \
               --dataset_root=data/coco_s3d_bw/ \
               --eval_set=test \
               --output_dir=output_gt_s3dbw/ \
               --semantic_classes=19 \
               --input_channels 3 \
               --disable_image_transform \
               --poly2seq \
               --seq_len 1024 \
               --num_bins 64
               # --plot_gt \
            #    --debug
            #    --plot_polys