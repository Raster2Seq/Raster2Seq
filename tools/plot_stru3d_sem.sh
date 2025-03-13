#!/usr/bin/env bash

# python plot_floor.py --dataset_name=stru3d \
#                --dataset_root=data/stru3d \
#                --eval_set=test \
#                --output_dir=/share/kuleshov/htp26/floorplan_datasets/stru3d_sem_rich/test \
#                --semantic_classes=19 \
#                --input_channels 1 \
#                --disable_image_transform

python plot_floor.py --dataset_name=stru3d \
               --dataset_root=/share/kuleshov/htp26/floorplan_datasets/stru3d_sem_rich/ \
               --eval_set=train \
               --output_dir=plot_poly_outputs \
               --semantic_classes=19 \
               --input_channels 3 \
               --disable_image_transform \
               --plot_polys