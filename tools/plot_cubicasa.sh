#!/usr/bin/env bash

python plot_floor.py --dataset_name=cubicasa \
               --dataset_root=/share/kuleshov/htp26/floorplan_datasets/coco_cubicasa5k/ \
               --eval_set=test \
               --output_dir=output_gt_cubi \
               --semantic_classes=19 \
               --input_channels 3 \
               --plot_gt_image \
            #    --plot_density
               # --plot_gt \
               # --disable_image_transform \