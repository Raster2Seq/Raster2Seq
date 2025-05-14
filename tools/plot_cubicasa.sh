#!/usr/bin/env bash

SPLIT=val
python plot_floor.py --dataset_name=cubicasa \
               --dataset_root=data/coco_cubicasa5k_nowalls_v4-2_refined \
               --eval_set=${SPLIT} \
               --output_dir=/share/elor/htp26/roomformer/output_gt_cc5k_refined_v4-2/${SPLIT} \
               --semantic_classes=-1 \
               --input_channels 3 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --plot_gt \
               --disable_image_transform \
               # --plot_density \
            #    --plot_gt_image \