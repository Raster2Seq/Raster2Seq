#!/usr/bin/env bash

python eval.py --dataset_name=stru3d \
               --dataset_root=data/stru3d_sem_rich \
               --eval_set=test \
               --checkpoint=output/stru3d_org_org \
               --output_dir=eval_stru3d \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 
