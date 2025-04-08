# for folder in train val test;
# do
#     echo $folder
#     python combine_json.py \
#         --input /share/elor/htp26/floorplan_datasets/coco_cubicasa5k_nowalls_v4_refined/ \
#         --output /share/elor/htp26/floorplan_datasets/coco_cubicasa5k_nowalls_v4_refined/annotations/${folder}.json \
#         --split ${folder}

# done


python combine_json.py \
    --input /share/elor/htp26/floorplan_datasets/coco_cubicasa5k_nowalls_v4-1_refined/ \
    --output /share/elor/htp26/floorplan_datasets/coco_cubicasa5k_nowalls_v4-1_refined/annotations/ \