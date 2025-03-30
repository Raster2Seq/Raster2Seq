for folder in train val test;
do
    echo $folder
    python combine_json.py \
        --input /share/elor/htp26/floorplan_datasets/coco_cubicasa5k_nowalls_v3_refined/annotations_json/${folder}/ \
        --output /share/elor/htp26/floorplan_datasets/coco_cubicasa5k_nowalls_v3_refined/annotations/${folder}.json

done