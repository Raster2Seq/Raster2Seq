# train 
for folder in val test;
do
    echo $folder
    python combine_json.py \
        --input /share/elor/htp26/floorplan_datasets/coco_cubicasa5k_nowalls_v2/annotations_json/${folder}/ \
        --output /share/elor/htp26/floorplan_datasets/coco_cubicasa5k_nowalls_v2/annotations/${folder}.json

done