for folder in train val test;
do
    echo $folder
    python combine_json.py \
        --input /home/htp26/fp_datasets/coco_cubicasa5k_org/annotations_json/${folder}/ \
        --output /home/htp26/fp_datasets/coco_cubicasa5k_org/annotations/${folder}.json

done