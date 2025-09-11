# python plot_from_json2.py --dataset_name r2g \
#     --dataset_root /share/elor/htp26/floorplan_datasets/R2G_dataset/ \
#     --eval_set test \
#     --json_root /home/htp26/Raster-to-Graph/output/r2g_data_lowres_v2/jsons \
#     --save_dir r2g_vis \
#     --image_size 512 \
#     --crop_white_space

# python plot_from_json2.py --dataset_name r2g \
#     --dataset_root data/R2G_hr_dataset_processed_v1/ \
#     --eval_set test \
#     --json_root /home/htp26/Raster-to-Graph/output/r2g_res256_test/jsons \
#     --save_dir r2g_res256_vis \
#     --image_size 512 \
#     --crop_white_space


python plot_from_json2.py --dataset_name r2g \
    --dataset_root data/R2G_hr_dataset_processed_v1/ \
    --eval_set test \
    --json_root /home/htp26/heat/results/npy_heat_r2g_256/test/ \
    --save_dir r2g_test_preds/heat_r2g_256_vis \
    --image_size 512 \
    --crop_white_space \
    --one_color