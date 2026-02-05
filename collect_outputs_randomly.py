import os
from glob import glob
import numpy as np
import shutil
import json

dataset_name = ["Raster2Graph", "CubiCasa5K", "WAFFLE", "Structured3D"][-1]
output_dir = f"web_data_100/{dataset_name}"

if dataset_name == "CubiCasa5K":
    gt_path = "output_gt_cc5k_refined_v4-1_wd/test"
    input_data_dirs = [
        "cc5k_test_preds_wd/cubi_v4-1refined_queries56x50_sem_v1/",
        "cc5k_test_preds_wd/cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_convertv3_fromnorder499_t1/",
        "cc5k_test_preds_wd/cc5k_frinet_nowd_256_ckpt/",
    ]
    method_names = ["RoomFormer", "Raster2Seq", "FRI-Net"]
    id_path = "web_data/cc5k_select_ids.txt"
    input_json_dirs = [
        "slurm_scripts4/cubi_v4-1refined_queries56x50_sem_v1/test/result_jsons/",
        "slurm_scripts4/cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_raster1_anchor_deccatsrc_smoothing_cls12_convertv3_fromorder499_t1/eval_cc5k_epoch0499/result_jsons/",
        "slurm_scripts4/cc5k_frinet_nowd_256_ckpt_test/test/result_jsons/",
    ]
    num_zero_pad = 5
elif dataset_name == "Raster2Graph":
    gt_path = "output_gt_r2g/test/"
    input_data_dirs = [
        "r2g_test_preds/r2g_queries56x50_sem13",
        "r2g_test_preds/r2g_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls13_convertv3_from849_t1/",
        "r2g_test_preds/r2g_res256_vis",
    ]
    method_names = ["RoomFormer", "Raster2Seq", "Raster2Graph"]
    input_json_dirs = [
        "slurm_scripts4/r2g_queries56x50_sem13/eval_epoch0799/result_jsons/",
        "slurm_scripts4/r2g_res512_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls13_convertv3_frompretrainckpt_t1/eval_r2g_epoch0549/result_jsons/",
        "slurm_scripts4/r2g_res256_ckpt/eval/result_jsons/",
    ]
    id_path = "web_data/r2g_select_ids.txt"
    num_zero_pad = 6
elif dataset_name == "WAFFLE":
    gt_path = "waffle_raster00000_preds/cubi_v4-1refined_queries56x50_sem_v1/"
    input_data_dirs = [
        "waffle_raster00000_preds/cubi_v4-1refined_queries56x50_sem_v1/",
        "waffle_raster00000_preds/cubi_v4-1refined_poly2seq_l512_bin32_sem1_coo20_cls5_anchor_deccatsrc_smoothing_cls12_convertv3_fromnorder499_t1/",
    ]
    method_names = [
        "RoomFormer",
        "Raster2Seq",
    ]
    input_json_dirs = None
    id_path = "web_data/waffle_select_ids.txt"
    num_zero_pad = 9
elif dataset_name == "Structured3D":
    gt_path = "output_gt_s3dbw_wd/"
    input_data_dirs = [
        "s3d_test_preds_wd/s3d_bw_ddp_queries40x30/",
        "s3d_test_preds_wd/s3d_bw_poly2seq_l512_sem_bs32_coo20_cls1_anchor_deccatsrc_smoothing_numcls19_pts_fromnorder1399_convertv3_t2/",
    ]
    method_names = [
        "RoomFormer",
        "Raster2Seq",
        # "FRI-Net"
    ]
    # id_path = "web_data/cc5k_select_ids.txt"
    input_json_dirs = [
        # 'slurm_scripts/s3d_bw_ddp_queries40x30/result_jsons/',
        "slurm_scripts/s3d_bw_ddp_queries40x30_2/result_jsons/",
        "slurm_scripts/s3d_bw_poly2seq_l512_sem_bs32_coo20_cls1_anchor_deccatsrc_smoothing_numcls19_pts_fromnorder1399_convertv3_t2/s3d_bw_poly2seq_l512_sem_bs32_coo20_cls1_anchor_deccatsrc_smoothing_numcls19_pts_fromnorder1399_convertv3_t2_0449/result_jsons",
    ]
    num_zero_pad = 5


# with open(id_path, 'r') as f:
#     ids = f.read().splitlines()

# "room_f1": 0.8947295014175665, "corner_f1": 0.6465464755879026, "angles_f1": 0.4568913849378508, "room_sem_f1": 0.7894666205341155, "window_door_f1": 0.7792147883624705

random_idxs = [
    47,
    117,
    192,
    323,
    251,
    195,
    359,
    9,
    211,
    277,
    242,
    292,
    87,
    70,
    88,
    396,
    314,
    193,
    39,
    87,
    174,
    88,
    337,
    165,
    25,
    333,
    72,
    265,
    115,
    243,
    197,
    335,
    338,
    99,
    177,
    243,
    285,
    147,
    147,
    398,
    288,
    265,
    185,
    127,
    32,
    31,
    202,
    244,
    151,
    163,
    370,
    183,
    28,
    290,
    128,
    128,
    53,
    389,
    38,
    244,
    273,
    335,
    388,
    105,
    42,
    31,
    376,
    257,
    321,
    57,
    291,
    358,
    119,
    267,
    82,
    91,
    384,
    398,
    99,
    53,
    396,
    121,
    84,
    203,
    324,
    262,
    47,
    127,
    131,
    356,
    180,
    334,
    143,
    148,
    227,
    279,
    207,
    397,
    373,
    341,
]

# max_samples = 250 # len(ids)
tmp = sorted(glob(f"{input_data_dirs[1]}/*_pred_room_map.png"))  # [-max_samples:]
# Create IDs from filenames
ids = [os.path.basename(f).split("_")[0] for f in tmp]
if dataset_name == "Structured3D":
    random_idxs = [i for i in random_idxs if i < 240]
ids = [ids[i] for i in random_idxs]  # take the first 100 random samples

if dataset_name == "WAFFLE":
    gt_files = [f"{gt_path}/{_id.zfill(9)}.png" for _id in ids]
# elif dataset_name == 'Structured3D':
#     gt_files = [f'{gt_path}/{_id.zfill(9)}.png' for _id in ids]
else:
    gt_files = [f"{gt_path}/{int(_id)}_gt_image.png" for _id in ids]


save_dir = os.path.join(output_dir, "GT")
os.makedirs(save_dir, exist_ok=True)
for idx, _path in enumerate(gt_files):
    print(_path)
    shutil.copy(_path, os.path.join(save_dir, f"{str(idx).zfill(6)}.png"))

for i, (data_dir, method_name) in enumerate(zip(input_data_dirs, method_names)):
    # method_name = data_dir.split('/')[-2]
    print(method_name)
    pred_files = [os.path.join(data_dir, f"{_id.zfill(num_zero_pad)}_pred_floorplan.png") for _id in ids]
    # ids = [os.path.basename(f).split('_')[0] for f in tmp]
    # map_files = [f'{data_dir}/{_id}_pred_room_map.png' for _id in ids]
    # floor_files = [f'{data_dir}/{_id}_pred_floorplan.png' for _id in ids]

    save_dir = os.path.join(output_dir, method_name)
    os.makedirs(save_dir, exist_ok=True)

    json_output = {"items": []}
    for j, _path in enumerate(pred_files):
        print(_path)
        if input_json_dirs is not None:
            json_file = os.path.join(input_json_dirs[i], ids[j][-5:] + ".json")

            with open(json_file, "r") as f:
                results = json.load(f)
            key_results = {
                "RoomF1": round(results["room_f1"] * 100, 1),
                "CornerF1": round(results["corner_f1"] * 100, 1),
                "AngleF1": round(results["angles_f1"] * 100, 1),
            }
            if "room_sem_f1" in results:
                key_results["RoomSemF1"] = round(results["room_sem_f1"] * 100, 1)
            if "window_door_f1" in results:
                key_results["WindowDoorF1"] = round(results["window_door_f1"] * 100, 1)
                if key_results["WindowDoorF1"] == 0:
                    key_results["WindowDoorF1"] = "N/A"

        else:
            key_results = None
        json_output["items"].append(
            {
                "input_image": f"{str(j).zfill(6)}.png",
                "output_image": f"{str(j).zfill(6)}.png",
                "results": key_results,
            }
        )
        shutil.copy(_path, os.path.join(save_dir, f"{str(j).zfill(6)}.png"))

    with open(os.path.join(save_dir, f"manifest.json"), "w") as f:
        json.dump(json_output, f, indent=2)
