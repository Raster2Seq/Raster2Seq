import os
from glob import glob
from html4vision import Col, imagetable

# CLASS2ID = {'living room': 0, 'kitchen': 1, 'bedroom': 2, 'bathroom': 3, 'balcony': 4, 'corridor': 5,
#             'dining room': 6, 'study': 7, 'studio': 8, 'store room': 9, 'garden': 10, 'laundry room': 11,
#             'office': 12, 'basement': 13, 'garage': 14, 'undefined': 15, 'door': 16, 'window': 17}

max_samples = 100
path_A = "output/cubi_v4-1refined_queries56x50_sem_v1/eval_cubi/"
# path_B = "output/cubi_v4-1refined_poly2seq_l512_bin32_sem_coo20_cls5_nopolyrefine_predecPE_fromckpt_ignorewd_smoothing_t1/eval_cubi/"
path_B = "slurm_scripts/cubi_v4-1refined_poly2seq_l512_bin32_nosem_coo20_cls1_anchor_deccatsrc_fromckpt2450_ignorewd_smoothing_clscoeffx5@6e-1_t1/eval_cc5k_epoch1899/"

results_A = sorted(glob(f'{path_A}/*_pred_room_map.png'))[:max_samples]
results_B = sorted(glob(f'{path_B}/*_pred_room_map.png'))[:max_samples]
results_A2 = sorted(glob(f'{path_A}/*_pred_floorplan.png'))[:max_samples]
results_B2 = sorted(glob(f'{path_B}/*_pred_floorplan.png'))[:max_samples]

# Create IDs from filenames
ids = [os.path.basename(f).split('_')[0] for f in results_A]

# table description
cols = [
    Col('id1', 'ID', ids),                                               # make a column of 1-based indices
    Col('img', 'RoomFormer', results_A),     # specify image content for column 3
    Col('img', 'RoomFormer Map', results_A2),     # specify image content for column 3
    Col('img', 'Poly2Seq', results_B), # specify image content for column 4
    Col('img', 'Poly2Seq Map', results_B2),     # specify image content for column 3
]

# html table generation
imagetable(cols, out_file='baseline_vis.html', imsize=(768, 768))