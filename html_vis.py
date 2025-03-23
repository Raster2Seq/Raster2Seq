import os
from glob import glob
from html4vision import Col, imagetable

CLASS2ID = {'living room': 0, 'kitchen': 1, 'bedroom': 2, 'bathroom': 3, 'balcony': 4, 'corridor': 5,
            'dining room': 6, 'study': 7, 'studio': 8, 'store room': 9, 'garden': 10, 'laundry room': 11,
            'office': 12, 'basement': 13, 'garage': 14, 'undefined': 15, 'door': 16, 'window': 17}

root_path = "output/cubi_queries50x30_sem_t1/eval_cubi/"
org_data_path = "output_gt_cubi/"
max_samples = 100

# Create elements from directory of images
originals = sorted(glob(f'{root_path}/*_gt.png'))[:max_samples]
# Create IDs from filenames
ids = [os.path.basename(f).split('_')[0] for f in originals]

gt_images = [f'{org_data_path}/{img_id}_gt_image.png' for img_id in ids]

results_A = [f'{root_path}/{img_id}_pred_room_map.png' for img_id in ids] 
results_B = [f'{root_path}/{img_id}_sem_rich_pred.png' for img_id in ids] # sorted(glob(f'{root_path}/*_pred.png'))

# Create IDs from filenames
ids = [os.path.basename(f).split('_')[0] for f in originals]

# table description
cols = [
    Col('id1', 'ID', ids),                                               # make a column of 1-based indices
    Col('img', 'GT Raster', gt_images),             # specify image content for column 2
    Col('img', 'GT Vector', originals),             # specify image content for column 2
    Col('img', 'Pred', results_A),     # specify image content for column 3
    Col('img', 'Pred Vector', results_B), # specify image content for column 4
    Col('text', 'Class Label', [str(CLASS2ID)] * max_samples)
]


# html table generation
imagetable(cols, out_file='pred_vis.html', imsize=(256, 256))