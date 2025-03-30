import os
from glob import glob
from html4vision import Col, imagetable

org_root_path = "data/coco_cubicasa5k_nowalls_v3"
root_path = "data/coco_cubicasa5k_nowalls_v3_refined"
split = 'train'
max_samples = 100

# Create elements from directory of images
gt_images = sorted(glob(f'{root_path}/{split}/*.png'))[:max_samples]

# Create IDs from filenames
ids = [os.path.basename(f).split('.')[0] for f in gt_images]
org_ids = [os.path.basename(f).split('.')[0].split('_')[1] for f in gt_images]

col1 = [f'{org_root_path}/{split}/{img_id}.png' for img_id in org_ids]
col2 = [f'{root_path}/{split}_aux/{img_id}_polylines.png' for img_id in org_ids]
col3 = [f'{root_path}/{split}_aux/{img_id}_org_floor.png' for img_id in org_ids] 
col4 = [f'{root_path}/{split}_aux/{img_id}_floor.png' for img_id in ids] 
col5 = [f'{root_path}/{split}/{img_id}.png' for img_id in ids]

# table description
cols = [
    Col('id1', 'ID', ids),                             # make a column of 1-based indices
    Col('img', 'Raster (org)', col1),             # specify image content for column 2
    Col('img', 'Floormap (org)', col3),             # specify image content for column 2
    Col('img', 'Polylines', col2),             # specify image content for column 2
    Col('img', 'Floormap (after)', col4),    # specify image content for column 2
    Col('img', 'Raster (after)', col5),    # specify image content for column 2
    # Col('text', 'Class Label', [str(CLASS2ID)] * max_samples)
]

# html table generation
imagetable(cols, out_file='gt_vis_2.html', imsize=[512,512])