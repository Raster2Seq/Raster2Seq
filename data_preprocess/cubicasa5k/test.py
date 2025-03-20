import os
import sys
from pathlib import Path

import argparse

from tqdm import tqdm
import shutil
import json
import numpy as np
import cv2
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
from multiprocessing import Pool

from loaders import FloorplanSVG, ROOM_NAMES

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from util.plot_utils import plot_semantic_rich_floorplan_tight

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common_utils import resort_corners
from stru3d.stru3d_utils import type2id


# ROOM_NAMES = {
#     0: "Background", 
#     1: "Outdoor", 
#     2: "Wall", 
#     3: "Kitchen", 
#     4: "Living Room",
#     5: "Bed Room", 
#     6: "Bath", 
#     7: "Entry", 
#     8: "Railing", 
#     9: "Storage", 
#     10: "Garage", 
#     11: "Undefined"
# }


CUBICASA_TO_S3D_MAPPING = {
    0: None,  # "Background" has no direct match
    1: type2id['balcony'],  # "Outdoor" -> balcony (4)
    2: None,  # "Wall" has no direct match
    3: type2id['kitchen'],  # Kitchen -> kitchen (1)
    4: type2id['living room'],  # Living Room -> living room (0)
    5: type2id['bedroom'],  # Bed Room -> bedroom (2)
    6: type2id['bathroom'],  # Bath -> bathroom (3)
    7: type2id['corridor'],  # Entry -> corridor (5) as closest match
    8: None,  # "Railing" has no direct match, it is a subset of balcony
    9: type2id['store room'],  # Storage -> store room (9)
    10: type2id['garage'],  # Garage -> garage (14)
    11: type2id['undefined'],  # Undefined -> undefined (15)
    12: type2id['window'], # Window -> window (17)
    13: type2id['door'], # Door -> door (16) 
}


def extract_room_polygons_cv2(mask, skip_classes=[]):
    room_ids = np.unique(mask)
    room_ids = room_ids[room_ids != 0]
    
    room_polygons = []
    
    for room_id in room_ids:
        # skip wall
        if room_id in skip_classes:
            continue
        # Create binary mask for this room
        room_mask = (mask == room_id).astype(np.uint8)
        
        # Find contours using OpenCV
        contours, _ = cv2.findContours(
            room_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            polygon = [tuple(point[0]) for point in largest_contour]
            if len(polygon) < 3:
                continue
            poly = Polygon(polygon)
            simplified_poly = poly.simplify(tolerance=0.5, preserve_topology=True)
            simplified_poly = list(simplified_poly.exterior.coords)
            room_polygons.append([simplified_poly, int(room_id)])

            # # Optional: Simplify polygon
            # epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            # approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # # Convert to list of (x, y) tuples
            # polygon = [tuple(point[0]) for point in polygon]
            
            # room_polygons[int(room_id)] = polygon
    
    return room_polygons

def extract_icon_cv2(mask):
    # room_ids = np.unique(mask)
    room_ids = [1, 2] # window, door
    room_polygons = []
    new_mask = np.zeros(mask.shape)
    
    for room_id in room_ids:
        # window, door
        # if int(room_id) not in [1, 2]:
        #     continue
        true_room_id = int(room_id)+11
        # Create binary mask for this room
        room_mask = (mask == room_id).astype(np.uint8)
        new_mask = np.where(room_mask, true_room_id, 0)
        
        # Find contours using OpenCV
        contours, _ = cv2.findContours(
            room_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # # Get the largest contour
            # largest_contour = max(contours, key=cv2.contourArea)
            for cnt in contours:
                polygon = [tuple(point[0]) for point in cnt]
                if len(polygon) < 3:
                    continue

                poly = Polygon(polygon)
                simplified_poly = poly.simplify(tolerance=0.5, preserve_topology=True)
                simplified_poly = list(simplified_poly.exterior.coords)
                room_polygons.append([simplified_poly, true_room_id])

    return room_polygons, new_mask



def visualize_room_polygons(mask, room_polygons, figsize=(10, 10), save_path='cubicasa_debug.jpg'):
    """
    Visualize the extracted room polygons.
    
    Args:
        mask: Original segmentation mask
        room_polygons: Dictionary of room polygons as returned by extract_room_polygons
        figsize: Figure size for the plot
    """
    # # Set figure size to exactly 256x256 pixels
    # dpi = 100  # Standard screen DPI
    # figsize = (256/dpi, 256/dpi)  # Convert pixels to inches

    plt.figure(figsize=figsize)
    
    # Show the original mask
    plt.imshow(mask, cmap='nipy_spectral', interpolation='nearest', alpha=0.6)
    
    # Plot each room polygon
    for room_id, polygon in room_polygons:
        polygon_array = np.array(polygon)
        plt.plot(polygon_array[:, 0], polygon_array[:, 1], 'k-', linewidth=2)
        
        # # Add room ID label at the centroid
        # centroid_x = np.mean(polygon_array[:, 0])
        # centroid_y = np.mean(polygon_array[:, 1])
        # plt.text(centroid_x, centroid_y, str(room_id), 
        #          fontsize=12, ha='center', va='center',
        #          bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title('Room Polygons Extracted from Segmentation Mask')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # plt.show()

def config():
    a = argparse.ArgumentParser(description='Generate coco format data for Structured3D')
    a.add_argument('--data_root', default='Structured3D_panorama', type=str, help='path to raw Structured3D_panorama folder')
    a.add_argument('--output', default='coco_cubicasa5k', type=str, help='path to output folder')
    
    args = a.parse_args()
    return args

def detele_iccfile(image_path, output_path):
    '''
    ref: https://github.com/ultralytics/ultralytics/issues/339
    '''
    img = Image.open(image_path).convert('RGB')
    img.info.pop('icc_profile', None)
    img.save(output_path)

def process_floorplan(image_set, scene_id, start_scene_id, args, save_dir, annos_folder):
    # image_set = dataset[scene_id]

    mask = image_set['label'].numpy()
    room_polygons = extract_room_polygons_cv2(mask[0], skip_classes=[2])
    icon_polygons, icon_mask = extract_icon_cv2(mask[1])

    combined_mask = mask[0]
    combined_mask = np.where(icon_mask != 0, icon_mask, combined_mask)
    # combined_mask[icon_mask != 0] = icon_mask[icon_mask != 0]
    room_polygons.extend(icon_polygons)

    # visualize_room_polygons(combined_mask, room_polygons, save_path='cubicasa_combined_debug4.jpg')
    # visualize_room_polygons(icon_mask, room_polygons, save_path='cubicasa_icon_debug3.jpg')
    
    image_height, image_width = mask.shape[1:]
    new_polygon_list = []
    coco_annotation_dict_list = []

    # for storing
    save_dict = prepare_dict()

    instance_id = 0
    img_id = int(scene_id) + start_scene_id
    img_dict = {}
    img_dict["file_name"] = str(img_id).zfill(5) + '.jpg'
    img_dict["id"] = img_id 
    img_dict["width"] = image_width
    img_dict["height"] = image_height

    detele_iccfile(f"{args.data_root}/{image_set['folder']}/F1_scaled.png", f"{save_dir}/{str(img_id).zfill(5) + '.jpg'}")
    # shutil.copy(f"{args.data_root}/{image_set['folder']}/F1_scaled.png", f"{save_dir}/{str(img_id).zfill(5) + '.png'}")

    for poly_ind, (polygon, poly_type) in enumerate(room_polygons):
        poly_shapely = Polygon(polygon)
        area = poly_shapely.area
        
        poly_type = CUBICASA_TO_S3D_MAPPING[poly_type]
        if poly_type is None:
            continue

        # assert area > 10
        # if area < 100:
        # 'door', 'window'
        if poly_type not in [16, 17] and area < 100:
            continue
        if poly_type in [16, 17] and area < 1:
            continue
        
        rectangle_shapely = poly_shapely.envelope
        polygon = np.array(polygon)

        ### here we convert door/window annotation into a single line
        if poly_type in [16, 17]:
            # convert to rect
            if polygon.shape[0] > 4:
                min_x = np.min(polygon[:, 0])
                max_x = np.max(polygon[:, 0])
                min_y = np.min(polygon[:, 1])
                max_y = np.max(polygon[:, 1])

                # The bounding rectangle
                bounding_rect = np.array([
                    [min_x, min_y],  # top-left
                    [min_x, max_y],  # bottom-left
                    [max_x, max_y],  # bottom-right
                    [max_x, min_y],  # top-right
                    # [min_x, min_y]   # back to start (closed shape)
                ])
                polygon = bounding_rect

            assert polygon.shape[0] == 4
            midp_1 = (polygon[0] + polygon[1])/2
            midp_2 = (polygon[1] + polygon[2])/2
            midp_3 = (polygon[2] + polygon[3])/2
            midp_4 = (polygon[3] + polygon[0])/2

            dist_1_3 = np.square(midp_1 -midp_3).sum()
            dist_2_4 = np.square(midp_2 -midp_4).sum()
            if dist_1_3 > dist_2_4:
                polygon = np.row_stack([midp_1, midp_3])
            else:
                polygon = np.row_stack([midp_2, midp_4])


        coco_seg_poly = []
        poly_sorted = resort_corners(polygon)
        # image = draw_polygon_on_image(image, poly_shapely, "test_poly.jpg")

        for p in poly_sorted:
            coco_seg_poly += list(p)

        # Slightly wider bounding box
        bound_pad = 2
        bb_x, bb_y = rectangle_shapely.exterior.xy
        bb_x = np.unique(bb_x)
        bb_y = np.unique(bb_y)
        bb_x_min = np.maximum(np.min(bb_x) - bound_pad, 0)
        bb_y_min = np.maximum(np.min(bb_y) - bound_pad, 0)

        bb_x_max = np.minimum(np.max(bb_x) + bound_pad, image_width - 1)
        bb_y_max = np.minimum(np.max(bb_y) + bound_pad, image_height - 1)

        bb_width = (bb_x_max - bb_x_min)
        bb_height = (bb_y_max - bb_y_min)

        coco_bb = [bb_x_min, bb_y_min, bb_width, bb_height]

        coco_annotation_dict = {
                "segmentation": [coco_seg_poly],
                "area": area,
                "iscrowd": 0,
                "image_id": img_id,
                "bbox": coco_bb,
                "category_id": poly_type,
                "id": instance_id}
        coco_annotation_dict_list.append(coco_annotation_dict)
        instance_id += 1


        # # modified for plotting
        # corners = polygon
        # corners_flip_y = corners.copy()
        # corners_flip_y[:,1] = image_height - corners_flip_y[:,1] - 1
        # corners = corners_flip_y
        # new_polygon_list.append([corners, poly_type])
    
    #### end split_file loop
    save_dict['images'].append(img_dict)
    save_dict["annotations"] += coco_annotation_dict_list


    json_path = f"{annos_folder}/{str(img_id).zfill(5) + '.json'}"
    with open(json_path, 'w') as f:
        json.dump(save_dict, f)

    # save_path = f"{save_dir}/plot_debug.jpg"
    # plot_semantic_rich_floorplan_tight(new_polygon_list, save_path, prec=1, rec=1, plot_text=False, is_bw=True, 
    #                                    img_w=image_width, img_h=image_height)

    # return save_dict


def prepare_dict():
    save_dict = {"images":[],"annotations":[],"categories":[]}
    for key, value in type2id.items():
        type_dict = {"supercategory": "room", "id": value, "name": key}
        save_dict["categories"].append(type_dict)
    return save_dict


if __name__ == '__main__':
    args = config()

    # data_folder = '/share/kuleshov/htp26/floorplan_datasets/cubicasa5k/'
    # data_file = 'test.txt'


    ### prepare
    outFolder = args.output
    if not os.path.exists(outFolder):
        os.mkdir(outFolder)

    annotation_outFolder = os.path.join(outFolder, 'annotations_json')
    if not os.path.exists(annotation_outFolder):
        os.mkdir(annotation_outFolder)
    
    annos_train_folder = os.path.join(annotation_outFolder, 'train')
    annos_val_folder = os.path.join(annotation_outFolder, 'val')
    annos_test_folder = os.path.join(annotation_outFolder, 'test')
    os.makedirs(annos_train_folder, exist_ok=True)
    os.makedirs(annos_val_folder, exist_ok=True)
    os.makedirs(annos_test_folder, exist_ok=True)

    train_img_folder = os.path.join(outFolder, 'train')
    val_img_folder = os.path.join(outFolder, 'val')
    test_img_folder = os.path.join(outFolder, 'test')

    for img_folder in [train_img_folder, val_img_folder, test_img_folder]:
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

    coco_train_json_path = os.path.join(annotation_outFolder, 'train.json')
    coco_val_json_path = os.path.join(annotation_outFolder, 'val.json')
    coco_test_json_path = os.path.join(annotation_outFolder, 'test.json')

    # coco_train_dict = {"images":[],"annotations":[],"categories":[]}
    # coco_val_dict = {"images":[],"annotations":[],"categories":[]}
    # coco_test_dict = {"images":[],"annotations":[],"categories":[]}

    # for key, value in type2id.items():
    #     type_dict = {"supercategory": "room", "id": value, "name": key}
    #     coco_train_dict["categories"].append(type_dict)
    #     coco_val_dict["categories"].append(type_dict)
    #     coco_test_dict["categories"].append(type_dict)

    ### begin processing
    start_scene_id = 3500 # following index of s3d data
    split_set = ['train.txt', 'val.txt', 'test.txt']
    save_folders = [train_img_folder, val_img_folder, test_img_folder]
    # save_dicts = [coco_train_dict, coco_val_dict, coco_test_dict]
    coco_json_paths = [coco_train_json_path, coco_val_json_path, coco_test_json_path]
    annos_folders = [annos_train_folder, annos_val_folder, annos_test_folder]

    def wrapper(scene_id):
        image_set = dataset[scene_id]
        process_floorplan(image_set, scene_id, start_scene_id, args, save_dir, annos_folder)

    def worker_init(dataset_obj):
        # Store dataset as global to avoid pickling issues
        global dataset
        dataset = dataset_obj

    for split_id, split_file in enumerate(split_set):
        dataset = FloorplanSVG(args.data_root, split_file, format='txt', original_size=False)
        save_dir = save_folders[split_id]
        # save_dict = save_dicts[split_id]
        json_path = coco_json_paths[split_id]
        print(f"############# {split_file}")

        annos_folder = annos_folders[split_id]

        # # for scene_id, image_set in enumerate(tqdm(dataset)):
        # for scene_id in tqdm(range(0, len(dataset), 1)):
        #     # process_floorplan(dataset, scene_id, start_scene_id, args, save_dir, annos_folder)
        #     wrapper(scene_id)

        num_processes = 16
        with Pool(num_processes, initializer=worker_init, initargs=(dataset,)) as p:
            # args = [(dataset[i], i) for i in range(len(dataset))]
            indices = range(len(dataset))
            list(tqdm(p.imap(wrapper, indices), total=len(dataset)))

        start_scene_id += len(dataset)
        # with open(json_path, 'w') as f:
        #     json.dump(save_dict, f)
