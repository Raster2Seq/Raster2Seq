from pathlib import Path

import torch
import torch.utils.data

from pycocotools.coco import COCO
from PIL import Image
import cv2
import torchvision

from util.poly_ops import resort_corners
from detectron2.data import transforms as T
from torch.utils.data import Dataset
import numpy as np
import os
from copy import deepcopy

from datasets.transforms import Resize, ResizeAndPad

from detectron2.data.detection_utils import annotations_to_instances, transform_instance_annotations
from detectron2.structures import BoxMode


class MultiPoly(Dataset):
    def __init__(self, img_folder, ann_file, transforms, semantic_classes, dataset_name='', image_norm=False):
        super(MultiPoly, self).__init__()

        self.root = img_folder
        self._transforms = transforms
        self.semantic_classes = semantic_classes
        self.dataset_name = dataset_name

        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.prepare = ConvertToCocoDict(self.root, self._transforms, image_norm)

    def get_image(self, path):
        return Image.open(os.path.join(self.root, path))
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: COCO format dict
        """
        coco = self.coco
        img_id = self.ids[index]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        ### Note: here is a hack which assumes door/window have category_id 16, 17 in structured3D
        if self.semantic_classes == -1:
            if self.dataset_name == 'stru3d':
                target = [t for t in target if t['category_id'] not in [16, 17]]
            elif self.dataset_name == 'rplan':
                target = [t for t in target if t['category_id'] not in [9, 11]]

        path = coco.loadImgs(img_id)[0]['file_name']

        record = self.prepare(img_id, path, target)

        return record


class ConvertToCocoDict(object):
    def __init__(self, root, augmentations, image_norm):
        self.root = root
        self.augmentations = augmentations
        if image_norm:
            self.image_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            self.image_normalize = None

    def _expand_image_dims(self, x):
        if len(x.shape) == 2:
            exp_img = np.expand_dims(x, 0)
        else:
            exp_img = x.transpose((2, 0, 1)) # (h,w,c) -> (c,h,w)
        return exp_img

    def __call__(self, img_id, path, target):

        file_name = os.path.join(self.root, path)

        img = np.array(Image.open(file_name))

        #### NEW
        if len(img.shape) >= 3:
            if img.shape[-1] > 3: # drop alpha channel
                img = img[:, :, :3]
            w, h = img.shape[:-1]
        else:
            w, h = img.shape
        #### NEW

        record = {}
        record["file_name"] = file_name
        record["height"] = h
        record["width"] = w
        record['image_id'] = img_id
        
        for obj in target: obj["bbox_mode"] = BoxMode.XYWH_ABS

        record['annotations'] = target

        if self.augmentations is None:
            record['image'] = (1/255) * torch.as_tensor(np.ascontiguousarray(self._expand_image_dims(img)))
            record['instances'] = annotations_to_instances(target, (h, w), mask_format="polygon")
        else:
            aug_input = T.AugInput(img)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
            record['image'] = (1/255) * torch.as_tensor(np.array(self._expand_image_dims(image)))
            
            annos = [
                transform_instance_annotations(
                    obj, transforms, image.shape[:2]
                    )
                    for obj in record.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                    ]
            # resort corners after augmentation: so that all corners start from upper-left counterclockwise
            for anno in annos:
                anno['segmentation'][0] = resort_corners(anno['segmentation'][0])

            record['instances'] = annotations_to_instances(annos, (h, w), mask_format="polygon")

        #### NEW ####
        if self.image_normalize is not None:
            record['image'] = self.image_normalize(record['image'])
            
        return record


def make_poly_transforms(dataset_name, image_set):
    
    trans_list = []
    if dataset_name == 'cubicasa':
        trans_list = [ResizeAndPad((256, 256), pad_value=255)]

    if image_set == 'train':
        trans_list.extend([
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomRotation([0.0, 90.0, 180.0, 270.0], expand=False, center=None, sample_style="choice")
            ]) 
        return T.AugmentationList(trans_list)
        
    if image_set == 'val' or image_set == 'test':
        return None if len(trans_list) == 0 else T.AugmentationList(trans_list)

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.dataset_root)
    assert root.exists(), f'provided data path {root} does not exist'

    PATHS = {
        "train": (root / "train", root / "annotations" / 'train.json'),
        "val": (root / "val", root / "annotations" / 'val.json'),
        "test": (root / "test", root / "annotations" / 'test.json')
    }

    img_folder, ann_file = PATHS[image_set]
    image_transform = None if getattr(args, 'disable_image_transform', False) else make_poly_transforms(args.dataset_name, image_set)
    
    dataset = MultiPoly(img_folder, 
                        ann_file, 
                        transforms=image_transform, 
                        semantic_classes=args.semantic_classes,
                        dataset_name=args.dataset_name,
                        image_norm=args.image_norm,
                        )
    
    return dataset
