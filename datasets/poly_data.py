from pathlib import Path

import torch
import torch.utils.data

from pycocotools.coco import COCO
from PIL import Image
import cv2
import torchvision
import math

from util.poly_ops import resort_corners
from detectron2.data import transforms as T
from torch.utils.data import Dataset
import numpy as np
import os
from copy import deepcopy

from enum import Enum

from datasets.transforms import Resize, ResizeAndPad
from datasets.discrete_tokenizer import DiscreteTokenizer

from detectron2.data.detection_utils import annotations_to_instances, transform_instance_annotations
from detectron2.structures import BoxMode

class TokenType(Enum):
    """0 for <coord>, 1 for <sep>, 2 for <eos>, 3 for <cls>"""
    coord = 0
    sep = 1
    eos = 2
    cls = 3


class MultiPoly(Dataset):
    def __init__(self, img_folder, ann_file, transforms, semantic_classes, dataset_name='', image_norm=False, poly2seq=False, **kwargs):
        super(MultiPoly, self).__init__()

        self.root = img_folder
        self._transforms = transforms
        self.semantic_classes = semantic_classes
        self.dataset_name = dataset_name

        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.poly2seq = poly2seq
        self.prepare = ConvertToCocoDict(self.root, self._transforms, image_norm, poly2seq, 
                                         add_cls_token=(semantic_classes!=-1), **kwargs)

    def get_image(self, path):
        return Image.open(os.path.join(self.root, path))
    
    def get_vocab_size(self):
        if self.poly2seq:
            return len(self.prepare.tokenizer)
        return None
    
    def get_tokenizer(self):
        if self.poly2seq:
            return self.prepare.tokenizer
        return None
    
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
    def __init__(self, root, augmentations, image_norm, poly2seq=False, add_cls_token=False, **kwargs):
        self.root = root
        self.augmentations = augmentations
        if image_norm:
            self.image_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            self.image_normalize = None
        
        self.poly2seq = poly2seq
        if poly2seq:
            self.tokenizer = DiscreteTokenizer(**kwargs)
            self.add_cls_token = add_cls_token

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
            # print(img.shape, file_name)
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
            h, w = image.shape[:2] # update size
            
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

        # convert polygons to sequences
        if self.poly2seq:
            polygons = [np.clip(np.array(inst).reshape(-1, 2) / (w - 1), 0, 1) for inst in record['instances'].gt_masks.polygons]
            polygons_label = [inst.item() for inst in record['instances'].gt_classes]
            record.update(self._get_bilinear_interpolation_coeffs(polygons, polygons_label, self.add_cls_token))
            
        return record

        
    def _get_bilinear_interpolation_coeffs(self, polygons, polygons_label, add_cls_token=False):
        num_bins = self.tokenizer.num_bins
        quant_poly = [poly * (num_bins - 1) for poly in polygons]
        index11 = [[math.floor(p[0])*num_bins + math.floor(p[1]) for p in poly] for poly in quant_poly]
        index21 = [[math.ceil(p[0])*num_bins + math.floor(p[1]) for p in poly] for poly in quant_poly]
        index12 = [[math.floor(p[0])*num_bins + math.ceil(p[1]) for p in poly] for poly in quant_poly]
        index22 = [[math.ceil(p[0])*num_bins + math.ceil(p[1]) for p in poly] for poly in quant_poly]

        seq11 = self.tokenizer(index11, add_bos=True, add_eos=False, dtype=torch.long)
        seq21 = self.tokenizer(index21, add_bos=True, add_eos=False, dtype=torch.long)
        seq12 = self.tokenizer(index12, add_bos=True, add_eos=False, dtype=torch.long)
        seq22 = self.tokenizer(index22, add_bos=True, add_eos=False, dtype=torch.long)
        
        # in real values insteads
        target_seq = []                      
        token_labels = [] # 0 for <coord>, 1 for <sep>, 2 for <eos>, 3 for <cls>
        for poly in polygons:
            token_labels.extend([TokenType.coord.value] * len(poly))
            if add_cls_token:
                token_labels.append(TokenType.cls.value) # cls token
            token_labels.append(TokenType.sep.value) # separator token
            target_seq.extend(poly)
            if add_cls_token:
                target_seq.append([0, 0]) # padding for cls token
            target_seq.append([0, 0]) # padding for sep/end token
        # remove last separator token
        token_labels[-1] = TokenType.eos.value

        mask = torch.ones(self.tokenizer.seq_len, dtype=torch.bool)
        mask[len(token_labels):] = 0
        target_seq = self.tokenizer._padding(target_seq, [0, 0], dtype=torch.float32)
        token_labels = self.tokenizer._padding(token_labels, -1, dtype=torch.long)

        delta_x1 = [0] # [0] for bos token
        for polygon in quant_poly:
            # delta_x1.extend([0])  # for cls token
            delta = [poly_point[0] - math.floor(poly_point[0]) for poly_point in polygon]
            delta_x1.extend(delta)
            delta_x1.extend([0])  # for separator token
        delta_x1 = delta_x1[:-1]  # there is no separator token in the end
        delta_x1 = self.tokenizer._padding(delta_x1, 0, dtype=torch.float32)
        delta_x2 = 1 - delta_x1

        delta_y1 = [0] # [0] for bos token
        for polygon in quant_poly:
            # delta_y1.extend([0])  # for cls token
            delta = [poly_point[1] - math.floor(poly_point[1]) for poly_point in polygon]
            delta_y1.extend(delta)
            delta_y1.extend([0])  # for separator token
        delta_y1 = delta_y1[:-1]  # there is no separator token in the end
        delta_y1 = self.tokenizer._padding(delta_y1, 0, dtype=torch.float32)
        delta_y2 = 1 - delta_y1

        target_polygon_labels = polygons_label
        max_label_length = self.tokenizer.seq_len
        if len(polygons_label) < max_label_length:
            target_polygon_labels.extend([-1]* (max_label_length - len(target_polygon_labels)))
        target_polygon_labels = torch.tensor(target_polygon_labels, dtype=torch.long)

        return {'delta_x1': delta_x1, 
                'delta_x2': delta_x2, 
                'delta_y1': delta_y1, 
                'delta_y2': delta_y2,
                'seq11': seq11,
                'seq21': seq21,
                'seq12': seq12,
                'seq22': seq22,
                'target_seq': target_seq,
                'token_labels': token_labels,
                'mask': mask,
                'target_polygon_labels': target_polygon_labels}


def make_poly_transforms(dataset_name, image_set, image_size=256):
    
    trans_list = []
    if dataset_name == 'cubicasa':
        trans_list = [ResizeAndPad((image_size, image_size), pad_value=255)]

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
    image_transform = None if getattr(args, 'disable_image_transform', False) else make_poly_transforms(args.dataset_name, image_set, image_size=args.image_size)
    
    dataset = MultiPoly(img_folder, 
                        ann_file, 
                        transforms=image_transform, 
                        semantic_classes=args.semantic_classes,
                        dataset_name=args.dataset_name,
                        image_norm=args.image_norm,
                        poly2seq=args.poly2seq,
                        num_bins=args.num_bins,
                        seq_len=args.seq_len
                        )
    
    return dataset
