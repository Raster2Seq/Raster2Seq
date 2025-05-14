import torch.utils.data

from .poly_data import build as build_poly


def build_dataset(image_set, args):
    if args.semantic_classes > 0:
        # assert args.dataset_name == 'stru3d', "Semantically-rich floorplans only support Structured3D"
        pass
    if args.dataset_name in ['stru3d', 'scenecad', 'rplan', 'cubicasa', 'waffle']:
        return build_poly(image_set, args)
    raise ValueError(f'dataset {args.dataset_name} not supported')

def get_dataset_class_labels(dataset_name):
    semantics_label = None
    
    if dataset_name == 'stru3d':
        semantics_label = {
            0: 'Living Room',
            1: 'Kitchen',
            2: 'Bedroom',
            3: 'Bathroom',
            4: 'Balcony',
            5: 'Corridor',
            6: 'Dining room',
            7: 'Study',
            8: 'Studio',
            9: 'Store room',
            10: 'Garden',
            11: 'Laundry room',
            12: 'Office',
            13: 'Basement',
            14: 'Garage',
            15: 'Misc.',
            16: 'Door',
            17: 'Window'
        }
    elif dataset_name == 'cubicasa':
        semantics_label = {
            "Outdoor": 0,
            "Kitchen": 1,
            "Living Room": 2,
            "Bed Room": 3,
            "Bath": 4,
            "Entry": 5,
            "Storage": 6,
            "Garage": 7,
            "Undefined": 8,
            "Window": 9,
            "Door": 10,
        }
    
    return semantics_label

    