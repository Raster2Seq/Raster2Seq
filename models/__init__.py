# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------------------

from .roomformer import build
from .raster2seq import build as build_v2


def build_model(args, train=True, tokenizer=None):
    if not args.poly2seq:
        return build(args, train)
    return build_v2(args, train, tokenizer=tokenizer)
