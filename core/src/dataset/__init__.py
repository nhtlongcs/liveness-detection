from registry import Registry
from PIL import Image
DATASET_REGISTRY = Registry("DATASET")


def default_loader(path):
    return Image.open(path).convert("RGB")


from .default import *
