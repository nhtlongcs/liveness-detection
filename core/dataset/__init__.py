from core.registry import Registry
from PIL import Image
import numpy as np

DATASET_REGISTRY = Registry("DATASET")


def default_loader(path):
    pillow_image = Image.open(path).convert("RGB")
    return np.array(pillow_image)


from .default import *
