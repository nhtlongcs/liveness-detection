# python tests/model.py
from core.augmentations import TRANSFORM_REGISTRY
from core.opt import Opts
from pathlib import Path
import torch
import pytest
import numpy as np


@pytest.mark.order(1)
@pytest.mark.parametrize('transform_name,img_size',
                         [("train_classify_tf", 384),
                          ("test_classify_tf", 380)])
def test_augmentation(transform_name, img_size):
    cfg = {"img_size": img_size}
    input_random_size = (np.random.randint(100,
                                           500), np.random.randint(100,
                                                                   500), 3)
    output_sanity_size = (3, img_size, img_size)
    input_tensor = np.random.rand(*input_random_size)
    tf = TRANSFORM_REGISTRY.get(transform_name)(**cfg)
    output = tf(image=input_tensor)
    assert output['image'].shape == output_sanity_size
