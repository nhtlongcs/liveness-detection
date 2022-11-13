# python tests/model.py
from core.extractors import EXTRCT_REGISTRY
from core.opt import Opts
from pathlib import Path
import torch
import pytest 

@pytest.mark.order(1)
@pytest.mark.parametrize('extractor_name,version,img_size', [("VitNetExtractor",'vit_base_patch16_384',384), ("EfficientNetExtractor",0,380)])
def test_extractor_model(extractor_name, version, img_size):
    cfg = {
        "version": version,
        "from_pretrained": True,
        "freeze": True,
    }
    input_size = (3, img_size, img_size)
    input_tensor = torch.randn(1, *input_size)
    print(EXTRCT_REGISTRY)
    model = EXTRCT_REGISTRY.get(extractor_name)(**cfg)
    output = model(input_tensor)
