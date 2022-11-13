# python tests/model.py
from src.extractors import EXTRCT_REGISTRY
from opt import Opts
from pathlib import Path
import torch

def test_extractor_model(extractor_name):
    cfg = {
        "version": 'vit_base_patch16_384',
        "from_pretrained": True,
        "freeze": True,
    }
    input_size = (3, 384, 384)
    input_tensor = torch.randn(1, *input_size)
    print(EXTRCT_REGISTRY)
    model = EXTRCT_REGISTRY.get(extractor_name)(**cfg)
    output = model(input_tensor)

test_extractor_model("VitNetExtractor")