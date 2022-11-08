# python tests/model.py
from src.models import MODEL_REGISTRY
from opt import Opts
from pathlib import Path


def test_model(model_name):
    cfg_path = "tests/configs/keyframes.yml"
    assert Path(cfg_path).exists(), "config file not found"
    cfg = Opts(cfg=cfg_path).parse_args([])

    print(MODEL_REGISTRY)
    model = MODEL_REGISTRY.get(model_name)(cfg)
    model.prepare_data()

test_model("FrameClassifier")