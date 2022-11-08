# python tests/model.py
from src.models import MODEL_REGISTRY
from opt import Opts
from pathlib import Path
import pytest


@pytest.mark.parametrize("model_name", ["UTS"])
def test_model(tmp_path, model_name):
    cfg_path = "tests/configs/default.yml"
    assert Path(cfg_path).exists(), "config file not found"
    cfg = Opts(cfg=cfg_path).parse_args([])

    print(MODEL_REGISTRY)
    model = MODEL_REGISTRY.get(model_name)(cfg)
    model.prepare_data()
