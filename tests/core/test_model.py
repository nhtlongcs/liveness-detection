# python tests/model.py
from core.models import MODEL_REGISTRY
from core.opt import Opts
from pathlib import Path
import pytest


@pytest.mark.order(1)
def test_model(model_name="FrameClassifier"):
    cfg_path = "tests/configs/keyframes.yml"
    assert Path(cfg_path).exists(), "config file not found"
    cfg = Opts(cfg=cfg_path).parse_args([])

    print(MODEL_REGISTRY)
    model = MODEL_REGISTRY.get(model_name)(cfg)
    model.prepare_data()


if __name__ == "__main__":
    test_model()
