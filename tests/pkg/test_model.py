# python tests/model.py
from core_pkg.models import MODEL_REGISTRY
from core.opt import Opts
from pathlib import Path
import pytest


@pytest.mark.order(1)
def test_model(model_name="DualClassifier"):
    cfg_path = "tests/configs/cps.yml"
    assert Path(cfg_path).exists(), "config file not found"
    cfg = Opts(cfg=cfg_path).parse_args([])

    print(MODEL_REGISTRY)
    model = MODEL_REGISTRY.get(model_name)(cfg)
    model.prepare_data()


if __name__ == "__main__":
    test_model()
