# python tests/parser.py -c configs/template.yml -o global.debug=False
from core.opt import Opts

import pytest
import yaml

@pytest.mark.order(1)
@pytest.fixture
def minimal_cfg(tmp_path):
    cfg = {"global": {"name": "test", "verbose": False, "SEED": 21318}}
    return cfg


def _save_cfg(tmp_path, cfg):
    with open(tmp_path, "w+") as f:
        yaml.safe_dump(cfg, f)


@pytest.mark.parametrize("exp_name", [None, "delete", "another_name"])
def test_opts_device_cpu(tmp_path, minimal_cfg, exp_name):
    def _fake_test():
        opts = Opts(cfg=cfg_path).parse_args(["-o", "global.name=another_name"])
        assert opts["global"]["name"] == "another_name"

    def _normal_test():
        opts = Opts(cfg=cfg_path).parse_args([])
        assert opts["global"]["name"] == "test"

    cfg_path = tmp_path / "default.yaml"
    _save_cfg(cfg_path, minimal_cfg)
    if exp_name == "another_name":
        _normal_test()  # opts.global.name is default
        _fake_test()  # opts.global.name is set to another_name
    elif exp_name is None:
        minimal_cfg["global"]["name"] = None  # global has the "name" key but don't set
    elif exp_name == "delete":
        del minimal_cfg["global"]["name"]  # global doesnt have the "name" key
