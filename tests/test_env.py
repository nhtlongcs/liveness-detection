import pytest 
import os 
from core.bugs_report.collect_env import info_system, info_cuda, info_packages, nice_print
import logging
@pytest.mark.order(1)
def test_collect_env():
    details = {"System": info_system(), "CUDA": info_cuda(), "Packages": info_packages()}
    details["Lightning"] = {k: v for k, v in details["Packages"].items() if "torch" in k or "lightning" in k}
    lines = nice_print(details)
    text = os.linesep.join(lines)
    print(text)
    # assert len(details["CUDA"]["GPU"]) > 0, "No GPU found, please check your CUDA installation"
    # assert details['CUDA']['version'] == '10.2', "CUDA must be 10.2 to reproduce the results"
    assert details['Lightning']['pytorch-lightning'] == '1.6.0', "pytorch-lightning must be 1.6.0 to reproduce the results"