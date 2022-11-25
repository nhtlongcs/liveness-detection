import torch
import os
# https://github.com/pytorch/pytorch/issues/33296#issuecomment-1014126598
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from tqdm.auto import tqdm
from pathlib import Path

from core.models import MODEL_REGISTRY
from core.dataset import DATASET_REGISTRY
from core.metrics import METRIC_REGISTRY
from core.augmentations import TRANSFORM_REGISTRY
from core.opt import Opts
import torchvision
import pytest


@pytest.mark.order(2)
@pytest.mark.skip(reason="no way of currently testing this")
def test_evaluate(model_name='FrameClassifier'):
    cfg_path = "tests/configs/keyframes.yml"
    assert Path(cfg_path).exists(), "config file not found"
    cfg = Opts(cfg=cfg_path).parse_args([])
    image_transform = TRANSFORM_REGISTRY.get('train_classify_tf')(img_size=380)

    ds = DATASET_REGISTRY.get(cfg["data"]["name"])(
        **cfg.data["args"]["train"],
        transform=image_transform,
    )
    dataloader = torch.utils.data.DataLoader(
        **cfg.data["args"]["train"]["loader"],
        dataset=ds,
        collate_fn=ds.collate_fn,
    )
    model = MODEL_REGISTRY.get(model_name)(cfg)
    metrics = [
        METRIC_REGISTRY.get(mcfg["name"])(**mcfg["args"])
        if mcfg["args"] else METRIC_REGISTRY.get(mcfg["name"])()
        for mcfg in cfg["metric"]
    ]
    for i, batch in tqdm(enumerate(dataloader), total=5):
        out = model(batch)
        for metric in metrics:
            metric.update(out, batch)

        if (i % 5 == 0) and (i > 0):
            for metric in metrics:
                metric_dict = metric.value()
                # Log string
                log_string = ""
                for metric_name, score in metric_dict.items():
                    if isinstance(score, (int, float)):
                        log_string += metric_name + ": " + f"{score:.5f}" + " | "
                log_string += "\n"
                print(log_string)

                # 4. Reset metric
                metric.reset()
            break


if __name__ == "__main__":
    test_evaluate()
