import torch
import os
# https://github.com/pytorch/pytorch/issues/33296#issuecomment-1014126598
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from tqdm.auto import tqdm
from pathlib import Path

# from core.models import MODEL_REGISTRY
from core.dataset import DATASET_REGISTRY
from core.metrics import METRIC_REGISTRY
from core.augmentations import TRANSFORM_REGISTRY
from core.opt import Opts

from core_pkg.dataloader import TwoStreamDataLoader
from core_pkg.models import MODEL_REGISTRY

import torchvision
import pytest


@pytest.mark.order(2)
@pytest.mark.skip(reason="no way of currently testing this")
def test_evaluate(model_name='DualClassifier'):
    cfg_path = "tests/configs/cps.yml"
    assert Path(cfg_path).exists(), "config file not found"
    cfg = Opts(cfg=cfg_path).parse_args([])
    image_transform = TRANSFORM_REGISTRY.get('train_classify_tf')(img_size=380)

    ds = DATASET_REGISTRY.get(cfg["data"]["name"])(
        **cfg.data["args"]["train"],
        transform=image_transform,
    )

    uds = DATASET_REGISTRY.get(cfg["data"]["name"])(
        **cfg.data["args"]["val"],
        transform=image_transform,
        return_lbl=False,
    )

    dataloader = TwoStreamDataLoader(
        dataset_l=ds,
        dataset_u=uds,
        batch_sizes=[2, 2],
        shuffle=False,
        num_workers=0,
    )
    model = MODEL_REGISTRY.get(model_name)(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    torch.set_grad_enabled(True)

    outs = []
    pbar = tqdm(enumerate(dataloader))
    # model.current_epoch = 0
    for batch_idx, batch in pbar:
        output = model.training_step(batch, batch_idx)
        loss = output['loss']
        outs.append(loss.detach())

        # clear gradients
        optimizer.zero_grad()

        # backward
        loss.backward()

        # update parameters
        optimizer.step()
        pbar.set_description(f"loss: {loss.item():.4f}")


if __name__ == "__main__":
    test_evaluate()
