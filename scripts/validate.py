import torch
from core.opt import Opts

from pathlib import Path
from core.models import MODEL_REGISTRY
from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl


def check(cfg, pretrained_ckpt=None):
    model = MODEL_REGISTRY.get(cfg.model["name"])(cfg)
    model = model.load_from_checkpoint(pretrained_ckpt, config = cfg, strict=True)
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.device_count() else None,  # Use all gpus available
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
    )
    trainer.validate(model)
    del trainer
    del cfg
    del model


if __name__ == "__main__":
    cfg = Opts().parse_args()
    check(
        cfg,
        pretrained_ckpt=cfg['global']['pretrained']
    )