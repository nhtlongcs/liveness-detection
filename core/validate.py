import torch
from opt import Opts

from pathlib import Path
from src.models import MODEL_REGISTRY
from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl


def check(cfg, pretrained_ckpt=None):
    model = MODEL_REGISTRY.get(cfg.model["name"])(cfg)
    model = model.load_from_checkpoint(pretrained_ckpt, config = cfg, strict=True)
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.device_count() else None,  # Use all gpus available
        accelerator="ddp" if torch.cuda.device_count() > 1 else None,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
    )
    trainer.validate(model)
    del trainer
    del cfg
    del model

pretrained_ckpt = "/home/nhtlong/playground/zalo-ai/liveness-det/clf/runs/zaloai-faceB7-clf/61p7x42k/checkpoints/faceB7-epoch=6-val/Accuracy=0.9730.ckpt"
cfg_path = 'configs/facevit384.yml'

cfg = Opts(cfg=cfg_path).parse_args([])
check(
    cfg,
    pretrained_ckpt=pretrained_ckpt
)