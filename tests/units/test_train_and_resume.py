# python tests/trainer.py

import torch
from core.opt import Opts

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from core.models import MODEL_REGISTRY
import pytest
from pathlib import Path


def train(model_name, cfg_path, resume_ckpt=None):
    cfg = Opts(cfg=cfg_path).parse_args([])
    model = MODEL_REGISTRY.get(model_name)(cfg)
    checkpoint_callback = ModelCheckpoint(verbose=True, save_last=True,)
    trainer = pl.Trainer(
        default_root_dir="./tmp",
        log_every_n_steps=1,
        max_steps=10,
        max_epochs=2,
        gpus=-1 if torch.cuda.device_count() else None,  # Use all gpus available
        check_val_every_n_epoch=cfg.trainer["evaluate_interval"],
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        precision=16 if cfg["global"]["use_fp16"] else 32,
        fast_dev_run=False if resume_ckpt is None else True,
        callbacks=[checkpoint_callback] if resume_ckpt is None else [],
        enable_checkpointing=True if resume_ckpt is None else False,
    )
    trainer.fit(model, ckpt_path=resume_ckpt)
    del trainer
    del cfg
    del model
    del checkpoint_callback

@pytest.mark.order(1)
def test_trainer(model_name="FrameClassifier"):
    cfg_path = "tests/configs/keyframes.yml"
    assert Path(cfg_path).exists(), "config file not found"
    print(cfg_path)
    train(model_name, cfg_path)
    train(
        model_name,
        cfg_path,
        resume_ckpt="./tmp/lightning_logs/version_0/checkpoints/last.ckpt",
    )

