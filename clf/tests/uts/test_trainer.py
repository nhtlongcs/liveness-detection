# python tests/trainer.py

import torch
from opt import Opts

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models import MODEL_REGISTRY

from pathlib import Path
import pytest


def train(model_name, cfg_path, resume_ckpt=None):
    cfg = Opts(cfg=cfg_path).parse_args([])
    model = MODEL_REGISTRY.get(model_name)(cfg)
    checkpoint_callback = ModelCheckpoint(verbose=True, save_last=True,)
    trainer = pl.Trainer(
        default_root_dir="./uts_runs",
        log_every_n_steps=1,
        max_steps=10,
        max_epochs=2,
        gpus=-1 if torch.cuda.device_count() else None,  # Use all gpus available
        check_val_every_n_epoch=cfg.trainer["evaluate_interval"],
        accelerator="ddp" if torch.cuda.device_count() > 1 else None,
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


@pytest.mark.parametrize("model_name", ["UTS"])
def test_trainer(tmp_path, model_name):
    cfg_path = "tests/configs/default.yml"
    assert Path(cfg_path).exists(), "config file not found"
    print(cfg_path)
    train(model_name, cfg_path)
    train(
        model_name,
        cfg_path,
        resume_ckpt="./uts_runs/lightning_logs/version_0/checkpoints/last.ckpt",
    )



