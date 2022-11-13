# python tests/trainer.py

import torch
from core.opt import Opts

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from core.models import MODEL_REGISTRY
from core.dataset import DATASET_REGISTRY

from pathlib import Path 
from torch.utils.data import DataLoader, Dataset
import torchvision


def predict(model_name, cfg_path, resume_ckpt=None):
    

    class WrapperDataModule(pl.LightningDataModule):
        def __init__(self, ds, batch_size):
            super().__init__()
            self.ds = ds
            self.batch_size = batch_size

        def predict_dataloader(self):
            return DataLoader(
                self.ds, batch_size=self.batch_size, collate_fn=self.ds.collate_fn,
            )
    
    cfg = Opts(cfg=cfg_path).parse_args([])
    model = MODEL_REGISTRY.get(model_name)(cfg)
    model = model.load_from_checkpoint(resume_ckpt, config=cfg, strict=True)

    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.device_count() else None,  # Use all gpus available
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        enable_checkpointing=False,
    )

    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((288, 288)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    ds = DATASET_REGISTRY.get(cfg["data"]["name"])(
        **cfg.data["args"]["train"],
        transform=image_transform,
    )
    dm = WrapperDataModule(ds, batch_size=2)
    prds = trainer.predict(model, dm)


def test_predict(model_name="FrameClassifier"):
    cfg_path = "tests/configs/keyframes.yml"
    assert Path(cfg_path).exists(), "config file not found"
    print(cfg_path)
    
    predict(
        model_name,
        cfg_path,
        resume_ckpt="./tmp/lightning_logs/version_0/checkpoints/last.ckpt",
    )

