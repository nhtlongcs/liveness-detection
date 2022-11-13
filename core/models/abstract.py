import abc
from typing import Any

import pytorch_lightning as pl
import torch
import torchvision
from core.dataset import DATASET_REGISTRY
from core.metrics import METRIC_REGISTRY
from core.utils.device import detach
from torch.utils.data import DataLoader

from . import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ClsBase(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.init_model()
        self.learning_rate = self.cfg.get("trainer", {}).get("learning_rate", 1e-3)

    @abc.abstractmethod
    def init_model(self):
        raise NotImplementedError

    def setup(self, stage: str):
        if stage != "predict":
            image_size = self.cfg["data"]["args"]["SIZE"]
            image_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((image_size, image_size)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            self.train_dataset = DATASET_REGISTRY.get(self.cfg["data"]["name"])(
                **self.cfg["data"]["args"]["train"],
                data_cfg=self.cfg["data"]["args"],
                transform=image_transform,
            )
            self.val_dataset = DATASET_REGISTRY.get(self.cfg["data"]["name"])(
                **self.cfg["data"]["args"]["val"],
                data_cfg=self.cfg["data"]["args"],
                transform=image_transform,
            )

            self.metric = [
                METRIC_REGISTRY.get(mcfg["name"])(**mcfg["args"])
                if mcfg["args"]
                else METRIC_REGISTRY.get(mcfg["name"])()
                for mcfg in self.cfg["metric"]
            ]

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, visual_embeddings, nlang_embeddings, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # 1. Get embeddings from model
        output = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(**output, batch=batch).mean()
        # 3. Update monitor
        self.log("train/loss", detach(loss))

        return {"loss": loss, "log": {"train_loss": detach(loss)}}


    def validation_step(self, batch, batch_idx):
        # 1. Get embeddings from model
        output = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(**output, batch=batch)
        # 3. Update metric for each batch
        for m in self.metric:
            m.update(output, batch)

        return {"loss": detach(loss)}

    def validation_epoch_end(self, outputs):

        # 1. Calculate average validation loss
        loss = torch.mean(torch.stack([o["loss"] for o in outputs], dim=0))
        # 2. Calculate metric value
        out = {"val_loss": loss}
        for m in self.metric:
            # 3. Update metric for each batch
            metric_dict = m.value()
            out.update(metric_dict)
            for k in metric_dict.keys():
                self.log(f"val/{k}", out[k])

        # Log string
        log_string = ""
        for metric, score in out.items():
            if isinstance(score, (int, float)):
                log_string += metric + ": " + f"{score:.5f}" + " | "
        log_string += "\n"
        print(log_string)

        # 4. Reset metric
        for m in self.metric:
            m.reset()

        self.log("val/loss", loss.cpu().numpy().item())
        return {**out, "log": out}
        

    def train_dataloader(self):
        train_loader = DataLoader(
            **self.cfg["data"]["args"]["train"]["loader"],
            dataset=self.train_dataset,
            collate_fn=self.train_dataset.collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            **self.cfg["data"]["args"]["val"]["loader"],
            dataset=self.val_dataset,
            collate_fn=self.val_dataset.collate_fn,
        )
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[3, 5, 7], 
        gamma=0.5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

