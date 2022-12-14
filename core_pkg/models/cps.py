import abc
from typing import Any

import pytorch_lightning as pl
import torch
import torchvision
from core.dataset import DATASET_REGISTRY
from core.metrics import METRIC_REGISTRY
from core.augmentations import TRANSFORM_REGISTRY
from core.utils.device import detach
from torch.utils.data import DataLoader
from core.models import MODEL_REGISTRY

from core_pkg.dataloader import TwoStreamDataLoader


@MODEL_REGISTRY.register()
class SemiSuperviseModel(pl.LightningModule):
    # NOTE: find lr not working with this model
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.branch1 = None  # this branch should be implement in subclass
        self.branch2 = None  # this branch should be implement in subclass
        self.init_model()

        self.consistency: float = self.cfg.model["args"]["consistency"]
        self.consistency_rampup = self.cfg.model["args"]["consistency_rampup"]

        self.learning_rate1 = self.cfg.get("trainer", {}).get("lr1", 1e-3)
        self.learning_rate2 = self.cfg.get("trainer", {}).get("lr2", 1e-3)

    @abc.abstractmethod
    def init_model(self):
        raise NotImplementedError

    def setup(self, stage: str):
        if stage != "predict":
            image_size = self.cfg["data"]["args"]["SIZE"]
            image_transform_train = TRANSFORM_REGISTRY.get(
                'train_classify_tf')(img_size=image_size)
            image_transform_test = TRANSFORM_REGISTRY.get('test_classify_tf')(
                img_size=image_size)

            self.labelled_train_dataset = DATASET_REGISTRY.get(
                self.cfg["data"]["name"])(
                    **self.cfg["data"]["args"]["train"],
                    data_cfg=self.cfg["data"]["args"],
                    transform=image_transform_train,
                )

            self.unlabelled_train_dataset = DATASET_REGISTRY.get(
                self.cfg["data"]["name"])(
                    **self.cfg["data"]["args"]["val"],
                    data_cfg=self.cfg["data"]["args"],
                    transform=image_transform_test,
                    return_lbl=False,
                )

            self.val_dataset = DATASET_REGISTRY.get(self.cfg["data"]["name"])(
                **self.cfg["data"]["args"]["val"],
                data_cfg=self.cfg["data"]["args"],
                transform=image_transform_test,
            )

            self.metric = [
                METRIC_REGISTRY.get(mcfg["name"])(**mcfg["args"])
                if mcfg["args"] else METRIC_REGISTRY.get(mcfg["name"])()
                for mcfg in self.cfg["metric"]
            ]

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, visual_embeddings, nlang_embeddings, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        # 1. Get embeddings from model
        output = self.forward(batch)
        # 2. Calculate loss
        # loss, loss_dict = self.compute_loss(**output, batch=batch).mean() # .mean?
        loss, loss_dict = self.compute_loss(output, batch=batch)
        # 3. Update monitor
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v)

        return {"loss": loss, "log": {"train_loss": detach(loss)}}

    def validation_step(self, batch, batch_idx):
        # 1. Get embeddings from model
        output = self.forward(batch)
        # 2. Calculate loss
        loss, _ = self.compute_loss(output, batch=batch)
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
        train_loader = TwoStreamDataLoader(
            **self.cfg["data"]["args"]["train"]["loader"],
            dataset_l=self.labelled_train_dataset,
            dataset_u=self.unlabelled_train_dataset,
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
        optimizer1 = torch.optim.AdamW(self.branch1.parameters(),
                                       lr=self.learning_rate1)
        optimizer2 = torch.optim.AdamW(self.branch2.parameters(),
                                       lr=self.learning_rate2)

        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1,
                                                          milestones=[3, 5, 7],
                                                          gamma=0.5)

        scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2,
                                                          milestones=[3, 5, 7],
                                                          gamma=0.5)

        return ({
            "optimizer": optimizer1,
            "lr_scheduler": {
                "scheduler": scheduler1,
                "interval": "epoch"
            },
        }, {
            "optimizer": optimizer2,
            "lr_scheduler": {
                "scheduler": scheduler2,
                "interval": "epoch"
            },
        })
