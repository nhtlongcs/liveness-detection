import json
import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


from core.dataset.default import ImageFolderFromCSV
from core.models.abstract import ClsBase
from core.dataset import DATASET_REGISTRY
from core.models import MODEL_REGISTRY

import torchvision
import pytorch_lightning as pl
from core.opt import Opts
import pandas as pd
from pathlib import Path

class WrapperDataModule(pl.LightningDataModule):
    def __init__(self, ds, batch_size):
        super().__init__()
        self.ds = ds
        self.batch_size = batch_size

    def predict_dataloader(self):
        return DataLoader(
            self.ds, batch_size=self.batch_size, collate_fn=self.ds.collate_fn, num_workers=20
        )


class ClsPredictor:
    def __init__(
        self,
        model: ClsBase,
        cfg: Opts,
        batch_size: int = 1,
    ):
        self.cfg = cfg
        self.model = model
        self.threshold = 0.5
        self.batch_size = batch_size
        self.setup()


    def setup(self):
        image_size = self.cfg['data']['SIZE']
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((image_size, image_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.ds = ImageFolderFromCSV(**self.cfg.data, transform=transform, num_rows=-1)

    def predict(self):
        trainer = pl.Trainer(
            gpus=-1 if torch.cuda.device_count() else None,  # Use all gpus available
            strategy="ddp" if torch.cuda.device_count() > 1 else None,
            sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
            enable_checkpointing=False,
        )

        dm = WrapperDataModule(self.ds, batch_size=self.batch_size)
        prds = trainer.predict(self.model, dm)
        return prds
    def predict_csv(self):
        prds = self.predict()
        video_ids = []
        frame_ids = []
        labels = []
        probs = []
        
        for _, batch_prds in enumerate(prds):
            video_bids = batch_prds['video_ids'] # batch ids
            frame_bids = batch_prds['frame_ids']
            # convert logits to probabilities using softmax
            probs_b = torch.softmax(batch_prds['logits'], dim=1).cpu().numpy()
            labels_b = torch.argmax(batch_prds['logits'], dim=1).cpu().numpy()
            # only get the probability of the positive class
            probs_b = probs_b[:, 1]
            video_ids.extend(video_bids)
            frame_ids.extend(frame_bids)
            labels.extend(labels_b)
            probs.extend(probs_b)
        result = pd.DataFrame({'video_id': video_ids, 'frame_id': frame_ids, 'label': labels, 'prob': probs})
        return result


    
if __name__ == "__main__":
    cfg = Opts().parse_args()
    resume_ckpt = cfg['global']['pretrained']
    save_path = Path(cfg['global']['save_path'])
    batch_sz = cfg['global']['batch_size'] 
    # if save_path is directory, then savepath = savepath / 'predict.csv'
    if save_path.is_dir():
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / 'predict.csv'
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        assert save_path.suffix == '.csv', f'Path {save_path} must be a csv file'

    model = MODEL_REGISTRY.get(cfg.model["name"])(cfg)
    model = model.load_from_checkpoint(resume_ckpt, config=cfg, strict=True)
    p = ClsPredictor(model, cfg, batch_size=batch_sz)
    df = p.predict_csv()
    df.to_csv(save_path, index=False)