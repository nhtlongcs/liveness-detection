import json
import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.models.abstract import AICBase
from src.utils.device import move_to
from src.utils.faiss_retrieval import FaissRetrieval

from src.dataset.default import AIC22TextJsonDataset, AIC22TrackJsonWithMotionDataset
from src.dataset.srl import AIC22TrackVehJsonDataset
from src.models.abstract import ClsBase
from src.dataset import DATASET_REGISTRY

import torchvision
import pytorch_lightning as pl
from opt import Opts


class WrapperDataModule(pl.LightningDataModule):
    def __init__(self, ds, batch_size):
        super().__init__()
        self.ds = ds
        self.batch_size = batch_size

    def predict_dataloader(self):
        return DataLoader(
            self.ds, batch_size=self.batch_size, collate_fn=self.ds.collate_fn,
        )


class Predictor(object):
    def __init__(
        self,
        model: AICBase,
        cfg: Opts,
        mode: str = "simple",
        batch_size: int = 1,
        top_k: int = 10,
        savedir: str = "./",
    ):
        self.cfg = cfg
        self.model = model
        self.savedir = savedir
        self.mode = mode
        self.batch_size = batch_size
        self.top_k = top_k
        self.setup()
        self.retriever = FaissRetrieval(dimension=model.cfg.model["args"]["EMBED_DIM"])

    def save(self, filename: str, data: dict):
        assert osp.exists(self.savedir), f"{self.savedir} does not exist"
        with open(osp.join(self.savedir, filename), "w") as f:
            json.dump(data, f)
        print(f"Saved {filename} to {self.savedir}")

    def setup(self):
        image_size = self.cfg['data']['track']['args']['image_size']
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((image_size, image_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.track_ds = DATASET_REGISTRY.get(self.cfg.data["track"]["name"])(
            **self.cfg.data["track"]['args'],
            transform=transform,
        )
        self.query_ds = DATASET_REGISTRY.get(self.cfg.data["text"]["name"])(
            **self.cfg.data["text"]['args']
        )


    def predict(self):
        self.model.eval()
        if self.mode == "simple":
            query_results, track_results = self.predict_simple()
        elif self.mode == "complex":
            query_results, track_results = self.predict_complex()
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        self.save("query_embeds.json", query_results)
        self.save("track_embeds.json", track_results)

        query_embeddings = np.stack(query_results.values(), axis=0).astype(np.float32)
        gallery_embeddings = np.stack(track_results.values(), axis=0).astype(np.float32)
        query_ids = list(query_results.keys())
        gallery_ids = list(track_results.keys())

        self.retriever.similarity_search(
            query_embeddings,
            gallery_embeddings,
            query_ids,
            gallery_ids,
            top_k=self.top_k,
            save_results=osp.join(self.savedir, "retrieval_results.json"),
        )

    def predict_simple(self):
        query_results = self.predict_simple_loop(self.query_ds, "query_embedding_head")
        track_results = self.predict_simple_loop(self.track_ds, "track_embedding_head")
        return query_results, track_results

    def predict_complex(self):
        def pred2dict(preds):
            results = {}
            for batch in preds:
                ids = batch["ids"]
                feats = batch["features"].numpy()
                for id, feat in zip(ids, feats):
                    results[id] = feat.tolist()
            return results

        trainer = pl.Trainer(
            gpus=-1 if torch.cuda.device_count() else None,  # Use all gpus available
            accelerator="ddp" if torch.cuda.device_count() > 1 else None,
            sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
            enable_checkpointing=False,
        )

        query_dm = WrapperDataModule(self.query_ds, batch_size=1)
        track_dm = WrapperDataModule(self.track_ds, batch_size=1)
        query_prds = trainer.predict(self.model, query_dm)
        track_prds = trainer.predict(self.model, track_dm)
        query_results = pred2dict(query_prds)
        track_results = pred2dict(track_prds)
        return query_results, track_results

    @torch.no_grad()
    def predict_simple_loop(self, ds: Dataset, embed_head_name: str):
        results = {}
        dataloader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, collate_fn=ds.collate_fn
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cpu = torch.device("cpu")
        self.model = self.model.to(device)
        extract_feat_fn = getattr(self.model, embed_head_name)
        for batch in tqdm(dataloader):
            batch = move_to(batch, device)
            out = extract_feat_fn(batch, inference=True)
            out = move_to(out, cpu)
            (ids, feats) = (out["ids"], out["features"].numpy())

            for id, feat_batch in zip(ids, feats):
                results[id] = feat_batch.tolist()
        return results


class ClsPredictor(object):
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

    def save(self, filepath: str, data: dict):
        with open(filepath, "w") as f:
            json.dump(data, f)
        print(f"Saved file to {filepath}")

    def setup(self):
        image_size = self.cfg['data']['image_size']
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((image_size, image_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.ds = AIC22TrackVehJsonDataset(**self.cfg.data, transform=transform)

    def predict(self):
        trainer = pl.Trainer(
            gpus=-1 if torch.cuda.device_count() else None,  # Use all gpus available
            accelerator="ddp" if torch.cuda.device_count() > 1 else None,
            sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
            enable_checkpointing=False,
        )

        dm = WrapperDataModule(self.ds, batch_size=self.batch_size)
        prds = trainer.predict(self.model, dm)
        return prds
    def predict_json(self):
        prds = self.predict()
        result = {}
        for _, prd in enumerate(prds):
            track_ids = prd['ids']
            logits = (prd['logits'] > 0) * 1
            for i, track_id in enumerate(track_ids):
                result[track_id] = logits[i].tolist()
        return result
