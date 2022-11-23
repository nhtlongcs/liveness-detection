import torch
import torch.nn.functional as F
from core.extractors import EXTRCT_REGISTRY
from core.utils.losses import FocalLoss

from . import MODEL_REGISTRY
from .base import SuperviseModel
from .utils import ClassifyBlock


@MODEL_REGISTRY.register()
class Classifier(SuperviseModel):

    def __init__(self, config):
        super().__init__(config)

    def init_model(self):
        embed_dim = self.cfg.model["args"]["EMBED_DIM"]

        assert self.cfg.extractors.get(
            "img_encoder", None) is not None, "img_encoder must be specified"
        self.visualExtrct = EXTRCT_REGISTRY.get(
            self.cfg.extractors["img_encoder"]["name"])(
                **self.cfg.extractors["img_encoder"]["args"])

        self.img_in_dim = self.visualExtrct.feature_dim
        assert self.cfg.model["args"].get(
            "NUM_CLASS", None) is not None, "NUM_CLASS must be specified"
        self.logits = ClassifyBlock(
            inp_dim=self.img_in_dim,
            num_cls=self.cfg.model["args"]["NUM_CLASS"],
            embed_dim=embed_dim,
        )

        self.loss = FocalLoss(num_classes=self.cfg.model["args"]["NUM_CLASS"])

    def normalize_head(
        self,
        embedding,
    ):
        return F.normalize(embedding, p=2, dim=-1)

    def forward(self, batch):
        assert "images" in batch.keys(), "Batch must contain images"
        embedding = self.visualExtrct(batch["images"])
        embedding = self.normalize_head(embedding)
        logits = self.logits(embedding, return_embed=False)
        return {"logits": logits}

    def compute_loss(self, logits, batch, **kwargs):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx):
        assert "images" in batch.keys(), "Batch must contain images"
        preds = self.forward(batch)
        # add meta data for inference stage
        preds.update({
            'filenames': batch['filenames'],
            'video_ids': batch['video_ids'],
            'frame_ids': batch['frame_ids']
        })
        return preds


@MODEL_REGISTRY.register()
class FrameClassifier(Classifier):

    def __init__(self, config):
        super().__init__(config)

    def compute_loss(self, logits, batch, **kwargs):
        return self.loss(logits, batch["labels"].long())
