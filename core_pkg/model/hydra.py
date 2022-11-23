import torch
import torch.nn as nn
import torch.nn.functional as F
from core.extractors import EXTRCT_REGISTRY
from core.utils.losses import FocalLoss
from . import MODEL_REGISTRY
from .ss_pipeline import SemiSuperviseModel
from .utils import ClassifyBlock


@MODEL_REGISTRY.register()
class DualClassifier(SemiSuperviseModel):

    def __init__(self, config):
        super().__init__(config)

    def init_model(self):

        assert cfg.model["args"].get(
            "NUM_CLASS", None) is not None, "NUM_CLASS must be specified"
        self.branch1 = self.init_branch(cfg.model['branch1'])
        self.branch2 = self.init_branch(cfg.model['branch2'])
        self.reduction = 'sum'

        self.loss = FocalLoss(num_classes=self.cfg.model["args"]["NUM_CLASS"])

    def init_branch(self, cfg):
        embed_dim = cfg.model["args"]["EMBED_DIM"]
        assert cfg.extractors.get(
            "img_encoder", None) is not None, "img_encoder must be specified"

        visualExtrct = EXTRCT_REGISTRY.get(
            cfg.extractors["img_encoder"]["name"])(
                **cfg.extractors["img_encoder"]["args"])

        img_in_dim = visualExtrct.feature_dim
        logits = ClassifyBlock(
            inp_dim=img_in_dim,
            num_cls=self.num_cls,
            embed_dim=embed_dim,
        )
        return nn.Sequential(visualExtrct, self.normalize_block, logits)

    def normalize_block(
        self,
        embedding,
    ):
        return F.normalize(embedding, p=2, dim=-1)

    def forward(self, batch):
        assert "images" in batch.keys(), "Batch must contain images"
        logits1 = self.branch1(batch["images"])
        logits2 = self.branch2(batch["images"])
        return {"logits1": logits1, "logits2": logits2}

    def compute_loss(self, logits, batch, **kwargs):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx, reduction='sum'):
        assert "images" in batch.keys(), "Batch must contain images"

        logits = self.forward(batch)

        logits1 = logits['logits1']
        logits2 = logits['logits2']

        prob1 = torch.softmax(logits1, dim=1)
        prob2 = torch.softmax(logits2, dim=1)

        output = torch.stack([prob1, prob2], dim=0)  # [2, B, C, H, W]
        if reduction == 'sum':
            output = output.sum(dim=0)  #[B, C, H, W]
        elif reduction == 'max':
            output, _ = output.max(dim=0)  #[B, C, H, W]
        elif reduction == 'first':
            output = logits1
        elif reduction == 'second':
            output = logits2

        preds = {'logits': output}
        # add meta data for inference stage
        preds.update({
            'filenames': batch['filenames'],
            'video_ids': batch['video_ids'],
            'frame_ids': batch['frame_ids']
        })
        return preds
