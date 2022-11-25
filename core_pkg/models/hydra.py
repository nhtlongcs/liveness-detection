import torch
import torch.nn as nn
import torch.nn.functional as F
from core.extractors import EXTRCT_REGISTRY
from core.utils.losses import FocalLoss
from core.models import MODEL_REGISTRY
from core.models.utils import ClassifyBlock, NormalizeBlock
from .cps import SemiSuperviseModel
from .utils import sigmoid_rampup, cosine_rampdown, linear_rampup


@MODEL_REGISTRY.register()
class DualClassifier(SemiSuperviseModel):

    def __init__(self, config):
        super().__init__(config)

    def init_model(self):

        assert self.cfg.model["args"].get(
            "NUM_CLASS", None) is not None, "NUM_CLASS must be specified"
        self.num_cls = self.cfg.model["args"]["NUM_CLASS"]
        self.branch1 = self.init_branch(self.cfg.model['branch1'],
                                        self.cfg.extractors['branch1'],
                                        'branch1')
        self.branch2 = self.init_branch(self.cfg.model['branch2'],
                                        self.cfg.extractors['branch2'],
                                        'branch2')
        self.reduction = 'sum'
        self.sup_loss_func = FocalLoss(
            num_classes=self.cfg.model["args"]["NUM_CLASS"])
        self.unsup_loss_func = FocalLoss(
            num_classes=self.cfg.model["args"]["NUM_CLASS"])

    def init_branch(self, cfg, extr_cfg, branch_name):
        embed_dim = cfg["args"]["EMBED_DIM"]
        assert extr_cfg.get("img_encoder",
                            None) is not None, "img_encoder must be specified"

        visualExtrct = EXTRCT_REGISTRY.get(
            extr_cfg["img_encoder"]["name"])(**extr_cfg["img_encoder"]["args"])

        img_in_dim = visualExtrct.feature_dim
        normalize_block = NormalizeBlock()
        logits = ClassifyBlock(
            inp_dim=img_in_dim,
            num_cls=self.num_cls,
            embed_dim=embed_dim,
        )
        return nn.Sequential(visualExtrct, normalize_block, logits)

    def forward(self, batch):
        assert "images" in batch.keys(), "Batch must contain images"
        logits1 = self.branch1(batch["images"])
        logits2 = self.branch2(batch["images"])
        logits = torch.softmax(logits1, dim=1) + torch.softmax(logits2, dim=1)
        return {"logits1": logits1, "logits2": logits2, "logits": logits}

    def compute_supervised_loss(self, logits, batch):
        # Supervised loss
        # import pdb; pdb.set_trace()
        return self.sup_loss_func(logits, batch["labels"].long())

    def compute_unsupervised_loss(self, logits, batch):
        # Unsupervised loss
        # import pdb; pdb.set_trace()
        return self.unsup_loss_func(logits, batch["labels"].long())

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.consistency * sigmoid_rampup(epoch,
                                                 self.consistency_rampup)

    def compute_loss(self, preds, batch, **kwargs):
        """
        Forward inputs and targets through multiple losses
        """
        logits1 = preds['logits1']
        logits2 = preds['logits2']
        if batch.get("split_pos", None) is None:
            split_pos = len(batch["labels"])
        else:
            split_pos = batch['split_pos']
            assert split_pos < len(
                batch["images"]
            ), "split_pos must be less than batch size in training config"

        outputs_soft1 = torch.softmax(logits1, dim=1)
        outputs_soft2 = torch.softmax(logits2, dim=1)

        pseudo_outputs1 = torch.argmax(outputs_soft1[split_pos:].detach(),
                                       dim=1,
                                       keepdim=False)
        pseudo_outputs2 = torch.argmax(outputs_soft2[split_pos:].detach(),
                                       dim=1,
                                       keepdim=False)

        supLoss1 = self.compute_supervised_loss(logits1[:split_pos], batch)
        supLoss2 = self.compute_supervised_loss(logits2[:split_pos], batch)

        # Unsupervised loss
        if split_pos == len(batch["images"]):
            unsupLoss1 = torch.tensor(0.0).to(self.device)
            unsupLoss2 = torch.tensor(0.0).to(self.device)
        else:
            unsupLoss1 = self.compute_unsupervised_loss(
                logits1[split_pos:], batch={'labels': pseudo_outputs2})
            unsupLoss2 = self.compute_unsupervised_loss(
                logits2[split_pos:], batch={'labels': pseudo_outputs1})
        # import pdb; pdb.set_trace()
        # https://github.com/Lightning-AI/lightning/issues/1424
        consistency_weight = self.get_current_consistency_weight(
            self.current_epoch)

        model1_loss = supLoss1 + consistency_weight * unsupLoss1
        model2_loss = supLoss2 + consistency_weight * unsupLoss2
        total_loss = model1_loss + model2_loss
        # import pdb; pdb.set_trace()

        total_loss_dict = {
            'loss': total_loss.item(),
            'model1_loss': model1_loss.item(),
            'model2_loss': model2_loss.item(),
            'supLoss1': supLoss1.item(),
            'supLoss2': supLoss2.item(),
            'unsupLoss1': unsupLoss1.item(),
            'unsupLoss2': unsupLoss2.item(),
            'consistency_weight': consistency_weight,
        }
        return total_loss, total_loss_dict

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
