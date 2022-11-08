from typing import List
import torch
from torch import nn
from . import METRIC_REGISTRY
import numpy as np


@METRIC_REGISTRY.register()
class Accuracy:
    """
    Acc Score
    """

    def __init__(self,label_key="labels", **kwargs):
        super().__init__(**kwargs)
        self.label_key = label_key
        self.threshold = kwargs.get("threshold", 0.5)
        self.reset()

    def update(self, preds, batch):
        """
        Perform calculation based on prediction and targets
        """
        preds = preds["logits"].argmax(dim=1)
        targets = batch[self.label_key]
        preds = (preds > self.threshold) * 1
        preds = preds.detach().cpu().long()
        targets = targets.detach().cpu().long()
        self.preds += preds.numpy().tolist()
        self.targets += targets.numpy().tolist()

    def reset(self):
        self.targets = []
        self.preds = []

    def value(self):
        score = np.mean(np.array(self.targets) == np.array(self.preds))
        return {f"Accuracy": score}

