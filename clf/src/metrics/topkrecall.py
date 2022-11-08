from typing import List
import numpy as np
from . import METRIC_REGISTRY

@METRIC_REGISTRY.register()
class TopKRecall:
    """
    Top k item recall
    """
    def __init__(self, top_k=(1,)):
        self.top_k = top_k
        self.reset()
        
    def update(self, output: List, target: List):

        for k in self.top_k:
            retrieved_labels = output[:k]

            # Number of targets
            n_targets = len(target)          

            # Number of corrects
            n_relevant_objs = len(np.intersect1d(target,retrieved_labels)) 

            # Recall score
            score = n_relevant_objs*1.0 / n_targets
            self.scores[k].append(score)

    def reset(self):
        self.scores = {k: [] for k in self.top_k}

    def value(self):
        mean_score = {k: np.mean(v) for k,v in self.scores.items()}
        return {f"recall@{k}" : score for k, score in mean_score.items()}