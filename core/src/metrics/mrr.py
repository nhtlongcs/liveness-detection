from typing import List
import numpy as np
from . import METRIC_REGISTRY

@METRIC_REGISTRY.register()
class MeanReciprocalRank:
    """
    Mean Reciprocal Rank
    https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    """
    def __init__(self):
        self.reset()

    def update(self, output: List, target: List):
        for rank, pred in enumerate(output):
            if pred in target:
                break   
        else:
            rank = -1
        
        # MRR score
        if rank > -1:
            score = 1.0 / (rank+1) 
        else:
            score = 0

        self.scores.append(score)

    def reset(self):
        self.scores = []

    def value(self):
        mean_score = np.mean(self.scores)
        return {f"mrr" : mean_score}