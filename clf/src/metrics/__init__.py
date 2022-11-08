from registry import Registry

METRIC_REGISTRY = Registry("METRIC")

from .metric_wrapper import RetrievalMetric
from .topkrecall import TopKRecall
from .mrr import MeanReciprocalRank

from .classification import *
