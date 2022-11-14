import torch.nn as nn

from .extrct_net import ExtractorNetwork
from efficientnet_pytorch import EfficientNet

from . import EXTRCT_REGISTRY


@EXTRCT_REGISTRY.register()
class EfficientNetExtractor(ExtractorNetwork):

    def __init__(self, version, from_pretrained=True, freeze=False):
        super().__init__()
        assert version in range(8)
        self.extractor = EfficientNet.from_name(f"efficientnet-b{version}")
        if from_pretrained:
            self.extractor = EfficientNet.from_pretrained(
                f"efficientnet-b{version}")
        self.feature_dim = self.extractor._fc.in_features
        if freeze:
            self.freeze()

    def forward(self, x):
        x = self.extractor.extract_features(x)
        x = self.extractor._avg_pooling(x)
        x = x.view(x.size(0), -1)
        return x
