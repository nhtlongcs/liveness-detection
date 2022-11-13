import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModel,
)

from .extrct_net import ExtractorNetwork
from . import EXTRCT_REGISTRY


class HuggingFaceBlock(nn.Module):
    def __init__(self, model_name, num_labels):
        super(HuggingFaceBlock, self).__init__()

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": num_labels,
            }
        )
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)

    def forward(self, input_ids, attention_mask):
        return self.transformer(input_ids, attention_mask)


@EXTRCT_REGISTRY.register()
class LangExtractor(ExtractorNetwork):
    def __init__(self, pretrained: str, freeze=False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained)
        self.extractor = AutoModel.from_pretrained(pretrained, config=self.config)
        if freeze:
            self.freeze()
        self.feature_dim = self.extractor.config.hidden_size

    def forward(self, batch):
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        transformer_out = self.extractor(
            input_ids=input_ids, attention_mask=attention_mask
        )
        feature = transformer_out.last_hidden_state
        feature = torch.mean(feature, dim=1)

        return feature
