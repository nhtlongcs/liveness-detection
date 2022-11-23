import torch.nn as nn


class ClassifyBlock(nn.Module):

    def __init__(self, inp_dim, num_cls, embed_dim=1024) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(inp_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )
        self.projection = nn.Linear(embed_dim, num_cls)

    def forward(self, x, return_embed=False):
        x = self.model(x)
        if return_embed:
            return x, self.projection(x)
        return self.projection(x)
