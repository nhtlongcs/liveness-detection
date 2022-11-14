import torch.nn as nn
from torchvision.ops.focal_loss import sigmoid_focal_loss


class FocalLoss(nn.Module):

    def __init__(self, num_classes, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
        if inputs.shape != targets.shape:
            targets = nn.functional.one_hot(targets,
                                            num_classes=self.num_classes)
        targets = targets.float()
        return sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma,
                                  self.reduction)
