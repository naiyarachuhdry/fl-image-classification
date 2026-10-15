"""Pretrained feature extractor and linear head for FedPFT."""
from collections import OrderedDict

import torch
from torch import nn
from torchvision import models

# name -> (constructor, pretrained weights, feature dimension)
_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
}


def build_feature_extractor(name: str = "resnet18") -> nn.Module:
    """Frozen ImageNet-pretrained backbone that outputs flat feature vectors."""
    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list(_BACKBONES)}")
    ctor, weights, dim = _BACKBONES[name]
    backbone = ctor(weights=weights)
    # drop the final fc layer, keep avgpool, flatten to (batch, dim)
    extractor = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
    for p in extractor.parameters():
        p.requires_grad_(False)
    extractor.eval()
    extractor.hidden_dimension = dim
    return extractor


class FedPFTNet(nn.Module):
    """Frozen extractor + trainable linear head (used for evaluation)."""

    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()
        self.extractor = build_feature_extractor(backbone_name)
        self.head = nn.Linear(self.extractor.hidden_dimension, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.extractor.eval()  # keep BatchNorm statistics frozen
        return self

    def forward(self, x):
        with torch.no_grad():
            feats = self.extractor(x)
        return self.head(feats)

    def load_head(self, state: "OrderedDict") -> None:
        self.head.load_state_dict(state)