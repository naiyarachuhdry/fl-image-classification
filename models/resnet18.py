import torch.nn as nn
from torchvision import models

from .base import BaseModel


class ResNet18Model(BaseModel):

    def build(self) -> nn.Module:
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False

        # Classification head
        model.fc = nn.Linear(512, 10)

        return model