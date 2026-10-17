import torch.nn as nn
from torchvision import models

from .base import BaseModel


class ResNet50Model(BaseModel):

    def build(self) -> nn.Module:
        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False

        # Classification head
        model.fc = nn.Linear(2048, 10)

        return model