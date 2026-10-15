
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel

class ResNet18MLBModel(BaseModel):
    def build(self):
        return ResNet18MLB(outputs=10)

def group_norm(channels: int) -> nn.Module:
    """Group Normalization used by the original FedMLB architecture."""
    return nn.GroupNorm(num_groups=2, num_channels=channels, eps=1e-5)


class ResBlock(nn.Module):
    """Two-convolution residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.gn1 = group_norm(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.gn2 = group_norm(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                group_norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))

        return F.relu(out + residual)


class ResNet18MLB(nn.Module):
    """FedMLB ResNet-18 with five intermediate feature outputs."""

    def __init__(self, outputs: int = 10):
        super().__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            ),
            group_norm(64),
            nn.ReLU(inplace=False),
        )

        self.layer1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
        )

        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
        )

        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
        )

        self.layer4 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, outputs)

        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        level: int = 0,
    ):
        """Return final logits or features for FedMLB branch predictions."""
        if level <= 0:
            out0 = self.layer0(x)
        else:
            out0 = x

        if level <= 1:
            out1 = self.layer1(out0)
        else:
            out1 = out0

        if level <= 2:
            out2 = self.layer2(out1)
        else:
            out2 = out1

        if level <= 3:
            out3 = self.layer3(out2)
        else:
            out3 = out2

        if level <= 4:
            out4 = self.layer4(out3)
            out4 = self.gap(out4).flatten(1)
        else:
            out4 = out3

        logits = self.fc(out4)

        if return_features:
            return out0, out1, out2, out3, out4, logits

        return logits