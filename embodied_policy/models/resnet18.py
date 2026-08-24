"""Minimal pure-PyTorch ResNet-18 compatible with torchvision ImageNet weights."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
            if stride != 1 or in_channels != channels
            else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        if self.downsample is not None:
            identity = self.downsample(inputs)
        return self.relu(outputs + identity)


class ResNet18Backbone(nn.Module):
    """ResNet-18 trunk returning one 512-D feature vector per image."""

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers: list[nn.Module] = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(BasicBlock(self.in_channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        return self.avgpool(outputs).flatten(1)

    def load_imagenet_weights(self, path: str | Path) -> None:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        filtered = {key: value for key, value in state_dict.items() if not key.startswith("fc.")}
        self.load_state_dict(filtered, strict=True)
