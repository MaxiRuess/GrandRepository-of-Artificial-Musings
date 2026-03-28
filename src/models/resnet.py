"""
ResNet model factories for CIFAR-10.

Provides:
- make_resnet18_cifar10: Standard ResNet-18 adapted for 32x32 images
- CheckpointedResNet18: Same model with activation checkpointing on layer1-4
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torchvision.models as models


def make_resnet18_cifar10():
    """Create a ResNet-18 adapted for CIFAR-10 (32x32 images, 10 classes)."""
    model = models.resnet18(weights=None)
    # Adapt for 32x32: smaller first conv, remove aggressive maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


class CheckpointedResNet18(nn.Module):
    """ResNet-18 with activation checkpointing on layer1-4."""

    def __init__(self):
        super().__init__()
        base = make_resnet18_cifar10()
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = base.fc

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = cp.checkpoint(self.layer1, x, use_reentrant=False)
        x = cp.checkpoint(self.layer2, x, use_reentrant=False)
        x = cp.checkpoint(self.layer3, x, use_reentrant=False)
        x = cp.checkpoint(self.layer4, x, use_reentrant=False)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
