"""
DASResNetClassifier — ResNet-34-like CNN for DAS event classification.

Input:  [B, 1, 8, T]  (normalized 500 Hz DAS patch, 8 fiber channels)
Output: [B, n_classes] logits (one linear head per class)

Architecture mirrors DOVE's pzresnet34 pattern: shared backbone + per-class heads.

Stem (2× asymmetric conv, stride-4 temporal)   [B, 64, 8, T/16]
Layer 1 (2× BasicBlock, stride=(1,2))           [B,  64, 8, T/32]
Layer 2 (2× BasicBlock, stride=(2,2))           [B, 128, 4, T/64]
Layer 3 (2× BasicBlock, stride=(2,2))           [B, 256, 2, T/128]
Layer 4 (2× BasicBlock, stride=(2,2))           [B, 512, 1, T/256]
AdaptiveAvgPool2d(1,1) + Flatten                [B, embed_dim]
Dropout
n_classes × Linear(embed_dim→1) heads          [B, n_classes]

For T=16384 all divisions are exact; AdaptiveAvgPool handles any other T.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=(1, 1)):
        super().__init__()
        stride_t = stride if isinstance(stride, tuple) else (stride, stride)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride_t, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride_t != (1, 1) or in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride_t, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class DASResNetClassifier(nn.Module):
    def __init__(self, n_classes: int = 4, embed_dim: int = 512, dropout: float = 0.35):
        super().__init__()
        self.n_classes = n_classes
        self.embed_dim = embed_dim

        # Two asymmetric convolutions compress the time axis by 16× before ResNet blocks.
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, (1, 7), stride=(1, 4), padding=(0, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (1, 7), stride=(1, 4), padding=(0, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64,  64,       stride=(1, 2))
        self.layer2 = self._make_layer(64,  128,      stride=(2, 2))
        self.layer3 = self._make_layer(128, 256,      stride=(2, 2))
        self.layer4 = self._make_layer(256, embed_dim, stride=(2, 2))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(n_classes)])

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, stride) -> nn.Sequential:
        return nn.Sequential(
            BasicBlock(in_ch, out_ch, stride),
            BasicBlock(out_ch, out_ch),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.drop(self.pool(x).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return torch.cat([h(feat) for h in self.heads], dim=1)
