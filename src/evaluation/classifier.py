"""
Simple Conv2D classifier for DAS event patches.

Input:  [B, 1, 32, 256]
Output: [B, n_classes] logits
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List


class DASClassifier(nn.Module):
    def __init__(self, n_classes: int = 9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # [B,32,16,128]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # [B,64,8,64]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # [B,128,4,32]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),               # [B,128,2,2]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def train_classifier(
    model: DASClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cuda",
) -> Dict[str, List[float]]:
    """
    Train classifier and return dict with train/val accuracy history.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {"train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        # Train
        model.train()
        correct, total = 0, 0
        for batch in train_loader:
            patch, class_idx, _ = batch[0], batch[1], batch[2]
            patch = patch.to(device)
            if isinstance(class_idx, torch.Tensor):
                class_idx = class_idx.to(device)
            else:
                class_idx = torch.tensor(class_idx, dtype=torch.long, device=device)
            logits = model(patch)
            loss = criterion(logits, class_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == class_idx).sum().item()
            total += len(class_idx)
        history["train_acc"].append(correct / total)

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                patch, class_idx, _ = batch[0], batch[1], batch[2]
                patch = patch.to(device)
                if isinstance(class_idx, torch.Tensor):
                    class_idx = class_idx.to(device)
                else:
                    class_idx = torch.tensor(class_idx, dtype=torch.long, device=device)
                logits = model(patch)
                correct += (logits.argmax(1) == class_idx).sum().item()
                total += len(class_idx)
        history["val_acc"].append(correct / total)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | "
                  f"train_acc={history['train_acc'][-1]:.3f} | "
                  f"val_acc={history['val_acc'][-1]:.3f}")

    return history
