"""
Evaluate CVAE augmentation benefit.

Experiment:
  A) Train DASClassifier on real data only
  B) Train DASClassifier on real + synthetic (generated via CVAE)

Usage:
    python scripts/evaluate_augmentation.py \
        --config configs/cvae_config.yaml \
        --generated-dir generated/
"""

import argparse
import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, random_split
from sklearn.metrics import classification_report

from src.data.das_patch_dataset import DASPatchDataset, CLASSES, N_CLASSES
from src.evaluation.classifier import DASClassifier, train_classifier


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_classifier(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            patch, cls_idx = batch[0], batch[1]
            patch = patch.to(device)
            if isinstance(cls_idx, torch.Tensor):
                cls_idx = cls_idx.to(device)
            else:
                cls_idx = torch.tensor(cls_idx, dtype=torch.long, device=device)
            preds = model(patch).argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(cls_idx.cpu().tolist())
    return all_preds, all_labels


def load_synthetic(generated_dir: str, classes: list) -> tuple:
    """Load generated .npy files → tensors."""
    patches, labels = [], []
    for cls_idx, cls_name in enumerate(classes):
        path = os.path.join(generated_dir, f"{cls_name}.npy")
        if not os.path.exists(path):
            print(f"  Warning: no synthetic samples for {cls_name}")
            continue
        arr = np.load(path)  # [N, 1, 32, 256]
        patches.append(torch.tensor(arr, dtype=torch.float32))
        labels.append(torch.full((len(arr),), cls_idx, dtype=torch.long))
    return torch.cat(patches), torch.cat(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cvae_config.yaml")
    parser.add_argument("--generated-dir", default="generated/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    classes = data_cfg["classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Real data ----
    dataset = DASPatchDataset(
        data_dir=data_cfg["data_dir"],
        patch_channels=data_cfg["patch_channels"],
        patch_time=data_cfg["patch_time"],
        bitmap_shift=data_cfg["bitmap_shift"],
        decimation=data_cfg["decimation"],
        classes=classes,
    )
    n_test = int(len(dataset) * data_cfg["test_split"])
    n_val = int(len(dataset) * data_cfg["val_split"])
    n_train = len(dataset) - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    # ---- Experiment A: Real only ----
    print("\n=== Experiment A: Train on REAL data only ===")
    clf_a = DASClassifier(n_classes=len(classes))
    train_classifier(clf_a, train_loader, val_loader, epochs=30, device=device)
    preds_a, labels_a = evaluate_classifier(clf_a, test_loader, device)
    print(classification_report(labels_a, preds_a, target_names=classes))

    # ---- Experiment B: Real + Synthetic ----
    print("\n=== Experiment B: Train on REAL + SYNTHETIC data ===")
    syn_patches, syn_labels = load_synthetic(args.generated_dir, classes)
    syn_onehot = torch.zeros(len(syn_labels), N_CLASSES)
    syn_onehot.scatter_(1, syn_labels.unsqueeze(1), 1.0)
    syn_dataset = TensorDataset(syn_patches, syn_labels, syn_onehot)

    aug_train_ds = ConcatDataset([train_ds, syn_dataset])
    aug_loader = DataLoader(aug_train_ds, batch_size=64, shuffle=True, num_workers=4)

    clf_b = DASClassifier(n_classes=len(classes))
    train_classifier(clf_b, aug_loader, val_loader, epochs=30, device=device)
    preds_b, labels_b = evaluate_classifier(clf_b, test_loader, device)
    print(classification_report(labels_b, preds_b, target_names=classes))


if __name__ == "__main__":
    main()
