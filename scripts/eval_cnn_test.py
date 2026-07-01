"""
Evaluate trained DASResNetClassifier on the held-out test set and generate
publication-quality figures for the presentation.

Saves to --output (default: ppt_results/):
  confusion_matrix_{name}.png  -- normalized confusion matrix
  roc_pr_{name}.png            -- ROC + PR curves, all classes
  gallery_{name}.png           -- one correct + one incorrect sample per class
  comparison_f1.png            -- F1 per class, comparing all runs (if >1)
  metrics.csv                  -- all metrics

Usage:
    # Single model
    python scripts/eval_cnn_test.py \\
        --config configs/cnn_classifier_config.yaml \\
        --runs baseline:checkpoints/das_cnn_classifier/cnn_best.pt

    # Compare multiple models
    python scripts/eval_cnn_test.py \\
        --config configs/cnn_classifier_config.yaml \\
        --runs baseline:checkpoints/das_cnn_classifier/cnn_best.pt \\
               bg_mixup:checkpoints/das_cnn_classifier/... \\
               ratio_0p5:checkpoints/das_cnn_classifier/synth_ratio_0p5/cnn_best.pt \\
               ratio_1p0:checkpoints/das_cnn_classifier/synth_ratio_1p0/cnn_best.pt \\
        --output ppt_results/
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.das_latent_patch_dataset import DASLatentPatchDataset
from src.data.splits import recording_level_split
from src.models.das_cnn_classifier import DASResNetClassifier

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ── data ──────────────────────────────────────────────────────────────────────

def build_test_loader(cfg: dict, batch_size: int = 64) -> DataLoader:
    data_cfg = cfg["data"]
    normalize = (data_cfg["normalize"]["mean"], data_cfg["normalize"]["std"])
    full_dataset = DASLatentPatchDataset(
        data_dir=data_cfg["data_dir"],
        patch_channels=data_cfg["patch_channels"],
        patch_time=data_cfg["patch_time"],
        event_offset_range=tuple(data_cfg["event_offset_range"]),
        decimation=data_cfg["decimation"],
        classes=data_cfg["classes"],
        normalize=normalize,
        seed=data_cfg["seed"],
        return_mixed=False,
        cache_in_ram=data_cfg.get("cache_in_ram", False),
        target_sample_rate=data_cfg.get("target_sample_rate", 500),
    )
    _, _, test_ds = recording_level_split(
        full_dataset,
        val_frac=data_cfg["val_split"],
        test_frac=data_cfg["test_split"],
        seed=data_cfg["seed"],
    )
    print(f"Test set: {len(test_ds)} patches")
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


# ── model ─────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, cfg: dict, device: str) -> torch.nn.Module:
    model_cfg = cfg["model"]
    model = DASResNetClassifier(
        n_classes=model_cfg["n_classes"],
        embed_dim=model_cfg["embed_dim"],
        dropout=model_cfg["dropout"],
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(device)


# ── inference ─────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, class_names):
    """Run inference on the loader. Returns metrics dict + per-class sample bank."""
    all_preds, all_targets, all_probs = [], [], []
    # sample bank: {cls: {"correct": [...], "wrong": [...]}}
    bank = {i: {"correct": [], "wrong": []} for i in range(len(class_names))}
    MAX_PER_BUCKET = 3

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating", leave=False):
            patch, class_idx = batch[0], batch[1]
            patch = patch.to(device)
            labels = class_idx.long().to(device)

            logits = model(patch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            targets = labels.cpu().numpy()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

            # collect samples for gallery
            patches_np = patch.cpu().numpy()
            for i in range(len(targets)):
                t, p = int(targets[i]), int(preds[i])
                bucket = "correct" if p == t else "wrong"
                if len(bank[t][bucket]) < MAX_PER_BUCKET:
                    bank[t][bucket].append({
                        "patch": patches_np[i],   # [1, C, T]
                        "pred": p,
                        "true": t,
                        "probs": probs[i].tolist(),
                    })

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    labels = list(range(len(class_names)))
    prec, rec, f1, sup = precision_recall_fscore_support(
        all_targets, all_preds, labels=labels, zero_division=0
    )
    f1_mac = f1_score(all_targets, all_preds, labels=labels, average="macro", zero_division=0)
    acc = accuracy_score(all_targets, all_preds)

    # per-class AUC-ROC
    auc_roc = {}
    try:
        for i, name in enumerate(class_names):
            binary = (all_targets == i).astype(int)
            auc_roc[name] = float(roc_auc_score(binary, all_probs[:, i]))
    except Exception:
        auc_roc = {n: float("nan") for n in class_names}

    return {
        "preds": all_preds,
        "targets": all_targets,
        "probs": all_probs,
        "acc": float(acc),
        "f1_macro": float(f1_mac),
        "precision": {class_names[i]: float(prec[i]) for i in labels},
        "recall":    {class_names[i]: float(rec[i])  for i in labels},
        "f1":        {class_names[i]: float(f1[i])   for i in labels},
        "support":   {class_names[i]: int(sup[i])    for i in labels},
        "auc_roc":   auc_roc,
        "bank":      bank,
    }


# ── figures ───────────────────────────────────────────────────────────────────

def plot_confusion_matrix(metrics, run_name: str, class_names, out_dir: str):
    cm = confusion_matrix(metrics["targets"], metrics["preds"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(class_names)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {run_name}\nAcc={metrics['acc']:.3f}  F1={metrics['f1_macro']:.3f}")

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                    ha="center", va="center", fontsize=9, color=color)

    fig.tight_layout()
    path = os.path.join(out_dir, f"confusion_matrix_{run_name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_roc_pr(metrics, run_name: str, class_names, out_dir: str):
    targets = metrics["targets"]
    probs = metrics["probs"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    ax = axes[0]
    for i, (name, color) in enumerate(zip(class_names, CLASS_COLORS)):
        binary = (targets == i).astype(int)
        fpr, tpr, _ = roc_curve(binary, probs[:, i])
        auc = metrics["auc_roc"][name]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {run_name}")
    ax.legend(loc="lower right")

    # PR
    ax = axes[1]
    for i, (name, color) in enumerate(zip(class_names, CLASS_COLORS)):
        binary = (targets == i).astype(int)
        prec_c, rec_c, _ = precision_recall_curve(binary, probs[:, i])
        ap = average_precision_score(binary, probs[:, i])
        ax.plot(rec_c, prec_c, color=color, lw=2, label=f"{name}  AP={ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR Curves — {run_name}")
    ax.legend(loc="lower left")

    fig.tight_layout()
    path = os.path.join(out_dir, f"roc_pr_{run_name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_gallery(metrics, run_name: str, class_names, out_dir: str):
    """One correct + one incorrect patch per class. 2 rows × n_cls columns."""
    bank = metrics["bank"]
    n_cls = len(class_names)
    n_rows = 2   # correct row, wrong row

    fig, axes = plt.subplots(
        n_rows * 2, n_cls,
        figsize=(3.5 * n_cls, 7),
        gridspec_kw={"height_ratios": [2, 1, 2, 1]},
    )
    axes = np.array(axes)

    row_labels = ["Correct", "Incorrect"]
    row_offsets = [0, 2]

    for row_i, (bucket, row_off) in enumerate(zip(["correct", "wrong"], row_offsets)):
        for col_i, cls_name in enumerate(class_names):
            samples = bank[col_i][bucket]
            ax_wave = axes[row_off, col_i]
            ax_bar  = axes[row_off + 1, col_i]

            if not samples:
                ax_wave.axis("off"); ax_bar.axis("off")
                if col_i == 0:
                    ax_wave.set_ylabel(row_labels[row_i], fontsize=9, labelpad=4)
                continue

            s = samples[0]
            patch_2d = np.array(s["patch"])[0]   # [C, T]
            probs = np.array(s["probs"])
            pred_idx = s["pred"]
            true_idx = s["true"]

            # Waveform heatmap (seismic)
            vmax = float(np.percentile(np.abs(patch_2d), 99)) or 1.0
            ax_wave.imshow(patch_2d, aspect="auto", cmap="seismic",
                           vmin=-vmax, vmax=vmax, origin="lower")
            correct = pred_idx == true_idx
            color = "green" if correct else "red"
            title = f"True: {class_names[true_idx]}" if correct else \
                    f"True: {class_names[true_idx]}\nPred: {class_names[pred_idx]}"
            ax_wave.set_title(title, fontsize=8, color=color, pad=2)
            ax_wave.set_xticks([]); ax_wave.set_yticks([])
            if col_i == 0:
                ax_wave.set_ylabel(row_labels[row_i], fontsize=9, labelpad=4)

            # Probability bar chart
            bar_colors = [
                "forestgreen" if j == true_idx else "steelblue"
                for j in range(n_cls)
            ]
            ax_bar.barh(class_names, probs, color=bar_colors, height=0.6)
            ax_bar.set_xlim(0, 1)
            ax_bar.tick_params(labelsize=7)
            ax_bar.set_xticks([0, 0.5, 1])

    fig.suptitle(f"Prediction Gallery — {run_name}", fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, f"gallery_{run_name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_comparison(all_metrics: dict, class_names, out_dir: str):
    """Grouped bar chart: F1 per class × run. One summary bar for macro-F1."""
    run_names = list(all_metrics.keys())
    n_runs = len(run_names)
    n_cls = len(class_names)

    fig, axes = plt.subplots(1, 2, figsize=(5 + 2 * n_runs, 5),
                             gridspec_kw={"width_ratios": [n_cls, 1]})

    # Left: per-class F1
    ax = axes[0]
    x = np.arange(n_cls)
    width = 0.8 / n_runs
    for i, (run_name, m) in enumerate(all_metrics.items()):
        f1_vals = [m["f1"][c] for c in class_names]
        offset = (i - n_runs / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1_vals, width, label=run_name, alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(class_names)
    ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1 — Test Set")
    ax.legend()
    ax.axhline(1.0, color="gray", lw=0.5, ls="--")

    # Right: macro-F1 summary
    ax = axes[1]
    macro_f1s = [all_metrics[r]["f1_macro"] for r in run_names]
    accs = [all_metrics[r]["acc"] for r in run_names]
    y = np.arange(n_runs)
    ax.barh(y - 0.2, macro_f1s, 0.35, label="Macro F1", color="#4C72B0", alpha=0.85)
    ax.barh(y + 0.2, accs, 0.35, label="Accuracy", color="#DD8452", alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(run_names)
    ax.set_xlim(0, 1.05)
    ax.set_title("Overall")
    ax.legend()
    for j, (f1v, acc) in enumerate(zip(macro_f1s, accs)):
        ax.text(f1v + 0.01, j - 0.2, f"{f1v:.3f}", va="center", fontsize=9)
        ax.text(acc + 0.01, j + 0.2, f"{acc:.3f}", va="center", fontsize=9)

    fig.suptitle("Model Comparison — Test Set", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_f1.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def save_csv(all_metrics: dict, class_names, out_dir: str):
    rows = []
    for run_name, m in all_metrics.items():
        row = {"run": run_name, "acc": m["acc"], "f1_macro": m["f1_macro"]}
        for cls in class_names:
            row[f"f1_{cls}"]        = m["f1"][cls]
            row[f"precision_{cls}"] = m["precision"][cls]
            row[f"recall_{cls}"]    = m["recall"][cls]
            row[f"auc_roc_{cls}"]   = m["auc_roc"][cls]
            row[f"support_{cls}"]   = m["support"][cls]
        rows.append(row)

    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(out_dir, "metrics.json")
    serializable = {
        run: {k: v for k, v in m.items() if k not in ("preds", "targets", "probs", "bank")}
        for run, m in all_metrics.items()
    }
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"  saved: {csv_path}")
    print(f"  saved: {json_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cnn_classifier_config.yaml")
    parser.add_argument(
        "--runs", nargs="+", required=True,
        metavar="NAME:CKPT_PATH",
        help="One or more name:checkpoint_path pairs",
    )
    parser.add_argument("--output", default="ppt_results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    class_names = cfg["data"]["classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    test_loader = build_test_loader(cfg, batch_size=cfg["training"]["batch_size"])

    all_metrics = {}

    for run_spec in args.runs:
        if ":" not in run_spec:
            raise ValueError(f"Expected name:path format, got: {run_spec!r}")
        run_name, ckpt_path = run_spec.split(":", 1)
        print(f"\n=== {run_name} ({ckpt_path}) ===")

        model = load_model(ckpt_path, cfg, device)
        metrics = evaluate(model, test_loader, device, class_names)
        all_metrics[run_name] = metrics

        print(f"  acc={metrics['acc']:.4f}  f1_macro={metrics['f1_macro']:.4f}")
        for cls in class_names:
            print(
                f"  {cls:10s}  P={metrics['precision'][cls]:.3f}  "
                f"R={metrics['recall'][cls]:.3f}  F1={metrics['f1'][cls]:.3f}  "
                f"AUC={metrics['auc_roc'][cls]:.3f}"
            )

        plot_confusion_matrix(metrics, run_name, class_names, args.output)
        plot_roc_pr(metrics, run_name, class_names, args.output)
        plot_gallery(metrics, run_name, class_names, args.output)

    if len(all_metrics) > 1:
        plot_comparison(all_metrics, class_names, args.output)

    save_csv(all_metrics, class_names, args.output)
    print(f"\nAll results saved to: {args.output}/")


if __name__ == "__main__":
    main()
