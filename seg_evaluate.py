"""
Publication-ready figures for segmentation results.
1. Training curves (loss, Dice, IoU)
2. Metric comparison bar chart (baseline vs synthetic)
3. Qualitative segmentation results on test set
"""
import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from seg_model import SegUNet
from seg_dataset import SegmentationDataset

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})


def plot_seg_training_curves(log, mode):
    """Loss, Dice, IoU curves for a single mode."""
    epochs = log["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))

    # Loss
    axes[0].plot(epochs, log["train_loss"], color="#2563EB", label="Train", alpha=0.85)
    axes[0].plot(epochs, log["val_loss"], color="#DC2626", label="Val", alpha=0.85)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Dice + BCE Loss")
    axes[0].set_title("Loss")
    axes[0].legend(framealpha=0.9, edgecolor="none")

    # Dice
    axes[1].plot(epochs, log["train_dice"], color="#2563EB", label="Train", alpha=0.85)
    axes[1].plot(epochs, log["val_dice"], color="#DC2626", label="Val", alpha=0.85)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].set_title("Dice Coefficient")
    axes[1].legend(framealpha=0.9, edgecolor="none")
    axes[1].set_ylim(0, 1)

    # IoU
    axes[2].plot(epochs, log["train_iou"], color="#2563EB", label="Train", alpha=0.85)
    axes[2].plot(epochs, log["val_iou"], color="#DC2626", label="Val", alpha=0.85)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].set_title("Intersection over Union")
    axes[2].legend(framealpha=0.9, edgecolor="none")
    axes[2].set_ylim(0, 1)

    for ax in axes:
        ax.set_xlim(1, max(epochs))

    fig.suptitle(f"Segmentation Training — {mode.capitalize()}", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"seg_curves_{mode}.jpg"), format="jpg")
    plt.close(fig)
    print(f"Saved seg_curves_{mode}.jpg")


def plot_metric_comparison(baseline_log, synthetic_log):
    """Bar chart comparing test metrics between baseline and synthetic-augmented."""
    metrics = ["dice", "iou", "precision", "recall", "accuracy", "specificity"]
    labels = ["Dice", "IoU", "Precision", "Recall", "Accuracy", "Specificity"]

    base_vals = [baseline_log["test"][m] for m in metrics]
    syn_vals = [synthetic_log["test"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - width / 2, base_vals, width, label="Baseline (Real Only)", color="#2563EB", alpha=0.85)
    bars2 = ax.bar(x + width / 2, syn_vals, width, label="Real + Synthetic", color="#16A34A", alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("Test Set Segmentation Metrics: Baseline vs. Synthetic Augmentation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.9, edgecolor="none")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "seg_metric_comparison.jpg"), format="jpg")
    plt.close(fig)
    print("Saved seg_metric_comparison.jpg")


def plot_qualitative_results(model, test_ds, mode, n=6):
    """Show input image, ground truth mask, predicted mask side by side."""
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(n * 2, 6.5))
    row_labels = ["Input Image", "Ground Truth", "Prediction"]

    for i in range(n):
        img, mask = test_ds[i]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(DEVICE))
            pred = (torch.sigmoid(pred) > 0.5).float()

        img_np = (img[0].numpy() * 255).astype(np.uint8)
        mask_np = (mask[0].numpy() * 255).astype(np.uint8)
        pred_np = (pred[0, 0].cpu().numpy() * 255).astype(np.uint8)

        axes[0, i].imshow(img_np, cmap="gray", vmin=0, vmax=255)
        axes[1, i].imshow(mask_np, cmap="gray", vmin=0, vmax=255)
        axes[2, i].imshow(pred_np, cmap="gray", vmin=0, vmax=255)

        for r in range(3):
            axes[r, i].axis("off")

    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=11, rotation=90, labelpad=12)

    fig.suptitle(f"Segmentation Results — {mode.capitalize()}", fontsize=13, y=1.01)
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    fig.savefig(os.path.join(FIG_DIR, f"seg_qualitative_{mode}.jpg"), format="jpg")
    plt.close(fig)
    print(f"Saved seg_qualitative_{mode}.jpg")


def main():
    test_ds = SegmentationDataset("data/test", augment=False)

    # Try baseline
    baseline_path = "seg_log_baseline.json"
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline_log = json.load(f)
        plot_seg_training_curves(baseline_log, "baseline")

        model = SegUNet().to(DEVICE)
        model.load_state_dict(torch.load("seg_checkpoints_baseline/best.pt", map_location=DEVICE, weights_only=True))
        plot_qualitative_results(model, test_ds, "baseline", n=6)

    # Try synthetic
    synthetic_path = "seg_log_synthetic.json"
    if os.path.exists(synthetic_path):
        with open(synthetic_path) as f:
            synthetic_log = json.load(f)
        plot_seg_training_curves(synthetic_log, "synthetic")

        model = SegUNet().to(DEVICE)
        model.load_state_dict(torch.load("seg_checkpoints_synthetic/best.pt", map_location=DEVICE, weights_only=True))
        plot_qualitative_results(model, test_ds, "synthetic", n=6)

    # Comparison plot if both exist
    if os.path.exists(baseline_path) and os.path.exists(synthetic_path):
        plot_metric_comparison(baseline_log, synthetic_log)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
