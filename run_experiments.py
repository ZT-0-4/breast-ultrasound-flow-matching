"""
Automated experiment: train segmentation model with increasing synthetic data.
Runs 10 experiments: 100, 200, 300, ..., 1000 synthetic samples added to real training data.
Each experiment trains a full segmentation model and evaluates on the held-out test set.
Results are saved to a summary JSON and publication-ready comparison plots are generated.
"""
import os
import sys
import json
import time
import random
import shutil
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from seg_model import SegUNet
from seg_dataset import SegmentationDataset

# ---- Config ----
BATCH_SIZE = 16
EPOCHS = 200
LR = 1e-3
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
SYNTHETIC_COUNTS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
FULL_SYNTHETIC_DIR = "generated_seg_data"
EXPERIMENT_DIR = "experiments"
SUMMARY_FILE = "experiment_summary.json"

os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("experiments.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def compute_metrics(pred, target, threshold=0.5):
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    tn = ((1 - pred_bin) * (1 - target)).sum()
    return {
        "dice": (2 * tp / (2 * tp + fp + fn + 1e-8)).item(),
        "iou": (tp / (tp + fp + fn + 1e-8)).item(),
        "precision": (tp / (tp + fp + 1e-8)).item(),
        "recall": (tp / (tp + fn + 1e-8)).item(),
        "accuracy": ((tp + tn) / (tp + tn + fp + fn + 1e-8)).item(),
        "specificity": (tn / (tn + fp + 1e-8)).item(),
    }


class DiceBCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum()
        dice_loss = 1 - (2 * intersection + 1) / (pred_sig.sum() + target.sum() + 1)
        return bce_loss + dice_loss


def prepare_synthetic_subset(n_samples, subset_dir):
    """Copy first n_samples from full synthetic dir to a subset dir."""
    src_img = os.path.join(FULL_SYNTHETIC_DIR, "images")
    src_mask = os.path.join(FULL_SYNTHETIC_DIR, "masks")
    dst_img = os.path.join(subset_dir, "images")
    dst_mask = os.path.join(subset_dir, "masks")

    if os.path.exists(subset_dir):
        shutil.rmtree(subset_dir)
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_mask, exist_ok=True)

    all_files = sorted(os.listdir(src_img))[:n_samples]
    for f in all_files:
        shutil.copy2(os.path.join(src_img, f), os.path.join(dst_img, f))
        shutil.copy2(os.path.join(src_mask, f), os.path.join(dst_mask, f))
    return subset_dir


def train_and_evaluate(synthetic_dir, n_syn, save_dir):
    """Train segmentation model and return test metrics."""
    train_ds = SegmentationDataset("data/train", augment=True, synthetic_dir=synthetic_dir)
    val_ds = SegmentationDataset("data/val", augment=False)
    test_ds = SegmentationDataset("data/test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    log.info(f"  Train samples: {len(train_ds)} (556 real + {n_syn} synthetic)")

    model = SegUNet(in_channels=1, out_channels=1).to(DEVICE)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs(save_dir, exist_ok=True)
    best_val_dice = 0.0
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": [], "epoch": []}

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_losses = []
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses, val_metrics_list = [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                val_losses.append(criterion(preds, masks).item())
                val_metrics_list.append(compute_metrics(preds, masks))

        train_avg = sum(train_losses) / len(train_losses)
        val_avg = sum(val_losses) / len(val_losses)
        val_dice = sum(m["dice"] for m in val_metrics_list) / len(val_metrics_list)
        val_iou = sum(m["iou"] for m in val_metrics_list) / len(val_metrics_list)

        history["train_loss"].append(train_avg)
        history["val_loss"].append(val_avg)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["epoch"].append(epoch)

        scheduler.step()

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))

        if epoch % 50 == 0 or epoch == 1:
            log.info(f"    Epoch {epoch:03d}/{EPOCHS} | Train: {train_avg:.4f} | Val: {val_avg:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

    # Test evaluation with best model
    model.load_state_dict(torch.load(os.path.join(save_dir, "best.pt"), map_location=DEVICE, weights_only=True))
    model.eval()
    test_metrics_list = []
    test_losses = []
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            test_losses.append(criterion(preds, masks).item())
            test_metrics_list.append(compute_metrics(preds, masks))

    test_results = {}
    for key in test_metrics_list[0]:
        test_results[key] = sum(m[key] for m in test_metrics_list) / len(test_metrics_list)
    test_results["loss"] = sum(test_losses) / len(test_losses)

    # Save training history
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f)

    return test_results


def generate_summary_plots(summary):
    """Generate publication-ready plots comparing all experiments."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 11, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05, "axes.linewidth": 0.8, "lines.linewidth": 1.8,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    })

    counts = [0] + [r["n_synthetic"] for r in summary["experiments"]]
    baseline = summary["baseline"]

    metrics_to_plot = ["dice", "iou", "precision", "recall"]
    labels = ["Dice Score", "IoU", "Precision", "Recall"]
    colors = ["#2563EB", "#16A34A", "#DC2626", "#7C3AED"]

    # Plot 1: All metrics vs synthetic count
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for idx, (metric, label, color) in enumerate(zip(metrics_to_plot, labels, colors)):
        ax = axes[idx // 2, idx % 2]
        values = [baseline[metric]] + [r["test"][metric] for r in summary["experiments"]]
        ax.plot(counts, values, color=color, marker="o", markersize=5, linewidth=1.8)
        ax.axhline(y=baseline[metric], color="gray", linestyle="--", alpha=0.5, label="Baseline (no synthetic)")
        ax.set_xlabel("Number of Synthetic Samples")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(framealpha=0.9, edgecolor="none", fontsize=9)
        ax.set_xticks(counts)

    fig.suptitle("Segmentation Performance vs. Synthetic Data Augmentation", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/experiment_metrics_vs_synthetic.jpg", format="jpg")
    plt.close(fig)
    log.info("Saved experiment_metrics_vs_synthetic.jpg")

    # Plot 2: Dice and IoU together
    fig, ax = plt.subplots(figsize=(7, 4.5))
    dice_vals = [baseline["dice"]] + [r["test"]["dice"] for r in summary["experiments"]]
    iou_vals = [baseline["iou"]] + [r["test"]["iou"] for r in summary["experiments"]]
    ax.plot(counts, dice_vals, color="#2563EB", marker="o", markersize=6, label="Dice Score", linewidth=2)
    ax.plot(counts, iou_vals, color="#16A34A", marker="s", markersize=6, label="IoU", linewidth=2)
    ax.axhline(y=baseline["dice"], color="#2563EB", linestyle="--", alpha=0.3)
    ax.axhline(y=baseline["iou"], color="#16A34A", linestyle="--", alpha=0.3)
    ax.set_xlabel("Number of Synthetic Samples Added")
    ax.set_ylabel("Score")
    ax.set_title("Effect of Synthetic Data on Segmentation Performance")
    ax.legend(framealpha=0.9, edgecolor="none")
    ax.set_xticks(counts)
    ax.set_ylim(0.5, 1.0)
    fig.savefig("figures/experiment_dice_iou.jpg", format="jpg")
    plt.close(fig)
    log.info("Saved experiment_dice_iou.jpg")

    # Plot 3: Summary table as figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    headers = ["Synthetic\nSamples", "Total\nTrain", "Dice", "IoU", "Precision", "Recall", "Accuracy", "Specificity"]
    rows = []
    rows.append(["0 (baseline)", "556", f"{baseline['dice']:.4f}", f"{baseline['iou']:.4f}",
                  f"{baseline['precision']:.4f}", f"{baseline['recall']:.4f}",
                  f"{baseline['accuracy']:.4f}", f"{baseline['specificity']:.4f}"])
    for r in summary["experiments"]:
        t = r["test"]
        rows.append([str(r["n_synthetic"]), str(556 + r["n_synthetic"]),
                      f"{t['dice']:.4f}", f"{t['iou']:.4f}", f"{t['precision']:.4f}",
                      f"{t['recall']:.4f}", f"{t['accuracy']:.4f}", f"{t['specificity']:.4f}"])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Highlight best dice row
    best_dice_idx = max(range(len(rows)), key=lambda i: float(rows[i][2]))
    for col in range(len(headers)):
        table[best_dice_idx + 1, col].set_facecolor("#E8F5E9")

    fig.suptitle("Experiment Results Summary", fontsize=13, y=0.95)
    fig.savefig("figures/experiment_table.jpg", format="jpg")
    plt.close(fig)
    log.info("Saved experiment_table.jpg")


def main():
    t_total = time.time()
    log.info("=" * 70)
    log.info("SYNTHETIC DATA AUGMENTATION EXPERIMENT")
    log.info(f"Synthetic counts: {SYNTHETIC_COUNTS}")
    log.info(f"Epochs per experiment: {EPOCHS}")
    log.info("=" * 70)

    # Load baseline results
    with open("seg_log_baseline.json") as f:
        baseline_log = json.load(f)
    baseline_test = baseline_log["test"]
    log.info(f"Baseline test Dice: {baseline_test['dice']:.4f}, IoU: {baseline_test['iou']:.4f}")
    log.info("-" * 70)

    summary = {
        "baseline": baseline_test,
        "experiments": [],
    }

    for n_syn in SYNTHETIC_COUNTS:
        log.info(f"\n{'='*70}")
        log.info(f"EXPERIMENT: {n_syn} synthetic samples")
        log.info(f"{'='*70}")
        t_exp = time.time()

        # Prepare synthetic subset
        subset_dir = os.path.join(EXPERIMENT_DIR, f"syn_{n_syn}")
        prepare_synthetic_subset(n_syn, subset_dir)

        # Train and evaluate
        save_dir = os.path.join(EXPERIMENT_DIR, f"ckpt_{n_syn}")
        test_results = train_and_evaluate(subset_dir, n_syn, save_dir)

        exp_time = time.time() - t_exp
        log.info(f"\n  RESULTS for {n_syn} synthetic samples:")
        log.info(f"  Test Dice:        {test_results['dice']:.4f}")
        log.info(f"  Test IoU:         {test_results['iou']:.4f}")
        log.info(f"  Test Precision:   {test_results['precision']:.4f}")
        log.info(f"  Test Recall:      {test_results['recall']:.4f}")
        log.info(f"  Test Accuracy:    {test_results['accuracy']:.4f}")
        log.info(f"  Test Specificity: {test_results['specificity']:.4f}")
        log.info(f"  Time: {exp_time/60:.1f} minutes")

        summary["experiments"].append({
            "n_synthetic": n_syn,
            "test": test_results,
            "time_minutes": exp_time / 60,
        })

        # Save running summary
        with open(SUMMARY_FILE, "w") as f:
            json.dump(summary, f, indent=2)

    # Generate plots
    log.info("\n" + "=" * 70)
    log.info("Generating summary plots...")
    generate_summary_plots(summary)

    total_time = time.time() - t_total
    log.info(f"\nAll experiments complete in {total_time/60:.1f} minutes")
    log.info(f"Summary saved to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
