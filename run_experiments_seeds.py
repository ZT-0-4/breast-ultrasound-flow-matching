"""
Re-run synthetic augmentation experiments with additional seeds.
Reads existing seed-1 results from experiment_summary.json,
runs seed 2 and seed 3, then combines all into multi-seed summary with mean±std.
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

BATCH_SIZE = 16
EPOCHS = 200
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
SYNTHETIC_COUNTS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
FULL_SYNTHETIC_DIR = "generated_seg_data"
SEEDS = [2, 3]  # seed 1 already done (default pytorch seed from first run)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("experiments_seeds.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def prepare_synthetic_subset(n_samples):
    subset_dir = f"/tmp/syn_{n_samples}"
    src_img = os.path.join(FULL_SYNTHETIC_DIR, "images")
    src_mask = os.path.join(FULL_SYNTHETIC_DIR, "masks")
    dst_img = os.path.join(subset_dir, "images")
    dst_mask = os.path.join(subset_dir, "masks")
    if os.path.exists(subset_dir):
        shutil.rmtree(subset_dir)
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_mask, exist_ok=True)
    for f in sorted(os.listdir(src_img))[:n_samples]:
        shutil.copy2(os.path.join(src_img, f), os.path.join(dst_img, f))
        shutil.copy2(os.path.join(src_mask, f), os.path.join(dst_mask, f))
    return subset_dir


def train_and_evaluate(synthetic_dir, n_syn, seed):
    set_seed(seed)
    train_ds = SegmentationDataset("data/train", augment=True, synthetic_dir=synthetic_dir)
    val_ds = SegmentationDataset("data/val", augment=False)
    test_ds = SegmentationDataset("data/test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = SegUNet(in_channels=1, out_channels=1).to(DEVICE)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_dice = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            loss = criterion(model(imgs), masks)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_metrics_list = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                val_metrics_list.append(compute_metrics(model(imgs), masks))

        val_dice = sum(m["dice"] for m in val_metrics_list) / len(val_metrics_list)
        scheduler.step()

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            log.info(f"      Epoch {epoch:03d}/{EPOCHS} | Val Dice: {val_dice:.4f}")

    # Test with best model
    model.load_state_dict(best_state)
    model.eval()
    test_metrics_list = []
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            test_metrics_list.append(compute_metrics(model(imgs), masks))

    test_results = {}
    for key in test_metrics_list[0]:
        test_results[key] = sum(m[key] for m in test_metrics_list) / len(test_metrics_list)
    return test_results


def train_baseline(seed):
    """Train baseline (no synthetic) with a given seed."""
    set_seed(seed)
    train_ds = SegmentationDataset("data/train", augment=True)
    val_ds = SegmentationDataset("data/val", augment=False)
    test_ds = SegmentationDataset("data/test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = SegUNet(in_channels=1, out_channels=1).to(DEVICE)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_dice = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            loss = criterion(model(imgs), masks)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_metrics_list = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                val_metrics_list.append(compute_metrics(model(imgs), masks))

        val_dice = sum(m["dice"] for m in val_metrics_list) / len(val_metrics_list)
        scheduler.step()

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            log.info(f"      Epoch {epoch:03d}/{EPOCHS} | Val Dice: {val_dice:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    test_metrics_list = []
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            test_metrics_list.append(compute_metrics(model(imgs), masks))

    test_results = {}
    for key in test_metrics_list[0]:
        test_results[key] = sum(m[key] for m in test_metrics_list) / len(test_metrics_list)
    return test_results


def generate_multiseed_plots(all_results):
    """Generate publication-ready plots with mean±std error bars."""
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

    counts = [0] + SYNTHETIC_COUNTS
    metrics_to_plot = ["dice", "iou", "precision", "recall"]
    labels = ["Dice Score", "IoU", "Precision", "Recall"]
    colors = ["#2563EB", "#16A34A", "#DC2626", "#7C3AED"]

    # Compute mean and std for each count and metric
    def get_stats(count, metric):
        if count == 0:
            vals = [r["baseline"] for r in all_results]
        else:
            vals = []
            for r in all_results:
                for exp in r["experiments"]:
                    if exp["n_synthetic"] == count:
                        vals.append(exp["test"])
        metric_vals = [v[metric] for v in vals]
        return np.mean(metric_vals), np.std(metric_vals)

    # Plot 1: All 4 metrics with error bars
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for idx, (metric, label, color) in enumerate(zip(metrics_to_plot, labels, colors)):
        ax = axes[idx // 2, idx % 2]
        means = [get_stats(c, metric)[0] for c in counts]
        stds = [get_stats(c, metric)[1] for c in counts]
        ax.errorbar(counts, means, yerr=stds, color=color, marker="o", markersize=5,
                     capsize=4, capthick=1.2, linewidth=1.8)
        ax.axhline(y=means[0], color="gray", linestyle="--", alpha=0.5, label="Baseline")
        ax.set_xlabel("Number of Synthetic Samples")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(framealpha=0.9, edgecolor="none", fontsize=9)
        ax.set_xticks(counts)

    fig.suptitle("Segmentation Performance vs. Synthetic Augmentation (3 seeds)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/experiment_metrics_3seeds.jpg", format="jpg")
    plt.close(fig)
    log.info("Saved experiment_metrics_3seeds.jpg")

    # Plot 2: Dice + IoU with error bands
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for metric, color, marker, label in [("dice", "#2563EB", "o", "Dice Score"), ("iou", "#16A34A", "s", "IoU")]:
        means = [get_stats(c, metric)[0] for c in counts]
        stds = [get_stats(c, metric)[1] for c in counts]
        ax.errorbar(counts, means, yerr=stds, color=color, marker=marker, markersize=6,
                     capsize=4, capthick=1.2, linewidth=2, label=label)
        ax.fill_between(counts, np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds), color=color, alpha=0.1)
    ax.axhline(y=get_stats(0, "dice")[0], color="#2563EB", linestyle="--", alpha=0.3)
    ax.axhline(y=get_stats(0, "iou")[0], color="#16A34A", linestyle="--", alpha=0.3)
    ax.set_xlabel("Number of Synthetic Samples Added")
    ax.set_ylabel("Score")
    ax.set_title("Effect of Synthetic Data on Segmentation (mean ± std, 3 seeds)")
    ax.legend(framealpha=0.9, edgecolor="none")
    ax.set_xticks(counts)
    ax.set_ylim(0.5, 1.0)
    fig.savefig("figures/experiment_dice_iou_3seeds.jpg", format="jpg")
    plt.close(fig)
    log.info("Saved experiment_dice_iou_3seeds.jpg")

    # Plot 3: Summary table
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis("off")
    headers = ["Synthetic", "Total\nTrain", "Dice", "IoU", "Precision", "Recall", "Accuracy", "Specificity"]
    rows = []
    for c in counts:
        total = 556 + c
        row = [str(c) + (" (base)" if c == 0 else ""), str(total)]
        for m in ["dice", "iou", "precision", "recall", "accuracy", "specificity"]:
            mean, std = get_stats(c, m)
            row.append(f"{mean:.3f}±{std:.3f}")
        rows.append(row)

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Highlight best dice row
    dice_means = [get_stats(c, "dice")[0] for c in counts]
    best_idx = np.argmax(dice_means)
    for col in range(len(headers)):
        table[best_idx + 1, col].set_facecolor("#E8F5E9")

    fig.suptitle("Experiment Results (mean ± std, 3 seeds)", fontsize=13, y=0.95)
    fig.savefig("figures/experiment_table_3seeds.jpg", format="jpg")
    plt.close(fig)
    log.info("Saved experiment_table_3seeds.jpg")


def main():
    t_total = time.time()

    # Load seed-1 results
    with open("experiment_summary.json") as f:
        seed1 = json.load(f)

    all_results = [seed1]
    log.info("=" * 70)
    log.info("MULTI-SEED EXPERIMENTS (seeds 2 and 3)")
    log.info("=" * 70)
    log.info(f"Seed 1 already complete (loaded from experiment_summary.json)")

    for seed in SEEDS:
        log.info(f"\n{'='*70}")
        log.info(f"SEED {seed}")
        log.info(f"{'='*70}")
        t_seed = time.time()

        # Baseline for this seed
        log.info(f"  Training baseline (seed={seed})...")
        baseline = train_baseline(seed)
        log.info(f"  Baseline Dice: {baseline['dice']:.4f}, IoU: {baseline['iou']:.4f}")

        seed_results = {"baseline": baseline, "experiments": []}

        for n_syn in SYNTHETIC_COUNTS:
            log.info(f"\n  Experiment: {n_syn} synthetic (seed={seed})")
            t_exp = time.time()
            subset_dir = prepare_synthetic_subset(n_syn)
            test_results = train_and_evaluate(subset_dir, n_syn, seed)
            exp_time = time.time() - t_exp

            log.info(f"    Dice: {test_results['dice']:.4f} | IoU: {test_results['iou']:.4f} | Time: {exp_time/60:.1f}min")

            seed_results["experiments"].append({
                "n_synthetic": n_syn,
                "test": test_results,
            })

        all_results.append(seed_results)

        # Save intermediate
        with open("experiment_summary_3seeds.json", "w") as f:
            json.dump(all_results, f, indent=2)

        seed_time = time.time() - t_seed
        log.info(f"\n  Seed {seed} complete in {seed_time/60:.1f} minutes")

    # Generate plots
    log.info("\n" + "=" * 70)
    log.info("Generating multi-seed plots...")
    generate_multiseed_plots(all_results)

    total_time = time.time() - t_total
    log.info(f"\nAll seeds complete in {total_time/60:.1f} minutes")
    log.info(f"Summary saved to experiment_summary_3seeds.json")


if __name__ == "__main__":
    main()
