"""Generate 1000 samples from large model, then run full multi-seed segmentation experiments."""
import os
import sys
import json
import time
import random
import shutil
import logging
import torch
import numpy as np
from PIL import Image
from diffusers import UNet2DModel
from torch.utils.data import DataLoader
from seg_model import SegUNet
from seg_dataset import SegmentationDataset

IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "generated_seg_data_large"
BATCH_SIZE_GEN = 50
STEPS = 100
BATCH_SIZE_SEG = 16
EPOCHS = 200
LR = 1e-3
NUM_WORKERS = 4
SYNTHETIC_COUNTS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
SEEDS = [1, 2, 3]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("experiments_large.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def create_large_unet(img_size=128):
    return UNet2DModel(
        sample_size=img_size, in_channels=2, out_channels=2,
        layers_per_block=2, block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


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


def generate_samples():
    """Generate 1000 image-mask pairs from large model."""
    log.info("Loading large model...")
    model = create_large_unet(IMG_SIZE)
    model.load_state_dict(torch.load("checkpoints_large/best.pt", map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()

    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "masks"), exist_ok=True)

    N_TOTAL = 1000
    for start in range(0, N_TOTAL, BATCH_SIZE_GEN):
        n = min(BATCH_SIZE_GEN, N_TOTAL - start)
        x = torch.randn(n, 2, IMG_SIZE, IMG_SIZE, device=DEVICE)
        dt = 1.0 / STEPS
        with torch.no_grad():
            for i in range(STEPS):
                t_scaled = torch.full((n,), int((i / STEPS) * 999), device=DEVICE, dtype=torch.long)
                v = model(x, t_scaled).sample
                x = x + v * dt

        for j in range(n):
            idx = start + j
            img_arr = np.clip((x[j, 0].cpu().numpy() + 1) / 2.0 * 255, 0, 255).astype(np.uint8)
            mask_arr = np.clip((x[j, 1].cpu().numpy() + 1) / 2.0 * 255, 0, 255).astype(np.uint8)
            mask_bin = ((mask_arr > 127) * 255).astype(np.uint8)
            Image.fromarray(img_arr, mode="L").save(os.path.join(OUT_DIR, "images", f"{idx:04d}.png"))
            Image.fromarray(mask_bin, mode="L").save(os.path.join(OUT_DIR, "masks", f"{idx:04d}.png"))

        log.info(f"  Generated {start + n}/{N_TOTAL}")

    del model
    torch.cuda.empty_cache()


def prepare_synthetic_subset(n_samples):
    subset_dir = f"/tmp/syn_large_{n_samples}"
    src_img = os.path.join(OUT_DIR, "images")
    src_mask = os.path.join(OUT_DIR, "masks")
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


def train_and_evaluate(synthetic_dir, seed):
    set_seed(seed)
    train_ds = SegmentationDataset("data/train", augment=True, synthetic_dir=synthetic_dir)
    val_ds = SegmentationDataset("data/val", augment=False)
    test_ds = SegmentationDataset("data/test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_SEG, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_SEG, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE_SEG, shuffle=False, num_workers=NUM_WORKERS)

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
            log.info(f"        Epoch {epoch:03d}/{EPOCHS} | Val Dice: {val_dice:.4f}")

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


def generate_plots(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 11, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05, "axes.linewidth": 0.8, "lines.linewidth": 1.8,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    })

    os.makedirs("figures_large", exist_ok=True)
    counts = [0] + SYNTHETIC_COUNTS

    def get_stats(count, metric):
        if count == 0:
            vals = [r["baseline"] for r in all_results]
        else:
            vals = []
            for r in all_results:
                for exp in r["experiments"]:
                    if exp["n_synthetic"] == count:
                        vals.append(exp["test"])
        return np.mean([v[metric] for v in vals]), np.std([v[metric] for v in vals])

    # All 4 metrics
    metrics = ["dice", "iou", "precision", "recall"]
    labels = ["Dice Score", "IoU", "Precision", "Recall"]
    colors = ["#2563EB", "#16A34A", "#DC2626", "#7C3AED"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for idx, (m, l, c) in enumerate(zip(metrics, labels, colors)):
        ax = axes[idx // 2, idx % 2]
        means = [get_stats(cnt, m)[0] for cnt in counts]
        stds = [get_stats(cnt, m)[1] for cnt in counts]
        ax.errorbar(counts, means, yerr=stds, color=c, marker="o", markersize=5, capsize=4, capthick=1.2)
        ax.axhline(y=means[0], color="gray", linestyle="--", alpha=0.5, label="Baseline")
        ax.set_xlabel("Number of Synthetic Samples")
        ax.set_ylabel(l)
        ax.set_title(l)
        ax.legend(framealpha=0.9, edgecolor="none", fontsize=9)
        ax.set_xticks(counts)
    fig.suptitle("Segmentation vs. Synthetic Augmentation — Large Model (3 seeds)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("figures_large/experiment_metrics_3seeds.jpg", format="jpg")
    plt.close(fig)

    # Dice + IoU with bands
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, c, mk, l in [("dice", "#2563EB", "o", "Dice"), ("iou", "#16A34A", "s", "IoU")]:
        means = [get_stats(cnt, m)[0] for cnt in counts]
        stds = [get_stats(cnt, m)[1] for cnt in counts]
        ax.errorbar(counts, means, yerr=stds, color=c, marker=mk, markersize=6, capsize=4, capthick=1.2, linewidth=2, label=l)
        ax.fill_between(counts, np.array(means)-np.array(stds), np.array(means)+np.array(stds), color=c, alpha=0.1)
    ax.axhline(y=get_stats(0, "dice")[0], color="#2563EB", linestyle="--", alpha=0.3)
    ax.axhline(y=get_stats(0, "iou")[0], color="#16A34A", linestyle="--", alpha=0.3)
    ax.set_xlabel("Number of Synthetic Samples Added")
    ax.set_ylabel("Score")
    ax.set_title("Effect of Synthetic Data — Large Model (mean \u00b1 std, 3 seeds)")
    ax.legend(framealpha=0.9, edgecolor="none")
    ax.set_xticks(counts)
    ax.set_ylim(0.5, 1.0)
    fig.savefig("figures_large/experiment_dice_iou_3seeds.jpg", format="jpg")
    plt.close(fig)

    # Table
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis("off")
    headers = ["Synthetic", "Total\nTrain", "Dice", "IoU", "Precision", "Recall", "Accuracy", "Specificity"]
    rows = []
    for c in counts:
        row = [str(c) + (" (base)" if c == 0 else ""), str(556 + c)]
        for m in ["dice", "iou", "precision", "recall", "accuracy", "specificity"]:
            mean, std = get_stats(c, m)
            row.append(f"{mean:.3f}\u00b1{std:.3f}")
        rows.append(row)
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    dice_means = [get_stats(c, "dice")[0] for c in counts]
    best_idx = np.argmax(dice_means)
    for col in range(len(headers)):
        table[best_idx + 1, col].set_facecolor("#E8F5E9")
    fig.suptitle("Experiment Results — Large Model (mean \u00b1 std, 3 seeds)", fontsize=13, y=0.95)
    fig.savefig("figures_large/experiment_table_3seeds.jpg", format="jpg")
    plt.close(fig)

    log.info("All plots saved to figures_large/")


def main():
    t_total = time.time()
    log.info("=" * 70)
    log.info("LARGE MODEL EXPERIMENT (106.4M params, 3 seeds)")
    log.info("=" * 70)

    # Step 1: Generate 1000 samples
    log.info("Generating 1000 synthetic samples from large model...")
    generate_samples()
    log.info("Generation complete.")

    # Step 2: Run experiments for each seed
    all_results = []
    for seed in SEEDS:
        log.info(f"\n{'='*70}")
        log.info(f"SEED {seed}")
        log.info(f"{'='*70}")

        # Baseline
        log.info(f"  Training baseline (seed={seed})...")
        baseline = train_and_evaluate(None, seed)
        log.info(f"  Baseline Dice: {baseline['dice']:.4f}, IoU: {baseline['iou']:.4f}")

        seed_results = {"baseline": baseline, "experiments": []}

        for n_syn in SYNTHETIC_COUNTS:
            log.info(f"  Experiment: {n_syn} synthetic (seed={seed})")
            subset_dir = prepare_synthetic_subset(n_syn)
            test_results = train_and_evaluate(subset_dir, seed)
            log.info(f"    Dice: {test_results['dice']:.4f} | IoU: {test_results['iou']:.4f}")
            seed_results["experiments"].append({"n_synthetic": n_syn, "test": test_results})

        all_results.append(seed_results)
        with open("experiment_summary_large_3seeds.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # Step 3: Plots
    log.info("\nGenerating plots...")
    generate_plots(all_results)

    total_time = time.time() - t_total
    log.info(f"\nAll done in {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
