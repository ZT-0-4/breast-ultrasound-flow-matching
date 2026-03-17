"""
Train segmentation UNet (baseline or with synthetic data).
Logs multiple metrics: Dice, IoU, Precision, Recall, Accuracy, Specificity.

Usage:
  Baseline (real data only):
    python seg_train.py --mode baseline

  With synthetic augmentation:
    python seg_train.py --mode synthetic --synthetic_dir generated_seg_data
"""
import sys
import os
import json
import time
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from seg_model import SegUNet
from seg_dataset import SegmentationDataset

# ---- Config ----
BATCH_SIZE = 16
EPOCHS = 200
LR = 1e-3
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
NUM_WORKERS = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "synthetic"])
    parser.add_argument("--synthetic_dir", type=str, default="generated_seg_data")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    return parser.parse_args()


# ---- Metrics ----
def compute_metrics(pred, target, threshold=0.5):
    """Compute segmentation metrics. pred: logits (B,1,H,W), target: binary (B,1,H,W)."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target = target.float()

    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    tn = ((1 - pred_bin) * (1 - target)).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "accuracy": accuracy.item(),
        "specificity": specificity.item(),
    }


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum()
        dice_loss = 1 - (2 * intersection + 1) / (pred_sig.sum() + target.sum() + 1)
        return bce_loss + dice_loss


def evaluate(model, loader, criterion, device):
    """Run evaluation, return avg loss and metrics."""
    model.eval()
    losses = []
    all_metrics = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            losses.append(loss.item())
            metrics = compute_metrics(preds, masks)
            all_metrics.append(metrics)

    avg_loss = sum(losses) / len(losses)
    avg_metrics = {}
    for key in all_metrics[0]:
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    return avg_loss, avg_metrics


def main():
    args = parse_args()
    mode = args.mode

    # Directories
    save_dir = f"seg_checkpoints_{mode}"
    log_file = f"seg_log_{mode}.json"
    train_log_file = f"seg_train_{mode}.log"
    os.makedirs(save_dir, exist_ok=True)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(train_log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info(f"Segmentation Training — Mode: {mode.upper()}")
    log.info("=" * 70)
    log.info(f"Device: {DEVICE}")

    # Datasets
    syn_dir = args.synthetic_dir if mode == "synthetic" else None
    train_ds = SegmentationDataset("data/train", augment=True, synthetic_dir=syn_dir)
    val_ds = SegmentationDataset("data/val", augment=False)
    test_ds = SegmentationDataset("data/test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    n_train_batches = len(train_loader)
    log.info(f"Train: {len(train_ds)} samples ({len(train_ds) - len(SegmentationDataset('data/train').filenames)} synthetic), {n_train_batches} batches")
    log.info(f"Val:   {len(val_ds)} samples")
    log.info(f"Test:  {len(test_ds)} samples")

    # Model
    model = SegUNet(in_channels=1, out_channels=1).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"Model params: {param_count:.1f}M")
    log.info(f"Batch size: {BATCH_SIZE}, Epochs: {args.epochs}, LR: {LR}")
    log.info(f"Loss: Dice + BCE")
    log.info("-" * 70)

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {
        "epoch": [], "lr": [],
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_iou": [], "val_iou": [],
        "train_precision": [], "val_precision": [],
        "train_recall": [], "val_recall": [],
        "train_accuracy": [], "val_accuracy": [],
        "train_specificity": [], "val_specificity": [],
    }

    best_val_dice = 0.0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # ---- Training ----
        model.train()
        train_losses = []
        train_metrics_list = []
        for batch_idx, (imgs, masks) in enumerate(train_loader, 1):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
            train_metrics_list.append(compute_metrics(preds.detach(), masks))

            if batch_idx % 5 == 0 or batch_idx == n_train_batches:
                avg_loss = sum(train_losses) / len(train_losses)
                avg_dice = sum(m["dice"] for m in train_metrics_list) / len(train_metrics_list)
                log.info(
                    f"  Epoch {epoch:03d}/{args.epochs} | Batch {batch_idx:03d}/{n_train_batches} | "
                    f"Loss: {loss.item():.5f} | Avg Loss: {avg_loss:.5f} | Avg Dice: {avg_dice:.4f}"
                )

        # ---- Validation ----
        val_loss, val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        # Train epoch averages
        train_avg_loss = sum(train_losses) / len(train_losses)
        train_avg_metrics = {}
        for key in train_metrics_list[0]:
            train_avg_metrics[key] = sum(m[key] for m in train_metrics_list) / len(train_metrics_list)

        lr_now = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - t_start
        eta = (elapsed / epoch) * (args.epochs - epoch)

        # Record history
        history["epoch"].append(epoch)
        history["lr"].append(lr_now)
        history["train_loss"].append(train_avg_loss)
        history["val_loss"].append(val_loss)
        for key in ["dice", "iou", "precision", "recall", "accuracy", "specificity"]:
            history[f"train_{key}"].append(train_avg_metrics[key])
            history[f"val_{key}"].append(val_metrics[key])

        scheduler.step()

        marker = ""
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
            marker = " ** BEST **"

        log.info(
            f">> Epoch {epoch:03d}/{args.epochs} DONE | "
            f"Train Loss: {train_avg_loss:.5f} | Val Loss: {val_loss:.5f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | Val IoU: {val_metrics['iou']:.4f} | "
            f"Val Prec: {val_metrics['precision']:.4f} | Val Rec: {val_metrics['recall']:.4f} | "
            f"LR: {lr_now:.6f} | Time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min{marker}"
        )
        log.info("-" * 70)

        # Save log every epoch
        with open(log_file, "w") as f:
            json.dump(history, f)

    # ---- Final test evaluation ----
    model.load_state_dict(torch.load(os.path.join(save_dir, "best.pt"), map_location=DEVICE, weights_only=True))
    test_loss, test_metrics = evaluate(model, test_loader, criterion, DEVICE)

    log.info("=" * 70)
    log.info(f"FINAL TEST RESULTS — Mode: {mode.upper()}")
    log.info("=" * 70)
    log.info(f"Test Loss:        {test_loss:.5f}")
    log.info(f"Test Dice:        {test_metrics['dice']:.4f}")
    log.info(f"Test IoU:         {test_metrics['iou']:.4f}")
    log.info(f"Test Precision:   {test_metrics['precision']:.4f}")
    log.info(f"Test Recall:      {test_metrics['recall']:.4f}")
    log.info(f"Test Accuracy:    {test_metrics['accuracy']:.4f}")
    log.info(f"Test Specificity: {test_metrics['specificity']:.4f}")
    log.info("=" * 70)

    # Save test results
    history["test"] = test_metrics
    history["test"]["loss"] = test_loss
    with open(log_file, "w") as f:
        json.dump(history, f)

    total_time = time.time() - t_start
    log.info(f"Training complete in {total_time/60:.1f} minutes")
    log.info(f"Best val Dice: {best_val_dice:.4f}")

    torch.save(model.state_dict(), os.path.join(save_dir, "final.pt"))


if __name__ == "__main__":
    main()
