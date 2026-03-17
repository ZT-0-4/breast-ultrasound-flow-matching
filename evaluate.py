"""
Generate publication-ready figures:
1. Training & validation loss curves
2. Learning rate schedule
3. Generated sample grid (image + mask pairs)
4. Comparison: real vs generated samples
5. Generated images with mask overlay
"""
import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import create_unet
from dataset import UltrasoundPairDataset
from train import sample, IMG_SIZE

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ---- Publication style ----
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


def to_image(tensor_ch, is_mask=False):
    """Convert a single-channel tensor from [-1,1] to [0,255] uint8."""
    arr = tensor_ch.cpu().numpy()
    arr = (arr + 1) / 2.0
    arr = np.clip(arr, 0, 1)
    if is_mask:
        arr = (arr > 0.5).astype(np.float32)
    return (arr * 255).astype(np.uint8)


def plot_training_curves(log):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    epochs = log["epoch"]
    ax.plot(epochs, log["train_loss"], color="#2563EB", label="Train loss", alpha=0.85)
    ax.plot(epochs, log["val_loss"], color="#DC2626", label="Val loss", alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Flow Matching Training Progress")
    ax.legend(framealpha=0.9, edgecolor="none")
    ax.set_xlim(1, max(epochs))
    fig.savefig(os.path.join(FIG_DIR, "training_curves.jpg"), format="jpg")
    plt.close(fig)
    print("Saved training_curves.jpg")


def plot_lr_schedule(log):
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    ax.plot(log["epoch"], log["lr"], color="#7C3AED", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Cosine Learning Rate Schedule")
    ax.set_xlim(1, max(log["epoch"]))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
    fig.savefig(os.path.join(FIG_DIR, "lr_schedule.jpg"), format="jpg")
    plt.close(fig)
    print("Saved lr_schedule.jpg")


def plot_generated_grid(model, n=8):
    samples = sample(model, n, DEVICE, steps=100)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.4, 3.2))
    for i in range(n):
        img = to_image(samples[i, 0])
        mask = to_image(samples[i, 1], is_mask=True)
        axes[0, i].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0, i].axis("off")
        axes[1, i].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Image", fontsize=11, rotation=90, labelpad=10)
    axes[1, 0].set_ylabel("Mask", fontsize=11, rotation=90, labelpad=10)
    fig.suptitle("Generated Ultrasound Image\u2013Mask Pairs", fontsize=13, y=1.02)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    fig.savefig(os.path.join(FIG_DIR, "generated_samples.jpg"), format="jpg")
    plt.close(fig)
    print("Saved generated_samples.jpg")


def plot_real_vs_generated(model, real_ds, n=6):
    samples = sample(model, n, DEVICE, steps=100)
    fig, axes = plt.subplots(4, n, figsize=(n * 1.5, 6.5))
    row_labels = ["Real Image", "Real Mask", "Generated Image", "Generated Mask"]
    for i in range(n):
        real = real_ds[i]
        real_img = to_image(real[0])
        real_mask = to_image(real[1], is_mask=True)
        gen_img = to_image(samples[i, 0])
        gen_mask = to_image(samples[i, 1], is_mask=True)
        axes[0, i].imshow(real_img, cmap="gray", vmin=0, vmax=255)
        axes[1, i].imshow(real_mask, cmap="gray", vmin=0, vmax=255)
        axes[2, i].imshow(gen_img, cmap="gray", vmin=0, vmax=255)
        axes[3, i].imshow(gen_mask, cmap="gray", vmin=0, vmax=255)
        for r in range(4):
            axes[r, i].axis("off")
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=12)
    fig.suptitle("Real vs. Generated Ultrasound Samples", fontsize=13, y=1.01)
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    fig.savefig(os.path.join(FIG_DIR, "real_vs_generated.jpg"), format="jpg")
    plt.close(fig)
    print("Saved real_vs_generated.jpg")


def plot_generated_overlay(model, n=8):
    samples = sample(model, n, DEVICE, steps=100)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.5, 2.0))
    for i in range(n):
        img = to_image(samples[i, 0])
        mask = to_image(samples[i, 1], is_mask=True)
        overlay = np.stack([img, img, img], axis=-1).astype(np.float32)
        mask_bool = mask > 127
        overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0] * 0.5 + 128, 0, 255)
        overlay[mask_bool, 1] = overlay[mask_bool, 1] * 0.5
        overlay[mask_bool, 2] = overlay[mask_bool, 2] * 0.5
        axes[i].imshow(overlay.astype(np.uint8))
        axes[i].axis("off")
    fig.suptitle("Generated Images with Segmentation Overlay", fontsize=13, y=1.02)
    fig.subplots_adjust(wspace=0.05)
    fig.savefig(os.path.join(FIG_DIR, "overlay_samples.jpg"), format="jpg")
    plt.close(fig)
    print("Saved overlay_samples.jpg")


def main():
    with open("training_log.json") as f:
        log = json.load(f)

    plot_training_curves(log)
    plot_lr_schedule(log)

    model = create_unet(IMG_SIZE)
    ckpt_path = "checkpoints/best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    print(f"Loaded {ckpt_path}")

    plot_generated_grid(model, n=8)

    test_ds = UltrasoundPairDataset("data/test")
    plot_real_vs_generated(model, test_ds, n=6)
    plot_generated_overlay(model, n=8)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
