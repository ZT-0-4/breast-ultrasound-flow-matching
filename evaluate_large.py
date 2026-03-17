"""Evaluate large model: training curves, generated samples, real vs generated, overlay."""
import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import UNet2DModel
from dataset import UltrasoundPairDataset

IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG_DIR = "figures_large"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "axes.linewidth": 0.8, "lines.linewidth": 1.5,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})


def create_large_unet(img_size=128):
    return UNet2DModel(
        sample_size=img_size, in_channels=2, out_channels=2,
        layers_per_block=2, block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


@torch.no_grad()
def sample(model, n_samples, device, steps=100):
    model.eval()
    x = torch.randn(n_samples, 2, IMG_SIZE, IMG_SIZE, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t_scaled = torch.full((n_samples,), int((i / steps) * 999), device=device, dtype=torch.long)
        v = model(x, t_scaled).sample
        x = x + v * dt
    return x


def to_image(tensor_ch, is_mask=False):
    arr = tensor_ch.cpu().numpy()
    arr = (arr + 1) / 2.0
    arr = np.clip(arr, 0, 1)
    if is_mask:
        arr = (arr > 0.5).astype(np.float32)
    return (arr * 255).astype(np.uint8)


def main():
    # Training curves
    with open("training_log_large.json") as f:
        log = json.load(f)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(log["epoch"], log["train_loss"], color="#2563EB", label="Train loss", alpha=0.85)
    ax.plot(log["epoch"], log["val_loss"], color="#DC2626", label="Val loss", alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Flow Matching Training Progress (106.4M)")
    ax.legend(framealpha=0.9, edgecolor="none")
    ax.set_xlim(1, max(log["epoch"]))
    fig.savefig(os.path.join(FIG_DIR, "training_curves.jpg"), format="jpg")
    plt.close(fig)
    print("Saved training_curves.jpg")

    # LR schedule
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    ax.plot(log["epoch"], log["lr"], color="#7C3AED", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Cosine Learning Rate Schedule (106.4M)")
    ax.set_xlim(1, max(log["epoch"]))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
    fig.savefig(os.path.join(FIG_DIR, "lr_schedule.jpg"), format="jpg")
    plt.close(fig)
    print("Saved lr_schedule.jpg")

    # Load model
    model = create_large_unet(IMG_SIZE)
    model.load_state_dict(torch.load("checkpoints_large/best.pt", map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    print("Loaded checkpoints_large/best.pt")

    # Generated samples grid
    samples = sample(model, 8, DEVICE)
    fig, axes = plt.subplots(2, 8, figsize=(8 * 1.4, 3.2))
    for i in range(8):
        axes[0, i].imshow(to_image(samples[i, 0]), cmap="gray", vmin=0, vmax=255)
        axes[0, i].axis("off")
        axes[1, i].imshow(to_image(samples[i, 1], is_mask=True), cmap="gray", vmin=0, vmax=255)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Image", fontsize=11, rotation=90, labelpad=10)
    axes[1, 0].set_ylabel("Mask", fontsize=11, rotation=90, labelpad=10)
    fig.suptitle("Generated Ultrasound Image\u2013Mask Pairs (106.4M)", fontsize=13, y=1.02)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    fig.savefig(os.path.join(FIG_DIR, "generated_samples.jpg"), format="jpg")
    plt.close(fig)
    print("Saved generated_samples.jpg")

    # Real vs generated
    test_ds = UltrasoundPairDataset("data/test")
    samples = sample(model, 6, DEVICE)
    fig, axes = plt.subplots(4, 6, figsize=(6 * 1.5, 6.5))
    for i in range(6):
        real = test_ds[i]
        axes[0, i].imshow(to_image(real[0]), cmap="gray", vmin=0, vmax=255)
        axes[1, i].imshow(to_image(real[1], is_mask=True), cmap="gray", vmin=0, vmax=255)
        axes[2, i].imshow(to_image(samples[i, 0]), cmap="gray", vmin=0, vmax=255)
        axes[3, i].imshow(to_image(samples[i, 1], is_mask=True), cmap="gray", vmin=0, vmax=255)
        for r in range(4):
            axes[r, i].axis("off")
    for r, label in enumerate(["Real Image", "Real Mask", "Generated Image", "Generated Mask"]):
        axes[r, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=12)
    fig.suptitle("Real vs. Generated Ultrasound Samples (106.4M)", fontsize=13, y=1.01)
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    fig.savefig(os.path.join(FIG_DIR, "real_vs_generated.jpg"), format="jpg")
    plt.close(fig)
    print("Saved real_vs_generated.jpg")

    # Overlay
    samples = sample(model, 8, DEVICE)
    fig, axes = plt.subplots(1, 8, figsize=(8 * 1.5, 2.0))
    for i in range(8):
        img = to_image(samples[i, 0])
        mask = to_image(samples[i, 1], is_mask=True)
        overlay = np.stack([img, img, img], axis=-1).astype(np.float32)
        mask_bool = mask > 127
        overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0] * 0.5 + 128, 0, 255)
        overlay[mask_bool, 1] *= 0.5
        overlay[mask_bool, 2] *= 0.5
        axes[i].imshow(overlay.astype(np.uint8))
        axes[i].axis("off")
    fig.suptitle("Generated Images with Segmentation Overlay (106.4M)", fontsize=13, y=1.02)
    fig.subplots_adjust(wspace=0.05)
    fig.savefig(os.path.join(FIG_DIR, "overlay_samples.jpg"), format="jpg")
    plt.close(fig)
    print("Saved overlay_samples.jpg")

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
