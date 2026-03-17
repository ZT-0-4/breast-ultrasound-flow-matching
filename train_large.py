"""
Train LARGE UNet with flow matching objective on 2-channel ultrasound (image + mask) data.
~100M param model for higher quality generation.
"""
import sys
import os
import json
import time
import logging
import torch
from torch.utils.data import DataLoader
from diffusers import UNet2DModel
from dataset import UltrasoundPairDataset

# ---- Config ----
BATCH_SIZE = 8  # smaller batch for larger model
EPOCHS = 300
LR = 1e-4
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SAVE_DIR = "checkpoints_large"
LOG_FILE = "training_log_large.json"
TRAIN_LOG = "train_large.log"
SAMPLE_EVERY = 50
NUM_WORKERS = 4

os.makedirs(SAVE_DIR, exist_ok=True)

# ---- Logging setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(TRAIN_LOG, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def create_large_unet(img_size=128):
    model = UNet2DModel(
        sample_size=img_size,
        in_channels=2,
        out_channels=2,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model


def flow_matching_loss(model, x_0, device):
    B = x_0.shape[0]
    t = torch.rand(B, device=device)
    eps = torch.randn_like(x_0)
    x_t = (1 - t[:, None, None, None]) * eps + t[:, None, None, None] * x_0
    t_scaled = (t * 999).long()
    v_pred = model(x_t, t_scaled).sample
    v_target = x_0 - eps
    loss = ((v_pred - v_target) ** 2).mean()
    return loss


@torch.no_grad()
def sample(model, n_samples, device, steps=50):
    model.eval()
    x = torch.randn(n_samples, 2, IMG_SIZE, IMG_SIZE, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t_scaled = torch.full((n_samples,), int((i / steps) * 999), device=device, dtype=torch.long)
        v = model(x, t_scaled).sample
        x = x + v * dt
    model.train()
    return x


def main():
    log.info("=" * 70)
    log.info("LARGE UNet Flow Matching Training (~100M params)")
    log.info("=" * 70)
    log.info(f"Device: {DEVICE}")

    train_ds = UltrasoundPairDataset("data/train")
    val_ds = UltrasoundPairDataset("data/val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    log.info(f"Train: {len(train_ds)} samples, {n_train_batches} batches")
    log.info(f"Val:   {len(val_ds)} samples, {n_val_batches} batches")

    model = create_large_unet(IMG_SIZE)
    model = model.to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"Model params: {param_count:.1f}M")
    log.info(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LR}")
    log.info("-" * 70)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"train_loss": [], "val_loss": [], "epoch": [], "lr": []}
    best_val = float("inf")
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        model.train()
        train_losses = []
        for batch_idx, batch in enumerate(train_loader, 1):
            batch = batch.to(DEVICE)
            loss = flow_matching_loss(model, batch, DEVICE)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

            if batch_idx % 5 == 0 or batch_idx == n_train_batches:
                log.info(
                    f"  Epoch {epoch:03d}/{EPOCHS} | Batch {batch_idx:03d}/{n_train_batches} | "
                    f"Batch Loss: {loss.item():.5f} | Running Avg: {sum(train_losses)/len(train_losses):.5f}"
                )

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                loss = flow_matching_loss(model, batch, DEVICE)
                val_losses.append(loss.item())

        train_avg = sum(train_losses) / len(train_losses)
        val_avg = sum(val_losses) / len(val_losses)
        lr_now = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - t_start
        eta = (elapsed / epoch) * (EPOCHS - epoch)

        history["train_loss"].append(train_avg)
        history["val_loss"].append(val_avg)
        history["epoch"].append(epoch)
        history["lr"].append(lr_now)

        scheduler.step()

        marker = ""
        if val_avg < best_val:
            best_val = val_avg
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best.pt"))
            marker = " ** BEST **"

        log.info(
            f">> Epoch {epoch:03d}/{EPOCHS} DONE | "
            f"Train: {train_avg:.5f} | Val: {val_avg:.5f} | "
            f"LR: {lr_now:.6f} | "
            f"Time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min{marker}"
        )
        log.info("-" * 70)

        if epoch % SAMPLE_EVERY == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch:03d}.pt"))
            log.info(f"Checkpoint saved: epoch_{epoch:03d}.pt")

        with open(LOG_FILE, "w") as f:
            json.dump(history, f)

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final.pt"))
    total_time = time.time() - t_start
    log.info("=" * 70)
    log.info(f"Training complete in {total_time/60:.1f} minutes")
    log.info(f"Best val loss: {best_val:.5f}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
