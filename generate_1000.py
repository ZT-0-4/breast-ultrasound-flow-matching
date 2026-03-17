"""Generate 1000 image-mask pairs and create t-SNE plot of real vs synthetic."""
import os
import torch
import numpy as np
from PIL import Image
from model import create_unet
from dataset import UltrasoundPairDataset
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "generated_seg_data"
os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "masks"), exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 0.8, "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})

# Load model
model = create_unet(IMG_SIZE)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location=DEVICE, weights_only=True))
model = model.to(DEVICE)
model.eval()
print("Loaded best.pt", flush=True)

# Generate in batches of 50
N_TOTAL = 1000
BATCH = 50
STEPS = 100
all_synthetic = []

for start in range(0, N_TOTAL, BATCH):
    n = min(BATCH, N_TOTAL - start)
    x = torch.randn(n, 2, IMG_SIZE, IMG_SIZE, device=DEVICE)
    dt = 1.0 / STEPS
    with torch.no_grad():
        for i in range(STEPS):
            t_scaled = torch.full((n,), int((i / STEPS) * 999), device=DEVICE, dtype=torch.long)
            v = model(x, t_scaled).sample
            x = x + v * dt

    for j in range(n):
        idx = start + j
        img_arr = x[j, 0].cpu().numpy()
        mask_arr = x[j, 1].cpu().numpy()
        img_arr = np.clip((img_arr + 1) / 2.0 * 255, 0, 255).astype(np.uint8)
        mask_arr = np.clip((mask_arr + 1) / 2.0 * 255, 0, 255).astype(np.uint8)
        mask_bin = ((mask_arr > 127) * 255).astype(np.uint8)

        Image.fromarray(img_arr, mode="L").save(os.path.join(OUT_DIR, "images", f"{idx:04d}.png"))
        Image.fromarray(mask_bin, mode="L").save(os.path.join(OUT_DIR, "masks", f"{idx:04d}.png"))

    all_synthetic.append(x.cpu().numpy())
    print(f"Generated {start + n}/{N_TOTAL}", flush=True)

all_synthetic = np.concatenate(all_synthetic, axis=0)  # (1000, 2, 128, 128)

# Load real training data
print("Loading real data...", flush=True)
real_ds = UltrasoundPairDataset("data/train")
all_real = np.stack([real_ds[i].numpy() for i in range(len(real_ds))], axis=0)  # (556, 2, 128, 128)

# Flatten for t-SNE: each sample -> flat vector
# Downsample to 32x32 first to make t-SNE tractable
from PIL import Image as PILImage

def downsample_batch(arr, size=32):
    """Downsample (N, 2, H, W) to (N, 2*size*size)."""
    N = arr.shape[0]
    out = np.zeros((N, 2 * size * size), dtype=np.float32)
    for i in range(N):
        for c in range(2):
            img = PILImage.fromarray(((arr[i, c] + 1) / 2.0 * 255).clip(0, 255).astype(np.uint8))
            img = img.resize((size, size), PILImage.BILINEAR)
            out[i, c * size * size:(c + 1) * size * size] = np.array(img, dtype=np.float32).flatten() / 255.0
    return out

print("Downsampling for t-SNE...", flush=True)
real_flat = downsample_batch(all_real)
syn_flat = downsample_batch(all_synthetic)

# Combine and run t-SNE
combined = np.vstack([real_flat, syn_flat])
labels = np.array([0] * len(real_flat) + [1] * len(syn_flat))  # 0=real, 1=synthetic

print(f"Running t-SNE on {combined.shape[0]} samples...", flush=True)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
embeddings = tsne.fit_transform(combined)

real_emb = embeddings[labels == 0]
syn_emb = embeddings[labels == 1]

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(real_emb[:, 0], real_emb[:, 1], c="#2563EB", s=12, alpha=0.6, label=f"Real (n={len(real_emb)})", edgecolors="none")
ax.scatter(syn_emb[:, 0], syn_emb[:, 1], c="#DC2626", s=12, alpha=0.4, label=f"Synthetic (n={len(syn_emb)})", edgecolors="none")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title("t-SNE: Real vs. Synthetic Ultrasound Samples")
ax.legend(framealpha=0.9, edgecolor="none", markerscale=2)
ax.grid(False)
fig.savefig("figures/tsne_real_vs_synthetic.jpg", format="jpg")
plt.close(fig)
print("Saved figures/tsne_real_vs_synthetic.jpg", flush=True)
print("Done!", flush=True)
