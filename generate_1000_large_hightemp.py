"""Generate 1000 samples from large model with temperature=1.3 for more diversity."""
import os
import torch
import numpy as np
from PIL import Image
from diffusers import UNet2DModel

IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "generated_seg_data_large_hightemp"
TEMPERATURE = 1.3
BATCH = 50
STEPS = 100
N_TOTAL = 1000

os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "masks"), exist_ok=True)


def create_large_unet(img_size=128):
    return UNet2DModel(
        sample_size=img_size, in_channels=2, out_channels=2,
        layers_per_block=2, block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


model = create_large_unet(IMG_SIZE)
model.load_state_dict(torch.load("checkpoints_large/best.pt", map_location=DEVICE, weights_only=True))
model = model.to(DEVICE)
model.eval()
print(f"Loaded best.pt | Temperature: {TEMPERATURE}", flush=True)

for start in range(0, N_TOTAL, BATCH):
    n = min(BATCH, N_TOTAL - start)
    x = torch.randn(n, 2, IMG_SIZE, IMG_SIZE, device=DEVICE) * TEMPERATURE
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

    print(f"Generated {start + n}/{N_TOTAL}", flush=True)

print(f"Done. Saved to {OUT_DIR}/", flush=True)
