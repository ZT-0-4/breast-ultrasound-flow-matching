"""Generate 10 samples from best checkpoint, save as individual image and mask PNGs."""
import os
import torch
import numpy as np
from PIL import Image
from model import create_unet

IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "generated_samples"
os.makedirs(OUT_DIR, exist_ok=True)

model = create_unet(IMG_SIZE)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location=DEVICE, weights_only=True))
model = model.to(DEVICE)
model.eval()
print("Loaded best.pt", flush=True)

n = 10
steps = 100
x = torch.randn(n, 2, IMG_SIZE, IMG_SIZE, device=DEVICE)
dt = 1.0 / steps
with torch.no_grad():
    for i in range(steps):
        t_scaled = torch.full((n,), int((i / steps) * 999), device=DEVICE, dtype=torch.long)
        v = model(x, t_scaled).sample
        x = x + v * dt

print("Generated samples", flush=True)

for i in range(n):
    img_arr = x[i, 0].cpu().numpy()
    mask_arr = x[i, 1].cpu().numpy()
    img_arr = np.clip((img_arr + 1) / 2.0 * 255, 0, 255).astype(np.uint8)
    mask_arr = np.clip((mask_arr + 1) / 2.0 * 255, 0, 255).astype(np.uint8)
    mask_bin = ((mask_arr > 127) * 255).astype(np.uint8)

    Image.fromarray(img_arr, mode="L").save(os.path.join(OUT_DIR, f"sample_{i:02d}_image.png"))
    Image.fromarray(mask_bin, mode="L").save(os.path.join(OUT_DIR, f"sample_{i:02d}_mask.png"))
    print(f"Saved sample {i}", flush=True)

print(f"All saved to {OUT_DIR}/", flush=True)
