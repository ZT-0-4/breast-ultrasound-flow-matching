"""
Prepare ultrasound breast lesion dataset:
- Aggregate multiple masks per image into a single binary mask
- Resize to 128x128
- Split into train/val/test
"""
import os
import re
import random
import shutil
from collections import defaultdict
from PIL import Image
import numpy as np

RAW_DIR = "raw_data/all"
OUT_DIR = "data"
IMG_SIZE = 128
SEED = 42

# Split ratios for ~647 samples
# train ~560, val ~44, test ~43
TRAIN_RATIO = 0.86
VAL_RATIO = 0.07
TEST_RATIO = 0.07


def find_samples(raw_dir):
    """Parse filenames and group images with their masks."""
    files = os.listdir(raw_dir)
    samples = defaultdict(lambda: {"image": None, "masks": []})

    for f in sorted(files):
        if not f.endswith(".png"):
            continue
        # Match mask files: "benign (1)_mask.png", "benign (1)_mask_1.png"
        mask_match = re.match(r"^(.+)_mask(?:_\d+)?\.png$", f)
        if mask_match:
            key = mask_match.group(1)
            samples[key]["masks"].append(f)
        else:
            key = f.replace(".png", "")
            samples[key]["image"] = f

    # Filter out samples without both image and mask
    valid = {k: v for k, v in samples.items() if v["image"] and v["masks"]}
    print(f"Found {len(valid)} valid samples ({len(files)} total files)")
    return valid


def aggregate_mask(mask_paths, raw_dir, size):
    """Load and OR-aggregate multiple masks into one binary mask."""
    combined = np.zeros((size, size), dtype=np.uint8)
    for mp in mask_paths:
        m = Image.open(os.path.join(raw_dir, mp)).convert("L").resize(
            (size, size), Image.NEAREST
        )
        m_arr = np.array(m)
        combined = np.maximum(combined, (m_arr > 127).astype(np.uint8) * 255)
    return Image.fromarray(combined, mode="L")


def main():
    random.seed(SEED)

    samples = find_samples(RAW_DIR)
    keys = sorted(samples.keys())
    random.shuffle(keys)

    n = len(keys)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": keys[:n_train],
        "val": keys[n_train : n_train + n_val],
        "test": keys[n_train + n_val :],
    }

    for split_name, split_keys in splits.items():
        img_dir = os.path.join(OUT_DIR, split_name, "images")
        mask_dir = os.path.join(OUT_DIR, split_name, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        for i, key in enumerate(split_keys):
            s = samples[key]
            # Load and resize image to grayscale
            img = Image.open(os.path.join(RAW_DIR, s["image"])).convert("L")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

            # Aggregate masks
            mask = aggregate_mask(s["masks"], RAW_DIR, IMG_SIZE)

            fname = f"{i:04d}.png"
            img.save(os.path.join(img_dir, fname))
            mask.save(os.path.join(mask_dir, fname))

        print(f"{split_name}: {len(split_keys)} samples")

    print("Done.")


if __name__ == "__main__":
    main()
