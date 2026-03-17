"""Dataset for 2-channel (image + mask) ultrasound pairs."""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class UltrasoundPairDataset(Dataset):
    """Loads grayscale ultrasound images and binary masks as 2-channel tensors.
    Channel 0: image normalized to [-1, 1]
    Channel 1: mask normalized to [-1, 1]
    """

    def __init__(self, split_dir):
        self.img_dir = os.path.join(split_dir, "images")
        self.mask_dir = os.path.join(split_dir, "masks")
        self.filenames = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = np.array(Image.open(os.path.join(self.img_dir, fname)), dtype=np.float32)
        mask = np.array(Image.open(os.path.join(self.mask_dir, fname)), dtype=np.float32)

        # Normalize to [-1, 1]
        img = img / 127.5 - 1.0
        mask = mask / 127.5 - 1.0

        # Stack as 2-channel: (2, H, W)
        x = np.stack([img, mask], axis=0)
        return torch.from_numpy(x)
