"""Segmentation dataset with augmentations for ultrasound breast lesion images."""
import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import numpy as np
import torchvision.transforms.functional as TF


class SegmentationDataset(Dataset):
    """
    Loads grayscale ultrasound images and binary masks for segmentation.
    Applies joint augmentations to both image and mask.
    """

    def __init__(self, split_dir, augment=False, synthetic_dir=None):
        """
        Args:
            split_dir: path to split folder with images/ and masks/ subdirs
            augment: whether to apply data augmentation
            synthetic_dir: optional path to synthetic data folder (images/ and masks/)
        """
        self.img_dir = os.path.join(split_dir, "images")
        self.mask_dir = os.path.join(split_dir, "masks")
        self.filenames = sorted(os.listdir(self.img_dir))
        self.augment = augment

        # Optionally add synthetic data
        self.syn_filenames = []
        self.syn_img_dir = None
        self.syn_mask_dir = None
        if synthetic_dir and os.path.isdir(synthetic_dir):
            self.syn_img_dir = os.path.join(synthetic_dir, "images")
            self.syn_mask_dir = os.path.join(synthetic_dir, "masks")
            self.syn_filenames = sorted(os.listdir(self.syn_img_dir))

    def __len__(self):
        return len(self.filenames) + len(self.syn_filenames)

    def _load_pair(self, idx):
        if idx < len(self.filenames):
            fname = self.filenames[idx]
            img = Image.open(os.path.join(self.img_dir, fname)).convert("L")
            mask = Image.open(os.path.join(self.mask_dir, fname)).convert("L")
        else:
            sidx = idx - len(self.filenames)
            fname = self.syn_filenames[sidx]
            img = Image.open(os.path.join(self.syn_img_dir, fname)).convert("L")
            mask = Image.open(os.path.join(self.syn_mask_dir, fname)).convert("L")
        return img, mask

    def __getitem__(self, idx):
        img, mask = self._load_pair(idx)

        if self.augment:
            img, mask = self._augment(img, mask)

        # To tensor and normalize image to [0, 1]
        img = torch.from_numpy(np.array(img, dtype=np.float32)) / 255.0
        mask = torch.from_numpy((np.array(mask, dtype=np.float32) > 127).astype(np.float32))

        # Add channel dim: (1, H, W)
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        return img, mask

    def _augment(self, img, mask):
        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        # Random rotation (-15 to 15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)

        # Random affine (scale + translate)
        if random.random() > 0.5:
            angle = 0
            translate = [random.randint(-10, 10), random.randint(-10, 10)]
            scale = random.uniform(0.85, 1.15)
            img = TF.affine(img, angle, translate, scale, shear=0, fill=0)
            mask = TF.affine(mask, angle, translate, scale, shear=0, fill=0)

        # Brightness/contrast jitter (image only)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = TF.adjust_brightness(img, factor)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = TF.adjust_contrast(img, factor)

        # Gaussian blur (image only)
        if random.random() > 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        return img, mask
