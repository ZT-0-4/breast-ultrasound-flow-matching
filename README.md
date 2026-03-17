# Synthetic Data Augmentation for Breast Ultrasound Lesion Segmentation via Generative Flow Matching

Code and experiments for the paper: *"Synthetic Data Augmentation for Breast Ultrasound Lesion Segmentation via Generative Flow Matching"*.

## Overview

This repository contains the complete pipeline for:
1. Training a UNet-based flow matching generative model to produce synthetic breast ultrasound image--mask pairs
2. Training a UNet segmentation model on real + synthetic data
3. Systematic evaluation of synthetic augmentation ratios across multiple seeds

## Dataset

We use the [Breast Ultrasound Images (BUSI) dataset](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset) (Al-Dhabyani et al., 2020). Download and place the images in `raw_data/all/`.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce

### 1. Prepare data splits
```bash
python prepare_data.py
```

### 2. Train generative model (large, 106.4M params)
```bash
python train_large.py
```

### 3. Generate 1000 synthetic samples
```bash
python generate_1000_large.py
```

### 4. Train baseline segmentation
```bash
python seg_train.py --mode baseline
```

### 5. Run synthetic augmentation experiments (3 seeds)
```bash
python run_experiments_seeds.py
```

### 6. Generate evaluation figures
```bash
python evaluate_large.py
python seg_evaluate.py
```

## Citation

If you use this code, please cite:

```bibtex
@article{rouzbayani2025synthetic,
  title={Synthetic Data Augmentation for Breast Ultrasound Lesion Segmentation via Generative Flow Matching},
  author={Rouzbayani, Ali and Todorova, Zlatitsa and Patel, Dev and Camlioglu, Errol and Sadri, Rayan},
  year={2025}
}
```
