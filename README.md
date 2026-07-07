# COMPASS-v1.0 Official Release

## Compact Prototype-Structural Graph Distillation with Local-Global Consistency for Unsupervised Multimodal Change Detection

This repository contains the official core implementation of COMPASS.

COMPASS learns modality-invariant structural commonality by combining compact prototype learning, structural graph distillation, and local-global consistency optimization.

## Method Components

### 1. Compact Graph Student Backbone

The student backbone extracts hierarchical structural representations and models local graph relationships without explicit graph construction.

### 2. Prototype-Structural Dual Teacher Distillation

The prototype teacher provides stable prototype-level commonality supervision.

The structural teacher transfers structural organization knowledge from graph-aware representations.

### 3. Local-Global Structural Consistency

Local and global structural views are aligned to improve robust multimodal representation learning.

## Inference

Only the student network is retained during inference.

```
Multimodal Image Pair
        |
Compact Graph Student Backbone
        |
Prototype Distribution Discrepancy
+
Structural Discrepancy
        |
Change Map
```

## Repository Structure

```
COMPASS-v1.0/

├── models/
│   └── compass.py

├── scripts/
│   └── train.py

├── configs/
│   └── (released after paper acceptance)

├── datasets/
│   └── (released after paper acceptance)

├── utils/
│   └── (released after paper acceptance)

└── README.md
```

## Running

Core training entry:

```bash
python scripts/train.py
```

Complete environment configuration and dataset preparation scripts will be released after paper acceptance.

## Current Release

Included:

- COMPASS model implementation
- Training pipeline

Pending:

- Dataset preparation
- Configuration files
- Environment requirements
- Additional evaluation utilities

## Citation

```bibtex
@article{COMPASS,
title={COMPASS: Compact Prototype-Structural Graph Distillation with Local-Global Consistency for Unsupervised Multimodal Change Detection},
author={Zhang, Bo and Chen, Yaxiong and Wen, Jie and Dong, Yanni and Xiong, Shengwu and Lu, Xiaoqiang},
journal={}
}
```
