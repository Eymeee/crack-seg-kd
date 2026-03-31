# Crack Segmentation with Knowledge Distillation

Binary semantic segmentation of cracks in concrete structures using feature-based knowledge distillation to compress a large teacher model into a lightweight student model.

**Course:** Deep Learning 2 — ENSIAS, Universite Mohammed V de Rabat (2025-2026)

---

## Overview

This project trains a **Teacher** (U-Net++ / ResNet34) and a **Student** (U-Net / MobileNetV2) on crack segmentation, then uses **feature-based knowledge distillation** to transfer the teacher's intermediate representations to the student. The result is a model ~10x smaller and ~4x faster that recovers most of the teacher's performance.

```text
Teacher (U-Net++ / ResNet34)  ->  31.2 M params  ~45 ms/img
Student (U-Net / MobileNetV2) ->   3.1 M params  ~12 ms/img
```

---

## Repository Structure

```text
crack-seg-kd/
├── src/
│   ├── models.py       # Teacher, Student, and FeatureAdapters architectures
│   ├── train.py        # Training pipeline (teacher / student / distill / eval)
│   ├── losses.py       # DiceLoss, SegmentationLoss, FeatureKDLoss, TotalDistillationLoss
│   ├── metrics.py      # IoU, F1, inference benchmark, comparison table
│   └── dataset.py      # CrackDataset, augmentation pipeline, dataloaders
├── notebook.ipynb      # Interactive exploration and visualization
├── requirements.txt
└── data/
    ├── train/          # 8 163 image-mask pairs
    ├── val/            # 1 440 image-mask pairs
    └── test/           # 1 695 image-mask pairs
```

---

## Method

### Architectures

| Model   | Encoder                | Decoder | Parameters |
|---------|------------------------|---------|------------|
| Teacher | ResNet34 (ImageNet)    | U-Net++ | 31.2 M     |
| Student | MobileNetV2 (ImageNet) | U-Net   | 3.1 M      |

### Knowledge Distillation

Feature maps are extracted from 4 intermediate layers of each model. Lightweight 1x1 convolution **adapters** project the student's feature channels to match the teacher's:

| Level | Student channels | Teacher channels |
|-------|------------------|------------------|
| 1     | 24               | 64               |
| 2     | 32               | 128              |
| 3     | 96               | 256              |
| 4     | 320              | 512              |

### Loss Functions

```text
L_seg   = a * DiceLoss + (1 - a) * BCEWithLogitsLoss     (a = 0.5)
L_kd    = (1/N) * sum MSE(Adapter(F_student_i), F_teacher_i)
L_total = L_seg + lambda * L_kd                           (lambda = 1.0)
```

---

## Dataset

Multi-source crack segmentation dataset aggregated from GAPS384, CRACK500, DeepCrack, CFD, Rissbilder, and Volker collections.

- **Images:** RGB, resized to 512x512
- **Masks:** Binary (crack = 1, background = 0), threshold at 127
- **Normalization:** ImageNet statistics

**Augmentation (train only):** horizontal/vertical flips, random 90 degree rotations, brightness/contrast jitter, Gaussian blur.

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python >= 3.8, PyTorch >= 2.0, CUDA recommended.

---

## Usage

All training modes are controlled via `src/train.py`.

### 1. Train the Teacher

```bash
python src/train.py --mode teacher --epochs 50 --batch_size 8
```

### 2. Train the Student (baseline, no distillation)

```bash
python src/train.py --mode student --epochs 50 --batch_size 8
```

### 3. Knowledge Distillation

```bash
python src/train.py --mode distill --epochs 50 --lambda_kd 1.0
```

### 4. Evaluate all models

```bash
python src/train.py --mode eval
```

Prints a comparison table with IoU, F1, parameter count, and inference latency for all three models.

### Key arguments

| Argument           | Default        | Description                                  |
|--------------------|----------------|----------------------------------------------|
| `--mode`           | —              | `teacher` / `student` / `distill` / `eval`   |
| `--epochs`         | 50             | Number of training epochs                    |
| `--batch_size`     | 8              | Batch size                                   |
| `--lr`             | 1e-4           | Learning rate (Adam)                         |
| `--lambda_kd`      | 1.0            | KD loss weight                               |
| `--img_size`       | 512            | Input image resolution                       |
| `--data_dir`       | `data/`        | Path to dataset root                         |
| `--checkpoint_dir` | `checkpoints/` | Where to save models                         |

---

## Results

| Model                               | IoU | F1 | Params | Latency |
|-------------------------------------|-----|----|--------|---------|
| Teacher (U-Net++ / ResNet34)        | —   | —  | 31.2 M | ~45 ms  |
| Student alone (U-Net / MobileNetV2) | —   | —  | 3.1 M  | ~12 ms  |
| Student distilled                   | —   | —  | 3.1 M  | ~12 ms  |

*Fill in after running evaluation.*

---

## Notebook

`notebook.ipynb` provides an end-to-end interactive workflow:

1. Dataset visualization
2. Architecture comparison (parameter counts, feature map shapes)
3. Training instructions
4. Evaluation and metric comparison
5. Feature map visualization (teacher vs. student)
6. Side-by-side prediction comparison

---

## License

For academic use only.
