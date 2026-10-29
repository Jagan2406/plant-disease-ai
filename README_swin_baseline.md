# Swin Transformer classification baseline

This repository contains a plant-disease research project. This document covers **only** the first baseline:

Raw image → Swin Transformer → 23-class classification (class count is read from `data/class_names.json`, not hardcoded).

SAM 2, Grad-CAM++, the disease knowledge base, the LLM recommender, FastAPI, and React are **not** part of this stage.

## Project purpose

Train a reproducible ImageNet-pretrained Swin-T classifier on an existing stratified plant-disease split. The later pipeline will add leaf segmentation and explanation layers; this baseline measures classification performance on unmodified RGB images.

## Dataset structure

Images live under `dataset/Dataset/`, one folder per class. Existing metadata:

- `data/train.csv`
- `data/val.csv`
- `data/test.csv`
- `data/class_names.json`

CSV columns in the current files:

- `image_path` — project-relative path
- `class_name`
- `class_index`

The loader discovers these columns instead of assuming names. Number of classes is `len(class_names.json)`.

Do **not** rerun `ai/data/create_split.py` for this baseline. The split is already created.

## Train / validation / test split

Existing split (seed 42):

- Train: 28,581
- Validation: 3,574
- Test: 3,570

The split was checked for cross-split duplicate-content leakage. Training and model selection use **train** and **validation** only. The **test** set is used once, after a checkpoint is selected.

## Swin architecture

`torchvision.models.swin_t` with ImageNet pretrained weights when available. The original classification head is replaced with:

- Dropout
- Linear(`num_classes` from `class_names.json`)

## Transfer-learning stages

Starting values, not claimed optima:

**Stage 1** (3 epochs by default)

- Freeze the Swin backbone
- Train the classification head
- Head learning rate: `1e-4`

**Stage 2** (15 epochs by default, configurable 10–20)

- Unfreeze the final two Swin stages (plus associated downsampling / norm)
- Backbone learning rate: `1e-5`
- Head learning rate: `1e-4`

Optimizer: AdamW. Weight decay: `1e-4`. Label smoothing: `0.1`. Loss: CrossEntropyLoss. LR schedule: linear warmup (when epoch budget allows) then cosine annealing. Early stopping patience is configurable (default 5) and watches **validation macro F1**.

Mixed precision is used only on CUDA. CPU training is supported.

## Augmentation

Training only:

- `RandomResizedCrop(224, scale=(0.7, 1.0))`
- `RandomHorizontalFlip`
- `RandomVerticalFlip`
- `RandomRotation(20)`
- mild `ColorJitter`
- ImageNet mean/std normalization

Validation and test (deterministic):

- `Resize(256)`
- `CenterCrop(224)`
- ImageNet normalization

## Checkpoint selection

Best weights are chosen by **validation macro F1**. The test set is never used for:

- checkpoint selection
- early stopping
- hyperparameter tuning

Saved file:

`models/swin/best_model.pth`

Contents: model state, class mapping, number of classes, configuration, epoch, validation metrics.

## Final test evaluation

After training:

```bash
python -m ai.classification.evaluate
```

This reads `data/test.csv` only and writes:

```
results/swin_baseline/metrics.json
results/swin_baseline/per_class_metrics.csv
results/swin_baseline/confusion_matrix.png
results/swin_baseline/training_curves.png
results/swin_baseline/evaluation_report.txt
```

Reported metrics: accuracy; macro and weighted precision / recall / F1; top-5 accuracy; per-class precision / recall / F1 / support; confusion matrix over the full class list.

Do not invent numbers. Files under `results/swin_baseline/` appear only after a real evaluation run.

## Commands

From the `plant-disease-ai` project root, with the intended virtual environment activated:

```bash
python -m ai.utils.device
python -m ai.verify_setup
python -m ai.classification.train
python -m ai.classification.evaluate
python -m ai.classification.inference --image <image_path>
```

Optional training knobs:

```bash
python -m ai.classification.train --batch-size 16 --stage1-epochs 3 --stage2-epochs 15 --head-lr 1e-4 --backbone-lr 1e-5 --patience 5
```

Non-PyTorch packages are listed in `requirements.txt`. Install a PyTorch / torchvision build that matches this machine (CPU or CUDA) separately. Do not pin a CUDA wheel in `requirements.txt`.

## Research notes

- Do not call this baseline state-of-the-art, best, or 100% accurate without measured, qualified evidence.
- Corrupt images raise errors instead of being silently dropped in bulk.
- Seed 42 is used for training randomness and matches the existing split seed.
