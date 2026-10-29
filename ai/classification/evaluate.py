"""Evaluate the selected Swin checkpoint on the held-out test split only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai.classification.config import TrainConfig
from ai.classification.metrics import compute_classification_metrics
from ai.classification.swin_model import build_swin_classifier
from ai.classification.transforms import get_eval_transforms
from ai.data.dataset import PlantDiseaseDataset
from ai.data.manifest import load_class_mapping
from ai.utils.device import get_device
from ai.utils.seed import set_seed


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    class_mapping_raw = checkpoint.get("class_mapping")
    if class_mapping_raw is None:
        class_mapping = load_class_mapping()
    else:
        class_mapping = {int(k): str(v) for k, v in class_mapping_raw.items()}
    num_classes = int(checkpoint.get("num_classes", len(class_mapping)))
    dropout = float(checkpoint.get("config", {}).get("dropout", 0.2))
    model = build_swin_classifier(num_classes, dropout=dropout, pretrained=False)
    model_state = checkpoint.get("model_state_dict", checkpoint.get("model_state"))
    if model_state is None:
        raise KeyError("Checkpoint is missing 'model_state_dict'.")
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return checkpoint, model, class_mapping


def _plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set_title("Swin baseline test confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_training_curves(history_path: Path, output_path: Path) -> bool:
    if not history_path.exists():
        return False
    history = json.loads(history_path.read_text(encoding="utf-8"))
    records = history.get("stage1", []) + history.get("stage2", [])
    if not records:
        return False
    epochs = [row["epoch"] for row in records]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in records], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in records], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_macro_f1"] for row in records], label="train macro F1")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in records], label="val macro F1")
    axes[1].set_title("Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _write_report(metrics: dict, checkpoint: dict, output_path: Path) -> None:
    lines = [
        "Swin Transformer baseline — test evaluation",
        "===========================================",
        "",
        "The test split was not used for training, early stopping, or checkpoint selection.",
        f"Checkpoint epoch: {checkpoint.get('epoch')}",
        f"Checkpoint stage: {checkpoint.get('stage')}",
        f"Validation metrics stored in checkpoint: {checkpoint.get('validation_metrics')}",
        "",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Macro precision: {metrics['macro_precision']:.4f}",
        f"Macro recall: {metrics['macro_recall']:.4f}",
        f"Macro F1: {metrics['macro_f1']:.4f}",
        f"Weighted precision: {metrics['weighted_precision']:.4f}",
        f"Weighted recall: {metrics['weighted_recall']:.4f}",
        f"Weighted F1: {metrics['weighted_f1']:.4f}",
        f"Top-5 accuracy: {metrics['top5_accuracy']:.4f}",
        "",
        "Per-class metrics",
        "-----------------",
    ]
    for row in metrics["per_class"]:
        lines.append(
            f"{row['class_index']:2d} {row['class_name']}: "
            f"P={row['precision']:.4f} R={row['recall']:.4f} "
            f"F1={row['f1']:.4f} support={row['support']}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(checkpoint_path: Path | None = None, results_dir: Path | None = None) -> dict:
    set_seed()
    config = TrainConfig()
    device = get_device()
    ckpt_path = checkpoint_path or Path(config.checkpoint_path)
    out_dir = results_dir or Path(config.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train first with python -m ai.classification.train"
        )

    checkpoint, model, class_mapping = load_checkpoint(ckpt_path, device)
    checkpoint_config = checkpoint.get("config", {})
    image_size = int(checkpoint_config.get("image_size", config.image_size))
    test_csv = checkpoint_config.get("test_csv", config.test_csv)
    dataset = PlantDiseaseDataset(
        "test",
        transform=get_eval_transforms(image_size),
        csv_path=test_csv,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[list[float]] = []
    with torch.no_grad():
        for images, labels, _paths in tqdm(loader, desc="test"):
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits.float(), dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(probabilities.argmax(dim=1).cpu().tolist())
            y_score.extend(probabilities.cpu().tolist())

    metrics = compute_classification_metrics(y_true, y_pred, class_mapping, y_score)
    summary = {key: value for key, value in metrics.items() if key not in {"per_class", "confusion_matrix"}}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(metrics["per_class"]).to_csv(out_dir / "per_class_metrics.csv", index=False)
    _plot_confusion_matrix(
        np.asarray(metrics["confusion_matrix"]),
        [class_mapping[i] for i in range(len(class_mapping))],
        out_dir / "confusion_matrix.png",
    )
    curves_written = _plot_training_curves(Path(config.history_path), out_dir / "training_curves.png")
    _write_report(metrics, checkpoint, out_dir / "evaluation_report.txt")
    if not curves_written:
        print("training_curves.png was not written because training history is missing.")
    print(f"Wrote test evaluation outputs to {out_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Swin baseline on data/test.csv")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.checkpoint, args.results_dir)
