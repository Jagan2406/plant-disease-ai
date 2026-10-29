"""Configurable hyperparameters for the Swin classification baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ai.data.manifest import CLASS_NAMES_PATH, PROJECT_ROOT, SPLIT_CSV_PATHS, num_classes
from ai.utils.seed import DEFAULT_SEED


@dataclass
class TrainConfig:
    seed: int = DEFAULT_SEED
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 0
    dropout: float = 0.2
    label_smoothing: float = 0.1
    weight_decay: float = 1e-4
    head_lr: float = 1e-4
    backbone_lr: float = 1e-5
    stage1_epochs: int = 3
    stage2_epochs: int = 15
    warmup_epochs: int = 1
    early_stopping_patience: int = 5
    pretrained: bool = True
    class_names_path: str = str(CLASS_NAMES_PATH)
    train_csv: str = str(SPLIT_CSV_PATHS["train"])
    val_csv: str = str(SPLIT_CSV_PATHS["val"])
    test_csv: str = str(SPLIT_CSV_PATHS["test"])
    checkpoint_path: str = str(PROJECT_ROOT / "models" / "swin" / "best_model.pth")
    latest_checkpoint_path: str = str(PROJECT_ROOT / "models" / "swin" / "latest_checkpoint.pth")
    results_dir: str = str(PROJECT_ROOT / "results" / "swin_baseline")
    history_path: str = str(PROJECT_ROOT / "models" / "swin" / "training_history.json")
    log_path: str = str(PROJECT_ROOT / "models" / "swin" / "training_log.jsonl")

    def num_classes(self) -> int:
        return num_classes(Path(self.class_names_path))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["num_classes"] = self.num_classes()
        return payload


def default_config() -> TrainConfig:
    return TrainConfig()
