"""Train Swin on the existing train split. Select checkpoints with validation macro F1 only."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai.classification.config import TrainConfig
from ai.classification.metrics import compute_classification_metrics
from ai.classification.swin_model import (
    build_swin_classifier,
    count_trainable_parameters,
    freeze_backbone,
    trainable_parameter_groups,
    unfreeze_last_two_stages,
)
from ai.classification.transforms import get_eval_transforms, get_train_transforms
from ai.data.dataset import PlantDiseaseDataset
from ai.data.manifest import load_class_mapping
from ai.utils.device import get_device
from ai.utils.seed import seed_worker, set_seed


def _amp_components(device: torch.device):
    use_cuda = device.type == "cuda"
    if not use_cuda:
        return False, lambda: nullcontext(), None

    try:
        scaler = torch.amp.GradScaler("cuda")
        autocast_ctx = lambda: torch.amp.autocast("cuda")
        return True, autocast_ctx, scaler
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
        autocast_ctx = torch.cuda.amp.autocast
        return True, autocast_ctx, scaler


def _make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    autocast_ctx,
    scaler,
    train: bool,
) -> tuple[float, list[int], list[int], list[list[float]]]:
    model.train(train)
    running_loss = 0.0
    seen = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[list[float]] = []

    context = torch.enable_grad() if train else torch.no_grad()
    iterator = tqdm(loader, leave=False)
    with context:
        for images, labels, _paths in iterator:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast_ctx():
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            running_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)
            probabilities = torch.softmax(logits.detach().float(), dim=1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(probabilities.argmax(dim=1).cpu().tolist())
            y_score.extend(probabilities.cpu().tolist())
            iterator.set_postfix(loss=running_loss / max(1, seen))

    return running_loss / max(1, seen), y_true, y_pred, y_score


def _build_scheduler(optimizer, epochs: int, warmup_epochs: int):
    warmup = max(0, min(warmup_epochs, epochs))
    if warmup == 0:
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    cosine_epochs = max(1, epochs - warmup)
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=cosine_epochs),
        ],
        milestones=[warmup],
    )


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            torch.save(payload, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, indent=2)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_checkpoint(path: Path, map_location="cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _config_from_dict(values: dict) -> TrainConfig:
    fields = TrainConfig.__dataclass_fields__
    return TrainConfig(**{key: value for key, value in values.items() if key in fields})


def _save_history(history: dict, path: Path) -> None:
    _atomic_json_save(history, path)


def _append_training_log(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _rewrite_training_log(history: dict, path: Path) -> None:
    records = sorted(
        (record for stage in ("stage1", "stage2") for record in history.get(stage, [])),
        key=lambda record: int(record["epoch"]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for record in records:
                temporary_file.write(json.dumps(record) + "\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def train(config: TrainConfig, resume_path: Path | None = None) -> Path:
    resume_checkpoint = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_checkpoint = _load_checkpoint(resume_path, map_location="cpu")
        config = _config_from_dict(resume_checkpoint["config"])

    if not config.pretrained:
        raise ValueError(
            "Actual training requires ImageNet-pretrained Swin-T weights; "
            "set pretrained=True."
        )

    set_seed(config.seed)
    if resume_checkpoint is None:
        log_path = Path(config.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
    device = get_device()
    class_mapping = load_class_mapping(Path(config.class_names_path))
    n_classes = len(class_mapping)

    if resume_checkpoint is not None:
        saved_mapping = {int(key): str(value) for key, value in resume_checkpoint["class_mapping"].items()}
        if saved_mapping != class_mapping or int(resume_checkpoint["num_classes"]) != n_classes:
            raise ValueError("Resume checkpoint class mapping does not match the current dataset mapping.")

    train_dataset = PlantDiseaseDataset(
        "train",
        transform=get_train_transforms(config.image_size),
        csv_path=config.train_csv,
    )
    val_dataset = PlantDiseaseDataset(
        "val",
        transform=get_eval_transforms(config.image_size),
        csv_path=config.val_csv,
    )
    train_loader = _make_loader(train_dataset, config.batch_size, True, config.num_workers, config.seed)
    val_loader = _make_loader(val_dataset, config.batch_size, False, config.num_workers, config.seed)

    model = build_swin_classifier(
        n_classes, dropout=config.dropout, pretrained=resume_checkpoint is None
    )
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    _, autocast_ctx, scaler = _amp_components(device)

    history = {"stage1": [], "stage2": []}
    if resume_checkpoint is not None:
        history = resume_checkpoint.get("history", history)
        if not history.get("stage1") and not history.get("stage2"):
            history_path = Path(config.history_path)
            if history_path.is_file():
                history = json.loads(history_path.read_text(encoding="utf-8"))
        completed_epoch = int(resume_checkpoint["current_epoch"])
        history = {
            stage: [row for row in history.get(stage, []) if int(row["epoch"]) <= completed_epoch]
            for stage in ("stage1", "stage2")
        }
        _save_history(history, Path(config.history_path))
        _rewrite_training_log(history, Path(config.log_path))
    else:
        completed_epoch = 0

    best_macro_f1 = float(
        resume_checkpoint.get("best_validation_macro_f1", -1.0)
        if resume_checkpoint is not None else -1.0
    )
    best_accuracy = float(
        resume_checkpoint.get("best_validation_accuracy", -1.0)
        if resume_checkpoint is not None else -1.0
    )
    best_path = Path(config.checkpoint_path)
    latest_path = Path(config.latest_checkpoint_path)
    if resume_checkpoint is not None and best_path.is_file():
        best_checkpoint = _load_checkpoint(best_path, map_location="cpu")
        best_validation = best_checkpoint.get("validation_metrics", {})
        best_macro_f1 = max(best_macro_f1, float(best_validation.get("macro_f1", -1.0)))
        best_accuracy = max(best_accuracy, float(best_validation.get("accuracy", -1.0)))
    global_epoch = completed_epoch
    global_step = int(resume_checkpoint.get("global_step", 0)) if resume_checkpoint else 0

    stages = [
        ("stage1", config.stage1_epochs, freeze_backbone, config.head_lr, 0.0),
        ("stage2", config.stage2_epochs, unfreeze_last_two_stages, config.head_lr, config.backbone_lr),
    ]

    print(f"Device: {device}")
    print(f"Classes: {n_classes} (from class_names.json)")
    print("Checkpoint selection metric: validation macro F1")
    print("Test split is not used during training.")

    resume_stage_index = int(resume_checkpoint.get("stage_index", 0)) if resume_checkpoint else 0
    resume_stage_epoch = int(resume_checkpoint.get("stage_epoch", 0)) if resume_checkpoint else 0
    early_state = resume_checkpoint.get("early_stopping_state", {}) if resume_checkpoint else {}
    stop_early = bool(early_state.get("stopped", False))
    if resume_checkpoint is not None:
        loader_states = resume_checkpoint.get("data_loader_generator_states", {})
        if "train" in loader_states:
            train_loader.generator.set_state(loader_states["train"])
        if "val" in loader_states:
            val_loader.generator.set_state(loader_states["val"])
        if "rng_state" in resume_checkpoint:
            _restore_rng_state(resume_checkpoint["rng_state"])

    for stage_index, (stage_name, epochs, setup_fn, head_lr, backbone_lr) in enumerate(stages):
        if epochs <= 0 or stop_early or stage_index < resume_stage_index:
            continue
        continuing_stage = resume_checkpoint is not None and stage_index == resume_stage_index
        stage_epoch_completed = resume_stage_epoch if continuing_stage else 0
        if stage_epoch_completed >= epochs:
            continue

        stage_best_macro_f1 = float(
            early_state.get("stage_best_macro_f1", -1.0) if continuing_stage else -1.0
        )
        stage_epochs_without_improvement = int(
            early_state.get("stage_epochs_without_improvement", 0) if continuing_stage else 0
        )
        setup_fn(model)
        param_groups = trainable_parameter_groups(
            model,
            backbone_lr=backbone_lr if backbone_lr > 0 else head_lr,
            head_lr=head_lr,
        )
        if stage_name == "stage1":
            optimizer = AdamW(
                [{"params": [p for p in model.parameters() if p.requires_grad], "lr": head_lr}],
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = AdamW(param_groups, weight_decay=config.weight_decay)

        scheduler = _build_scheduler(optimizer, epochs, config.warmup_epochs)
        if continuing_stage:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            if scaler is not None and resume_checkpoint.get("grad_scaler_state_dict") is not None:
                scaler.load_state_dict(resume_checkpoint["grad_scaler_state_dict"])
        print(f"\n{stage_name}: trainable parameters = {count_trainable_parameters(model):,}")

        for local_epoch in range(stage_epoch_completed + 1, epochs + 1):
            global_epoch += 1
            train_loss, train_true, train_pred, train_score = _run_epoch(
                model, train_loader, criterion, optimizer, device, autocast_ctx, scaler, train=True
            )
            val_loss, val_true, val_pred, val_score = _run_epoch(
                model, val_loader, criterion, None, device, autocast_ctx, None, train=False
            )
            scheduler.step()
            global_step += len(train_loader)

            train_metrics = compute_classification_metrics(
                train_true, train_pred, class_mapping, train_score
            )
            val_metrics = compute_classification_metrics(val_true, val_pred, class_mapping, val_score)
            record = {
                "epoch": global_epoch,
                "stage": stage_name,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_macro_f1": train_metrics["macro_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
            }
            history[stage_name].append(record)
            print(
                f"{stage_name} epoch {global_epoch}: "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

            if val_metrics["macro_f1"] > stage_best_macro_f1:
                stage_best_macro_f1 = val_metrics["macro_f1"]
                stage_epochs_without_improvement = 0
            else:
                stage_epochs_without_improvement += 1

            improved_best = val_metrics["macro_f1"] > best_macro_f1
            if improved_best:
                best_macro_f1 = val_metrics["macro_f1"]
            best_accuracy = max(best_accuracy, float(val_metrics["accuracy"]))

            if improved_best:
                best_payload = {
                    "model_state_dict": model.state_dict(),
                    "class_mapping": {str(k): v for k, v in class_mapping.items()},
                    "num_classes": n_classes,
                    "config": config.to_dict(),
                    "epoch": global_epoch,
                    "stage": stage_name,
                    "validation_metrics": {
                        key: value for key, value in val_metrics.items()
                        if key not in {"per_class", "confusion_matrix"}
                    },
                }
                _atomic_torch_save(best_payload, best_path)
                print(f"Saved best checkpoint to {best_path} (val macro F1={best_macro_f1:.4f})")

            if stage_epochs_without_improvement >= config.early_stopping_patience:
                print(
                    f"Early stopping {stage_name} after {config.early_stopping_patience} "
                    "epochs without validation macro F1 improvement."
                )
                stop_early = True
            _save_history(history, Path(config.history_path))
            _append_training_log(record, Path(config.log_path))
            latest_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "grad_scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "stage": stage_name,
                "stage_index": stage_index,
                "stage_epoch": local_epoch,
                "current_epoch": global_epoch,
                "global_step": global_step,
                "best_validation_macro_f1": best_macro_f1,
                "best_validation_accuracy": best_accuracy,
                "early_stopping_state": {
                    "stage_best_macro_f1": stage_best_macro_f1,
                    "stage_epochs_without_improvement": stage_epochs_without_improvement,
                    "patience": config.early_stopping_patience,
                    "stopped": stop_early,
                },
                "rng_state": _capture_rng_state(),
                "data_loader_generator_states": {
                    "train": train_loader.generator.get_state(),
                    "val": val_loader.generator.get_state(),
                },
                "history": history,
                "config": config.to_dict(),
                "class_mapping": {str(k): v for k, v in class_mapping.items()},
                "num_classes": n_classes,
                "validation_metrics": {
                    key: value for key, value in val_metrics.items()
                    if key not in {"per_class", "confusion_matrix"}
                },
            }
            _atomic_torch_save(latest_payload, latest_path)
            if stop_early:
                break

    if best_macro_f1 < 0:
        raise RuntimeError("Training finished without saving a validation checkpoint.")
    print(f"Best validation macro F1: {best_macro_f1:.4f}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Latest resumable checkpoint: {latest_path}")
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Swin plant-disease baseline.")
    parser.add_argument(
        "--resume", nargs="?", const="", default=None, metavar="CHECKPOINT",
        help="Resume from latest_checkpoint.pth, or from CHECKPOINT when provided.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig()
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.stage1_epochs is not None:
        config.stage1_epochs = args.stage1_epochs
    if args.stage2_epochs is not None:
        config.stage2_epochs = args.stage2_epochs
    if args.head_lr is not None:
        config.head_lr = args.head_lr
    if args.backbone_lr is not None:
        config.backbone_lr = args.backbone_lr
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.label_smoothing is not None:
        config.label_smoothing = args.label_smoothing
    if args.patience is not None:
        config.early_stopping_patience = args.patience
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    return config


if __name__ == "__main__":
    args = parse_args()
    if args.resume is not None:
        overrides = (
            args.batch_size, args.stage1_epochs, args.stage2_epochs, args.head_lr,
            args.backbone_lr, args.weight_decay, args.label_smoothing, args.patience,
            args.num_workers,
        )
        if any(value is not None for value in overrides):
            raise SystemExit("Do not combine --resume with hyperparameter overrides; saved config is restored.")
        resume_path = Path(args.resume) if args.resume else Path(TrainConfig().latest_checkpoint_path)
        if not resume_path.is_file():
            raise SystemExit(f"Resume checkpoint not found: {resume_path}")
        saved_checkpoint = _load_checkpoint(resume_path, map_location="cpu")
        config = _config_from_dict(saved_checkpoint["config"])
        train(config, resume_path=resume_path)
    else:
        train(config_from_args(args))
