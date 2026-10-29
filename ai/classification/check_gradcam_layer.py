"""Inspect the forward activation shape of the selected Swin Grad-CAM layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from ai.classification.swin_model import build_swin_classifier
from ai.classification.transforms import get_eval_transforms
from ai.data.manifest import (
    CLASS_NAMES_PATH,
    PROJECT_ROOT,
    load_class_mapping,
    load_split_frame,
    resolve_image_path,
    resolve_split_columns,
)
from ai.utils.device import get_device


CHECKPOINT_PATH = PROJECT_ROOT / "models" / "swin" / "best_model.pth"
TARGET_LAYER_NAME = "model.features[7][1]"


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must contain a dictionary, got {type(checkpoint).__name__}.")
    return checkpoint


def _first_valid_test_image() -> tuple[Path, Image.Image]:
    frame = load_split_frame("test")
    columns = resolve_split_columns(frame)
    for value in frame[columns["image"]]:
        image_path = resolve_image_path(value, PROJECT_ROOT)
        if not image_path.is_file():
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                return image_path, image.convert("RGB")
        except (UnidentifiedImageError, OSError):
            continue
    raise FileNotFoundError("No valid image was found in data/test.csv.")


def _tensor_leaves(value: Any, location: str = "output") -> list[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        return [(location, value)]
    if isinstance(value, (tuple, list)):
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(_tensor_leaves(item, f"{location}[{index}]"))
        return leaves
    return []


def _describe_activation(value: Any) -> torch.Tensor:
    print(f"Target activation type: {type(value).__name__}")
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        print(f"Target activation structure: {type(value).__name__} with {len(value)} item(s)")
        for index, item in enumerate(value):
            shape = tuple(item.shape) if isinstance(item, torch.Tensor) else "not a tensor"
            print(f"  [{index}] type={type(item).__name__}, shape={shape}")
        leaves = _tensor_leaves(value)
        if len(leaves) == 1:
            location, tensor = leaves[0]
            print(f"Extracted the only tensor activation at {location}.")
            return tensor
        if not leaves:
            raise TypeError("Target layer output contains no tensor activation.")
        locations = ", ".join(location for location, _ in leaves)
        raise TypeError(
            "Target layer output contains multiple tensors; refusing to guess which "
            f"is the activation. Tensor locations: {locations}"
        )
    raise TypeError(f"Target layer output is not a tensor: {type(value).__name__}.")


def main() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Trained Swin checkpoint not found: {CHECKPOINT_PATH}")

    class_mapping = load_class_mapping(CLASS_NAMES_PATH)
    if len(class_mapping) != 23:
        raise ValueError(f"Expected 23 classes in class_names.json, found {len(class_mapping)}.")

    device = get_device()
    checkpoint = _load_checkpoint(CHECKPOINT_PATH, device)
    if "model_state_dict" in checkpoint:
        state_key = "model_state_dict"
    elif "model_state" in checkpoint:
        state_key = "model_state"
    else:
        raise KeyError("Checkpoint contains neither 'model_state_dict' nor 'model_state'.")

    model = build_swin_classifier(num_classes=23, pretrained=False)
    model.load_state_dict(checkpoint[state_key])
    model.to(device)
    model.eval()

    if not hasattr(model, "features") or len(model.features) <= 7 or len(model.features[7]) <= 1:
        raise AttributeError(f"The model does not contain the requested target layer {TARGET_LAYER_NAME}.")
    target_layer = model.features[7][1]

    image_path, original_image = _first_valid_test_image()
    input_tensor = get_eval_transforms()(original_image).unsqueeze(0).to(device)

    captured: dict[str, Any] = {}

    def capture_activation(_module, _inputs, output) -> None:
        captured["activation"] = output

    hook = target_layer.register_forward_hook(capture_activation)
    try:
        with torch.inference_mode():
            logits = model(input_tensor)
    finally:
        hook.remove()

    if "activation" not in captured:
        raise RuntimeError(f"Forward hook on {TARGET_LAYER_NAME} did not capture an output.")

    print("=" * 60)
    print("Grad-CAM++ Target Layer Diagnostic")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Image path: {image_path}")
    print(f"Target layer: {TARGET_LAYER_NAME}")
    print()

    activation = _describe_activation(captured["activation"])
    probabilities = torch.softmax(logits, dim=1)
    predicted_index = int(torch.argmax(probabilities, dim=1).item())
    confidence = float(probabilities[0, predicted_index].item())

    print(f"Target activation shape: {tuple(activation.shape)}")
    print(f"Model output shape: {tuple(logits.shape)}")
    print()
    print(f"Predicted class index: {predicted_index}")
    print(f"Predicted class name: {class_mapping[predicted_index]}")
    print(f"Predicted confidence: {confidence:.6f}")
    print()
    print(f"Number of classes: {len(class_mapping)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
