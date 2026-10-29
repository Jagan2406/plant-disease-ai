"""Validate Grad-CAM++ derivatives against finite differences on a real Swin activation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from decimal import Decimal, localcontext

from ai.classification.gradcam import (
    CHECKPOINT_PATH,
    TARGET_LAYER_NAME,
    _layer_norm_score_derivatives,
)
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

# One batch/row/column/channel coordinate per entry. A few channels at
# different spatial positions keep the higher-order reference inexpensive.
SELECTED_COORDINATES = ((0, 1, 2, 127), (0, 3, 3, 383), (0, 5, 4, 641))
FINITE_DIFFERENCE_STEP = 1e-2
ERROR_THRESHOLDS = {
    "first": {"absolute": 1e-9, "relative": 2e-3},
    "second": {"absolute": 1e-10, "relative": 2e-2},
    "third": {"absolute": 1e-11, "relative": 1e-1},
}


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dictionary, got {type(checkpoint).__name__}.")
    return checkpoint


def _first_valid_test_image() -> tuple[Path, Image.Image]:
    frame = load_split_frame("test")
    columns = resolve_split_columns(frame)
    for image_value in frame[columns["image"]]:
        image_path = resolve_image_path(image_value, PROJECT_ROOT)
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


def _tail_score(
    activation_token: torch.Tensor,
    model: torch.nn.Module,
    class_index: int,
    offset_coordinate: int,
    offset: Decimal,
    spatial_count: int,
) -> Decimal:
    """Evaluate one token's LayerNorm/linear score directly in high precision."""
    norm = model.norm
    classifier = model.head[-1]
    with localcontext() as context:
        context.prec = 60
        values = [Decimal.from_float(float(value)) for value in activation_token]
        values[offset_coordinate] += offset
        gamma = [Decimal.from_float(float(value)) for value in norm.weight.detach()]
        beta = (
            [Decimal.from_float(float(value)) for value in norm.bias.detach()]
            if norm.bias is not None
            else [Decimal(0)] * len(values)
        )
        weights = [
            Decimal.from_float(float(value))
            for value in classifier.weight[class_index].detach()
        ]
        mean = sum(values) / Decimal(len(values))
        centered = [value - mean for value in values]
        variance = sum(value * value for value in centered) / Decimal(len(values))
        inverse_std = (variance + Decimal.from_float(float(norm.eps))).sqrt()
        normalized = [centered[i] / inverse_std * gamma[i] + beta[i] for i in range(len(values))]
        token_score = sum(weights[i] * normalized[i] for i in range(len(values)))
        # Other tokens and classifier bias are constant under this coordinate
        # perturbation and therefore vanish from all finite differences.
        return token_score / Decimal(spatial_count)


def _finite_difference_derivatives(
    activation: torch.Tensor,
    coordinate: tuple[int, int, int, int],
    model: torch.nn.Module,
    class_index: int,
    step: float,
) -> tuple[float, float, float]:
    """Estimate scalar-coordinate derivatives without using CAM formulas."""
    batch_index, row_index, column_index, channel_index = coordinate
    token = activation[batch_index, row_index, column_index].detach().cpu()
    spatial_count = int(activation.shape[1] * activation.shape[2])

    def evaluate(offset: float) -> float:
        score = _tail_score(
            token,
            model,
            class_index,
            channel_index,
            Decimal(str(offset)),
            spatial_count,
        )
        return float(score)

    f0 = evaluate(0.0)
    f_plus = evaluate(step)
    f_minus = evaluate(-step)
    f_plus2 = evaluate(2.0 * step)
    f_minus2 = evaluate(-2.0 * step)

    def estimates(h: float, values: tuple[float, float, float, float, float]) -> tuple[float, float, float]:
        center, plus, minus, plus2, minus2 = values
        first = (minus2 - 8.0 * minus + 8.0 * plus - plus2) / (12.0 * h)
        second = (-plus2 + 16.0 * plus - 30.0 * center + 16.0 * minus - minus2) / (
            12.0 * h**2
        )
        third = (plus2 - 2.0 * plus + 2.0 * minus - minus2) / (2.0 * h**3)
        return first, second, third

    coarse = estimates(step, (f0, f_plus, f_minus, f_plus2, f_minus2))
    half_step = step / 2.0
    half_values = (
        f0,
        evaluate(half_step),
        evaluate(-half_step),
        evaluate(2.0 * half_step),
        evaluate(-2.0 * half_step),
    )
    fine = estimates(half_step, half_values)

    # The five-point third-derivative stencil is O(h^2); Richardson
    # extrapolation removes that leading truncation term.
    return fine[0], fine[1], fine[2] + (fine[2] - coarse[2]) / 3.0


def _error(analytical: float, reference: float) -> tuple[float, float]:
    absolute = abs(analytical - reference)
    relative = absolute / max(abs(reference), 1e-12)
    return absolute, relative


def main() -> int:
    class_mapping = load_class_mapping(CLASS_NAMES_PATH)
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Trained Swin checkpoint not found: {CHECKPOINT_PATH}")

    device = get_device()
    checkpoint = _load_checkpoint(CHECKPOINT_PATH, device)
    if "model_state_dict" in checkpoint:
        state_key = "model_state_dict"
    elif "model_state" in checkpoint:
        state_key = "model_state"
    else:
        raise KeyError("Checkpoint contains neither 'model_state_dict' nor 'model_state'.")

    if int(checkpoint.get("num_classes", len(class_mapping))) != len(class_mapping):
        raise ValueError("Checkpoint class count does not match class_names.json.")
    checkpoint_mapping = checkpoint.get("class_mapping")
    if checkpoint_mapping is not None:
        checkpoint_mapping = {int(index): str(name) for index, name in checkpoint_mapping.items()}
        if checkpoint_mapping != class_mapping:
            raise ValueError("Checkpoint class mapping does not match class_names.json.")

    dropout = float(checkpoint.get("config", {}).get("dropout", 0.2))
    image_size = int(checkpoint.get("config", {}).get("image_size", 224))
    model = build_swin_classifier(len(class_mapping), dropout=dropout, pretrained=False)
    model.load_state_dict(checkpoint[state_key])
    model.to(device).eval()

    image_path, image = _first_valid_test_image()
    input_tensor = get_eval_transforms(image_size)(image.copy()).unsqueeze(0).to(device)
    captured: dict[str, torch.Tensor] = {}

    def capture_target(_module, _inputs, output) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected target activation tensor, got {type(output).__name__}.")
        captured["activation"] = output

    target_layer = model.features[7][1]
    hook = target_layer.register_forward_hook(capture_target)
    try:
        with torch.no_grad():
            logits = model(input_tensor)
    finally:
        hook.remove()

    activation = captured["activation"]
    expected_shape = (1, 7, 7, 768)
    if tuple(activation.shape) != expected_shape:
        raise ValueError(f"Expected activation {expected_shape}, got {tuple(activation.shape)}.")
    predicted_class = int(logits.argmax(dim=1).item())
    analytical = _layer_norm_score_derivatives(activation, model, predicted_class)
    analytical_by_order = tuple(
        tensor.detach().to(dtype=torch.float64) for tensor in analytical
    )

    errors: dict[str, list[tuple[float, float]]] = {"first": [], "second": [], "third": []}
    order_names = ("first", "second", "third")
    print("Grad-CAM++ derivative validation")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Image: {image_path}")
    print(f"Device: {device}")
    print(f"Target layer: {TARGET_LAYER_NAME}")
    print(f"Activation shape: {tuple(activation.shape)}")
    print(f"Target class: {class_mapping[predicted_class]} (index {predicted_class})")
    print(f"Finite-difference step: {FINITE_DIFFERENCE_STEP:g} (float64 tail evaluation)")
    print("Reference: independent central finite differences through final LayerNorm, mean pool, and linear head")

    for coordinate in SELECTED_COORDINATES:
        reference = _finite_difference_derivatives(
            activation, coordinate, model, predicted_class, FINITE_DIFFERENCE_STEP
        )
        print(f"\nCoordinate {coordinate}:")
        for index, order in enumerate(order_names):
            analytical_value = float(analytical_by_order[index][coordinate].item())
            absolute, relative = _error(analytical_value, reference[index])
            errors[order].append((absolute, relative))
            print(
                f"  {order}: analytical={analytical_value:.10e}, "
                f"reference={reference[index]:.10e}, abs_error={absolute:.3e}, "
                f"rel_error={relative:.3e}"
            )

    all_passed = True
    print("\nDerivative-order results:")
    for order in order_names:
        limits = ERROR_THRESHOLDS[order]
        passed = all(
            absolute <= limits["absolute"] or relative <= limits["relative"]
            for absolute, relative in errors[order]
        )
        all_passed = all_passed and passed
        print(
            f"  {order}: {'PASS' if passed else 'FAIL'} "
            f"(abs_tol={limits['absolute']:.1e}, rel_tol={limits['relative']:.1e})"
        )
    print(f"OVERALL: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
