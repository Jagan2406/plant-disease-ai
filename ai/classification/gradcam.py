"""Grad-CAM++ explanations for the trained torchvision Swin-T classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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


def _layer_norm_score_derivatives(
    activation_nhwc: torch.Tensor,
    model: torch.nn.Module,
    class_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact diagonal score derivatives through Swin's final LayerNorm/head.

    After ``features[7][1]``, torchvision Swin applies channel-wise LayerNorm,
    spatial mean pooling, and a linear classifier. For one class score this is
    separable by spatial token, while LayerNorm couples channels within each
    token. The expressions below are the analytic first, second, and third
    derivatives of that exact path, including LayerNorm epsilon and affine
    scale. This avoids approximating higher derivatives with powers of the
    first derivative or materializing a prohibitively large derivative tensor.
    All returned tensors keep the original Swin ``[B,H,W,C]`` layout.
    """
    if activation_nhwc.ndim != 4 or activation_nhwc.shape[0] != 1:
        raise ValueError(
            "Expected one rank-4 NHWC activation tensor, got "
            f"{tuple(activation_nhwc.shape)}."
        )
    norm = getattr(model, "norm", None)
    head = getattr(model, "head", None)
    if not isinstance(norm, torch.nn.LayerNorm):
        raise TypeError(f"Expected final LayerNorm, got {type(norm).__name__}.")
    if not isinstance(head, torch.nn.Sequential) or not isinstance(head[-1], torch.nn.Linear):
        raise TypeError("Expected the trained Swin Dropout + Linear classification head.")
    classifier = head[-1]

    batch_size, height, width, channels = activation_nhwc.shape
    if channels != classifier.in_features or channels != norm.normalized_shape[-1]:
        raise ValueError("Target activation channels do not match LayerNorm and classifier dimensions.")
    if not norm.elementwise_affine or norm.weight is None:
        raise ValueError("Expected the trained torchvision Swin LayerNorm to use affine parameters.")
    if not 0 <= class_index < classifier.out_features:
        raise ValueError(f"Class index {class_index} is outside the classifier output range.")

    # In eval mode Dropout is the identity and AdaptiveAvgPool2d(1) is a mean
    # over H*W. Fold both the classifier row and that spatial scale into a.
    spatial_count = height * width
    a = classifier.weight[class_index] * norm.weight / spatial_count
    b = a - a.mean()

    centered = activation_nhwc - activation_nhwc.mean(dim=-1, keepdim=True)
    inverse_std = torch.rsqrt(centered.pow(2).mean(dim=-1, keepdim=True) + norm.eps)
    projected = (centered * b).sum(dim=-1, keepdim=True)
    channel_count = float(channels)

    first = b * inverse_std - projected * inverse_std.pow(3) * centered / channel_count
    second = (
        -2.0 * b * inverse_std.pow(3) * centered / channel_count
        + 3.0 * projected * inverse_std.pow(5) * centered.pow(2) / (channel_count**2)
        - projected * inverse_std.pow(3) * (channels - 1) / (channel_count**2)
    )
    third = (
        9.0 * b * inverse_std.pow(5) * centered.pow(2) / (channel_count**2)
        - 3.0 * b * inverse_std.pow(3) * (channels - 1) / (channel_count**2)
        - 15.0 * projected * inverse_std.pow(7) * centered.pow(3) / (channel_count**3)
        + 9.0 * projected * inverse_std.pow(5) * centered * (channels - 1) / (channel_count**3)
    )
    return first.expand(batch_size, height, width, channels), second, third


def _gradcam_plus_plus(
    activation_nhwc: torch.Tensor,
    first_derivative_nhwc: torch.Tensor,
    second_derivative_nhwc: torch.Tensor,
    third_derivative_nhwc: torch.Tensor,
) -> torch.Tensor:
    """Apply standard Grad-CAM++ alpha and positive-gradient weights to NHWC data."""
    derivative_tensors = (
        first_derivative_nhwc,
        second_derivative_nhwc,
        third_derivative_nhwc,
    )
    if activation_nhwc.ndim != 4 or any(
        derivative.shape != activation_nhwc.shape for derivative in derivative_tensors
    ):
        shapes = [tuple(activation_nhwc.shape), *(tuple(item.shape) for item in derivative_tensors)]
        raise ValueError(f"Expected matching rank-4 NHWC activation/derivative tensors, got {shapes}.")
    if activation_nhwc.shape[0] != 1:
        raise ValueError(f"This CLI expects one image, got batch size {activation_nhwc.shape[0]}.")

    # Convert NHWC to NCHW only for the standard channel-wise spatial sums and
    # channel aggregation in Grad-CAM++.
    activations = activation_nhwc.permute(0, 3, 1, 2)
    first = first_derivative_nhwc.permute(0, 3, 1, 2)
    second = second_derivative_nhwc.permute(0, 3, 1, 2)
    third = third_derivative_nhwc.permute(0, 3, 1, 2)

    channel_activation_sum = activations.sum(dim=(2, 3), keepdim=True)
    alpha_denominator = 2.0 * second + channel_activation_sum * third
    eps = torch.finfo(activation_nhwc.dtype).eps
    alpha = second / torch.where(
        alpha_denominator.abs() > eps,
        alpha_denominator,
        torch.where(alpha_denominator < 0, -torch.full_like(alpha_denominator, eps),
                    torch.full_like(alpha_denominator, eps)),
    )
    channel_weights = (alpha * F.relu(first)).sum(dim=(2, 3), keepdim=True)
    cam = F.relu((channel_weights * activations).sum(dim=1, keepdim=True))

    minimum = cam.amin(dim=(2, 3), keepdim=True)
    maximum = cam.amax(dim=(2, 3), keepdim=True)
    return (cam - minimum) / (maximum - minimum).clamp_min(eps)


def _map_crop_cam_to_original(
    cam: torch.Tensor,
    original_image: Image.Image,
    resized_image: Image.Image,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-map a crop CAM through Resize(256) and CenterCrop geometry."""
    resized_height, resized_width = resized_image.height, resized_image.width
    top = int(round((resized_height - crop_size) / 2.0))
    left = int(round((resized_width - crop_size) / 2.0))
    if top < 0 or left < 0 or top + crop_size > resized_height or left + crop_size > resized_width:
        raise ValueError("CenterCrop geometry falls outside the resized image.")

    crop_cam = F.interpolate(cam, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
    resized_cam = cam.new_zeros((1, 1, resized_height, resized_width))
    resized_coverage = cam.new_zeros((1, 1, resized_height, resized_width))
    resized_cam[:, :, top : top + crop_size, left : left + crop_size] = crop_cam
    resized_coverage[:, :, top : top + crop_size, left : left + crop_size] = 1.0

    original_size = (original_image.height, original_image.width)
    original_cam = F.interpolate(
        resized_cam, size=original_size, mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()
    original_coverage = F.interpolate(
        resized_coverage, size=original_size, mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy().clip(0.0, 1.0)
    return original_cam, original_coverage


def _jet_colorize(heatmap: np.ndarray) -> Image.Image:
    value = np.clip(heatmap, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * value - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * value - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * value - 1.0), 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), mode="RGB")


def generate_explanation(
    image_path: Path,
    checkpoint_path: Path = CHECKPOINT_PATH,
    output_path: Path | None = None,
) -> tuple[Path, int, str, float, tuple[int, ...]]:
    class_mapping = load_class_mapping(CLASS_NAMES_PATH)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained Swin checkpoint not found: {checkpoint_path}")

    device = get_device()
    checkpoint = _load_checkpoint(checkpoint_path, device)
    if "model_state_dict" in checkpoint:
        state_key = "model_state_dict"
    elif "model_state" in checkpoint:
        state_key = "model_state"
    else:
        raise KeyError("Checkpoint contains neither 'model_state_dict' nor 'model_state'.")

    saved_classes = int(checkpoint.get("num_classes", len(class_mapping)))
    if saved_classes != len(class_mapping):
        raise ValueError(
            f"Checkpoint has {saved_classes} classes but class_names.json has {len(class_mapping)}."
        )
    checkpoint_mapping = checkpoint.get("class_mapping")
    if checkpoint_mapping is not None:
        normalized_mapping = {int(index): str(name) for index, name in checkpoint_mapping.items()}
        if normalized_mapping != class_mapping:
            raise ValueError("Checkpoint class mapping does not match data/class_names.json.")

    dropout = float(checkpoint.get("config", {}).get("dropout", 0.2))
    image_size = int(checkpoint.get("config", {}).get("image_size", 224))
    model = build_swin_classifier(
        num_classes=len(class_mapping), dropout=dropout, pretrained=False
    )
    model.load_state_dict(checkpoint[state_key])
    model.to(device)
    model.eval()

    if not hasattr(model, "features") or len(model.features) <= 7 or len(model.features[7]) <= 1:
        raise AttributeError(f"The model does not contain the target layer {TARGET_LAYER_NAME}.")
    target_layer = model.features[7][1]
    captured: dict[str, torch.Tensor] = {}

    def capture_activation(_module, _inputs, output) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"{TARGET_LAYER_NAME} must return a tensor, got {type(output).__name__}."
            )
        if output.ndim != 4:
            raise ValueError(
                f"Expected {TARGET_LAYER_NAME} output in [B,H,W,C] rank-4 layout, "
                f"got shape {tuple(output.shape)}."
            )
        captured["activation"] = output

    try:
        with Image.open(image_path) as opened_image:
            original_image = opened_image.convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as error:
        raise RuntimeError(f"Could not read image: {image_path} ({error})") from error

    input_tensor = get_eval_transforms(image_size)(original_image.copy()).unsqueeze(0).to(device)
    hook = target_layer.register_forward_hook(capture_activation)
    try:
        with torch.enable_grad():
            logits = model(input_tensor)
            if "activation" not in captured:
                raise RuntimeError(f"Forward hook on {TARGET_LAYER_NAME} did not capture an activation.")
            activation = captured["activation"]
            predicted_index = int(logits.argmax(dim=1).item())
    finally:
        hook.remove()

    if tuple(activation.shape) != (1, 7, 7, 768):
        raise ValueError(
            f"Expected verified Swin target shape (1, 7, 7, 768), got {tuple(activation.shape)}."
        )
    first_derivative, second_derivative, third_derivative = _layer_norm_score_derivatives(
        activation.detach(), model, predicted_index
    )
    cam = _gradcam_plus_plus(
        activation.detach(), first_derivative, second_derivative, third_derivative
    ).detach()
    confidence = float(torch.softmax(logits.detach(), dim=1)[0, predicted_index].item())

    resize_transform = get_eval_transforms(image_size).transforms[0]
    resized_image = resize_transform(original_image.copy())
    original_cam, coverage = _map_crop_cam_to_original(
        cam, original_image, resized_image, image_size
    )
    colored_heatmap = np.asarray(_jet_colorize(original_cam), dtype=np.float32)
    original_pixels = np.asarray(original_image, dtype=np.float32)
    alpha = (0.4 * coverage[..., None]).astype(np.float32)
    overlay = Image.fromarray(
        np.clip(np.round(original_pixels * (1.0 - alpha) + colored_heatmap * alpha), 0, 255).astype(
            np.uint8
        ),
        mode="RGB",
    )

    destination = output_path or (PROJECT_ROOT / "results" / "gradcam" / f"{image_path.stem}_gradcam_plus_plus.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(destination)
    return destination, predicted_index, class_mapping[predicted_index], confidence, tuple(activation.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one Swin Grad-CAM++ explanation.")
    parser.add_argument(
        "--image", type=Path, default=None,
        help="Image to explain. Defaults to the first valid entry in data/test.csv.",
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output", type=Path, default=None, help="Output overlay PNG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image is None:
        image_path, _original_image = _first_valid_test_image()
    else:
        image_path = args.image.resolve()
    output_path, class_index, class_name, confidence, activation_shape = generate_explanation(
        image_path=image_path,
        checkpoint_path=args.checkpoint.resolve(),
        output_path=args.output.resolve() if args.output else None,
    )
    print("Grad-CAM++ explanation generated")
    print(f"Image: {image_path}")
    print(f"Target layer: {TARGET_LAYER_NAME}")
    print(f"Activation shape (B,H,W,C): {activation_shape}")
    print(f"Predicted class: {class_name} (index {class_index})")
    print(f"Predicted confidence: {confidence:.6f}")
    print(f"Overlay saved: {output_path}")


if __name__ == "__main__":
    main()
