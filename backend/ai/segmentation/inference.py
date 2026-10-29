"""Official SAM 2.1 image inference wrapper.

This module does not install dependencies or download checkpoints. The SAM 2
package and a local checkpoint must be available before model loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from PIL import Image

from ai.segmentation.config import SegmentationConfig, default_config


RGBImage: TypeAlias = np.ndarray | Image.Image
CandidateMask: TypeAlias = dict[str, Any]

_MODEL_CONFIGS = {
    "facebook/sam2.1-hiera-small": "configs/sam2.1/sam2.1_hiera_s.yaml",
}


@dataclass
class SegmentationResult:
    """Final leaf-focused segmentation result for one image."""

    final_mask: np.ndarray
    segmented_image: np.ndarray
    candidate_metadata: list[CandidateMask]
    selected_candidate_score: float | None
    fallback_used: bool


@dataclass
class SAM2Inference:
    """Reusable SAM 2.1 model, predictor, and automatic mask generator."""

    model: Any
    predictor: Any
    mask_generator: Any
    config: SegmentationConfig

    def generate_candidate_masks(self, image: RGBImage) -> list[CandidateMask]:
        """Generate and return raw SAM candidate masks with their metadata.

        The returned dictionaries are the unmodified records produced by
        ``SAM2AutomaticMaskGenerator``. They include fields such as
        ``segmentation``, ``area``, ``bbox``, ``predicted_iou``,
        ``stability_score``, and crop/point metadata when provided by SAM 2.
        No leaf-mask selection or image rendering is performed here.
        """

        import gc
        import torch
        rgb_array = _as_rgb_array(image)
        with torch.inference_mode():
            masks = self.mask_generator.generate(rgb_array)
        gc.collect()
        return masks

    def segment_image(self, image: RGBImage) -> SegmentationResult:
        """Select one leaf-foreground mask and render a masked RGB image.

        Candidate masks remain independent instances from SAM. The selected
        foreground can contain multiple leaves; this method does not attempt
        to identify a single leaf instance.
        """

        rgb_array = _as_rgb_array(image)
        candidates = self.generate_candidate_masks(rgb_array)
        selected, metadata = _select_leaf_candidate(candidates, rgb_array, self.config)
        if selected is None:
            return SegmentationResult(
                final_mask=np.zeros(rgb_array.shape[:2], dtype=bool),
                segmented_image=rgb_array.copy(),
                candidate_metadata=metadata,
                selected_candidate_score=None,
                fallback_used=True,
            )

        final_mask = np.asarray(selected["segmentation"], dtype=bool)
        segmented_image = np.zeros_like(rgb_array)
        segmented_image[final_mask] = rgb_array[final_mask]
        return SegmentationResult(
            final_mask=final_mask,
            segmented_image=segmented_image,
            candidate_metadata=metadata,
            selected_candidate_score=float(selected["selection_score"]),
            fallback_used=False,
        )


def load_sam2_model(
    config: SegmentationConfig | None = None,
    *,
    device: str | None = None,
) -> SAM2Inference:
    """Load a configured SAM 2.1 model once for reuse across images.

    Args:
        config: Segmentation settings. Its checkpoint must point to an
            existing local SAM 2.1 checkpoint.
        device: Optional torch device string passed to the official builder.

    Raises:
        ValueError: If the model identifier is unsupported or no checkpoint is
            configured.
        FileNotFoundError: If the configured checkpoint does not exist.
        RuntimeError: If the official SAM 2 package is unavailable or loading
            fails.
    """

    segmentation_config = config or default_config()
    model_config = _MODEL_CONFIGS.get(segmentation_config.model_id)
    if model_config is None:
        supported = ", ".join(sorted(_MODEL_CONFIGS))
        raise ValueError(
            f"Unsupported SAM 2 model_id '{segmentation_config.model_id}'. "
            f"Supported model IDs: {supported}."
        )

    checkpoint_path = segmentation_config.checkpoint_path
    if checkpoint_path is None:
        raise ValueError(
            "SegmentationConfig.checkpoint_path must point to a local SAM 2.1 "
            "checkpoint. Checkpoint downloads are intentionally not automatic."
        )
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"SAM 2.1 checkpoint was not found: {checkpoint_path}. "
            "Provide a downloaded local checkpoint in SegmentationConfig."
        )

    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as error:
        raise RuntimeError(
            "The official SAM 2 package is required for segmentation inference "
            "but is not installed. Install it separately before calling "
            "load_sam2_model; this project does not install it automatically."
        ) from error

    try:
        model = build_sam2(model_config, str(checkpoint_path), device=device)
        predictor = SAM2ImagePredictor(model)
        mask_generator = SAM2AutomaticMaskGenerator(
    model,
    points_per_side=segmentation_config.points_per_side,
    pred_iou_thresh=segmentation_config.pred_iou_thresh,
    stability_score_thresh=segmentation_config.stability_score_thresh,
    crop_n_layers=segmentation_config.crop_n_layers,
    min_mask_region_area=segmentation_config.min_mask_region_area,
)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load SAM 2.1 model '{segmentation_config.model_id}' "
            f"from checkpoint '{checkpoint_path}'."
        ) from error

    return SAM2Inference(
        model=model,
        predictor=predictor,
        mask_generator=mask_generator,
        config=segmentation_config,
    )


def _as_rgb_array(image: RGBImage) -> np.ndarray:
    """Validate and normalize one RGB image for the official SAM API."""

    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            raise ValueError(f"Image must be RGB, got mode '{image.mode}'.")
        array = np.asarray(image)
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise TypeError("image must be a PIL.Image.Image or a NumPy RGB array.")

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Image must have shape (height, width, 3), got {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Image must have non-zero height and width.")
    if array.dtype != np.uint8:
        raise TypeError(f"Image array must use dtype uint8, got {array.dtype}.")

    return np.ascontiguousarray(array)


def _select_leaf_candidate(
    candidates: list[CandidateMask],
    image: np.ndarray,
    config: SegmentationConfig,
) -> tuple[CandidateMask | None, list[CandidateMask]]:
    image_area = image.shape[0] * image.shape[1]
    image_leaf_fraction = float(np.mean(_green_leaf_pixels(image)))
    metadata: list[CandidateMask] = []
    eligible: list[CandidateMask] = []

    for candidate in candidates:
        mask = np.asarray(candidate.get("segmentation"), dtype=bool)
        if mask.shape != image.shape[:2]:
            continue
        area_ratio = float(mask.mean())
        record = dict(candidate)
        record["area_ratio"] = area_ratio
        record["green_leaf_fraction"] = _masked_fraction(_green_leaf_pixels(image), mask)
        record["largest_component_ratio"] = _largest_component_ratio(mask)
        record["area_score"] = _reasonable_area_score(area_ratio)

        is_closeup = (
            config.allow_closeup_full_image
            and image_leaf_fraction >= config.closeup_leaf_fraction_threshold
        )

        # ---------------------------------------------------------------
        # Quality gates
        # ---------------------------------------------------------------

        # Extremely small masks are usually isolated spots/fragments,
        # not complete leaf instances.
        if area_ratio < max(config.min_mask_area_ratio, 0.03):
            record["rejection_reason"] = "mask_area_too_small"
            metadata.append(record)
            continue

        # Very large masks with almost no vegetation pixels are usually
        # background/scene masks rather than leaves.
        if (
            area_ratio > 0.45
            and record["green_leaf_fraction"] < 0.15
            and not is_closeup
        ):
            record["rejection_reason"] = "large_low_leaf_content_mask"
            metadata.append(record)
            continue

        # ---------------------------------------------------------------
        # Candidate quality score
        # ---------------------------------------------------------------

        iou_score = float(candidate.get("predicted_iou", 0.0))
        green_score = record["green_leaf_fraction"]
        component_score = record["largest_component_ratio"]

        # Prefer masks with a reasonable amount of image coverage.
        # This prevents tiny fragments from winning.
        if area_ratio < 0.35:
            size_quality = min(
                1.0,
                max(0.0, (area_ratio - 0.03) / 0.32)
            )
        else:
            size_quality = max(
                0.0,
                1.0 - (area_ratio - 0.35) / 0.30
            )

        # Combine SAM confidence, vegetation content,
        # spatial coverage, and connectedness.
        score = (
            0.35 * iou_score
            + 0.25 * green_score
            + 0.25 * size_quality
            + 0.15 * component_score
        )
        record["size_quality"] = float(size_quality)
        record["selection_score"] = float(score)
        eligible.append(record)
        metadata.append(record)

    if not eligible:
        return None, metadata
    return max(eligible, key=lambda item: (item["selection_score"], item["area_ratio"])), metadata


def _green_leaf_pixels(image: np.ndarray) -> np.ndarray:
    red = image[..., 0].astype(np.int16)
    green = image[..., 1].astype(np.int16)
    blue = image[..., 2].astype(np.int16)
    return (green > red * 1.05) & (green > blue * 1.05) & (green > 35)


def _masked_fraction(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if mask.any() else 0.0


def _reasonable_area_score(area_ratio: float) -> float:
    target = 0.35
    spread = 0.35
    return max(0.0, 1.0 - abs(area_ratio - target) / spread)


def _largest_component_ratio(mask: np.ndarray) -> float:
    """Return largest connected component divided by total mask area."""

    total = int(mask.sum())

    if total == 0:
        return 0.0

    try:
        from scipy import ndimage

        labeled, count = ndimage.label(
            mask,
            structure=np.array(
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                ],
                dtype=np.uint8,
            ),
        )

        if count == 0:
            return 0.0

        component_sizes = np.bincount(
            labeled.ravel()
        )[1:]

        return float(component_sizes.max() / total)

    except ImportError:
        # Lightweight fallback if scipy is unavailable.
        return 1.0