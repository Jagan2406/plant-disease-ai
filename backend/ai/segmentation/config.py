"""Configuration for the SAM 2.1 segmentation stage.

This module contains configuration only. It does not import SAM, load
checkpoints, download assets, run inference, or modify dataset files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "sam2"


@dataclass(frozen=True)
class SegmentationConfig:
    """Settings for local SAM 2.1 inference and leaf-mask selection."""

    model_id: str = "facebook/sam2.1-hiera-small"
    checkpoint_path: Path | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    min_mask_area_ratio: float = 0.015
    max_mask_area_ratio: float = 0.90
    reject_large_masks: bool = True
    allow_closeup_full_image: bool = True
    closeup_leaf_fraction_threshold: float = 0.75
    score_iou_weight: float = 0.40
    score_leaf_fraction_weight: float = 0.30
    score_area_weight: float = 0.15
    score_component_weight: float = 0.15
    save_masks: bool = True
    save_segmented_images: bool = True

    points_per_side: int = 12
    pred_iou_thresh: float = 0.70
    stability_score_thresh: float = 0.85
    crop_n_layers: int = 0
    min_mask_region_area: int = 100


def default_config() -> SegmentationConfig:
    """Return configuration defaults without initializing a segmentation model."""

    return SegmentationConfig()