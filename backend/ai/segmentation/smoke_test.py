"""Run SAM 2.1 segmentation on 3-5 images and create a contact sheet."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ai.segmentation.config import default_config
from ai.segmentation.inference import load_sam2_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", action="append", required=True, dest="images")
    parser.add_argument("--output", type=Path, default=Path("results/sam2/smoke_contact_sheet.jpg"))
    args = parser.parse_args()
    if not 3 <= len(args.images) <= 5:
        parser.error("Provide between 3 and 5 --image arguments")

    inference = load_sam2_model(
        replace(default_config(), checkpoint_path=args.checkpoint.resolve())
    )
    panels: list[Image.Image] = []
    for image_arg in args.images:
        image_path = Path(image_arg)
        with Image.open(image_path) as image:
            original = image.convert("RGB")
        result = inference.segment_image(original)
        area_ratio = float(np.mean(result.final_mask))
        score = "fallback" if result.selected_candidate_score is None else f"{result.selected_candidate_score:.3f}"
        print(
            f"{image_path}: mask_area_ratio={area_ratio:.4f} "
            f"selected_score={score} fallback={result.fallback_used}"
        )
        panel = _panel(original, result.segmented_image, image_path.name, area_ratio, score)
        panels.append(panel)

    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels)
    contact_sheet = Image.new("RGB", (width, height), "white")
    offset = 0
    for panel in panels:
        contact_sheet.paste(panel, (0, offset))
        offset += panel.height
    args.output.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.save(args.output)
    print(f"Saved contact sheet: {args.output}")


def _panel(
    original: Image.Image,
    segmented: np.ndarray,
    name: str,
    area_ratio: float,
    score: str,
) -> Image.Image:
    thumbnail_width = 320
    thumbnail_height = max(1, round(original.height * thumbnail_width / original.width))
    left = original.resize((thumbnail_width, thumbnail_height))
    right = Image.fromarray(segmented).resize((thumbnail_width, thumbnail_height))
    panel = Image.new("RGB", (thumbnail_width * 2, thumbnail_height + 42), "white")
    panel.paste(left, (0, 42))
    panel.paste(right, (thumbnail_width, 42))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), f"{name} | area={area_ratio:.3f} | score={score}", fill="black")
    return panel


if __name__ == "__main__":
    main()