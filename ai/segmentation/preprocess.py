"""Create a derived segmented dataset from the existing split CSVs.

This script never edits the source CSVs or original dataset. It is intentionally
standalone and must be invoked explicitly for full-dataset processing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from PIL import Image

from ai.segmentation.config import SegmentationConfig, default_config
from ai.segmentation.inference import load_sam2_model


def process_split(
    inference,
    split: str,
    source_csv: Path,
    project_root: Path,
    output_root: Path,
    progress_every: int,
    process_count: int | None,
    failures_path: Path,
) -> None:
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "The preprocessing script requires the existing pandas dependency. "
            "Install project requirements before running preprocessing."
        ) from error

    frame = pd.read_csv(source_csv)
    required = {"image_path", "class_name", "class_index"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source_csv} is missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    limit = len(frame) if process_count is None else min(process_count, len(frame))
    for row_number, row in frame.iloc[:limit].iterrows():
        source_path = project_root / str(row["image_path"])
        relative_output = Path("dataset_segmented") / "Dataset" / str(row["class_name"]) / source_path.name
        output_path = output_root / relative_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not output_path.exists():
                with Image.open(source_path) as image:
                    rgb_image = image.convert("RGB")
                result = inference.segment_image(rgb_image)
                Image.fromarray(result.segmented_image, mode="RGB").save(output_path)
            rows.append(
                {
                    "image_path": relative_output.as_posix(),
                    "class_name": row["class_name"],
                    "class_index": row["class_index"],
                }
            )
        except Exception as error:
            failure = {
                "split": split,
                "row": int(row_number),
                "source_path": str(source_path),
                "error": f"{type(error).__name__}: {error}",
            }
            with failures_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure) + "\n")

        if progress_every > 0 and (len(rows) + row_number + 1) % progress_every == 0:
            _save_progress(output_root, split, row_number + 1, limit)

    output_csv = project_root / "data" / f"segmented_{split}.csv"
    pd.DataFrame(rows, columns=["image_path", "class_name", "class_index"]).to_csv(
        output_csv, index=False
    )
    _save_progress(output_root, split, limit, limit)


def _save_progress(output_root: Path, split: str, completed: int, total: int) -> None:
    progress_path = output_root / "segmentation_progress.json"
    progress = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress[split] = {"completed": completed, "total": total}
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved processing-group size.")
    parser.add_argument("--process-count", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.process_count is not None and args.process_count < 1):
        parser.error("--batch-size and --process-count must be positive")

    project_root = args.project_root.resolve()
    input_root = (args.input_root or project_root).resolve()
    output_root = (args.output_root or project_root).resolve()
    failures_path = output_root / "segmentation_failures.jsonl"
    config = replace(default_config(), checkpoint_path=args.checkpoint.resolve())
    inference = load_sam2_model(config)

    for split in ("train", "val", "test"):
        process_split(
            inference=inference,
            split=split,
            source_csv=project_root / "data" / f"{split}.csv",
            project_root=input_root,
            output_root=output_root,
            progress_every=args.progress_every,
            process_count=args.process_count,
            failures_path=failures_path,
        )


if __name__ == "__main__":
    main()