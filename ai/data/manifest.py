"""Load existing split CSVs and class names. Does not create or modify splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CLASS_NAMES_PATH = DATA_DIR / "class_names.json"
SPLIT_CSV_PATHS = {
    "train": DATA_DIR / "train.csv",
    "val": DATA_DIR / "val.csv",
    "test": DATA_DIR / "test.csv",
}

_IMAGE_COLUMNS = ("image_path", "filepath", "file_path", "path", "image", "filename", "file")
_INDEX_COLUMNS = ("class_index", "label_index", "target", "label", "y")
_NAME_COLUMNS = ("class_name", "classname", "class", "label_name", "disease")


def get_project_root() -> Path:
    return PROJECT_ROOT


def load_class_mapping(path: Path | None = None) -> dict[int, str]:
    json_path = path or CLASS_NAMES_PATH
    if not json_path.exists():
        raise FileNotFoundError(f"Class mapping not found: {json_path}")

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"class_names.json must be a non-empty object: {json_path}")

    mapping: dict[int, str] = {}
    for key, value in raw.items():
        mapping[int(key)] = str(value)

    expected = set(range(len(mapping)))
    if set(mapping) != expected:
        raise ValueError(
            f"class_names.json indices must be contiguous from 0 to {len(mapping) - 1}."
        )
    return mapping


def num_classes(path: Path | None = None) -> int:
    return len(load_class_mapping(path))


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def resolve_split_columns(frame: pd.DataFrame) -> dict[str, str]:
    columns = list(frame.columns)
    image_col = _pick_column(columns, _IMAGE_COLUMNS)
    index_col = _pick_column(columns, _INDEX_COLUMNS)
    name_col = _pick_column(columns, _NAME_COLUMNS)

    if image_col is None:
        raise ValueError(f"Could not find an image-path column in {columns}")
    if index_col is None and name_col is None:
        raise ValueError(f"Could not find a class index or class name column in {columns}")

    resolved = {"image": image_col}
    if index_col is not None:
        resolved["index"] = index_col
    if name_col is not None:
        resolved["name"] = name_col
    return resolved


def load_split_frame(split: str) -> pd.DataFrame:
    if split not in SPLIT_CSV_PATHS:
        raise ValueError(f"Unknown split '{split}'. Expected one of {tuple(SPLIT_CSV_PATHS)}.")
    csv_path = SPLIT_CSV_PATHS[split]
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def resolve_image_path(image_path: str | Path, project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    path = Path(str(image_path))
    if not path.is_absolute():
        path = root / path
    return path
