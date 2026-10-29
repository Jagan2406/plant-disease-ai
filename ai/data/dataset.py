"""Image dataset over the existing train/val/test CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from ai.data.manifest import (
    load_class_mapping,
    load_split_frame,
    resolve_image_path,
    resolve_split_columns,
)

MAX_CORRUPT_IMAGES = 5


class CorruptImageError(RuntimeError):
    """Raised when an image cannot be opened or converted to RGB."""


class PlantDiseaseDataset(Dataset):
    def __init__(
        self,
        split: str,
        transform=None,
        project_root: Path | None = None,
        csv_path: str | Path | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.split = split
        self.transform = transform
        self.project_root = project_root
        self.class_mapping = load_class_mapping()
        self.name_to_index = {name: idx for idx, name in self.class_mapping.items()}
        self._corrupt_count = 0

        frame = pd.read_csv(csv_path) if csv_path is not None else load_split_frame(split)
        columns = resolve_split_columns(frame)
        self.image_paths: list[Path] = []
        self.labels: list[int] = []
        self.class_names: list[str] = []

        for _, row in frame.iterrows():
            image_path = resolve_image_path(row[columns["image"]], self.project_root)
            label, class_name = self._label_from_row(row, columns)
            self.image_paths.append(image_path)
            self.labels.append(label)
            self.class_names.append(class_name)

        if not self.image_paths:
            raise ValueError(f"Split '{split}' contains no rows.")

    def _label_from_row(self, row: pd.Series, columns: dict[str, str]) -> tuple[int, str]:
        if "index" in columns:
            label = int(row[columns["index"]])
            class_name = self.class_mapping.get(label)
            if class_name is None:
                raise ValueError(f"class_index {label} is not present in class_names.json")
            if "name" in columns:
                csv_name = str(row[columns["name"]])
                if csv_name != class_name:
                    raise ValueError(
                        f"CSV class_name '{csv_name}' does not match class_names.json "
                        f"index {label} ('{class_name}')."
                    )
            return label, class_name

        class_name = str(row[columns["name"]])
        if class_name not in self.name_to_index:
            raise ValueError(f"Unknown class_name '{class_name}' in split '{self.split}'")
        return self.name_to_index[class_name], class_name

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        label = self.labels[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            self._corrupt_count += 1
            message = f"Corrupt or unreadable image in {self.split} split: {path} ({exc})"
            if self._corrupt_count > MAX_CORRUPT_IMAGES:
                raise CorruptImageError(
                    f"{message}. Stopped after {self._corrupt_count} failures; "
                    "fix the dataset instead of skipping broken files."
                ) from exc
            raise CorruptImageError(message) from exc

        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(path)
