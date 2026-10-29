#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "Dataset"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data"
RANDOM_SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def discover_class_folders(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    class_folders = sorted(
        [path for path in dataset_root.iterdir() if path.is_dir()],
        key=lambda p: p.name.lower(),
    )
    if not class_folders:
        raise ValueError(f"No class folders found under {dataset_root}")
    return class_folders


def find_valid_image_files(class_dir: Path) -> list[Path]:
    image_files: list[Path] = []
    for path in sorted(class_dir.rglob("*"), key=lambda p: p.as_posix()):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            image_files.append(path)
    return image_files


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_split_counts(total_images: int) -> tuple[int, int, int]:
    if total_images >= 3:
        train_count = max(1, int(round(total_images * 0.8)))
        val_count = max(1, int(round(total_images * 0.1)))
        test_count = max(1, total_images - train_count - val_count)
        if train_count + val_count + test_count != total_images:
            difference = total_images - (train_count + val_count + test_count)
            train_count += difference
        if train_count < 1:
            train_count = 1
        if val_count < 1:
            val_count = 1
        if test_count < 1:
            test_count = 1
        # Ensure exact total and maintain the requested proportions as closely as possible.
        while train_count + val_count + test_count > total_images:
            if train_count > 1:
                train_count -= 1
            elif val_count > 1:
                val_count -= 1
            else:
                test_count -= 1
        while train_count + val_count + test_count < total_images:
            if train_count >= val_count and train_count >= test_count:
                train_count += 1
            elif val_count >= test_count:
                val_count += 1
            else:
                test_count += 1
        return train_count, val_count, test_count

    if total_images == 2:
        return 1, 1, 0
    if total_images == 1:
        return 1, 0, 0
    return 0, 0, 0


def stratify_class_images(class_to_paths: dict[str, list[Path]], seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    split_paths: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for class_name in sorted(class_to_paths):
        paths = class_to_paths[class_name][:]
        rng.shuffle(paths)
        total = len(paths)
        if total == 0:
            continue

        train_count, val_count, test_count = compute_split_counts(total)

        # Guarantee representation in every split when total >= 3.
        if total >= 3:
            train_count = max(1, int(round(total * 0.8)))
            val_count = max(1, int(round(total * 0.1)))
            test_count = total - train_count - val_count
            if test_count < 1:
                test_count = 1
                train_count = total - val_count - test_count
            if train_count < 1:
                train_count = 1
                val_count = total - train_count - test_count
            if val_count < 1:
                val_count = 1
                train_count = total - val_count - test_count
            if train_count + val_count + test_count != total:
                train_count = total - val_count - test_count

        if total >= 3:
            train_paths = paths[:train_count]
            val_paths = paths[train_count : train_count + val_count]
            test_paths = paths[train_count + val_count : train_count + val_count + test_count]
        elif total == 2:
            train_paths = [paths[0]]
            val_paths = [paths[1]]
            test_paths = []
        elif total == 1:
            train_paths = [paths[0]]
            val_paths = []
            test_paths = []
        else:
            train_paths = []
            val_paths = []
            test_paths = []

        split_paths["train"].extend([p.relative_to(PROJECT_ROOT).as_posix() for p in train_paths])
        split_paths["val"].extend([p.relative_to(PROJECT_ROOT).as_posix() for p in val_paths])
        split_paths["test"].extend([p.relative_to(PROJECT_ROOT).as_posix() for p in test_paths])

    return split_paths


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "class_name", "class_index"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_metadata(class_names: list[str], split_paths: dict[str, list[str]]) -> tuple[dict[str, list[dict[str, str | int]]], dict[str, str]]:
    name_to_index = {name: idx for idx, name in enumerate(class_names)}
    metadata: dict[str, list[dict[str, str | int]]] = {"train": [], "val": [], "test": []}

    for split_name in ("train", "val", "test"):
        for image_path in split_paths[split_name]:
            full_path = PROJECT_ROOT / image_path
            class_name = full_path.parent.name
            metadata[split_name].append(
                {
                    "image_path": image_path,
                    "class_name": class_name,
                    "class_index": name_to_index[class_name],
                }
            )

    class_index_map = {str(idx): class_name for idx, class_name in enumerate(class_names)}
    return metadata, class_index_map


def summary_table(class_names: list[str], class_to_total: dict[str, int], split_paths: dict[str, list[str]]) -> tuple[str, dict[str, int], dict[str, set[str]], dict[str, int]]:
    class_names_sorted = sorted(class_names)
    split_class_sets: dict[str, set[str]] = {split_name: set() for split_name in ("train", "val", "test")}
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for split_name, image_paths in split_paths.items():
        split_counts[split_name] = len(image_paths)
        for image_path in image_paths:
            class_name = (PROJECT_ROOT / image_path).parent.name
            split_class_sets[split_name].add(class_name)

    lines: list[str] = []
    lines.append("DATASET SUMMARY")
    lines.append("---------------")
    lines.append(f"Total classes: {len(class_names_sorted)}")
    lines.append(f"Total images: {sum(class_to_total.values())}")
    lines.append("")
    lines.append("TRAIN")
    lines.append("-----")
    lines.append(f"Images: {split_counts['train']}")
    lines.append(f"Percentage: {split_counts['train'] / max(1, sum(class_to_total.values())) * 100:.2f}%")
    lines.append(f"Classes represented: {len(split_class_sets['train'])}")
    lines.append("")
    lines.append("VALIDATION")
    lines.append("----------")
    lines.append(f"Images: {split_counts['val']}")
    lines.append(f"Percentage: {split_counts['val'] / max(1, sum(class_to_total.values())) * 100:.2f}%")
    lines.append(f"Classes represented: {len(split_class_sets['val'])}")
    lines.append("")
    lines.append("TEST")
    lines.append("----")
    lines.append(f"Images: {split_counts['test']}")
    lines.append(f"Percentage: {split_counts['test'] / max(1, sum(class_to_total.values())) * 100:.2f}%")
    lines.append(f"Classes represented: {len(split_class_sets['test'])}")
    lines.append("")
    lines.append("PER-CLASS DISTRIBUTION")
    lines.append("----------------------")
    lines.append("Class | Total | Train | Validation | Test")

    for class_name in class_names_sorted:
        total = class_to_total[class_name]
        train_count = sum(1 for path in split_paths["train"] if (PROJECT_ROOT / path).parent.name == class_name)
        val_count = sum(1 for path in split_paths["val"] if (PROJECT_ROOT / path).parent.name == class_name)
        test_count = sum(1 for path in split_paths["test"] if (PROJECT_ROOT / path).parent.name == class_name)
        lines.append(f"{class_name} | {total} | {train_count} | {val_count} | {test_count}")

    lines.append("")
    lines.append("DATA LEAKAGE CHECK")
    lines.append("------------------")
    all_paths = split_paths["train"] + split_paths["val"] + split_paths["test"]
    seen: set[str] = set()
    duplicates = 0
    for path in all_paths:
        if path in seen:
            duplicates += 1
        else:
            seen.add(path)
    lines.append(f"Duplicate paths across splits: {duplicates}")

    hash_to_paths: dict[str, list[str]] = defaultdict(list)
    for split_name in ("train", "val", "test"):
        for image_path in split_paths[split_name]:
            hash_to_paths[file_sha256(PROJECT_ROOT / image_path)].append(image_path)
    duplicate_content = sum(1 for paths in hash_to_paths.values() if len(paths) > 1)
    lines.append(f"Duplicate filenames/content if detectable: {duplicate_content}")

    lines.append("")
    lines.append("CLASS COVERAGE")
    lines.append("--------------")
    lines.append(f"Every class present in train: {'YES' if set(class_names_sorted) <= split_class_sets['train'] else 'NO'}")
    lines.append(f"Every class present in validation: {'YES' if set(class_names_sorted) <= split_class_sets['val'] else 'NO'}")
    lines.append(f"Every class present in test: {'YES' if set(class_names_sorted) <= split_class_sets['test'] else 'NO'}")
    lines.append("")
    lines.append("REPRODUCIBILITY")
    lines.append("---------------")
    lines.append(f"Random seed: {RANDOM_SEED}")

    return "\n".join(lines), split_counts, split_class_sets, {"duplicate_paths": duplicates, "duplicate_content": duplicate_content}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_folders = discover_class_folders(DATASET_ROOT)
    class_names = [folder.name for folder in class_folders]
    class_to_paths: dict[str, list[Path]] = {}
    class_to_total: dict[str, int] = {}

    for class_folder in class_folders:
        image_paths = find_valid_image_files(class_folder)
        class_to_paths[class_folder.name] = image_paths
        class_to_total[class_folder.name] = len(image_paths)

    total_images = sum(class_to_total.values())
    split_paths = stratify_class_images(class_to_paths, RANDOM_SEED)

    # Ensure each split contains every class for the real dataset.
    for split_name in ("train", "val", "test"):
        if not split_paths[split_name]:
            raise RuntimeError(f"Split {split_name} is empty; dataset may be too small for a valid stratified split.")

    rows_by_split: dict[str, list[dict[str, str | int]]] = {"train": [], "val": [], "test": []}
    for split_name in ("train", "val", "test"):
        for image_path in split_paths[split_name]:
            full_path = PROJECT_ROOT / image_path
            class_name = full_path.parent.name
            rows_by_split[split_name].append(
                {
                    "image_path": image_path,
                    "class_name": class_name,
                    "class_index": class_names.index(class_name),
                }
            )

    write_csv(OUTPUT_DIR / "train.csv", rows_by_split["train"])
    write_csv(OUTPUT_DIR / "val.csv", rows_by_split["val"])
    write_csv(OUTPUT_DIR / "test.csv", rows_by_split["test"])

    class_index_map = {str(idx): name for idx, name in enumerate(class_names)}
    (OUTPUT_DIR / "class_names.json").write_text(json.dumps(class_index_map, indent=4), encoding="utf-8")

    summary_text, split_counts, split_class_sets, _ = summary_table(class_names, class_to_total, split_paths)
    print(summary_text)

    # Final verification checks.
    checks = {
        "train_total": len(split_paths["train"]),
        "val_total": len(split_paths["val"]),
        "test_total": len(split_paths["test"]),
        "all_classes_train": set(class_names) <= split_class_sets["train"],
        "all_classes_val": set(class_names) <= split_class_sets["val"],
        "all_classes_test": set(class_names) <= split_class_sets["test"],
    }
    print("\nVERIFICATION")
    print("-------------")
    print(f"train + validation + test = total images: {checks['train_total'] + checks['val_total'] + checks['test_total']} == {total_images}")
    print(f"All 23 classes in train: {'YES' if checks['all_classes_train'] else 'NO'}")
    print(f"All 23 classes in validation: {'YES' if checks['all_classes_val'] else 'NO'}")
    print(f"All 23 classes in test: {'YES' if checks['all_classes_test'] else 'NO'}")

    all_paths = split_paths["train"] + split_paths["val"] + split_paths["test"]
    seen = set()
    duplicates = 0
    for path in all_paths:
        if path in seen:
            duplicates += 1
        else:
            seen.add(path)
    print(f"No image path appears in more than one split: {'YES' if duplicates == 0 else 'NO'}")


if __name__ == "__main__":
    main()
