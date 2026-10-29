"""Lightweight environment and pipeline checks. Does not train or scan the full dataset."""

from __future__ import annotations

import importlib
import sys
import traceback

REQUIRED_PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
}


def _ok(message: str) -> None:
    print(f"[PASS] {message}")


def _fail(message: str) -> None:
    print(f"[FAIL] {message}")


def main() -> int:
    failures = 0
    print("Swin baseline lightweight setup verification")
    print("===========================================")

    python_version = sys.version_info
    if python_version >= (3, 11):
        _ok(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        _fail(f"Python {python_version.major}.{python_version.minor} (target is 3.11+)")
        failures += 1

    missing = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            _ok(f"{package_name} ({version})")
        except ImportError:
            missing.append(package_name)
            _fail(f"Missing dependency: {package_name}")
            failures += 1

    if missing:
        print("===========================================")
        print("Missing dependencies: " + ", ".join(missing))
        print("Install them, then re-run: python -m ai.verify_setup")
        return 1

    from ai.classification.config import TrainConfig
    from ai.classification.swin_model import (
        build_swin_classifier,
        freeze_backbone,
        unfreeze_last_two_stages,
    )
    from ai.classification.transforms import get_eval_transforms, get_train_transforms
    from ai.data.dataset import PlantDiseaseDataset
    from ai.data.manifest import CLASS_NAMES_PATH, SPLIT_CSV_PATHS, load_class_mapping, load_split_frame
    from ai.utils.device import collect_device_info, get_device
    import torch
    from torch.utils.data import DataLoader

    for key, value in collect_device_info().items():
        print(f"       {key}: {value}")

    for split, path in SPLIT_CSV_PATHS.items():
        if path.exists():
            frame = load_split_frame(split)
            _ok(f"{split}.csv exists ({len(frame)} rows)")
        else:
            _fail(f"Missing split CSV: {path}")
            failures += 1

    if CLASS_NAMES_PATH.exists():
        mapping = load_class_mapping()
        _ok(f"class_names.json exists with {len(mapping)} classes")
        if len(mapping) != TrainConfig().num_classes():
            _fail("Config num_classes does not match class_names.json")
            failures += 1
    else:
        _fail(f"Missing {CLASS_NAMES_PATH}")
        failures += 1
        return 1

    try:
        train_tf = get_train_transforms()
        eval_tf = get_eval_transforms()
        _ok(f"Transforms constructed (train={len(train_tf.transforms)}, eval={len(eval_tf.transforms)})")
    except Exception as exc:
        _fail(f"Transform construction failed: {exc}")
        failures += 1
        return 1

    try:
        n_classes = len(mapping)
        model = build_swin_classifier(n_classes, pretrained=False)
        _ok(
            f"Swin-T constructed without pretrained weights "
            f"(weights=None), head outputs {n_classes} classes"
        )
        freeze_backbone(model)
        _ok("Stage 1 freeze_backbone ran")
        unfreeze_last_two_stages(model)
        _ok("Stage 2 unfreeze_last_two_stages ran")
    except Exception as exc:
        _fail(f"Swin model construction failed: {exc}")
        traceback.print_exc()
        failures += 1
        return 1

    try:
        dataset = PlantDiseaseDataset("train", transform=eval_tf)
        if len(dataset) == 0:
            _fail("Train dataset is empty")
            failures += 1
        else:
            sample_path = dataset.image_paths[0]
            if not sample_path.exists():
                _fail(f"First train image is missing: {sample_path}")
                failures += 1
            else:
                loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
                images, labels, paths = next(iter(loader))
                _ok(
                    f"Sample batch loaded: images={tuple(images.shape)}, "
                    f"labels={labels.tolist()}, paths={len(paths)}"
                )
                device = get_device()
                model.to(device)
                model.eval()
                with torch.no_grad():
                    logits = model(images.to(device))
                _ok(f"Forward pass logits shape={tuple(logits.shape)}")
    except Exception as exc:
        _fail(f"Sample batch check failed: {exc}")
        traceback.print_exc()
        failures += 1

    print("===========================================")
    if failures:
        print(f"Verification finished with {failures} failure(s).")
        return 1
    print("Lightweight verification passed. Do not start full training until you choose to.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
