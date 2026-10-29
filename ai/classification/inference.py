"""Top-5 inference for a single image using the selected Swin checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from ai.classification.config import TrainConfig
from ai.classification.evaluate import load_checkpoint
from ai.classification.transforms import get_eval_transforms
from ai.utils.device import get_device


def predict_image(image_path: Path, checkpoint_path: Path | None = None, top_k: int = 5) -> list[tuple[str, float]]:
    config = TrainConfig()
    device = get_device()
    ckpt_path = checkpoint_path or Path(config.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train first with python -m ai.classification.train"
        )

    _checkpoint, model, class_mapping = load_checkpoint(ckpt_path, device)
    transform = get_eval_transforms(config.image_size)
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(f"Could not read image: {image_path} ({exc})") from exc

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor).float(), dim=1)[0]
    k = min(top_k, probabilities.numel())
    scores, indices = torch.topk(probabilities, k=k)
    return [(class_mapping[int(idx)], float(score)) for score, idx in zip(scores, indices)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swin baseline top-5 inference")
    parser.add_argument("--image", type=Path, required=True, help="Path to an input image")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = predict_image(args.image, args.checkpoint, args.top_k)
    for rank, (class_name, probability) in enumerate(predictions, start=1):
        print(f"{rank}. {class_name} — {probability:.4f}")


if __name__ == "__main__":
    main()
