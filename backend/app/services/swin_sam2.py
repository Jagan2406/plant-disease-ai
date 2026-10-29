from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAM2_REPO_PATH = Path(
    r"C:\Users\jagan\OneDrive\Desktop\cp review\sam2"
)
DEFAULT_SAM2_CHECKPOINT_PATH = Path(
    r"C:\Users\jagan\Downloads\sam2.1_hiera_small.pt"
)
DEFAULT_SWIN_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "swin_sam2" / "best_model.pth"


def _add_import_path(path: Path) -> None:
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)


class SwinSAM2Predictor:
    def __init__(
        self,
        segmenter: Any,
        classifier: torch.nn.Module,
        class_mapping: dict[int, str],
        image_transform: Any,
    ) -> None:
        self.segmenter = segmenter
        self.classifier = classifier
        self.class_mapping = class_mapping
        self.image_transform = image_transform

    @classmethod
    def load(cls) -> SwinSAM2Predictor:
        _add_import_path(PROJECT_ROOT)

        sam2_repo_path = Path(
            os.environ.get("SAM2_REPO_PATH", str(DEFAULT_SAM2_REPO_PATH))
        ).expanduser().resolve()
        sam2_checkpoint_path = Path(
            os.environ.get(
                "SAM2_CHECKPOINT_PATH", str(DEFAULT_SAM2_CHECKPOINT_PATH)
            )
        ).expanduser().resolve()
        swin_checkpoint_path = Path(
            os.environ.get(
                "SWIN_SAM2_CHECKPOINT_PATH", str(DEFAULT_SWIN_CHECKPOINT_PATH)
            )
        ).expanduser().resolve()

        if not sam2_repo_path.is_dir():
            raise FileNotFoundError(f"SAM2 repository not found: {sam2_repo_path}")
        if not sam2_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {sam2_checkpoint_path}"
            )
        if not swin_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Swin-SAM2 checkpoint not found: {swin_checkpoint_path}"
            )

        _add_import_path(sam2_repo_path)

        from ai.classification.swin_model import build_swin_classifier
        from ai.classification.transforms import get_eval_transforms
        from ai.segmentation.config import default_config
        from ai.segmentation.inference import load_sam2_model

        sam2_config = replace(
            default_config(),
            checkpoint_path=sam2_checkpoint_path,
            save_masks=False,
            save_segmented_images=False,
        )
        segmenter = load_sam2_model(sam2_config, device="cpu")

        checkpoint = torch.load(
            swin_checkpoint_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise TypeError("Swin-SAM2 checkpoint must contain a dictionary.")

        raw_class_mapping = checkpoint.get("class_mapping")
        if not isinstance(raw_class_mapping, dict) or not raw_class_mapping:
            raise ValueError(
                "Swin-SAM2 checkpoint is missing its class mapping."
            )
        class_mapping = {
            int(class_id): str(class_name)
            for class_id, class_name in raw_class_mapping.items()
        }
        num_classes = int(checkpoint.get("num_classes", len(class_mapping)))
        if num_classes != len(class_mapping):
            raise ValueError(
                "Swin-SAM2 checkpoint class count does not match its class mapping."
            )
        if set(class_mapping) != set(range(num_classes)):
            raise ValueError(
                "Swin-SAM2 checkpoint class mapping indices are not contiguous."
            )

        model_state = checkpoint.get(
            "model_state_dict", checkpoint.get("model_state")
        )
        if model_state is None:
            raise KeyError(
                "Swin-SAM2 checkpoint is missing its model state dictionary."
            )
        checkpoint_config = checkpoint.get("config", {})
        dropout = float(checkpoint_config.get("dropout", 0.2))
        image_size = int(checkpoint_config.get("image_size", 224))
        classifier = build_swin_classifier(
            num_classes=num_classes,
            dropout=dropout,
            pretrained=False,
        )
        classifier.load_state_dict(model_state)
        classifier.to(torch.device("cpu"))
        classifier.eval()

        return cls(
            segmenter=segmenter,
            classifier=classifier,
            class_mapping=class_mapping,
            image_transform=get_eval_transforms(image_size),
        )

    def predict(self, image: Image.Image, top_k: int = 5) -> dict[str, Any]:
        segmentation = self.segmenter.segment_image(image)
        mask = np.asarray(segmentation.final_mask, dtype=bool)
        segmented_image = Image.fromarray(
            np.asarray(segmentation.segmented_image, dtype=np.uint8),
            mode="RGB",
        )
        input_tensor = self.image_transform(segmented_image).unsqueeze(0)

        with torch.inference_mode():
            probabilities = torch.softmax(
                self.classifier(input_tensor).float(), dim=1
            )[0]

        count = min(top_k, probabilities.numel())
        scores, indices = torch.topk(probabilities, k=count)
        predictions = [
            {
                "class_id": int(class_id),
                "class_name": self.class_mapping[int(class_id)],
                "confidence": float(score),
            }
            for score, class_id in zip(scores, indices)
        ]
        predicted = predictions[0]

        return {
            "predicted_class": predicted["class_name"],
            "class_id": predicted["class_id"],
            "confidence": predicted["confidence"],
            "top_k_predictions": predictions,
            "mask_area_ratio": float(mask.mean()),
            "fallback_used": bool(segmentation.fallback_used),
            "segmentation_metadata": {
                "candidate_count": len(segmentation.candidate_metadata),
                "selected_candidate_score": segmentation.selected_candidate_score,
                "mask_shape": list(mask.shape),
            },
        }
