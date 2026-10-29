from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "swin_sam2" / "best_model.pth"
TARGET_LAYER_NAME = "model.features[7][-1]"
IMAGE_SIZE = 224
CAM_EPSILON = 1e-8


class GradCAMServiceError(RuntimeError):
    """Raised when Grad-CAM++ cannot be generated for the supplied inputs."""


@dataclass(frozen=True)
class GradCAMResult:
    target_class_id: int
    target_class_name: str
    confidence: float
    activation_resolution: tuple[int, int]
    normalized_cam: np.ndarray
    leaf_coverage: float
    cam_inside_leaf: float | None


def _add_project_import_path() -> None:
    project_path = str(PROJECT_ROOT)
    if project_path not in sys.path:
        sys.path.insert(0, project_path)


class GradCAMPlusPlusService:
    def __init__(
        self,
        model: torch.nn.Module,
        class_mapping: dict[int, str],
        image_transform: Any,
    ) -> None:
        self.model = model
        self.class_mapping = class_mapping
        self.image_transform = image_transform
        self._inference_lock = threading.RLock()

    @classmethod
    def load(
        cls, checkpoint_path: Path = CHECKPOINT_PATH
    ) -> GradCAMPlusPlusService:
        _add_project_import_path()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Swin-SAM2 checkpoint not found: {checkpoint_path}"
            )

        from ai.classification.swin_model import build_swin_classifier
        from ai.classification.transforms import get_eval_transforms

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=torch.device("cpu"),
                weights_only=False,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise GradCAMServiceError(
                f"Could not load Swin-SAM2 checkpoint: {checkpoint_path}"
            ) from error

        if not isinstance(checkpoint, dict):
            raise GradCAMServiceError(
                "Swin-SAM2 checkpoint must contain a dictionary."
            )
        raw_mapping = checkpoint.get("class_mapping")
        if not isinstance(raw_mapping, dict) or not raw_mapping:
            raise GradCAMServiceError(
                "Swin-SAM2 checkpoint is missing its class mapping."
            )
        try:
            class_mapping = {
                int(class_id): str(class_name)
                for class_id, class_name in raw_mapping.items()
            }
        except (TypeError, ValueError) as error:
            raise GradCAMServiceError(
                "Swin-SAM2 checkpoint has an invalid class mapping."
            ) from error

        num_classes = int(checkpoint.get("num_classes", len(class_mapping)))
        if num_classes != len(class_mapping) or set(class_mapping) != set(
            range(num_classes)
        ):
            raise GradCAMServiceError(
                "Swin-SAM2 checkpoint class mapping does not match its class count."
            )

        model_state = checkpoint.get(
            "model_state_dict", checkpoint.get("model_state")
        )
        if model_state is None:
            raise GradCAMServiceError(
                "Swin-SAM2 checkpoint is missing its model state dictionary."
            )
        checkpoint_config = checkpoint.get("config", {})
        dropout = float(checkpoint_config.get("dropout", 0.2))
        model = build_swin_classifier(
            num_classes=num_classes,
            dropout=dropout,
            pretrained=False,
        )
        try:
            model.load_state_dict(model_state)
        except RuntimeError as error:
            raise GradCAMServiceError(
                "Swin-SAM2 checkpoint weights do not match the Swin-T architecture."
            ) from error
        model.to(torch.device("cpu"))
        model.eval()

        if (
            not hasattr(model, "features")
            or len(model.features) <= 7
            or len(model.features[7]) == 0
        ):
            raise GradCAMServiceError(
                f"The Swin-T model does not contain target layer {TARGET_LAYER_NAME}."
            )

        return cls(
            model=model,
            class_mapping=class_mapping,
            image_transform=get_eval_transforms(IMAGE_SIZE),
        )

    def generate(
        self,
        original_image: Image.Image,
        segmentation_result: Any = None,
        mask: np.ndarray | None = None,
    ) -> GradCAMResult:
        if not isinstance(original_image, Image.Image):
            raise TypeError("original_image must be a PIL image.")

        rgb_image = original_image.convert("RGB")
        image_array = np.asarray(rgb_image)
        if mask is None:
            if hasattr(segmentation_result, "final_mask"):
                mask = segmentation_result.final_mask
            else:
                mask = np.ones(image_array.shape[:2], dtype=bool)
        final_mask = np.asarray(mask, dtype=bool)
        if final_mask.shape != image_array.shape[:2]:
            raise GradCAMServiceError(
                "SAM2 final mask dimensions do not match the original image."
            )

        masked_array = np.zeros_like(image_array)
        masked_array[final_mask] = image_array[final_mask]
        masked_image = Image.fromarray(masked_array, mode="RGB")
        input_tensor = self.image_transform(masked_image).unsqueeze(0)

        target_layer = self.model.features[7][-1]
        captured: dict[str, torch.Tensor] = {}

        def capture_activation(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise GradCAMServiceError(
                    f"{TARGET_LAYER_NAME} must return a rank-4 tensor."
                )
            captured["activation"] = output

        with self._inference_lock:
            hook = target_layer.register_forward_hook(capture_activation)
            try:
                with torch.enable_grad():
                    logits = self.model(input_tensor)
                    activation = captured.get("activation")
                    if activation is None:
                        raise GradCAMServiceError(
                            f"Forward hook on {TARGET_LAYER_NAME} did not capture activations."
                        )
                    if activation.shape[0] != 1:
                        raise GradCAMServiceError(
                            "Grad-CAM++ expects exactly one image per request."
                        )
                    if activation.shape[1:3] != (7, 7):
                        raise GradCAMServiceError(
                            "Expected 7x7 channels-last Swin activations, "
                            f"got {tuple(activation.shape[1:3])}."
                        )

                    target_class_id = int(logits[0].argmax().item())
                    gradients = torch.autograd.grad(
                        logits[0, target_class_id],
                        activation,
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                    confidence = float(
                        torch.softmax(logits.float(), dim=1)[
                            0, target_class_id
                        ].item()
                    )
            finally:
                hook.remove()

        activations_nhwc = activation[0]
        gradients_nhwc = gradients[0]
        grad_squared = gradients_nhwc**2
        grad_cubed = gradients_nhwc**3
        denominator = 2.0 * grad_squared + (
            activations_nhwc * grad_cubed
        ).sum(dim=(0, 1), keepdim=True)
        alpha = grad_squared / denominator
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        weights = (alpha * torch.relu(gradients_nhwc)).sum(dim=-1)
        cam = torch.relu(
            (activations_nhwc * weights.unsqueeze(-1)).sum(dim=-1)
        )

        cam_min = cam.amin()
        cam_max = cam.amax()
        value_range = cam_max - cam_min
        if torch.isfinite(value_range) and value_range > CAM_EPSILON:
            normalized_cam = (cam - cam_min) / value_range
        else:
            normalized_cam = torch.zeros_like(cam)
        normalized_cam = functional.interpolate(
            normalized_cam[None, None],
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0].clamp_(0.0, 1.0)

        mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=InterpolationMode.NEAREST
                ),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.PILToTensor(),
            ]
        )
        transformed_mask = Image.fromarray(
            final_mask.astype(np.uint8) * 255, mode="L"
        )
        leaf_mask = mask_transform(transformed_mask)[0] > 0
        cam_sum = normalized_cam.sum()
        cam_inside_leaf = (
            float(normalized_cam[leaf_mask].sum().div(cam_sum).item())
            if bool(leaf_mask.any()) and float(cam_sum.item()) > CAM_EPSILON
            else None
        )

        return GradCAMResult(
            target_class_id=target_class_id,
            target_class_name=self.class_mapping[target_class_id],
            confidence=confidence,
            activation_resolution=(
                int(activation.shape[1]),
                int(activation.shape[2]),
            ),
            normalized_cam=normalized_cam.detach().cpu().numpy(),
            leaf_coverage=float(final_mask.mean()),
            cam_inside_leaf=cam_inside_leaf,
        )
