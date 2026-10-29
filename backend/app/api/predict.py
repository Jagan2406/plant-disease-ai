from __future__ import annotations

import base64
import io
import logging
import threading
import time
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFilter, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from app.services.gradcam_service import GradCAMPlusPlusService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["prediction"]
)


class InvalidInputError(Exception):
    pass


class ClassPrediction(BaseModel):
    class_id: int
    class_name: str
    confidence: float


class GradCAMResponse(BaseModel):
    target_class_id: int
    target_class_name: str
    confidence: float
    activation_resolution: tuple[int, int]
    cam_shape: tuple[int, int]
    cam_min: float
    cam_max: float
    leaf_coverage: float
    cam_inside_leaf: float | None


class KnowledgeResponse(BaseModel):
    status: str
    class_id: int
    class_name: str
    confidence: float
    advice_available: bool
    record: dict[str, Any] | None


class GroundedContextResponse(BaseModel):
    status: str
    uncertain: bool
    knowledge: dict[str, Any] | None
    top_k_predictions: list[ClassPrediction]


class SegmentationResponse(BaseModel):
    mask_area_ratio: float
    fallback_used: bool
    candidate_count: int
    selected_candidate_score: float | None


class AnalysisImages(BaseModel):
    original: str
    sam2_mask: str
    leaf_only: str
    gradcam_pp: str


class DiseaseInformation(BaseModel):
    name: str
    description: str | None
    symptoms: list[str]
    cause: str | None
    treatment: list[str]
    management: list[str]
    prevention: list[str]
    warnings: list[str]


class KnowledgeBaseResponse(BaseModel):
    status: str
    source_ids: list[str]


class PipelineTimings(BaseModel):
    sam2: float
    classification: float
    gradcam: float
    knowledge_base: float
    total: float


class PredictionResponse(BaseModel):
    predicted_class: str
    class_id: int
    confidence: float
    top_k_predictions: list[ClassPrediction]
    mask_area_ratio: float
    fallback_used: bool
    segmentation_metadata: dict[str, Any]
    gradcam: GradCAMResponse
    knowledge: KnowledgeResponse
    grounded_context: GroundedContextResponse
    analysis_images: AnalysisImages
    segmentation: SegmentationResponse
    disease_information: DiseaseInformation
    knowledge_base: KnowledgeBaseResponse
    timings: PipelineTimings


def _browser_image(image: Image.Image, *, jpeg: bool = False) -> str:
    image = ImageOps.contain(image.convert("RGB"), (960, 960), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    if jpeg:
        image.save(buffer, format="JPEG", quality=88, optimize=True, progressive=True)
        media_type = "image/jpeg"
    else:
        image.save(buffer, format="PNG", optimize=True)
        media_type = "image/png"
    return f"data:{media_type};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _mask_visualization(original: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(original.convert("RGB"), dtype=np.uint8)
    mask = np.asarray(mask, dtype=bool)
    tint = np.array(rgb, dtype=np.float32)
    tint[mask] = tint[mask] * 0.38 + np.array((0, 178, 104), dtype=np.float32) * 0.62
    visual = Image.fromarray(np.clip(tint, 0, 255).astype(np.uint8), mode="RGB")
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    boundary = mask_image.filter(ImageFilter.MaxFilter(3))
    inner = mask_image.filter(ImageFilter.MinFilter(3))
    boundary_array = np.asarray(boundary, dtype=np.int16) - np.asarray(inner, dtype=np.int16)
    draw = ImageDraw.Draw(visual)
    draw.point([(x, y) for y in range(boundary_array.shape[0]) for x in range(boundary_array.shape[1]) if boundary_array[y, x] > 0], fill=(255, 255, 255))
    return visual


def _leaf_only(original: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(original.convert("RGB"), dtype=np.uint8)
    masked = np.zeros_like(rgb)
    masked[np.asarray(mask, dtype=bool)] = rgb[np.asarray(mask, dtype=bool)]
    return Image.fromarray(masked, mode="RGB")


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]


def _disease_information(predicted_class: str, record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {
            "name": predicted_class,
            "description": None,
            "symptoms": [],
            "cause": None,
            "treatment": [],
            "management": [],
            "prevention": [],
            "warnings": [],
        }
    cultural = _string_list(record.get("cultural_management"))
    biological = _string_list(record.get("biological_management"))
    chemical = _string_list(record.get("chemical_management"))
    return {
        "name": str(record.get("class_name", predicted_class)),
        "description": record.get("description") if isinstance(record.get("description"), str) else None,
        "symptoms": _string_list(record.get("symptoms")),
        "cause": str(record["scientific_name"]).strip() if isinstance(record.get("scientific_name"), str) and record.get("scientific_name") else None,
        "treatment": chemical,
        "management": cultural + biological,
        "prevention": cultural + _string_list(record.get("transmission")),
        "warnings": _string_list(record.get("safety_notes")),
    }


def _gradcam_visualization(original: Image.Image, cam: np.ndarray) -> Image.Image:
    cam = np.asarray(cam, dtype=np.float32)
    cam = np.maximum(cam, 0)
    maximum = float(cam.max())
    if maximum > 0:
        cam /= maximum
    cam_image = Image.fromarray(np.uint8(cam * 255), mode="L").resize(original.size, Image.Resampling.BILINEAR)
    heatmap = ImageOps.colorize(cam_image, black=(15, 18, 24), white=(255, 63, 52))
    # Tint the image to preserve visual structure while retaining the real activation field.
    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    heat = np.asarray(heatmap.convert("RGB"), dtype=np.float32)
    weight = np.asarray(cam_image, dtype=np.float32)[..., None] / 255.0
    blended = np.clip(base * (1.0 - 0.55 * weight) + heat * (0.55 * weight), 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def _is_plausible_leaf(segmentation: Any, image: Image.Image) -> bool:
    mask = np.asarray(segmentation.final_mask, dtype=bool)
    area_ratio = float(mask.mean()) if mask.size else 0.0
    if segmentation.fallback_used or not 0.03 <= area_ratio <= 0.90:
        return False

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != mask.shape or not mask.any():
        return False
    red, green, blue = rgb.transpose(2, 0, 1).astype(np.int16)
    vegetation = (green > red * 1.05) & (green > blue * 1.05) & (green > 35)
    green_fraction = float(vegetation[mask].mean()) if mask.any() else 0.0

    try:
        from scipy import ndimage
        labeled_mask, component_count = ndimage.label(mask)
        if component_count == 0:
            largest_component_ratio = 0.0
        else:
            component_sizes = np.bincount(labeled_mask.ravel())[1:]
            largest_component_ratio = float(component_sizes.max() / mask.sum())
    except Exception:
        largest_component_ratio = 1.0

    quality_scores = []
    for candidate in segmentation.candidate_metadata:
        if not isinstance(candidate, dict):
            continue
        try:
            candidate_quality = float(candidate["selection_score"])
            if np.isfinite(candidate_quality):
                quality_scores.append(candidate_quality)
        except (KeyError, TypeError, ValueError):
            continue
    mask_quality = max(quality_scores, default=segmentation.selected_candidate_score or 0.0)
    if segmentation.selected_candidate_score is not None:
        mask_quality = float(segmentation.selected_candidate_score)

    return green_fraction >= 0.08 and largest_component_ratio >= 0.25 and mask_quality >= 0.20


def _predict_with_gradcam(
    predictor: Any,
    gradcam_service: GradCAMPlusPlusService,
    image: Image.Image,
    top_k: int,
    inference_lock: threading.RLock,
) -> dict[str, Any]:
    pipeline_started = time.perf_counter()
    with inference_lock:
        sam2_started = time.perf_counter()
        segmentation = predictor.segmenter.segment_image(image)
        mask = np.asarray(segmentation.final_mask, dtype=bool)
        sam2_elapsed = time.perf_counter() - sam2_started
        if not _is_plausible_leaf(segmentation, image):
            raise InvalidInputError()
        segmented_image = Image.fromarray(
            np.asarray(segmentation.segmented_image, dtype=np.uint8),
            mode="RGB",
        )
        classification_started = time.perf_counter()
        input_tensor = predictor.image_transform(segmented_image).unsqueeze(0)

        with torch.inference_mode():
            probabilities = torch.softmax(
                predictor.classifier(input_tensor).float(), dim=1
            )[0]

        count = min(top_k, probabilities.numel())
        scores, indices = torch.topk(probabilities, k=count)
        predictions = [
            {
                "class_id": int(class_id),
                "class_name": predictor.class_mapping[int(class_id)],
                "confidence": float(score),
            }
            for score, class_id in zip(scores, indices)
        ]
        predicted = predictions[0]
        classification_elapsed = time.perf_counter() - classification_started

        from ai.explainability.grounded_context import build_grounded_context
        from ai.knowledge.kb_lookup import lookup_disease

        kb_started = time.perf_counter()
        try:
            kb_result = lookup_disease(
                class_id=predicted["class_id"],
                class_name=predicted["class_name"],
                confidence=predicted["confidence"],
                top_k_predictions=predictions,
            )
        except Exception:
            logger.exception("Disease knowledge-base lookup failed.")
            kb_result = {
                "status": "unavailable",
                "record": None,
                "advice_available": False,
                "confidence_threshold": 1.0,
                "top_k_predictions": [],
                "message": "Knowledge-base lookup failed; disease guidance is withheld.",
            }
        kb_elapsed = time.perf_counter() - kb_started

        grounded_result = build_grounded_context(
            predicted_class=predicted["class_name"],
            confidence=predicted["confidence"],
            top_k_predictions=predictions,
            kb_result=kb_result,
        )
        gradcam_started = time.perf_counter()
        gradcam_result = gradcam_service.generate(image, segmentation)
        cam = gradcam_result.normalized_cam
        gradcam_elapsed = time.perf_counter() - gradcam_started

        total_elapsed = time.perf_counter() - pipeline_started
        logger.warning(
            "[PIPELINE] SAM2: %.4fs | classification: %.4fs | Grad-CAM++: %.4fs | "
            "KB lookup: %.4fs | total: %.4fs",
            sam2_elapsed,
            classification_elapsed,
            gradcam_elapsed,
            kb_elapsed,
            total_elapsed,
        )

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
            "gradcam": {
                "target_class_id": gradcam_result.target_class_id,
                "target_class_name": gradcam_result.target_class_name,
                "confidence": gradcam_result.confidence,
                "activation_resolution": list(
                    gradcam_result.activation_resolution
                ),
                "cam_shape": list(cam.shape),
                "cam_min": float(np.min(cam)),
                "cam_max": float(np.max(cam)),
                "leaf_coverage": gradcam_result.leaf_coverage,
                "cam_inside_leaf": gradcam_result.cam_inside_leaf,
            },
            "knowledge": {
                "status": kb_result["status"],
                "class_id": predicted["class_id"],
                "class_name": predicted["class_name"],
                "confidence": predicted["confidence"],
                "advice_available": bool(kb_result.get("advice_available", False)),
                "record": kb_result.get("record"),
            },
            "grounded_context": {
                "status": grounded_result["status"],
                "uncertain": bool(grounded_result["uncertain"]),
                "knowledge": grounded_result.get("knowledge"),
                "top_k_predictions": grounded_result["top_k_predictions"],
            },
            "segmentation": {
                "mask_area_ratio": float(mask.mean()),
                "fallback_used": bool(segmentation.fallback_used),
                "candidate_count": len(segmentation.candidate_metadata),
                "selected_candidate_score": segmentation.selected_candidate_score,
            },
            "disease_information": _disease_information(
                predicted["class_name"], grounded_result.get("knowledge")
            ),
            "knowledge_base": {
                "status": kb_result["status"],
                "source_ids": _string_list(
                    grounded_result.get("knowledge", {}).get("source_ids")
                    if isinstance(grounded_result.get("knowledge"), dict)
                    else []
                ),
            },
            "timings": {
                "sam2": sam2_elapsed,
                "classification": classification_elapsed,
                "gradcam": gradcam_elapsed,
                "knowledge_base": kb_elapsed,
                "total": total_elapsed,
            },
            "analysis_images": {
                "original": _browser_image(image, jpeg=True),
                "sam2_mask": _browser_image(_mask_visualization(image, mask)),
                "leaf_only": _browser_image(_leaf_only(image, mask)),
                "gradcam_pp": _browser_image(_gradcam_visualization(image, cam)),
            },
        }


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=100),
) -> PredictionResponse:
    contents = await file.read()
    try:
        with Image.open(io.BytesIO(contents)) as uploaded_image:
            uploaded_image.verify()
        with Image.open(io.BytesIO(contents)) as uploaded_image:
            image = uploaded_image.convert("RGB")
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must contain a valid image.",
        ) from error

    predictor = getattr(request.app.state, "swin_sam2_predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is not available.",
        )

    gradcam_service = getattr(request.app.state, "gradcam_service", None)
    if gradcam_service is None:
        gradcam_service = GradCAMPlusPlusService(
            model=predictor.classifier,
            class_mapping=predictor.class_mapping,
            image_transform=predictor.image_transform,
        )
        request.app.state.gradcam_service = gradcam_service
    inference_lock = getattr(request.app.state, "prediction_inference_lock", None)
    if inference_lock is None:
        inference_lock = threading.RLock()
        request.app.state.prediction_inference_lock = inference_lock

    try:
        result = await run_in_threadpool(
            _predict_with_gradcam,
            predictor,
            gradcam_service,
            image,
            top_k,
            inference_lock,
        )
    except InvalidInputError:
        return JSONResponse(
            status_code=422,
            content={
                "error": "INVALID_INPUT",
                "message": "Please upload a clear image of a plant leaf. The uploaded image does not appear to contain a detectable leaf.",
            },
        )
    except Exception:
        logger.exception("Swin-SAM2 prediction and Grad-CAM++ failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again later.",
        ) from None

    return PredictionResponse(**result)
