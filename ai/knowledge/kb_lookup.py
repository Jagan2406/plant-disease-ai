from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

KB_PATH = Path(__file__).with_name("disease_knowledge_base.json")
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


class KnowledgeBaseError(RuntimeError):
    """Base error for knowledge-base loading or validation failures."""


class KnowledgeBaseValidationError(KnowledgeBaseError):
    """Raised when the KB data is malformed or internally inconsistent."""


@lru_cache(maxsize=None)
def _load_knowledge_base(path: Path = KB_PATH) -> dict[str, Any]:
    """Load and validate a knowledge-base document once per file path."""
    try:
        with path.open("r", encoding="utf-8") as source_file:
            payload = json.load(source_file)
    except (OSError, json.JSONDecodeError) as error:
        raise KnowledgeBaseError(f"Unable to load knowledge base: {path}") from error
    if not isinstance(payload, dict):
        raise KnowledgeBaseValidationError("Knowledge base root must be an object.")
    return payload


@lru_cache(maxsize=None)
def _get_index(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    return _index_records(_load_knowledge_base(path))


def _index_records(
    payload: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    records = payload.get("classes")
    if not isinstance(records, list):
        raise KnowledgeBaseValidationError("Knowledge base classes must be a list.")

    by_id: dict[int, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    registry = payload.get("source_registry", {})
    if not isinstance(registry, dict):
        raise KnowledgeBaseValidationError("source_registry must be an object.")

    for record in records:
        if not isinstance(record, dict):
            raise KnowledgeBaseValidationError("Each class record must be an object.")
        try:
            class_id = int(record["class_id"])
            class_name = record["class_name"]
        except (KeyError, TypeError, ValueError) as error:
            raise KnowledgeBaseValidationError(
                "Each class record must have an integer class_id and exact class_name."
            ) from error
        if not isinstance(class_name, str) or not class_name:
            raise KnowledgeBaseValidationError("class_name must be a non-empty string.")
        if class_id in by_id or class_name in by_name:
            raise KnowledgeBaseValidationError(
                "Duplicate class_id or class_name in knowledge base."
            )
        source_ids = record.get("source_ids")
        if not isinstance(source_ids, list) or any(
            source_id not in registry for source_id in source_ids
        ):
            raise KnowledgeBaseValidationError(
                f"Invalid or unregistered source_id in class {class_id}."
            )
        by_id[class_id] = record
        by_name[class_name] = record
    return by_id, by_name


def get_confidence_threshold() -> float:
    raw_threshold = os.environ.get("PLANT_DISEASE_KB_CONFIDENCE_THRESHOLD")
    if raw_threshold is None:
        raw_threshold = _load_knowledge_base().get(
            "confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD
        )
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError) as error:
        raise KnowledgeBaseValidationError(
            "Confidence threshold must be a number between 0 and 1."
        ) from error
    if not 0.0 <= threshold <= 1.0:
        raise KnowledgeBaseValidationError(
            "Confidence threshold must be between 0 and 1."
        )
    return threshold


def _prediction_context(
    top_k_predictions: Iterable[Mapping[str, Any]] | None,
    by_id: Mapping[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    for prediction in top_k_predictions or ():
        try:
            class_id = int(prediction["class_id"])
            class_name = prediction["class_name"]
            confidence = float(prediction["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        record = by_id.get(class_id)
        if (
            record is None
            or class_name != record["class_name"]
            or not 0.0 <= confidence <= 1.0
        ):
            continue
        context.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
            }
        )
    return context


def lookup_disease(
    *,
    confidence: float,
    class_id: int | None = None,
    class_name: str | None = None,
    top_k_predictions: Iterable[Mapping[str, Any]] | None = None,
    confidence_threshold: float | None = None,
    knowledge_base_path: Path = KB_PATH,
) -> dict[str, Any]:
    """Perform exact class lookup and fail closed for uncertain/invalid input."""

    by_id, by_name = _get_index(knowledge_base_path)
    threshold = (
        get_confidence_threshold()
        if confidence_threshold is None
        else float(confidence_threshold)
    )
    if not 0.0 <= threshold <= 1.0:
        raise KnowledgeBaseValidationError(
            "Confidence threshold must be between 0 and 1."
        )

    try:
        prediction_confidence = float(confidence)
    except (TypeError, ValueError):
        prediction_confidence = -1.0
    top_k_context = _prediction_context(top_k_predictions, by_id)
    safe_result = {
        "record": None,
        "top_k_predictions": top_k_context,
        "confidence": prediction_confidence,
        "confidence_threshold": threshold,
    }
    if not 0.0 <= prediction_confidence <= 1.0:
        return {
            **safe_result,
            "status": "unknown",
            "advice_available": False,
            "message": "Prediction confidence is invalid; disease guidance is withheld.",
        }
    if prediction_confidence < threshold:
        return {
            **safe_result,
            "status": "uncertain",
            "advice_available": False,
            "message": "Prediction confidence is below the configured threshold; verify the diagnosis.",
        }
    if class_id is None and class_name is None:
        return {
            **safe_result,
            "status": "unknown",
            "advice_available": False,
            "message": "An exact class_id or class_name is required; disease guidance is withheld.",
        }

    record_by_id = None
    record_by_name = None
    if class_id is not None:
        try:
            record_by_id = by_id.get(int(class_id))
        except (TypeError, ValueError):
            record_by_id = None
    if class_name is not None:
        if isinstance(class_name, str):
            record_by_name = by_name.get(class_name)

    if class_id is not None and class_name is not None:
        if (
            record_by_id is None
            or record_by_name is None
            or record_by_id["class_id"] != record_by_name["class_id"]
        ):
            return {
                **safe_result,
                "status": "unknown",
                "advice_available": False,
                "message": "The supplied class_id and exact class_name do not match; disease guidance is withheld.",
            }
        record = record_by_id
    else:
        record = record_by_id if class_id is not None else record_by_name

    if record is None or record.get("scope_status") == "out_of_scope":
        return {
            **safe_result,
            "status": "unknown",
            "advice_available": False,
            "message": "No exact in-scope knowledge-base record exists; disease guidance is withheld.",
        }

    return {
        **safe_result,
        "status": "grounded",
        "advice_available": True,
        "message": "Information is limited to the exact matched knowledge-base record.",
        "record": record,
    }


lookup_by_class_id = lookup_disease
lookup_by_class_name = lookup_disease
