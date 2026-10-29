from __future__ import annotations

from typing import Any, Mapping


def build_grounded_context(
    *,
    predicted_class: str,
    confidence: float,
    top_k_predictions: list[Mapping[str, Any]] | None,
    kb_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a context containing only the exact result supplied by KB lookup."""

    status = kb_result.get("status")
    record = kb_result.get("record")
    try:
        confidence_value = float(confidence)
        confidence_threshold = float(kb_result.get("confidence_threshold", 1.0))
    except (TypeError, ValueError):
        confidence_value = -1.0
        confidence_threshold = 1.0
    grounded_top_k = kb_result.get("top_k_predictions", [])
    if not isinstance(grounded_top_k, list):
        grounded_top_k = []

    if (
        confidence_value < 0.0
        or confidence_value > 1.0
        or confidence_value < confidence_threshold
    ):
        return {
            "status": "uncertain",
            "uncertain": True,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_k_predictions": grounded_top_k,
            "message": "Prediction confidence is below the configured threshold or invalid; verify the diagnosis.",
            "knowledge": None,
        }

    if status != "grounded" or not isinstance(record, Mapping):
        return {
            "status": "uncertain" if status == "uncertain" else "unavailable",
            "uncertain": True,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_k_predictions": grounded_top_k,
            "message": str(
                kb_result.get(
                    "message",
                    "Knowledge-base grounding is unavailable; do not provide disease advice.",
                )
            ),
            "knowledge": None,
        }

    if record.get("class_name") != predicted_class:
        return {
            "status": "unavailable",
            "uncertain": True,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_k_predictions": grounded_top_k,
            "message": "Prediction and knowledge-base record do not match; disease advice is withheld.",
            "knowledge": None,
        }

    is_healthy = record.get("scope_status") == "healthy_class"
    is_viral = record.get("causal_agent_type") == "virus"
    is_spider_mite = record.get("causal_agent_type") == "arthropod_pest"
    grounded_record = dict(record)
    if is_healthy:
        grounded_record["safety_notes"] = list(
            dict.fromkeys(
                [
                    *grounded_record.get("safety_notes", []),
                    "A healthy-class prediction is not confirmation that the plant has no disease.",
                ]
            )
        )
        grounded_record["chemical_management"] = [
            "No treatment recommendation is made from a healthy-class prediction."
        ]
    if is_viral:
        grounded_record["curative_chemical_available"] = False
        grounded_record["chemical_management"] = [
            "No curative chemical treatment claim is permitted for this viral class."
        ]
        grounded_record["safety_notes"] = list(
            dict.fromkeys(
                [
                    *grounded_record.get("safety_notes", []),
                    "Viral diseases must not be described as chemically curable.",
                ]
            )
        )
    if is_spider_mite:
        grounded_record["safety_notes"] = list(
            dict.fromkeys(
                [
                    *grounded_record.get("safety_notes", []),
                    "Treat this as an arthropod pest/IPM case, not a pathogen infection.",
                ]
            )
        )

    return {
        "status": "grounded",
        "uncertain": False,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_k_predictions": grounded_top_k,
        "message": str(kb_result.get("message", "Grounded to the exact KB record.")),
        "knowledge": grounded_record,
    }


grounded_context = build_grounded_context
