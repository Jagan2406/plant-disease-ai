from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

DEFAULT_MODEL = "gemini-2.5-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

_GROUNDED_FIELDS = (
    "transmission",
    "symptoms",
    "favorable_conditions",
    "cultural_management",
    "biological_management",
    "chemical_management",
    "differential_diagnosis",
    "safety_notes",
)
_DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|mL|l|L|oz|lb|ppm|ppb|"
    r"teaspoons?|tablespoons?|tsp|tbsp|percent|%)\b",
    re.IGNORECASE,
)
_TANK_MIX_PATTERN = re.compile(r"\btank[\s-]+mix(?:es|ing)?\b", re.IGNORECASE)
_GUARANTEED_CURE_PATTERN = re.compile(
    r"\b(?:guaranteed?|will\s+cure|cures?\s+all|eradicates?\s+completely)\b",
    re.IGNORECASE,
)
_PRODUCT_NAME_PATTERN = re.compile(
    r"\b(?:brand|product)\s+name\b|\b(?:buy|purchase|apply)\s+[A-Z][a-zA-Z0-9-]+",
    re.IGNORECASE,
)
_CAM_CAUSALITY_PATTERN = re.compile(
    r"\b(?:grad[\s-]?cam|heatmap|activation map)\b.{0,50}"
    r"\b(?:proves?|confirms?|diagnoses?|causes?)\b|"
    r"\b(?:proves?|confirms?|diagnoses?|causes?)\b.{0,50}"
    r"\b(?:grad[\s-]?cam|heatmap|activation map)\b",
    re.IGNORECASE,
)


def _unavailable(model: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "response": None,
        "model": model,
        "grounded": False,
    }


def _error(model: str) -> dict[str, Any]:
    return {
        "status": "error",
        "response": None,
        "model": model,
        "grounded": False,
    }


def _allowed_statements(record: Mapping[str, Any]) -> list[str]:
    statements: list[str] = []
    for field in _GROUNDED_FIELDS:
        values = record.get(field, [])
        if isinstance(values, list):
            statements.extend(value for value in values if isinstance(value, str))
    return list(dict.fromkeys(statements))


def _validate_statements(
    statements: Any,
    allowed_statements: list[str],
    record: Mapping[str, Any],
) -> str | None:
    if not isinstance(statements, list) or not statements:
        return None
    if any(
        not isinstance(statement, str)
        or statement not in allowed_statements
        for statement in statements
    ):
        return None

    response = "\n".join(statements)
    if (
        _DOSAGE_PATTERN.search(response)
        or _TANK_MIX_PATTERN.search(response)
        or _GUARANTEED_CURE_PATTERN.search(response)
        or _PRODUCT_NAME_PATTERN.search(response)
        or _CAM_CAUSALITY_PATTERN.search(response)
    ):
        return None
    if (
        record.get("causal_agent_type") == "virus"
        and re.search(
            r"\b(?:curative|cure|cures|cured)\b.{0,80}\bchemical\b|"
            r"\bchemical\b.{0,80}\b(?:cure|cures|cured)\b",
            response,
            re.IGNORECASE,
        )
    ):
        return None
    if (
        record.get("scope_status") == "healthy_class"
        and re.search(
            r"\b(?:disease[- ]free|free of disease|has no disease)\b",
            response,
            re.IGNORECASE,
        )
    ):
        return None
    return response


def generate_grounded_response(
    grounded_context: Mapping[str, Any],
) -> dict[str, Any]:
    model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _unavailable(model)

    if (
        grounded_context.get("status") != "grounded"
        or grounded_context.get("uncertain") is not False
    ):
        return _unavailable(model)
    record = grounded_context.get("knowledge")
    if not isinstance(record, Mapping):
        return _unavailable(model)

    allowed_statements = _allowed_statements(record)
    if not allowed_statements:
        return _unavailable(model)

    model_input = {
        "predicted_class": grounded_context.get("predicted_class"),
        "confidence": grounded_context.get("confidence"),
        "knowledge_record": record,
        "allowed_statements": allowed_statements,
    }
    instruction = (
        "Select only relevant exact strings from allowed_statements. "
        "Return JSON only in the shape {\"statements\": [\"...\"]}. "
        "Every statement must be copied character-for-character from "
        "allowed_statements. Do not add or paraphrase any claim. "
        "Do not provide pesticide dosages, product names, tank mixes, "
        "guaranteed cures, or claims that Grad-CAM proves causality or "
        "diagnosis. Follow all safety notes in the knowledge record."
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": instruction
                        + "\nGrounded context:\n"
                        + json.dumps(model_input, ensure_ascii=False)
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    url = f"{GEMINI_API_URL}/{quote(model, safe='')}:generateContent"
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        candidates = response_payload.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", [])
        response_text = "".join(
            part.get("text", "") for part in parts if isinstance(part, dict)
        )
        parsed = json.loads(response_text)
        validated_response = _validate_statements(
            parsed.get("statements") if isinstance(parsed, dict) else None,
            allowed_statements,
            record,
        )
    except HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            error_body = str(exc)

        return {
            "status": "error",
            "response": None,
            "model": model,
            "grounded": False,
            "error": f"HTTP {exc.code}: {error_body}",
        }

    except (URLError, TimeoutError, OSError, ValueError, KeyError, IndexError) as exc:
        return {
            "status": "error",
            "response": None,
            "model": model,
            "grounded": False,
            "error": str(exc),
        }

    if validated_response is None:
        return {
            "status": "blocked",
            "response": None,
            "model": model,
            "grounded": False,
        }
    return {
        "status": "success",
        "response": validated_response,
        "model": model,
        "grounded": True,
    }
