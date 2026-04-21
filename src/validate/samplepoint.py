from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.models import SamplePoint
from src.ontology.catalog import Ontology


def load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_sample_point(point: SamplePoint, ontology: Ontology, schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    payload = point.to_dict()

    for field_name in schema.get("required", []):
        if field_name not in payload:
            errors.append(f"Missing required field: {field_name}")

    if not ontology.has_record_class(point.record_class):
        errors.append(f"Invalid record_class: {point.record_class}")

    if not ontology.has_category(point.indicator_type):
        errors.append(f"Invalid indicator_type: {point.indicator_type}")

    if point.coordinate_source not in {"reported", "inferred_text", "inferred_map"}:
        errors.append(f"Invalid coordinate_source: {point.coordinate_source}")

    if point.latitude is not None and not (-90 <= point.latitude <= 90):
        errors.append("Latitude out of range")

    if point.longitude is not None and not (-180 <= point.longitude <= 180):
        errors.append("Longitude out of range")

    if point.confidence_score is not None and not (0 <= point.confidence_score <= 1):
        errors.append("Confidence score must be between 0 and 1")

    if not point.source_locator.quote_or_paraphrase and not point.notes:
        errors.append("Provenance note missing: add source quote/paraphrase or explanatory notes")

    return errors


def score_confidence(point: SamplePoint) -> float:
    completeness = 0.0
    completeness += 0.15 if point.latitude is not None and point.longitude is not None else 0.0
    completeness += 0.15 if point.elevation_m is not None else 0.0
    completeness += 0.15 if point.age_ka is not None else 0.0
    completeness += 0.15 if point.source_locator.quote_or_paraphrase else 0.0
    completeness += 0.10 if point.coordinate_source == "reported" else 0.05 if point.coordinate_source == "inferred_text" else 0.02
    completeness += 0.15 if point.record_class == "sea_level_indicator" else 0.08
    completeness += 0.15 if point.indicative_range_m is not None else 0.05
    return round(min(completeness, 1.0), 3)
