from __future__ import annotations

import csv
import json
from pathlib import Path

from src.common.io import CSV_COLUMNS
from src.common.models import AgeModel, DerivedObservations, ReportedObservations, SamplePoint, SourceLocator
from src.extract.base import BaseExtractor, ExtractionResult
from src.extract.document_loader import load_document_payload
from src.extract.interpreter import interpret_document


class CsvExtractor(BaseExtractor):
    supported_suffixes = (".csv",)

    def extract(self, source_path: Path, source_id: str) -> ExtractionResult:
        with source_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        if reader.fieldnames and set(CSV_COLUMNS).issubset(set(reader.fieldnames)):
            points = [self._row_to_sample_point(row) for row in rows]
            summary = [
                f"# Summary for {source_path.name}",
                "",
                f"- Source ID: `{source_id}`",
                "- File type: `.csv`",
                f"- Row count: `{len(rows)}`",
                f"- Structured SamplePoints extracted: `{len(points)}`",
                "- Extraction mode: direct canonical CSV import.",
            ]
            return ExtractionResult(source_id=source_id, summary_lines=summary, sample_points=points, unresolved_lines=[])

        payload = load_document_payload(source_path)
        return interpret_document(source_path, source_id, payload)

    def _row_to_sample_point(self, row: dict[str, str]) -> SamplePoint:
        return SamplePoint(
            id=row["id"],
            source_id=row["source_id"],
            record_class=row["record_class"],
            site_name=row["site_name"],
            sample_id=row["sample_id"],
            latitude=float(row["latitude"]) if row["latitude"] else None,
            longitude=float(row["longitude"]) if row["longitude"] else None,
            coordinate_source=row["coordinate_source"],
            coordinate_uncertainty_m=float(row["coordinate_uncertainty_m"]) if row["coordinate_uncertainty_m"] else None,
            elevation_m=row["elevation_m"] or None,
            elevation_reference=row["elevation_reference"],
            depth_source=row["depth_source"],
            indicator_type=row["indicator_type"],
            indicator_subtype=row["indicator_subtype"],
            indicative_range_m=None,
            age_ka=row["age_ka"] or None,
            dating_method=row["dating_method"],
            description=row["description"],
            location_name=row["location_name"],
            bibliographic_reference=row["bibliographic_reference"],
            doi_or_url=row["doi_or_url"],
            confidence_score=float(row["confidence_score"]) if row["confidence_score"] else None,
            notes=row["notes"],
            source_locator=SourceLocator(**_json_or_default(row["source_locator"], {"quote_or_paraphrase": ""})),
            reported_observations=ReportedObservations(**_json_or_default(row["reported_observations"], {})),
            derived_observations=DerivedObservations(**_json_or_default(row["derived_observations"], {})),
            age_models=[AgeModel(**item) for item in _json_or_default(row["age_models"], [])],
        )


def _json_or_default(value: str, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default
