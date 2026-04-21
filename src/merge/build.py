from __future__ import annotations

import csv
import json
from pathlib import Path

from src.common.io import write_csv, write_geojson
from src.common.models import AgeModel, DerivedObservations, ReportedObservations, SamplePoint, SourceLocator


def read_per_source_csvs(directory: Path) -> list[SamplePoint]:
    points: list[SamplePoint] = []
    for csv_path in sorted(directory.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                points.append(
                    SamplePoint(
                        id=row["id"],
                        source_id=row["source_id"],
                        record_class=row["record_class"],
                        site_name=row["site_name"],
                        sample_id=row["sample_id"],
                        latitude=_coerce_float(row["latitude"]),
                        longitude=_coerce_float(row["longitude"]),
                        coordinate_source=row["coordinate_source"],
                        coordinate_uncertainty_m=_coerce_float(row["coordinate_uncertainty_m"]),
                        elevation_m=_coerce_json_scalar(row["elevation_m"]),
                        elevation_reference=row["elevation_reference"],
                        depth_source=row["depth_source"],
                        indicator_type=row["indicator_type"],
                        indicator_subtype=row["indicator_subtype"],
                        indicative_range_m=json.loads(row["indicative_range_m"]),
                        age_ka=_coerce_json_scalar(row["age_ka"]),
                        dating_method=row["dating_method"],
                        description=row["description"],
                        location_name=row["location_name"],
                        bibliographic_reference=row["bibliographic_reference"],
                        doi_or_url=row["doi_or_url"],
                        confidence_score=_coerce_float(row["confidence_score"]),
                        notes=row["notes"],
                        source_locator=SourceLocator(**json.loads(row["source_locator"])),
                        reported_observations=ReportedObservations(**json.loads(row["reported_observations"])),
                        derived_observations=DerivedObservations(**json.loads(row["derived_observations"])),
                        age_models=[AgeModel(**item) for item in json.loads(row["age_models"])],
                    )
                )
    return points


def deduplicate(points: list[SamplePoint]) -> list[SamplePoint]:
    unique: dict[str, SamplePoint] = {}
    for point in points:
        unique[point.id] = point
    return list(unique.values())


def build_master_outputs(per_source_dir: Path, merged_dir: Path) -> tuple[Path, Path, int]:
    points = deduplicate(read_per_source_csvs(per_source_dir))
    csv_path = merged_dir / "master_dataset.csv"
    geojson_path = merged_dir / "master_dataset.geojson"
    write_csv(csv_path, points)
    write_geojson(geojson_path, points)
    return csv_path, geojson_path, len(points)


def _coerce_float(value: str) -> float | None:
    if value in {"", "null", "None"}:
        return None
    return float(value)


def _coerce_json_scalar(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
