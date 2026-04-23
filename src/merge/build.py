from __future__ import annotations

import csv
import json
from dataclasses import fields
import os
from pathlib import Path
import tempfile

from src.common.io import CSV_COLUMNS, sample_point_csv_row, write_geojson
from src.common.models import AgeModel, DerivedObservations, ReportedObservations, SamplePoint, SourceLocator


MERGE_MODES = {"append", "overwrite", "skip"}


def read_per_source_csvs(directory: Path) -> list[SamplePoint]:
    points: list[SamplePoint] = []
    for csv_path in sorted(directory.glob("*.csv")):
        points.extend(read_sample_points_csv(csv_path))
    return points


def read_sample_points_csv(csv_path: Path) -> list[SamplePoint]:
    points: list[SamplePoint] = []
    if not csv_path.exists():
        return points
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append(_row_to_sample_point(row))
    return points


def deduplicate(points: list[SamplePoint]) -> list[SamplePoint]:
    unique: dict[str, SamplePoint] = {}
    for point in points:
        existing = unique.get(point.id)
        unique[point.id] = _preferred_point(existing, point) if existing is not None else point
    return list(unique.values())


def build_master_outputs(per_source_dir: Path, merged_dir: Path, mode: str = "append") -> tuple[Path, Path, int]:
    if mode not in MERGE_MODES:
        raise ValueError(f"Unsupported merge mode `{mode}`. Expected one of: {', '.join(sorted(MERGE_MODES))}")
    csv_path = merged_dir / "master_dataset.csv"
    geojson_path = merged_dir / "master_dataset.geojson"

    if mode == "skip" and (csv_path.exists() or geojson_path.exists()):
        return csv_path, geojson_path, len(read_sample_points_csv(csv_path))

    points = deduplicate(read_per_source_csvs(per_source_dir))
    if mode == "overwrite":
        _write_master_csv(csv_path, points)
        write_geojson(geojson_path, points)
        return csv_path, geojson_path, len(points)

    existing_points = read_sample_points_csv(csv_path)
    existing_ids = {point.id for point in existing_points}
    new_points = [point for point in points if point.id not in existing_ids]

    combined_points = deduplicate(existing_points + new_points)
    _write_master_csv(csv_path, combined_points)
    if not geojson_path.exists() or new_points:
        write_geojson(geojson_path, combined_points)
    return csv_path, geojson_path, len(combined_points)


def _write_master_csv(csv_path: Path, points: list[SamplePoint]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = CSV_COLUMNS
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=csv_path.parent,
        prefix=f"{csv_path.name}.tmp-",
        suffix=".part",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(sample_point_csv_row(point))
        temp_path = Path(handle.name)
    os.replace(temp_path, csv_path)


def _row_to_sample_point(row: dict[str, str]) -> SamplePoint:
    return SamplePoint(
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


def _coerce_float(value: str) -> float | None:
    if value in {"", "null", "None"}:
        return None
    return float(value)


def _coerce_json_scalar(value: str):
    if value in {"", "null", "None"}:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _preferred_point(existing: SamplePoint, candidate: SamplePoint) -> SamplePoint:
    preferred = candidate
    alternate = existing

    existing_has_top_elevation = _is_present(existing.elevation_m)
    candidate_has_top_elevation = _is_present(candidate.elevation_m)
    if existing_has_top_elevation != candidate_has_top_elevation:
        preferred = candidate if candidate_has_top_elevation else existing
        alternate = existing if candidate_has_top_elevation else candidate
        return _fill_missing_fields(preferred, alternate)

    existing_has_elevation = _has_any_elevation(existing)
    candidate_has_elevation = _has_any_elevation(candidate)
    if existing_has_elevation != candidate_has_elevation:
        preferred = candidate if candidate_has_elevation else existing
        alternate = existing if candidate_has_elevation else candidate
        return _fill_missing_fields(preferred, alternate)

    existing_score = _point_completeness(existing)
    candidate_score = _point_completeness(candidate)
    if candidate_score > existing_score:
        preferred = candidate
        alternate = existing
    elif existing_score > candidate_score:
        preferred = existing
        alternate = candidate
    return _fill_missing_fields(preferred, alternate)


def _has_any_elevation(point: SamplePoint) -> bool:
    return any(
        _is_present(value)
        for value in (
            point.elevation_m,
            point.reported_observations.reported_elevation_m,
            point.derived_observations.derived_elevation_m,
        )
    )


def _point_completeness(point: SamplePoint) -> int:
    score = 0
    fields = (
        point.latitude,
        point.longitude,
        point.coordinate_uncertainty_m,
        point.elevation_m,
        point.age_ka,
        point.indicative_range_m,
        point.confidence_score,
        point.reported_observations.reported_elevation_m,
        point.reported_observations.reported_depth_m,
        point.reported_observations.reported_datum,
        point.derived_observations.derived_elevation_m,
        point.derived_observations.derived_depth_m,
    )
    for value in fields:
        if _is_present(value):
            score += 1
    return score


def _is_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized not in {"", "null", "none"}
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def _fill_missing_fields(preferred: SamplePoint, alternate: SamplePoint) -> SamplePoint:
    fill_attrs = (
        "latitude",
        "longitude",
        "coordinate_uncertainty_m",
        "elevation_m",
        "elevation_reference",
        "depth_source",
        "indicative_range_m",
        "age_ka",
        "dating_method",
        "description",
        "location_name",
        "bibliographic_reference",
        "doi_or_url",
        "confidence_score",
        "notes",
    )
    for attr in fill_attrs:
        preferred_value = getattr(preferred, attr)
        alternate_value = getattr(alternate, attr)
        if not _is_present(preferred_value) and _is_present(alternate_value):
            setattr(preferred, attr, alternate_value)

    _fill_missing_nested(preferred.reported_observations, alternate.reported_observations)
    _fill_missing_nested(preferred.derived_observations, alternate.derived_observations)
    if not preferred.age_models and alternate.age_models:
        preferred.age_models = alternate.age_models
    return preferred


def _fill_missing_nested(preferred: object, alternate: object) -> None:
    for field in fields(preferred):
        preferred_value = getattr(preferred, field.name)
        alternate_value = getattr(alternate, field.name)
        if not _is_present(preferred_value) and _is_present(alternate_value):
            setattr(preferred, field.name, alternate_value)
