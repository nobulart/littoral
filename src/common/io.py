from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from src.common.models import SamplePoint


CSV_COLUMNS = [
    "id",
    "source_id",
    "record_class",
    "site_name",
    "sample_id",
    "latitude",
    "longitude",
    "coordinate_source",
    "coordinate_uncertainty_m",
    "elevation_m",
    "elevation_reference",
    "depth_source",
    "indicator_type",
    "indicator_subtype",
    "indicative_range_m",
    "age_ka",
    "dating_method",
    "description",
    "location_name",
    "bibliographic_reference",
    "doi_or_url",
    "confidence_score",
    "notes",
    "source_locator",
    "reported_observations",
    "derived_observations",
    "age_models",
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_summary(path: Path, lines: list[str]) -> None:
    ensure_parent(path)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, sample_points: Iterable[SamplePoint]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for point in sample_points:
            writer.writerow(sample_point_csv_row(point))


def append_csv(path: Path, sample_points: Iterable[SamplePoint]) -> None:
    ensure_parent(path)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for point in sample_points:
            writer.writerow(sample_point_csv_row(point))


def sample_point_csv_row(point: SamplePoint) -> dict:
    row = point.to_dict()
    row["elevation_m"] = json.dumps(row["elevation_m"])
    row["indicative_range_m"] = json.dumps(row["indicative_range_m"])
    row["age_ka"] = json.dumps(row["age_ka"])
    row["source_locator"] = json.dumps(row["source_locator"])
    row["reported_observations"] = json.dumps(row["reported_observations"])
    row["derived_observations"] = json.dumps(row["derived_observations"])
    row["age_models"] = json.dumps(row["age_models"])
    return row


def write_geojson(path: Path, sample_points: Iterable[SamplePoint]) -> None:
    ensure_parent(path)
    features = []
    for point in sample_points:
        feature = {
            "type": "Feature",
            "geometry": None,
            "properties": point.to_dict(),
        }
        if point.latitude is not None and point.longitude is not None:
            feature["geometry"] = {
                "type": "Point",
                "coordinates": [point.longitude, point.latitude],
            }
        features.append(feature)
    payload = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_log(path: Path, line: str) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")
