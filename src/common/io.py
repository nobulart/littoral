from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
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
    "elevation_min",
    "elevation_max",
    "elevation_avg",
    "elevation_mean",
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text_atomic(path: Path, content: str, encoding: str = "utf-8") -> None:
    ensure_parent(path)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding=encoding,
        dir=path.parent,
        prefix=f"{path.name}.tmp-",
        suffix=".part",
        delete=False,
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def write_json_atomic(path: Path, payload: object) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_summary(path: Path, lines: list[str]) -> None:
    write_text_atomic(path, "\n".join(lines).rstrip() + "\n")


def write_csv(path: Path, sample_points: Iterable[SamplePoint]) -> None:
    ensure_parent(path)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f"{path.name}.tmp-",
        suffix=".part",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for point in sample_points:
            writer.writerow(sample_point_csv_row(point))
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


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
    row.update(elevation_derivative_values(point.elevation_m))
    return row


def elevation_derivative_values(value: float | list[float | None] | str | None) -> dict[str, str]:
    numbers = _numeric_elevation_values(value)
    if not numbers:
        return {
            "elevation_min": "",
            "elevation_max": "",
            "elevation_avg": "",
            "elevation_mean": "",
        }
    minimum = min(numbers)
    maximum = max(numbers)
    average = sum(numbers) / len(numbers)
    formatted_average = _format_float(average)
    return {
        "elevation_min": _format_float(minimum),
        "elevation_max": _format_float(maximum),
        "elevation_avg": formatted_average,
        "elevation_mean": formatted_average,
    }


def _numeric_elevation_values(value: float | list[float | None] | str | None) -> list[float]:
    values: list[float] = []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (int, float)):
                values.append(float(item))
            elif isinstance(item, str):
                parsed = _parse_numeric_string(item)
                if parsed is not None:
                    values.append(parsed)
        return values
    if isinstance(value, str):
        parsed = _parse_numeric_string(value)
        return [parsed] if parsed is not None else []
    return values


def _parse_numeric_string(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _format_float(value: float) -> str:
    rounded = round(value, 6)
    return f"{rounded:.6f}".rstrip("0").rstrip(".")


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
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_log(path: Path, line: str) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")
