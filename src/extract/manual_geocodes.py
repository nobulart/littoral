from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


LATITUDE_COLUMNS = ("latitude", "lat", "y")
LONGITUDE_COLUMNS = ("longitude", "lon", "lng", "long", "x")
ELEVATION_COLUMNS = ("elevation_m", "elevation", "altitude_m", "altitude", "height_m", "height")
DEPTH_COLUMNS = ("depth_m", "depth", "water_depth_m", "water_depth")
AGE_COLUMNS = ("age_ka", "age", "ka")
UNCERTAINTY_COLUMNS = ("coordinate_uncertainty_m", "uncertainty_m", "coord_uncertainty_m")
KEY_COLUMNS = (
    "sample_id",
    "site_name",
    "site",
    "site_no",
    "site_number",
    "specimen_no",
    "specimen",
    "lab_no",
    "lab_number",
    "locality",
    "name",
    "id",
)


@dataclass(frozen=True, slots=True)
class ManualGeocodeMatch:
    row: dict[str, str]
    matched_key: str
    latitude: float | None
    longitude: float | None
    elevation_m: float | None
    depth_m: float | None
    age_ka: float | None
    coordinate_uncertainty_m: float | None
    figure: str | None
    description: str | None


@dataclass(frozen=True, slots=True)
class ManualGeocodeTable:
    source_id: str
    path: Path
    rows: tuple[dict[str, str], ...]
    latitude_column: str | None
    longitude_column: str | None

    @property
    def has_rows(self) -> bool:
        return bool(self.rows)

    @property
    def has_coordinate_columns(self) -> bool:
        return self.latitude_column is not None and self.longitude_column is not None

    @property
    def suppresses_fuzzy_geocoding(self) -> bool:
        return self.has_rows

    def match(self, values: list[str | int | float | None]) -> ManualGeocodeMatch | None:
        wanted: set[str] = set()
        for value in values:
            wanted.update(_key_variants(value))
        if not wanted:
            return None

        for row in self.rows:
            row_keys: set[str] = set()
            for key, value in row.items():
                if _canonical_column(key) in KEY_COLUMNS:
                    row_keys.update(_key_variants(value))
            matched = sorted(wanted.intersection(row_keys), key=len, reverse=True)
            if matched:
                return self._match_from_row(row, matched[0])
        return None

    def _match_from_row(self, row: dict[str, str], matched_key: str) -> ManualGeocodeMatch:
        latitude = _number(_row_value(row, [self.latitude_column] if self.latitude_column else []))
        longitude = _number(_row_value(row, [self.longitude_column] if self.longitude_column else []))
        return ManualGeocodeMatch(
            row=row,
            matched_key=matched_key,
            latitude=latitude,
            longitude=longitude,
            elevation_m=_number(_row_value(row, ELEVATION_COLUMNS)),
            depth_m=_number(_row_value(row, DEPTH_COLUMNS)),
            age_ka=_number(_row_value(row, AGE_COLUMNS)),
            coordinate_uncertainty_m=_number(_row_value(row, UNCERTAINTY_COLUMNS)),
            figure=_text(_row_value(row, ("figure", "fig", "map", "source_figure"))),
            description=_text(_row_value(row, ("description", "notes", "note", "comment"))),
        )


@lru_cache(maxsize=256)
def load_manual_geocodes(source_id: str, source_path: Path) -> ManualGeocodeTable:
    manual_path = _manual_path(source_id, source_path)
    if manual_path is None or not manual_path.exists():
        return ManualGeocodeTable(source_id=source_id, path=Path(), rows=(), latitude_column=None, longitude_column=None)
    with manual_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = tuple({str(key or "").strip(): str(value or "").strip() for key, value in row.items()} for row in reader)
        fieldnames = tuple(str(field or "").strip() for field in (reader.fieldnames or ()))
    latitude_column = _first_column(fieldnames, LATITUDE_COLUMNS)
    longitude_column = _first_column(fieldnames, LONGITUDE_COLUMNS)
    return ManualGeocodeTable(source_id=source_id, path=manual_path, rows=rows, latitude_column=latitude_column, longitude_column=longitude_column)


def manual_geocode_note(table: ManualGeocodeTable, match: ManualGeocodeMatch | None) -> str:
    if not table.has_rows:
        return ""
    if not table.has_coordinate_columns:
        return (
            f" Authoritative manual geocode table {table.path.name} is present but lacks latitude/longitude columns; "
            "fuzzy gazetteer geocoding was suppressed."
        )
    if match is None:
        return (
            f" Authoritative manual geocode table {table.path.name} is present but no row matched this record; "
            "fuzzy gazetteer geocoding was suppressed."
        )
    if match.latitude is None or match.longitude is None:
        return (
            f" Authoritative manual geocode row matched key '{match.matched_key}' but lacks usable coordinates; "
            "fuzzy gazetteer geocoding was suppressed."
        )
    return f" Coordinate taken from authoritative manual geocode table {table.path.name} using key '{match.matched_key}'."


def _manual_path(source_id: str, source_path: Path) -> Path | None:
    for parent in (source_path.parent, *source_path.parents):
        candidate = parent / "manual_geocodes" / f"{source_id}.csv"
        if candidate.exists():
            return candidate
        if parent.name == "data":
            candidate = parent / "manual_geocodes" / f"{source_id}.csv"
            return candidate
    return source_path.parent.parent / "manual_geocodes" / f"{source_id}.csv" if len(source_path.parents) >= 2 else None


def _first_column(fieldnames: tuple[str, ...], candidates: tuple[str, ...]) -> str | None:
    canonical = {_canonical_column(field): field for field in fieldnames}
    for candidate in candidates:
        if candidate in canonical:
            return canonical[candidate]
    return None


def _row_value(row: dict[str, str], candidates: tuple[str, ...] | list[str]) -> str:
    by_canonical = {_canonical_column(key): value for key, value in row.items()}
    for candidate in candidates:
        if candidate and _canonical_column(candidate) in by_canonical:
            value = by_canonical[_canonical_column(candidate)]
            if value:
                return value
    return ""


def _canonical_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _key_variants(value: str | int | float | None) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    normalized = re.sub(r"[^a-z0-9]+", "", text.lower())
    variants = {normalized} if normalized else set()
    if normalized.isdigit():
        variants.add(str(int(normalized)))
        variants.add(normalized.zfill(2))
    return variants


def _number(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _text(value: str) -> str | None:
    text = str(value or "").strip()
    return text or None
