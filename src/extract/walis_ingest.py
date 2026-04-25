from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.common.io import write_csv, write_summary
from src.common.models import (
    AgeModel,
    DerivedObservations,
    ReportedObservations,
    SamplePoint,
    SourceLocator,
    deterministic_sample_id,
)
from src.merge.build import build_master_outputs
from src.validate.samplepoint import score_confidence


DEFAULT_WALIS_SUMMARY = (
    Path("data")
    / "WALIS"
    / "Atlas_Versions"
    / "Ver_1"
    / "Ver1_0_post_review"
    / "Output"
    / "Data"
    / "csv"
    / "Summary.csv"
)

QUALITY_MODES = {"accepted", "positive-quality", "all"}
SOURCE_ID = "walis_v1_0_post_review"


@dataclass(slots=True)
class WalisIngestResult:
    source_id: str
    source_csv: Path
    per_source_csv: Path
    summary_path: Path
    points_written: int
    rows_seen: int
    rows_skipped: int
    skip_reasons: Counter[str]
    indicator_counts: Counter[str]
    record_class_counts: Counter[str]
    merged_csv: Path | None = None
    merged_geojson: Path | None = None
    merged_count: int | None = None


def ingest_walis_summary(
    summary_csv: Path,
    per_source_dir: Path,
    *,
    source_id: str = SOURCE_ID,
    quality_mode: str = "accepted",
    include_limiting: bool = False,
    merge: bool = False,
    merged_dir: Path | None = None,
    merge_mode: str = "append",
) -> WalisIngestResult:
    if quality_mode not in QUALITY_MODES:
        raise ValueError(f"Unsupported quality mode `{quality_mode}`. Expected one of: {', '.join(sorted(QUALITY_MODES))}")
    if merge and merged_dir is None:
        raise ValueError("merged_dir is required when merge=True")

    points: list[SamplePoint] = []
    skip_reasons: Counter[str] = Counter()
    indicator_counts: Counter[str] = Counter()
    record_class_counts: Counter[str] = Counter()

    with summary_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    for row_index, row in enumerate(rows, start=2):
        reason = _skip_reason(row, quality_mode=quality_mode, include_limiting=include_limiting)
        if reason:
            skip_reasons[reason] += 1
            continue

        point = _row_to_sample_point(row, source_id, row_index)
        if point is None:
            skip_reasons["unmapped_indicator"] += 1
            continue

        points.append(point)
        indicator_counts[point.indicator_type] += 1
        record_class_counts[point.record_class] += 1

    per_source_dir.mkdir(parents=True, exist_ok=True)
    per_source_csv = per_source_dir / f"{source_id}.csv"
    summary_path = per_source_dir / f"{source_id}.summary.md"
    write_csv(per_source_csv, points)
    write_summary(
        summary_path,
        _summary_lines(
            summary_csv=summary_csv,
            source_id=source_id,
            quality_mode=quality_mode,
            include_limiting=include_limiting,
            rows_seen=len(rows),
            points_written=len(points),
            skip_reasons=skip_reasons,
            indicator_counts=indicator_counts,
            record_class_counts=record_class_counts,
        ),
    )

    result = WalisIngestResult(
        source_id=source_id,
        source_csv=summary_csv,
        per_source_csv=per_source_csv,
        summary_path=summary_path,
        points_written=len(points),
        rows_seen=len(rows),
        rows_skipped=len(rows) - len(points),
        skip_reasons=skip_reasons,
        indicator_counts=indicator_counts,
        record_class_counts=record_class_counts,
    )

    if merge:
        assert merged_dir is not None
        csv_path, geojson_path, merged_count = build_master_outputs(per_source_dir, merged_dir, mode=merge_mode)
        result.merged_csv = csv_path
        result.merged_geojson = geojson_path
        result.merged_count = merged_count

    return result


def _skip_reason(row: dict[str, str], *, quality_mode: str, include_limiting: bool) -> str | None:
    if not _text(row, "WALIS_ID"):
        return "missing_walis_id"
    if _float(row, "Latitude") is None or _float(row, "Longitude") is None:
        return "missing_coordinates"
    if not include_limiting and _record_class(row) != "sea_level_indicator":
        return "limiting_excluded"
    if quality_mode == "accepted" and _text(row, "Originally accepted?").lower() != "yes":
        return "not_originally_accepted"
    if quality_mode == "positive-quality" and (_quality(row, "Quality of RSL information") <= 0 or _quality(row, "Quality of age information") <= 0):
        return "non_positive_quality"
    return None


def _row_to_sample_point(row: dict[str, str], source_id: str, row_index: int) -> SamplePoint | None:
    indicator_type = _indicator_type(row)
    if indicator_type is None:
        return None

    lat = _float(row, "Latitude")
    lon = _float(row, "Longitude")
    site_name = _site_name(row)
    sample_id = _sample_id(row, row_index)
    locator = SourceLocator(
        table="WALIS Summary.csv",
        section=sample_id,
        quote_or_paraphrase=_quote_or_paraphrase(row),
    )
    age_value, age_uncertainty, age_notes = _best_age(row)
    paleo_rsl = _float(row, "Paleo RSL (m)")
    elevation = _float(row, "Elevation (m)")
    elevation_m = paleo_rsl if paleo_rsl is not None else elevation
    elevation_uncertainty = _float(row, "Paleo RSL uncertainty (m)") or _float(row, "Elevation error (m)")
    reported_datum = _text(row, "Vertical datum") or None

    point = SamplePoint(
        id=deterministic_sample_id(source_id, site_name, sample_id, lat, lon, indicator_type, locator),
        source_id=source_id,
        record_class=_record_class(row),
        site_name=site_name,
        sample_id=sample_id,
        latitude=lat,
        longitude=lon,
        coordinate_source="reported",
        coordinate_uncertainty_m=_coordinate_uncertainty(row),
        elevation_m=elevation_m,
        elevation_reference=_elevation_reference(row),
        depth_source="reported",
        indicator_type=indicator_type,
        indicator_subtype=_text(row, "RSL Indicator"),
        indicative_range_m=_indicative_range(row),
        age_ka=age_value,
        dating_method=_dating_method(row),
        description=_description(row),
        location_name=_location_name(row),
        bibliographic_reference=_text(row, "Reference(s)"),
        doi_or_url="https://doi.org/10.5281/zenodo.5979520",
        confidence_score=None,
        notes=_notes(row, age_notes=age_notes, used_paleo_rsl=paleo_rsl is not None),
        source_locator=locator,
        reported_observations=ReportedObservations(
            reported_elevation_m=elevation,
            reported_depth_m=_reported_depth(row, elevation),
            reported_datum=reported_datum,
            reported_uncertainty_m=elevation_uncertainty,
        ),
        derived_observations=DerivedObservations(
            derived_elevation_m=paleo_rsl,
            derivation_notes="WALIS Paleo RSL used as LITTORAL analytical elevation when available; measured indicator elevation preserved as reported observation.",
        ),
        age_models=[
            AgeModel(
                method=_dating_method(row),
                relation=_age_relation(row),
                age_ka=age_value,
                uncertainty_ka=age_uncertainty,
                material=_text(row, "Material_type") or None,
                notes=age_notes,
            )
        ],
    )
    point.confidence_score = _walis_confidence(point, row)
    return point


def _indicator_type(row: dict[str, str]) -> str | None:
    raw = _text(row, "RSL Indicator").lower()
    desc = _text(row, "RSL indicator description").lower()
    combined = f"{raw} {desc}"
    if "marine terrace" in combined or "shoreline angle" in combined:
        return "marine_terrace"
    if "single coral" in combined:
        return "coral_head"
    if "coral" in combined or "reef" in combined or "algal reef" in combined:
        return "coral_reef"
    if "tidal notch" in combined:
        return "tidal_notch"
    if "beachrock" in combined:
        return "beachrock"
    if "beach" in combined or "foreshore" in combined or "barrier" in combined:
        return "raised_beach"
    if "lagoon" in combined:
        return "lagoonal_sediment"
    if "estuar" in combined:
        return "estuarine_sediment"
    if "salt marsh" in combined or "basal peat" in combined or "peat" in combined:
        return "saltmarsh_peat_or_mud"
    if "mangrove" in combined:
        return "mangrove_peat_or_mud"
    if "tidal flat" in combined or "intertidal" in combined:
        return "tidal_flat_sediment"
    if "shore platform" in combined:
        return "shore_platform"
    if "speleothem" in combined:
        return "speleothem_sea_level_proxy"
    if "marine fauna" in combined or "shell" in combined:
        return "marine_shell_bed"
    if "limiting indicator" in combined:
        return "subtidal_facies" if _record_class(row) == "marine_limiting" else "marine_over_terrestrial_contact"
    return None


def _record_class(row: dict[str, str]) -> str:
    value = _text(row, "Type of datapoint").lower()
    if "marine limiting" in value:
        return "marine_limiting"
    if "terrestrial limiting" in value:
        return "terrestrial_limiting"
    return "sea_level_indicator"


def _dating_method(row: dict[str, str]) -> str:
    value = _text(row, "Dating technique").lower()
    if "u-series" in value or "uranium" in value:
        return "u_series"
    if "luminescence" in value:
        return "luminescence"
    if value == "aar":
        return "aar"
    if value == "esr":
        return "esr"
    if "stratigraphic" in value:
        return "stratigraphic_context"
    if "radiocarbon" in value or "14c" in value:
        return "radiocarbon"
    return "other"


def _best_age(row: dict[str, str]) -> tuple[float | list[float | None] | str | None, float | None, str]:
    candidates = [
        ("U-Series corrected age (speleothems, ka)", "U-Series corrected age uncertainty (speleothems, ka)", "WALIS corrected U-Series speleothem age."),
        ("U-Series recalculated age (ka)", "U-Series recalculate age uncertainty (ka)", "WALIS recalculated U-Series age."),
        ("Reported age (ka)", "Reported age uncertainty (ka)", "WALIS reported age."),
    ]
    for age_key, uncertainty_key, note in candidates:
        age = _float(row, age_key)
        if age is not None:
            return age, _float(row, uncertainty_key), note

    upper = _float(row, "Stratigraphy Upper Age (ka)")
    lower = _float(row, "Stratigraphy Lower Age (ka)")
    if upper is not None or lower is not None:
        return [lower, upper], None, "WALIS stratigraphic upper/lower age bounds."

    mis = _text(row, "Marine Isotopic Stage")
    if mis:
        return mis, None, "WALIS Marine Isotopic Stage used as non-numeric age attribution."

    return None, None, "No direct numeric age present in WALIS Summary row."


def _age_relation(row: dict[str, str]) -> str:
    value = _text(row, "Timing constraint").lower()
    if "older" in value:
        return "older_than"
    if "younger" in value:
        return "younger_than"
    if "equal" in value:
        return "equal"
    if _text(row, "Stratigraphy Upper Age (ka)") or _text(row, "Stratigraphy Lower Age (ka)"):
        return "range"
    return "unknown"


def _indicative_range(row: dict[str, str]) -> list[float | None] | None:
    upper = _float(row, "Upper limit of living range (m)")
    lower = _float(row, "Lower limit of living range (m)")
    if upper is None and lower is None:
        ir = _float(row, "IR")
        rwl = _float(row, "RWL")
        if ir is None or rwl is None:
            return None
        half = abs(ir) / 2.0
        return [rwl - half, rwl + half]
    return [lower, upper]


def _elevation_reference(row: dict[str, str]) -> str:
    datum = _text(row, "Vertical datum").lower()
    if "lowest astronomical" in datum:
        return "LAT"
    if "mean sea level" in datum or "egm" in datum or "geoid" in datum or "datum" in datum:
        return "MSL"
    return "unknown"


def _reported_depth(row: dict[str, str], elevation: float | None) -> float | None:
    depth = _float(row, "Paleo water depth estimate (m)")
    if depth is not None and depth > 0:
        return depth
    if elevation is not None and elevation < 0:
        return abs(elevation)
    return None


def _coordinate_uncertainty(row: dict[str, str]) -> float:
    distance = _float(row, "Distance from sample")
    if distance is not None and distance >= 0:
        return distance
    return 100.0


def _walis_confidence(point: SamplePoint, row: dict[str, str]) -> float:
    base = score_confidence(point)
    rsl_quality = _quality(row, "Quality of RSL information")
    age_quality = _quality(row, "Quality of age information")
    quality_bonus = ((rsl_quality + age_quality) / 10.0) * 0.25 if rsl_quality or age_quality else 0.0
    accepted_bonus = 0.08 if _text(row, "Originally accepted?").lower() == "yes" else 0.0
    accepted_other_bonus = 0.03 if _text(row, "Accepted by other study?").lower() == "yes" else 0.0
    limiting_penalty = 0.05 if point.record_class != "sea_level_indicator" else 0.0
    return round(max(0.0, min(1.0, base * 0.75 + quality_bonus + accepted_bonus + accepted_other_bonus - limiting_penalty)), 3)


def _description(row: dict[str, str]) -> str:
    parts = [_text(row, "RSL Indicator"), _text(row, "RSL indicator description")]
    return "; ".join(part for part in parts if part)


def _location_name(row: dict[str, str]) -> str:
    parts = [_text(row, "Site"), _text(row, "Subsite"), _text(row, "Region"), _text(row, "Nation")]
    return ", ".join(part for part in parts if part)


def _site_name(row: dict[str, str]) -> str:
    site = _text(row, "Site")
    subsite = _text(row, "Subsite")
    if site and subsite:
        return f"{site} - {subsite}"
    return site or subsite or _text(row, "WALIS_ID")


def _sample_id(row: dict[str, str], row_index: int) -> str:
    # WALIS_ID is a useful public identifier, but it is not unique in the
    # denormalized Summary.csv because one RSL point can carry many analyses.
    return f"{_text(row, 'WALIS_ID')}@row{row_index}"


def _quote_or_paraphrase(row: dict[str, str]) -> str:
    return (
        f"WALIS row {_text(row, 'WALIS_ID')}: {_text(row, 'Type of datapoint')}; "
        f"{_text(row, 'RSL Indicator')}; elevation {_text(row, 'Elevation (m)')} m; "
        f"paleo RSL {_text(row, 'Paleo RSL (m)') or 'not reported'} m; "
        f"dating {_text(row, 'Dating technique') or 'not reported'}."
    )


def _notes(row: dict[str, str], *, age_notes: str, used_paleo_rsl: bool) -> str:
    pieces = [
        f"WALIS quality RSL={_text(row, 'Quality of RSL information') or 'blank'}, age={_text(row, 'Quality of age information') or 'blank'}.",
        f"Originally accepted={_text(row, 'Originally accepted?') or 'blank'}; accepted by other study={_text(row, 'Accepted by other study?') or 'blank'}.",
        f"Elevation source: {'Paleo RSL (m)' if used_paleo_rsl else 'Elevation (m)'}.",
        age_notes,
    ]
    for key in ["Originally reported ID", "Analysis ID", "WALIS U-series ID", "WALIS AAR ID", "WALIS ESR ID", "WALIS LUM ID", "WALIS strat ID", "WALIS other ID", "Record Created by", "Last Update"]:
        value = _text(row, key)
        if value:
            pieces.append(f"{key}: {value}.")
    return " ".join(pieces)


def _quality(row: dict[str, str], key: str) -> int:
    value = _text(row, key)
    try:
        return int(float(value))
    except ValueError:
        return 0


def _float(row: dict[str, str], key: str) -> float | None:
    value = _text(row, key)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _text(row: dict[str, str], key: str) -> str:
    return " ".join(str(row.get(key) or "").split())


def _summary_lines(
    *,
    summary_csv: Path,
    source_id: str,
    quality_mode: str,
    include_limiting: bool,
    rows_seen: int,
    points_written: int,
    skip_reasons: Counter[str],
    indicator_counts: Counter[str],
    record_class_counts: Counter[str],
) -> list[str]:
    lines = [
        "# Summary for WALIS post-review atlas import",
        "",
        f"- Source ID: `{source_id}`",
        f"- Source CSV: `{summary_csv}`",
        f"- Quality mode: `{quality_mode}`",
        f"- Include limiting records: `{include_limiting}`",
        f"- WALIS rows seen: `{rows_seen}`",
        f"- Canonical SamplePoints written: `{points_written}`",
        f"- Rows skipped: `{rows_seen - points_written}`",
        "- Extraction mode: deterministic WALIS Summary.csv mapper.",
        "- Source: https://alerovere.github.io/WALIS/",
        "",
        "## Record Classes",
    ]
    lines.extend(f"- `{key}`: `{count}`" for key, count in record_class_counts.most_common())
    lines.append("")
    lines.append("## Indicator Types")
    lines.extend(f"- `{key}`: `{count}`" for key, count in indicator_counts.most_common())
    lines.append("")
    lines.append("## Skip Reasons")
    lines.extend(f"- `{key}`: `{count}`" for key, count in skip_reasons.most_common())
    return lines
