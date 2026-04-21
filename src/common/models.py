from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any


@dataclass(slots=True)
class SourceLocator:
    page: str | None = None
    figure: str | None = None
    table: str | None = None
    section: str | None = None
    quote_or_paraphrase: str = ""


@dataclass(slots=True)
class ReportedObservations:
    reported_elevation_m: float | None = None
    reported_depth_m: float | None = None
    reported_datum: str | None = None
    reported_uncertainty_m: float | None = None


@dataclass(slots=True)
class DerivedObservations:
    raw_raster_value_m: float | None = None
    applied_correction_m: float | None = None
    derived_elevation_m: float | None = None
    derived_depth_m: float | None = None
    derivation_notes: str = ""


@dataclass(slots=True)
class AgeModel:
    method: str = "other"
    relation: str = "unknown"
    age_ka: float | list[float | None] | str | None = None
    uncertainty_ka: float | None = None
    material: str | None = None
    notes: str = ""


@dataclass(slots=True)
class SamplePoint:
    id: str
    source_id: str
    record_class: str
    site_name: str
    sample_id: str
    latitude: float | None
    longitude: float | None
    coordinate_source: str
    coordinate_uncertainty_m: float | None
    elevation_m: float | list[float | None] | str | None
    elevation_reference: str
    depth_source: str
    indicator_type: str
    indicator_subtype: str
    indicative_range_m: list[float | None] | None
    age_ka: float | list[float | None] | str | None
    dating_method: str
    description: str
    location_name: str
    bibliographic_reference: str
    doi_or_url: str
    confidence_score: float | None
    notes: str
    source_locator: SourceLocator = field(default_factory=SourceLocator)
    reported_observations: ReportedObservations = field(default_factory=ReportedObservations)
    derived_observations: DerivedObservations = field(default_factory=DerivedObservations)
    age_models: list[AgeModel] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deterministic_sample_id(
    source_id: str,
    site_name: str,
    sample_id: str,
    latitude: float | None,
    longitude: float | None,
    indicator_type: str,
    locator: SourceLocator,
) -> str:
    payload = "|".join(
        [
            source_id.strip(),
            site_name.strip().lower(),
            sample_id.strip().lower(),
            "" if latitude is None else f"{latitude:.6f}",
            "" if longitude is None else f"{longitude:.6f}",
            indicator_type.strip(),
            locator.page or "",
            locator.figure or "",
            locator.table or "",
            locator.section or "",
        ]
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def make_empty_sample_point(source_id: str, site_name: str = "", sample_id: str = "") -> SamplePoint:
    locator = SourceLocator(quote_or_paraphrase="")
    return SamplePoint(
        id=deterministic_sample_id(source_id, site_name, sample_id, None, None, "unknown", locator),
        source_id=source_id,
        record_class="sea_level_indicator",
        site_name=site_name,
        sample_id=sample_id,
        latitude=None,
        longitude=None,
        coordinate_source="inferred_text",
        coordinate_uncertainty_m=None,
        elevation_m=None,
        elevation_reference="unknown",
        depth_source="other",
        indicator_type="raised_beach",
        indicator_subtype="",
        indicative_range_m=None,
        age_ka=None,
        dating_method="other",
        description="",
        location_name="",
        bibliographic_reference="",
        doi_or_url="",
        confidence_score=None,
        notes="",
        source_locator=locator,
    )
