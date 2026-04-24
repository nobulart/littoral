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


def source_locator_from_dict(payload: dict[str, Any] | None) -> SourceLocator:
    if not isinstance(payload, dict):
        return SourceLocator()
    return SourceLocator(
        page=payload.get("page"),
        figure=payload.get("figure"),
        table=payload.get("table"),
        section=payload.get("section"),
        quote_or_paraphrase=str(payload.get("quote_or_paraphrase") or ""),
    )


def reported_observations_from_dict(payload: dict[str, Any] | None) -> ReportedObservations:
    if not isinstance(payload, dict):
        return ReportedObservations()
    return ReportedObservations(
        reported_elevation_m=payload.get("reported_elevation_m"),
        reported_depth_m=payload.get("reported_depth_m"),
        reported_datum=payload.get("reported_datum"),
        reported_uncertainty_m=payload.get("reported_uncertainty_m"),
    )


def derived_observations_from_dict(payload: dict[str, Any] | None) -> DerivedObservations:
    if not isinstance(payload, dict):
        return DerivedObservations()
    return DerivedObservations(
        raw_raster_value_m=payload.get("raw_raster_value_m"),
        applied_correction_m=payload.get("applied_correction_m"),
        derived_elevation_m=payload.get("derived_elevation_m"),
        derived_depth_m=payload.get("derived_depth_m"),
        derivation_notes=str(payload.get("derivation_notes") or ""),
    )


def age_model_from_dict(payload: dict[str, Any] | None) -> AgeModel:
    if not isinstance(payload, dict):
        return AgeModel()
    return AgeModel(
        method=str(payload.get("method") or "other"),
        relation=str(payload.get("relation") or "unknown"),
        age_ka=payload.get("age_ka"),
        uncertainty_ka=payload.get("uncertainty_ka"),
        material=payload.get("material"),
        notes=str(payload.get("notes") or ""),
    )


def sample_point_from_dict(payload: dict[str, Any]) -> SamplePoint:
    return SamplePoint(
        id=str(payload.get("id") or ""),
        source_id=str(payload.get("source_id") or ""),
        record_class=str(payload.get("record_class") or ""),
        site_name=str(payload.get("site_name") or ""),
        sample_id=str(payload.get("sample_id") or ""),
        latitude=payload.get("latitude"),
        longitude=payload.get("longitude"),
        coordinate_source=str(payload.get("coordinate_source") or ""),
        coordinate_uncertainty_m=payload.get("coordinate_uncertainty_m"),
        elevation_m=payload.get("elevation_m"),
        elevation_reference=str(payload.get("elevation_reference") or ""),
        depth_source=str(payload.get("depth_source") or ""),
        indicator_type=str(payload.get("indicator_type") or ""),
        indicator_subtype=str(payload.get("indicator_subtype") or ""),
        indicative_range_m=payload.get("indicative_range_m"),
        age_ka=payload.get("age_ka"),
        dating_method=str(payload.get("dating_method") or ""),
        description=str(payload.get("description") or ""),
        location_name=str(payload.get("location_name") or ""),
        bibliographic_reference=str(payload.get("bibliographic_reference") or ""),
        doi_or_url=str(payload.get("doi_or_url") or ""),
        confidence_score=payload.get("confidence_score"),
        notes=str(payload.get("notes") or ""),
        source_locator=source_locator_from_dict(payload.get("source_locator")),
        reported_observations=reported_observations_from_dict(payload.get("reported_observations")),
        derived_observations=derived_observations_from_dict(payload.get("derived_observations")),
        age_models=[age_model_from_dict(item) for item in payload.get("age_models", []) if isinstance(item, dict)],
    )
