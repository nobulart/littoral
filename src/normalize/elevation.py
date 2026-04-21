from __future__ import annotations

from pathlib import Path

from src.common.models import SamplePoint


def apply_elevation_normalization(sample_point: SamplePoint, raster_path: Path) -> SamplePoint:
    """Preserve reported values and leave placeholders for future raster sampling.

    This function is intentionally conservative until the real corpus arrives.
    """
    if not raster_path.exists():
        sample_point.derived_observations.derivation_notes = "SRTM15+V2 raster not found."
        return sample_point

    if sample_point.reported_observations.reported_elevation_m is not None:
        sample_point.elevation_m = sample_point.reported_observations.reported_elevation_m
        sample_point.depth_source = "reported"
        sample_point.derived_observations.derivation_notes = "Reported elevation preserved; no raster overwrite applied."
        return sample_point

    sample_point.derived_observations.derivation_notes = (
        "Raster-backed elevation derivation scaffold present, but point sampling is deferred until real coordinates are available."
    )
    return sample_point
