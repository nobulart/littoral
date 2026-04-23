from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.common.models import SamplePoint


_RASTER_CACHE: dict[Path, "_TiledGeoTiffSampler"] = {}


def apply_elevation_normalization(sample_point: SamplePoint, raster_path: Path) -> SamplePoint:
    if sample_point.reported_observations.reported_elevation_m is not None:
        sample_point.elevation_m = sample_point.reported_observations.reported_elevation_m
        sample_point.depth_source = "reported"
        sample_point.derived_observations.derivation_notes = "Reported elevation preserved; no raster overwrite applied."
        return sample_point

    if sample_point.elevation_m is not None:
        if not sample_point.derived_observations.derivation_notes:
            sample_point.derived_observations.derivation_notes = "Existing elevation preserved; no raster overwrite applied."
        return sample_point

    if not raster_path.exists():
        sample_point.derived_observations.derivation_notes = "SRTM15+V2 raster not found."
        return sample_point

    if sample_point.latitude is None or sample_point.longitude is None:
        sample_point.derived_observations.derivation_notes = "No coordinates available for DEM elevation sampling."
        return sample_point

    try:
        elevation_m = _sampler(raster_path).sample(float(sample_point.longitude), float(sample_point.latitude))
    except Exception as exc:  # pragma: no cover - defensive around optional TIFF codecs.
        sample_point.derived_observations.derivation_notes = f"SRTM15+V2 sampling failed: {exc}"
        return sample_point

    if elevation_m is None:
        sample_point.derived_observations.derivation_notes = "SRTM15+V2 returned no valid elevation at the point coordinates."
        return sample_point

    sample_point.elevation_m = round(elevation_m, 2)
    sample_point.elevation_reference = "MSL"
    sample_point.depth_source = "SRTM15+V2"
    sample_point.derived_observations.raw_raster_value_m = elevation_m
    sample_point.derived_observations.derived_elevation_m = round(elevation_m, 2)
    sample_point.derived_observations.derived_depth_m = round(-elevation_m, 2) if elevation_m < 0 else None
    sample_point.derived_observations.derivation_notes = _sampling_note(sample_point)
    return sample_point


def _sampling_note(sample_point: SamplePoint) -> str:
    if _has_authoritative_coordinates(sample_point):
        return "Elevation sampled from SRTM15+V2 at authoritative coordinates because source elevation was not reported."
    return (
        "Elevation sampled from SRTM15+V2 at inferred coordinates because source elevation was not reported. "
        "Treat this as approximate spatial context, not a surveyed sample elevation."
    )


def _has_authoritative_coordinates(sample_point: SamplePoint) -> bool:
    notes = sample_point.notes.lower()
    if sample_point.coordinate_source == "reported":
        return True
    return "authoritative manual geocode" in notes or "coordinate manually geocoded" in notes


def _sampler(raster_path: Path) -> "_TiledGeoTiffSampler":
    path = raster_path.resolve()
    if path not in _RASTER_CACHE:
        _RASTER_CACHE[path] = _TiledGeoTiffSampler(path)
    return _RASTER_CACHE[path]


class _TiledGeoTiffSampler:
    def __init__(self, raster_path: Path) -> None:
        import tifffile

        self._tiff = tifffile.TiffFile(raster_path)
        self._page = self._tiff.pages[0]
        self._height, self._width = self._page.shape[:2]
        self._tile_width = int(self._page.tilewidth or self._width)
        self._tile_height = int(self._page.tilelength or self._height)
        self._tiles_across = math.ceil(self._width / self._tile_width)
        self._origin_x, self._origin_y, self._pixel_width, self._pixel_height = self._geotransform()
        self._nodata = self._nodata_value()

    def sample(self, lon: float, lat: float) -> float | None:
        column = int((lon - self._origin_x) / self._pixel_width)
        row = int((self._origin_y - lat) / self._pixel_height)
        if row < 0 or column < 0 or row >= self._height or column >= self._width:
            return None

        tile_column = column // self._tile_width
        tile_row = row // self._tile_height
        tile_index = tile_row * self._tiles_across + tile_column
        tile = self._decode_tile(tile_index)
        local_row = row - tile_row * self._tile_height
        local_column = column - tile_column * self._tile_width
        value = float(tile[local_row, local_column])
        if math.isnan(value):
            return None
        if self._nodata is not None and value == self._nodata:
            return None
        return value

    def _decode_tile(self, tile_index: int) -> Any:
        offset = self._page.dataoffsets[tile_index]
        bytecount = self._page.databytecounts[tile_index]
        self._tiff.filehandle.seek(offset)
        data = self._tiff.filehandle.read(bytecount)
        decoded = self._page.decode(data, tile_index)
        array = decoded[0] if isinstance(decoded, tuple) else decoded
        while getattr(array, "ndim", 0) > 2:
            array = array[0] if array.shape[0] == 1 else array[..., 0]
        return array

    def _geotransform(self) -> tuple[float, float, float, float]:
        scale_tag = self._page.tags.get("ModelPixelScaleTag")
        tiepoint_tag = self._page.tags.get("ModelTiepointTag")
        if scale_tag is None or tiepoint_tag is None:
            raise ValueError("GeoTIFF lacks ModelPixelScaleTag/ModelTiepointTag")
        scale = scale_tag.value
        tiepoint = tiepoint_tag.value
        return float(tiepoint[3]), float(tiepoint[4]), float(scale[0]), float(scale[1])

    def _nodata_value(self) -> float | None:
        tag = self._page.tags.get("GDAL_NODATA")
        if tag is None:
            return None
        try:
            return float(tag.value)
        except (TypeError, ValueError):
            return None
