#!/usr/bin/env python3
"""
13_littoral_geodetic_orientation_inverse.py

Invert a site elevation/depth into candidate geodetic pole-shift orientations.

Example:
  python scripts/13_littoral_geodetic_orientation_inverse.py \
    --lat 32.2 --lon -64.6 --elevation -2800 --name "Beebe 1931 Bermuda"

  python scripts/13_littoral_geodetic_orientation_inverse.py \
    --location "Bermuda" --elevation -2800

  python scripts/13_littoral_geodetic_orientation_inverse.py \
    --lat 32.2 --lon -64.6

If elevation is omitted, the script samples:
  ../data/elevation/SRTM15+V2.tiff
relative to the project root.

Model:
  Present pole P0 = geographic north pole.
  Candidate displaced pole P1 is sampled by offset angle alpha and azimuth phi.
  Local geodetic sea-level change is approximated from the change in the
  rotational/equatorial-bulge field:

      ΔSL = B * [ (P0·r)^2 - (P1·r)^2 ]

  where B defaults to 11000 m, matching Fairbridge's spheroid-radius
  scale used in the 245 m/degree first-order estimate.

Sign convention:
  predicted_elevation_m = -ΔSL

Thus a submerged beach at -2800 m requires ΔSL ≈ +2800 m.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "geospatial_13"
DEFAULT_DEM = PROJECT_ROOT / "data" / "elevation" / "SRTM15+V2.tiff"

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "littoral_matplotlib"),
)

EARTH_BULGE_M = 11000.0
FAIRBRIDGE_M_PER_DEG = 245.0
DEFAULT_PROJECTION = "robinson"


def normalize_longitude(lon):
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def latlon_to_unit(lat_deg, lon_deg):
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    return np.column_stack(
        [
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ]
    )


def unit_to_latlon(v):
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    lat = np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    return lat, normalize_longitude(lon)


def displaced_pole(alpha_deg, azimuth_deg):
    alpha = np.radians(np.asarray(alpha_deg, dtype=float))
    phi = np.radians(np.asarray(azimuth_deg, dtype=float))
    return np.column_stack(
        [
            np.sin(alpha) * np.cos(phi),
            np.sin(alpha) * np.sin(phi),
            np.cos(alpha),
        ]
    )


def predicted_elevation(site_vec, pole_vec, bulge_m):
    p0_dot = site_vec[2]
    p1_dot = pole_vec @ site_vec
    delta_sl = bulge_m * (p0_dot * p0_dot - p1_dot * p1_dot)
    return -delta_sl


def geocode_location(name):
    try:
        from geopy.geocoders import Nominatim
    except ImportError as exc:
        raise RuntimeError("Install geopy or provide --lat and --lon.") from exc

    geocoder = Nominatim(user_agent="littoral_geodetic_inverse")
    result = geocoder.geocode(name, timeout=20)
    if result is None:
        raise RuntimeError(f"Could not geocode location: {name}")
    return float(result.latitude), float(result.longitude), result.address


def sample_dem(lat, lon, dem_path):
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Install rasterio or provide --elevation.") from exc

    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    with rasterio.open(dem_path) as src:
        val = next(src.sample([(float(lon), float(lat))]))[0]
        if src.nodata is not None and val == src.nodata:
            raise RuntimeError(f"DEM returned nodata at lat={lat}, lon={lon}")
        return float(val)


def projection(name, ccrs):
    key = name.strip().lower()
    if key == "platecarree":
        return ccrs.PlateCarree()
    if key == "mollweide":
        return ccrs.Mollweide()
    if key == "equalearth":
        return ccrs.EqualEarth()
    if key == "robinson":
        return ccrs.Robinson()
    raise ValueError(f"Unsupported projection: {name}")


def solve_grid(
    lat,
    lon,
    target_elevation_m,
    alpha_min,
    alpha_max,
    alpha_step,
    azimuth_step,
    tolerance_m,
    bulge_m,
):
    site = latlon_to_unit([lat], [lon])[0]

    alphas = np.arange(alpha_min, alpha_max + 0.5 * alpha_step, alpha_step)
    azimuths = np.arange(0.0, 360.0, azimuth_step)

    rows = []
    all_rows = []

    for alpha in alphas:
        poles = displaced_pole(np.full_like(azimuths, alpha), azimuths)
        pred = predicted_elevation(site, poles, bulge_m)
        residual = pred - target_elevation_m

        pole_lat, pole_lon = unit_to_latlon(poles)

        block = pd.DataFrame(
            {
                "offset_deg": alpha,
                "orientation_azimuth_deg": azimuths,
                "candidate_pole_lat": pole_lat,
                "candidate_pole_lon": pole_lon,
                "predicted_elevation_m": pred,
                "target_elevation_m": target_elevation_m,
                "residual_m": residual,
                "abs_residual_m": np.abs(residual),
            }
        )
        all_rows.append(block)
        rows.append(block[block["abs_residual_m"] <= tolerance_m])

    candidates = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    full = pd.concat(all_rows, ignore_index=True)

    best = (
        full.sort_values("abs_residual_m")
        .groupby("offset_deg", as_index=False)
        .first()
        .sort_values("abs_residual_m")
    )

    return candidates.sort_values("abs_residual_m"), best, full


def make_plot(candidates, best, lat, lon, target_elevation_m, out_path, projection_name):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    data_crs = ccrs.PlateCarree()
    proj = projection(projection_name, ccrs)

    fig = plt.figure(figsize=(15, 8.5))
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.set_facecolor("#f7f7f4")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#eeeeea", edgecolor="none")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f8fbff", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.55)
    ax.gridlines(crs=data_crs, draw_labels=True, linewidth=0.35, alpha=0.35, linestyle="--")

    ax.scatter([lon], [lat], transform=data_crs, s=95, marker="*", c="black", zorder=10)

    if not candidates.empty:
        sc = ax.scatter(
            candidates["candidate_pole_lon"],
            candidates["candidate_pole_lat"],
            transform=data_crs,
            c=candidates["offset_deg"],
            s=18,
            alpha=0.75,
            linewidths=0,
            zorder=7,
        )
        cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.72, pad=0.06)
        cbar.set_label("Candidate polar offset satisfying target elevation (degrees)")

    top = best.head(20)
    ax.scatter(
        top["candidate_pole_lon"],
        top["candidate_pole_lat"],
        transform=data_crs,
        s=35,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
        zorder=8,
    )

    ax.set_title(
        "13 LITTORAL geodetic orientation inverse\n"
        f"site=({lat:.4f}, {lon:.4f}), target elevation={target_elevation_m:.1f} m",
        fontsize=15,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="site")
    parser.add_argument("--location", default=None)
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--elevation", type=float, default=None)
    parser.add_argument("--dem", default=str(DEFAULT_DEM))
    parser.add_argument("--projection", default=DEFAULT_PROJECTION)

    parser.add_argument("--alpha-min", type=float, default=0.1)
    parser.add_argument("--alpha-max", type=float, default=45.0)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--azimuth-step", type=float, default=0.25)
    parser.add_argument("--tolerance-m", type=float, default=25.0)
    parser.add_argument("--bulge-m", type=float, default=EARTH_BULGE_M)

    args = parser.parse_args()

    if args.lat is None or args.lon is None:
        if not args.location:
            raise SystemExit("Provide either --lat/--lon or --location.")
        lat, lon, geocode_label = geocode_location(args.location)
        name = args.name if args.name != "site" else args.location
    else:
        lat, lon = float(args.lat), float(args.lon)
        geocode_label = None
        name = args.name

    lon = float(normalize_longitude(lon))

    if args.elevation is None:
        target_elevation_m = sample_dem(lat, lon, args.dem)
        elevation_source = str(args.dem)
    else:
        target_elevation_m = float(args.elevation)
        elevation_source = "command_line"

    candidates, best, full = solve_grid(
        lat=lat,
        lon=lon,
        target_elevation_m=target_elevation_m,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        azimuth_step=args.azimuth_step,
        tolerance_m=args.tolerance_m,
        bulge_m=args.bulge_m,
    )

    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())[:80]
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_csv = out_dir / f"13_{safe}_candidate_orientations.csv"
    best_csv = out_dir / f"13_{safe}_best_by_offset.csv"
    full_csv = out_dir / f"13_{safe}_full_grid.csv"
    summary_json = out_dir / f"13_{safe}_summary.json"
    plot_png = out_dir / f"13_{safe}_candidate_orientations.png"

    candidates.to_csv(candidates_csv, index=False)
    best.to_csv(best_csv, index=False)
    full.to_csv(full_csv, index=False)

    fairbridge_linear_offset_deg = abs(target_elevation_m) / FAIRBRIDGE_M_PER_DEG

    summary = {
        "name": name,
        "geocode_label": geocode_label,
        "latitude": lat,
        "longitude": lon,
        "target_elevation_m": target_elevation_m,
        "elevation_source": elevation_source,
        "model": "delta_sl = bulge_m * [(P0 dot r)^2 - (P1 dot r)^2]; predicted_elevation = -delta_sl",
        "bulge_m": args.bulge_m,
        "fairbridge_m_per_degree_reference": FAIRBRIDGE_M_PER_DEG,
        "fairbridge_linear_abs_offset_estimate_deg": fairbridge_linear_offset_deg,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_step": args.alpha_step,
        "azimuth_step": args.azimuth_step,
        "tolerance_m": args.tolerance_m,
        "candidate_count": int(len(candidates)),
        "best_solution": None if best.empty else best.iloc[0].to_dict(),
        "outputs": {
            "candidates_csv": str(candidates_csv),
            "best_by_offset_csv": str(best_csv),
            "full_grid_csv": str(full_csv),
            "plot_png": str(plot_png),
            "summary_json": str(summary_json),
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_plot(
        candidates=candidates,
        best=best,
        lat=lat,
        lon=lon,
        target_elevation_m=target_elevation_m,
        out_path=plot_png,
        projection_name=args.projection,
    )

    print("\nLITTORAL 13 geodetic orientation inverse complete.")
    print(f"Site: {name}")
    print(f"Lat/Lon: {lat:.6f}, {lon:.6f}")
    print(f"Target elevation: {target_elevation_m:.2f} m")
    print(f"Fairbridge linear abs estimate: {fairbridge_linear_offset_deg:.3f} deg")
    print(f"Candidate orientations within ±{args.tolerance_m:.1f} m: {len(candidates)}")
    if not best.empty:
        row = best.iloc[0]
        print(
            "Best solution: "
            f"offset={row['offset_deg']:.3f} deg, "
            f"azimuth={row['orientation_azimuth_deg']:.3f} deg, "
            f"pole=({row['candidate_pole_lat']:.3f}, {row['candidate_pole_lon']:.3f}), "
            f"pred={row['predicted_elevation_m']:.2f} m, "
            f"resid={row['residual_m']:.2f} m"
        )
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()