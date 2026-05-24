#!/usr/bin/env python3
"""
16_littoral_time_regime_inverse.py

Time-regime inverse test for LITTORAL.

Purpose:
  Split reported geocoded shoreline / paleolittoral records into temporal regimes,
  run the Script 14 geodetic pole-orientation inverse independently for each
  regime, and compare whether different age windows prefer distinct polar-offset
  families.

Default binning:
  autodynamic quantile-style bins over parsed age_ka values, constrained by
  --min-records and capped by --auto-max-bins.

Optional binning/windowing:
  --time-bins "name:min_ka,max_ka;name:min_ka,max_ka"
  --fixed-bin-width-ka 25
  --time-min-ka 0 --time-max-ka 250

Writes:
  outputs/geospatial_16/16_time_bin_summary.csv
  outputs/geospatial_16/16_time_bin_pairwise_angular_separation.csv
  outputs/geospatial_16/16_<time_bin>_preference_grid.csv
  outputs/geospatial_16/16_<time_bin>_top_candidates.csv
  outputs/geospatial_16/16_<time_bin>_offset_azimuth_heatmap.png
  outputs/geospatial_16/16_<time_bin>_polar_preference.png
  outputs/geospatial_16/16_time_bin_best_poles_map.png
  outputs/geospatial_16/16_time_regime_composite.png
  outputs/geospatial_16/16_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "merged" / "master_dataset.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "geospatial_16"

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "littoral_matplotlib"),
)

EARTH_BULGE_M = 11000.0
DEFAULT_PROJECTION = "robinson"
MAP_OUTLINE_COLOR = "#cfcfcf"


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


def angular_distance_deg(lat1, lon1, lat2, lon2):
    a = latlon_to_unit([lat1], [lon1])[0]
    b = latlon_to_unit([lat2], [lon2])[0]
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def displaced_poles(alpha_deg, azimuth_deg):
    alpha = np.radians(np.asarray(alpha_deg, dtype=float))
    phi = np.radians(np.asarray(azimuth_deg, dtype=float))
    return np.column_stack(
        [
            np.sin(alpha) * np.cos(phi),
            np.sin(alpha) * np.sin(phi),
            np.cos(alpha),
        ]
    )


def reported_observation_value(value, key: str) -> float:
    if pd.isna(value):
        return np.nan
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return np.nan
    return pd.to_numeric(payload.get(key), errors="coerce")


def reported_depth_source_mask(df):
    if "depth_source" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["depth_source"].map(
        lambda value: str(value).strip().lower() == "reported"
        if not pd.isna(value)
        else False
    )


def choose_reported_z(df):
    out = df.copy()

    for col in ["latitude", "longitude", "elevation_m", "z_m"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    reported_source = reported_depth_source_mask(out)
    z = pd.Series(np.nan, index=out.index, dtype=float)
    source = pd.Series("", index=out.index, dtype=object)

    if "z_m" in out.columns:
        mask = reported_source & out["z_m"].notna()
        z.loc[mask] = out.loc[mask, "z_m"]
        source.loc[mask] = "z_m"

    if "elevation_m" in out.columns:
        mask = reported_source & z.isna() & out["elevation_m"].notna()
        z.loc[mask] = out.loc[mask, "elevation_m"]
        source.loc[mask] = "elevation_m"

    if "reported_observations" in out.columns:
        reported_elevation = out["reported_observations"].map(
            lambda value: reported_observation_value(value, "reported_elevation_m")
        )
        reported_depth = out["reported_observations"].map(
            lambda value: reported_observation_value(value, "reported_depth_m")
        )

        mask = reported_source & z.isna() & reported_elevation.notna()
        z.loc[mask] = reported_elevation.loc[mask]
        source.loc[mask] = "reported_observations.reported_elevation_m"

        mask = reported_source & z.isna() & reported_depth.notna()
        z.loc[mask] = -reported_depth.loc[mask]
        source.loc[mask] = "reported_observations.reported_depth_m"

    out["z_m"] = z
    out["z_source"] = source
    return out


def read_input_csv(path):
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception as exc:
        if "NULL byte" not in str(exc) and "NUL" not in str(exc):
            raise
        return pd.read_csv(path, engine="c")


def _numeric_age_token(token: str) -> float:
    return float(token.replace(",", ""))


def _same_magnitude_range(left: float, right: float, right_token: str) -> tuple[float, float]:
    if "," not in right_token or left <= 0 or right <= 0 or left >= 1000 or right < 1000:
        return left, right
    scale = 10 ** max(0, int(math.floor(math.log10(right))) - int(math.floor(math.log10(left))))
    return left * scale, right


def numeric_age_values(text: str) -> list[float]:
    number = r"\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"
    range_match = re.search(
        rf"(?<![\d.+-])({number})\s*(?:-|to|through|–|—)\s*({number})(?![\d.])",
        text,
    )
    if range_match:
        left = _numeric_age_token(range_match.group(1))
        right = _numeric_age_token(range_match.group(2))
        left, right = _same_magnitude_range(left, right, range_match.group(2))
        return [left, right]
    return [_numeric_age_token(match.group(0)) for match in re.finditer(rf"[-+]?{number}", text)]


def parse_age_ka_value(value, large_number_scale="auto") -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        parsed = [parse_age_ka_value(item, large_number_scale) for item in value]
        parsed = [item for item in parsed if np.isfinite(item)]
        return float(np.mean(parsed)) if parsed else np.nan

    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
        if not np.isfinite(out):
            return np.nan
        if large_number_scale == "years" or (large_number_scale == "auto" and out > 1000.0):
            return out / 1000.0
        return out

    text = str(value).strip().strip('"').strip("'")
    if not text or text.lower() in {"nan", "none", "null", "unknown"}:
        return np.nan

    lowered = text.lower()
    numbers = numeric_age_values(lowered)
    if not numbers:
        return np.nan

    out = float(np.mean(numbers))
    looks_like_years = any(token in lowered for token in [" bp", "yr", "year", "cal"])
    looks_like_million = any(token in lowered for token in [" ma", "million"])
    looks_like_ka = any(token in lowered for token in [" ka", "kyr", "kya"])

    if looks_like_million:
        return out * 1000.0
    if large_number_scale == "years":
        return out / 1000.0
    if large_number_scale == "ka":
        return out
    if looks_like_years and not looks_like_ka:
        return out / 1000.0
    if out > 1000.0:
        return out / 1000.0
    return out


def age_model_value(value, large_number_scale="auto") -> float:
    if pd.isna(value):
        return np.nan
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return np.nan
    if not isinstance(payload, list):
        payload = [payload]

    ages = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        for key in ("age_ka", "age", "age_bp"):
            parsed = parse_age_ka_value(item.get(key), large_number_scale)
            if np.isfinite(parsed):
                if key == "age_bp" and parsed > 1000.0:
                    parsed = parsed / 1000.0
                ages.append(parsed)
                break
    if not ages:
        return np.nan
    return float(np.mean(ages))


def choose_time_ka(df, large_number_scale="auto"):
    out = df.copy()
    time_ka = pd.Series(np.nan, index=out.index, dtype=float)
    source = pd.Series("", index=out.index, dtype=object)

    if "age_ka" in out.columns:
        parsed = out["age_ka"].map(lambda value: parse_age_ka_value(value, large_number_scale))
        mask = parsed.notna()
        time_ka.loc[mask] = parsed.loc[mask]
        source.loc[mask] = "age_ka"

    if "age_models" in out.columns:
        parsed = out["age_models"].map(lambda value: age_model_value(value, large_number_scale))
        mask = time_ka.isna() & parsed.notna()
        time_ka.loc[mask] = parsed.loc[mask]
        source.loc[mask] = "age_models"

    out["time_ka"] = time_ka
    out["time_source"] = source
    return out


def load_records(path, large_number_scale="auto", time_min_ka=None, time_max_ka=None):
    df = read_input_csv(path)
    df = choose_reported_z(df)
    df = choose_time_ka(df, large_number_scale=large_number_scale)

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["z_m"] = pd.to_numeric(df["z_m"], errors="coerce")
    df["time_ka"] = pd.to_numeric(df["time_ka"], errors="coerce")

    out = df[
        reported_depth_source_mask(df)
        & df["latitude"].notna()
        & df["longitude"].notna()
        & df["z_m"].notna()
        & df["time_ka"].notna()
    ].copy()

    if time_min_ka is not None:
        out = out[out["time_ka"] >= time_min_ka].copy()
    if time_max_ka is not None:
        out = out[out["time_ka"] <= time_max_ka].copy()

    out["longitude"] = normalize_longitude(out["longitude"])
    return out.reset_index(drop=True)


def build_candidate_pole_grid(alpha_min, alpha_max, alpha_step, azimuth_step):
    alphas = np.arange(alpha_min, alpha_max + 0.5 * alpha_step, alpha_step)
    azimuths = np.arange(0.0, 360.0, azimuth_step)

    aa, pp = np.meshgrid(alphas, azimuths, indexing="ij")
    alpha_flat = aa.ravel()
    azimuth_flat = pp.ravel()

    poles = displaced_poles(alpha_flat, azimuth_flat)
    pole_lat, pole_lon = unit_to_latlon(poles)

    return pd.DataFrame(
        {
            "offset_deg": alpha_flat,
            "orientation_azimuth_deg": azimuth_flat,
            "candidate_pole_lat": pole_lat,
            "candidate_pole_lon": pole_lon,
            "px": poles[:, 0],
            "py": poles[:, 1],
            "pz": poles[:, 2],
        }
    )


def score_dataset(records, pole_grid, bulge_m, sigma_m, chunk_size):
    site_xyz = latlon_to_unit(
        records["latitude"].to_numpy(float),
        records["longitude"].to_numpy(float),
    )
    target_z = records["z_m"].to_numpy(float)
    weights = records["weight"].to_numpy(float)

    weight_total = float(np.sum(weights))
    p0_dot_sq = site_xyz[:, 2] ** 2
    poles = pole_grid[["px", "py", "pz"]].to_numpy(float)

    pref_sum = np.zeros(len(poles), dtype=float)
    hit_count = np.zeros(len(poles), dtype=float)
    mae_sum = np.zeros(len(poles), dtype=float)
    rmse_sum = np.zeros(len(poles), dtype=float)

    for start in range(0, len(poles), chunk_size):
        end = min(start + chunk_size, len(poles))
        block = poles[start:end]

        p1_dot = site_xyz @ block.T
        predicted_z = -bulge_m * (p0_dot_sq[:, None] - p1_dot**2)
        residual = predicted_z - target_z[:, None]
        abs_residual = np.abs(residual)
        score = np.exp(-0.5 * (residual / sigma_m) ** 2)

        w = weights[:, None]
        pref_sum[start:end] = (score * w).sum(axis=0)
        hit_count[start:end] = ((abs_residual <= sigma_m) * w).sum(axis=0)
        mae_sum[start:end] = (abs_residual * w).sum(axis=0)
        rmse_sum[start:end] = (np.square(residual) * w).sum(axis=0)

    out = pole_grid.copy()
    out["preference_raw"] = pref_sum
    out["preference_norm"] = pref_sum / max(weight_total, 1.0)
    out["hit_count_within_sigma"] = hit_count
    out["hit_fraction_within_sigma"] = hit_count / max(weight_total, 1.0)
    out["mae_m"] = mae_sum / max(weight_total, 1.0)
    out["rmse_m"] = np.sqrt(rmse_sum / max(weight_total, 1.0))

    max_pref = float(out["preference_norm"].max())
    out["preference_0_1"] = out["preference_norm"] / max_pref if max_pref > 0 else 0.0

    return out


def spherical_centroid(lat, lon, weights=None):
    vectors = latlon_to_unit(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
    if weights is None:
        mean = vectors.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=float)
        if np.sum(w) <= 0:
            mean = vectors.mean(axis=0)
        else:
            mean = np.average(vectors, axis=0, weights=w)
    if np.linalg.norm(mean) == 0:
        return float(np.mean(lat)), float(np.mean(lon))
    out_lat, out_lon = unit_to_latlon(mean.reshape(1, 3))
    return float(out_lat[0]), float(out_lon[0])


def summarize_grid(regime_name, records, grid, time_min_ka, time_max_ka, args):
    best = grid.sort_values("preference_0_1", ascending=False).iloc[0]

    q90 = grid[grid["preference_0_1"] >= grid["preference_0_1"].quantile(0.90)]
    q95 = grid[grid["preference_0_1"] >= grid["preference_0_1"].quantile(0.95)]
    q99 = grid[grid["preference_0_1"] >= grid["preference_0_1"].quantile(0.99)]

    c90 = spherical_centroid(q90["candidate_pole_lat"], q90["candidate_pole_lon"], q90["preference_0_1"])
    c95 = spherical_centroid(q95["candidate_pole_lat"], q95["candidate_pole_lon"], q95["preference_0_1"])
    c99 = spherical_centroid(q99["candidate_pole_lat"], q99["candidate_pole_lon"], q99["preference_0_1"])

    return {
        "time_bin": regime_name,
        "time_min_ka": float(time_min_ka),
        "time_max_ka": float(time_max_ka),
        "n_records": int(len(records)),
        "weight_total": float(records["weight"].sum()),
        "time_mean_ka": float(records["time_ka"].mean()) if len(records) else np.nan,
        "time_median_ka": float(records["time_ka"].median()) if len(records) else np.nan,
        "z_mean": float(records["z_m"].mean()) if len(records) else np.nan,
        "z_median": float(records["z_m"].median()) if len(records) else np.nan,
        "best_offset_deg": float(best["offset_deg"]),
        "best_azimuth_deg": float(best["orientation_azimuth_deg"]),
        "best_pole_lat": float(best["candidate_pole_lat"]),
        "best_pole_lon": float(best["candidate_pole_lon"]),
        "preference_norm": float(best["preference_norm"]),
        "preference_0_1": float(best["preference_0_1"]),
        "hit_fraction_within_sigma": float(best["hit_fraction_within_sigma"]),
        "mae_m": float(best["mae_m"]),
        "rmse_m": float(best["rmse_m"]),
        "top_10pct_centroid_lat": c90[0],
        "top_10pct_centroid_lon": c90[1],
        "top_5pct_centroid_lat": c95[0],
        "top_5pct_centroid_lon": c95[1],
        "top_1pct_centroid_lat": c99[0],
        "top_1pct_centroid_lon": c99[1],
    }


def parse_time_bins(text):
    bins = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        name, bounds = part.split(":")
        time_min, time_max = bounds.split(",")
        bins.append((name.strip(), float(time_min), float(time_max)))
    return bins


def make_time_bin_label(prefix, time_min_ka, time_max_ka):
    return f"{prefix}_{time_min_ka:g}_{time_max_ka:g}ka"


def time_bin_mask(values, time_min_ka, time_max_ka, include_right_edge=False):
    if include_right_edge:
        return (values >= time_min_ka) & (values <= time_max_ka)
    return (values >= time_min_ka) & (values < time_max_ka)


def build_fixed_time_bins(records, width_ka, min_records):
    if width_ka <= 0:
        raise ValueError("--fixed-bin-width-ka must be positive.")
    start = math.floor(float(records["time_ka"].min()) / width_ka) * width_ka
    stop = math.ceil(float(records["time_ka"].max()) / width_ka) * width_ka
    bins = []
    cur = start
    while cur < stop:
        nxt = min(cur + width_ka, stop)
        count = int(time_bin_mask(records["time_ka"], cur, nxt, include_right_edge=nxt >= stop).sum())
        if count >= min_records:
            bins.append((make_time_bin_label("fixed", cur, nxt), cur, nxt))
        cur = nxt
    return bins


def build_auto_time_bins(records, min_records, auto_max_bins, auto_target_records=None):
    ages = np.sort(records["time_ka"].dropna().to_numpy(float))
    if len(ages) < min_records:
        return []

    if auto_target_records is None:
        auto_target_records = max(min_records, int(math.ceil(math.sqrt(len(ages)))))
    auto_target_records = max(min_records, int(auto_target_records))
    n_bins = max(1, int(math.ceil(len(ages) / auto_target_records)))
    n_bins = min(n_bins, max(1, int(auto_max_bins)))

    edges = np.quantile(ages, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = ages[0]
    edges[-1] = ages[-1]

    bins = []
    edge_pairs = list(zip(edges[:-1], edges[1:]))
    for idx, (left, right) in enumerate(edge_pairs, start=1):
        if right <= left:
            continue
        count = int(
            time_bin_mask(
                records["time_ka"],
                left,
                right,
                include_right_edge=idx == len(edge_pairs),
            ).sum()
        )
        if count < min_records:
            continue
        label = f"auto_{idx:02d}_{left:g}_{right:g}ka"
        bins.append((label, float(left), float(right)))

    return bins


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


def pole_lat_extent(grid, pad_deg=0.0):
    lat = pd.to_numeric(grid["candidate_pole_lat"], errors="coerce")
    lat = lat[np.isfinite(lat)]
    if lat.empty:
        return 30.0, 90.0
    lat_min = max(-90.0, float(lat.min()) - pad_deg)
    lat_max = min(90.0, float(lat.max()) + pad_deg)
    return lat_min, max(lat_min + 0.1, lat_max)


def alpha_lat_extent(alpha_min, alpha_max, pad_deg=0.0):
    lat_min = max(-90.0, 90.0 - float(alpha_max) - pad_deg)
    lat_max = min(90.0, 90.0 - float(alpha_min) + pad_deg)
    return lat_min, max(lat_min + 0.1, lat_max)


def overprint_map_outlines(ax, cfeature, scale="50m", linewidth=0.5):
    ax.add_feature(
        cfeature.COASTLINE.with_scale(scale),
        edgecolor=MAP_OUTLINE_COLOR,
        linewidth=linewidth,
        zorder=20,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale(scale),
        edgecolor=MAP_OUTLINE_COLOR,
        linewidth=linewidth * 0.75,
        alpha=0.85,
        zorder=20,
    )


def make_offset_azimuth_heatmap(grid, out_path, title):
    import matplotlib.pyplot as plt

    pivot = grid.pivot_table(
        index="offset_deg",
        columns="orientation_azimuth_deg",
        values="preference_0_1",
        aggfunc="mean",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[
            float(pivot.columns.min()),
            float(pivot.columns.max()),
            float(pivot.index.min()),
            float(pivot.index.max()),
        ],
        cmap="inferno",
        interpolation="nearest",
    )

    ax.set_xlabel("Orientation azimuth, degrees")
    ax.set_ylabel("Polar offset, degrees")
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Normalized polar path preference, 0–1")

    plt.tight_layout()
    plt.savefig(out_path, dpi=260)
    plt.close(fig)


def make_polar_preference_map(grid, summary, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    data_crs = ccrs.PlateCarree()
    proj = ccrs.NorthPolarStereo()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=proj)
    lat_min, lat_max = pole_lat_extent(grid)
    ax.set_extent([-180, 180, lat_min, lat_max], crs=data_crs)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#eeeeea", edgecolor="none")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f8fbff", edgecolor="none")

    sc = ax.scatter(
        grid["candidate_pole_lon"],
        grid["candidate_pole_lat"],
        c=grid["preference_0_1"],
        s=12,
        transform=data_crs,
        cmap="inferno",
        norm=PowerNorm(gamma=0.45, vmin=0, vmax=1),
        linewidths=0,
        alpha=0.9,
        rasterized=True,
    )

    ax.scatter(
        [summary["best_pole_lon"]],
        [summary["best_pole_lat"]],
        transform=data_crs,
        s=180,
        marker="*",
        c="cyan",
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
    )

    ax.scatter(
        [summary["top_5pct_centroid_lon"]],
        [summary["top_5pct_centroid_lat"]],
        transform=data_crs,
        s=120,
        marker="X",
        c="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
    )
    overprint_map_outlines(ax, cfeature, linewidth=0.45)

    gl = ax.gridlines(draw_labels=True, linewidth=0.35, alpha=0.35, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.72, pad=0.06)
    cbar.set_label("Normalized polar path preference, gamma-stretched")

    ax.set_title(
        f"LITTORAL 16 - {summary['time_bin']} polar preference\n"
        f"best offset={summary['best_offset_deg']:.2f}°, azimuth={summary['best_azimuth_deg']:.2f}°",
        fontsize=13,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=260)
    plt.close(fig)


def make_best_poles_map(summary_df, out_path, projection_name, alpha_min, alpha_max):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    data_crs = ccrs.PlateCarree()
    proj = projection(projection_name, ccrs)

    fig = plt.figure(figsize=(15, 8.5))
    ax = plt.axes(projection=proj)
    lat_min, lat_max = alpha_lat_extent(alpha_min, alpha_max, pad_deg=1.0)
    ax.set_extent([-180, 180, lat_min, lat_max], crs=data_crs)
    ax.set_facecolor("#f7f7f4")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#eeeeea", edgecolor="none")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f8fbff", edgecolor="none")

    sizes = 80 + 28 * summary_df["best_offset_deg"].to_numpy(float)

    sc = ax.scatter(
        summary_df["best_pole_lon"],
        summary_df["best_pole_lat"],
        c=summary_df["best_offset_deg"],
        s=sizes,
        transform=data_crs,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
    )

    for _, row in summary_df.iterrows():
        ax.text(
            row["best_pole_lon"],
            row["best_pole_lat"],
            f" {row['time_bin']} ({row['best_offset_deg']:.2f}°)",
            transform=data_crs,
            fontsize=8,
            zorder=11,
        )
    overprint_map_outlines(ax, cfeature, linewidth=0.55)

    gl = ax.gridlines(draw_labels=True, linewidth=0.35, alpha=0.32, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.78, pad=0.06)
    cbar.set_label("Best polar offset by time bin, degrees")

    ax.set_title("LITTORAL 16 - best candidate poles by time bin", fontsize=15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=260)
    plt.close(fig)


def make_composite_plot(summary_df, grid_by_regime, pair_df, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import PowerNorm
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    regimes = summary_df["time_bin"].tolist()
    n_regimes = len(regimes)
    n_cols = n_regimes

    fig = plt.figure(figsize=(4.8 * n_cols + 0.9, 10.0))
    gs = GridSpec(
        2,
        n_cols + 1,
        figure=fig,
        width_ratios=[1.0] * n_cols + [0.045],
        height_ratios=[1.0, 1.08],
        hspace=0.34,
        wspace=0.22,
    )

    heatmap_image = None
    polar_image = None
    data_crs = ccrs.PlateCarree()
    for idx, regime in enumerate(regimes):
        ax = fig.add_subplot(gs[0, idx])
        grid = grid_by_regime[regime]
        summary = summary_df.loc[summary_df["time_bin"] == regime].iloc[0]

        pivot = grid.pivot_table(
            index="offset_deg",
            columns="orientation_azimuth_deg",
            values="preference_0_1",
            aggfunc="mean",
        ).sort_index()

        heatmap_image = ax.imshow(
            pivot.values,
            origin="lower",
            aspect="auto",
            extent=[
                float(pivot.columns.min()),
                float(pivot.columns.max()),
                float(pivot.index.min()),
                float(pivot.index.max()),
            ],
            cmap="inferno",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.scatter(
            [summary["best_azimuth_deg"]],
            [summary["best_offset_deg"]],
            marker="*",
            s=95,
            c="cyan",
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
        )
        ax.set_title(
            f"{regime}: n={int(summary['n_records'])}, "
            f"best={summary['best_offset_deg']:.2f}°/{summary['best_azimuth_deg']:.1f}°",
            fontsize=9.5,
        )
        ax.set_xlabel("Azimuth, degrees")
        ax.set_ylabel("Offset, degrees")

        polar_ax = fig.add_subplot(gs[1, idx], projection=ccrs.NorthPolarStereo())
        lat_min, lat_max = pole_lat_extent(grid)
        polar_ax.set_extent([-180, 180, lat_min, lat_max], crs=data_crs)
        polar_ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#eeeeea", edgecolor="none")
        polar_ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f8fbff", edgecolor="none")

        polar_image = polar_ax.scatter(
            grid["candidate_pole_lon"],
            grid["candidate_pole_lat"],
            c=grid["preference_0_1"],
            s=8,
            transform=data_crs,
            cmap="inferno",
            norm=PowerNorm(gamma=0.45, vmin=0, vmax=1),
            linewidths=0,
            alpha=0.9,
            rasterized=True,
        )
        polar_ax.scatter(
            [summary["best_pole_lon"]],
            [summary["best_pole_lat"]],
            transform=data_crs,
            s=115,
            marker="*",
            c="cyan",
            edgecolors="black",
            linewidths=0.6,
            zorder=10,
        )
        polar_ax.scatter(
            [summary["top_5pct_centroid_lon"]],
            [summary["top_5pct_centroid_lat"]],
            transform=data_crs,
            s=70,
            marker="X",
            c="white",
            edgecolors="black",
            linewidths=0.6,
            zorder=10,
        )
        overprint_map_outlines(polar_ax, cfeature, linewidth=0.35)
        gl = polar_ax.gridlines(draw_labels=False, linewidth=0.25, alpha=0.32, linestyle="--")
        polar_ax.set_title(f"{regime} polar preference", fontsize=9.5)

    if heatmap_image is not None:
        cax = fig.add_subplot(gs[0, n_cols])
        cbar = fig.colorbar(
            heatmap_image,
            cax=cax,
            orientation="vertical",
        )
        cbar.set_label("Offset–azimuth normalized preference, 0–1")

    if polar_image is not None:
        cax = fig.add_subplot(gs[1, n_cols])
        cbar = fig.colorbar(
            polar_image,
            cax=cax,
            orientation="vertical",
        )
        cbar.set_label("North-polar preference, gamma-stretched")

    fig.suptitle(
        "LITTORAL 16 - time-regime inverse composite\n"
        "offset–azimuth parameter space above; north-polar candidate preference below",
        fontsize=15,
        y=0.995,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--projection", default=DEFAULT_PROJECTION)

    parser.add_argument(
        "--time-bins",
        default=None,
        help='Fixed named windows like "early:0,20;mid:20,80;late:80,250", in ka.',
    )
    parser.add_argument(
        "--fixed-bin-width-ka",
        type=float,
        default=None,
        help="Use equal-width time bins of this width in ka instead of autodynamic bins.",
    )
    parser.add_argument("--time-min-ka", type=float, default=None)
    parser.add_argument("--time-max-ka", type=float, default=None)
    parser.add_argument("--auto-max-bins", type=int, default=8)
    parser.add_argument(
        "--auto-target-records",
        type=int,
        default=None,
        help="Target records per autodynamic bin. Defaults to sqrt(n), bounded by --min-records.",
    )
    parser.add_argument(
        "--large-age-number-scale",
        choices=["auto", "years", "ka"],
        default="auto",
        help="How to treat age numbers above 1000 when parsing text/numeric age fields.",
    )

    parser.add_argument("--alpha-min", type=float, default=0.1)
    parser.add_argument("--alpha-max", type=float, default=45.0)
    parser.add_argument("--alpha-step", type=float, default=0.25)
    parser.add_argument("--azimuth-step", type=float, default=0.5)

    parser.add_argument("--sigma-m", type=float, default=25.0)
    parser.add_argument("--bulge-m", type=float, default=EARTH_BULGE_M)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument("--min-records", type=int, default=3)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSV and JSON outputs without rendering Cartopy/Matplotlib figures.",
    )

    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records = load_records(
        args.input,
        large_number_scale=args.large_age_number_scale,
        time_min_ka=args.time_min_ka,
        time_max_ka=args.time_max_ka,
    )

    if args.time_bins:
        regimes = parse_time_bins(args.time_bins)
        binning_mode = "fixed_named"
    elif args.fixed_bin_width_ka:
        regimes = build_fixed_time_bins(all_records, args.fixed_bin_width_ka, args.min_records)
        binning_mode = "fixed_width"
    else:
        regimes = build_auto_time_bins(
            all_records,
            min_records=args.min_records,
            auto_max_bins=args.auto_max_bins,
            auto_target_records=args.auto_target_records,
        )
        binning_mode = "autodynamic"

    if not regimes:
        raise SystemExit("No time bins could be built from the available dated records.")

    pole_grid = build_candidate_pole_grid(
        args.alpha_min,
        args.alpha_max,
        args.alpha_step,
        args.azimuth_step,
    )

    summaries = []
    grids_written = []
    grid_by_regime = {}

    for idx, (regime_name, time_min_ka, time_max_ka) in enumerate(regimes):
        records = all_records[
            time_bin_mask(
                all_records["time_ka"],
                time_min_ka,
                time_max_ka,
                include_right_edge=idx == len(regimes) - 1,
            )
        ].copy()

        if len(records) < args.min_records:
            print(f"Skipping {regime_name}: only {len(records)} records.")
            continue

        records["weight"] = 1.0

        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in regime_name)

        records_path = OUT_DIR / f"16_{safe}_records.csv"
        grid_path = OUT_DIR / f"16_{safe}_preference_grid.csv"
        top_path = OUT_DIR / f"16_{safe}_top_candidates.csv"
        heatmap_path = OUT_DIR / f"16_{safe}_offset_azimuth_heatmap.png"
        polar_path = OUT_DIR / f"16_{safe}_polar_preference.png"

        scored = score_dataset(
            records=records,
            pole_grid=pole_grid,
            bulge_m=args.bulge_m,
            sigma_m=args.sigma_m,
            chunk_size=args.chunk_size,
        )

        summary = summarize_grid(regime_name, records, scored, time_min_ka, time_max_ka, args)
        summaries.append(summary)
        grid_by_regime[regime_name] = scored

        records.to_csv(records_path, index=False)
        scored.to_csv(grid_path, index=False)
        scored.sort_values("preference_0_1", ascending=False).head(args.top_n).to_csv(top_path, index=False)

        if not args.skip_plots:
            make_offset_azimuth_heatmap(
                scored,
                heatmap_path,
                f"LITTORAL 16 - {regime_name} offset-azimuth preference",
            )

            make_polar_preference_map(
                scored,
                summary,
                polar_path,
            )

        grids_written.append(
            {
                "time_bin": regime_name,
                "time_min_ka": float(time_min_ka),
                "time_max_ka": float(time_max_ka),
                "records_csv": str(records_path),
                "grid_csv": str(grid_path),
                "top_candidates_csv": str(top_path),
                "offset_azimuth_heatmap": str(heatmap_path) if not args.skip_plots else None,
                "polar_preference_png": str(polar_path) if not args.skip_plots else None,
            }
        )

        print(
            f"{regime_name}: n={len(records)}, "
            f"time={time_min_ka:.3g}-{time_max_ka:.3g} ka, "
            f"best offset={summary['best_offset_deg']:.3f}°, "
            f"azimuth={summary['best_azimuth_deg']:.3f}°, "
            f"pole=({summary['best_pole_lat']:.3f}, {summary['best_pole_lon']:.3f}), "
            f"hit={summary['hit_fraction_within_sigma']:.4f}, "
            f"MAE={summary['mae_m']:.2f} m"
        )

    if not summaries:
        raise SystemExit("No time bins had enough records to process.")

    summary_df = pd.DataFrame(summaries)
    summary_csv = OUT_DIR / "16_time_bin_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    pair_rows = []
    for i, a in summary_df.iterrows():
        for j, b in summary_df.iterrows():
            if j <= i:
                continue
            pair_rows.append(
                {
                    "time_bin_a": a["time_bin"],
                    "time_bin_b": b["time_bin"],
                    "time_min_ka_a": a["time_min_ka"],
                    "time_max_ka_a": a["time_max_ka"],
                    "time_min_ka_b": b["time_min_ka"],
                    "time_max_ka_b": b["time_max_ka"],
                    "best_pole_angular_separation_deg": angular_distance_deg(
                        a["best_pole_lat"],
                        a["best_pole_lon"],
                        b["best_pole_lat"],
                        b["best_pole_lon"],
                    ),
                    "offset_difference_deg": abs(a["best_offset_deg"] - b["best_offset_deg"]),
                    "azimuth_difference_deg_raw": abs(a["best_azimuth_deg"] - b["best_azimuth_deg"]),
                    "mae_difference_m": abs(a["mae_m"] - b["mae_m"]),
                }
            )

    pair_df = pd.DataFrame(pair_rows)
    pair_csv = OUT_DIR / "16_time_bin_pairwise_angular_separation.csv"
    pair_df.to_csv(pair_csv, index=False)

    best_map = OUT_DIR / "16_time_bin_best_poles_map.png"
    composite_plot = OUT_DIR / "16_time_regime_composite.png"
    if not args.skip_plots:
        make_best_poles_map(summary_df, best_map, args.projection, args.alpha_min, args.alpha_max)
        make_composite_plot(summary_df, grid_by_regime, pair_df, composite_plot)

    summary_json = OUT_DIR / "16_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_csv": str(args.input),
                "output_dir": str(OUT_DIR),
                "binning_mode": binning_mode,
                "time_min_ka": args.time_min_ka,
                "time_max_ka": args.time_max_ka,
                "time_bins": [
                    {"time_bin": name, "time_min_ka": float(left), "time_max_ka": float(right)}
                    for name, left, right in regimes
                ],
                "large_age_number_scale": args.large_age_number_scale,
                "alpha_min_deg": args.alpha_min,
                "alpha_max_deg": args.alpha_max,
                "alpha_step_deg": args.alpha_step,
                "azimuth_step_deg": args.azimuth_step,
                "sigma_m": args.sigma_m,
                "bulge_m": args.bulge_m,
                "skip_plots": args.skip_plots,
                "time_bin_summary_csv": str(summary_csv),
                "pairwise_angular_separation_csv": str(pair_csv),
                "best_poles_map_png": str(best_map) if not args.skip_plots else None,
                "composite_plot_png": str(composite_plot) if not args.skip_plots else None,
                "time_bin_outputs": grids_written,
                "summaries": summaries,
            },
            f,
            indent=2,
        )

    print("\nLITTORAL 16 time-regime inverse complete.")
    print(f"Output: {OUT_DIR}")
    print(f"Summary: {summary_csv}")
    print(f"Pairwise separations: {pair_csv}")
    if not args.skip_plots:
        print(f"Composite plot: {composite_plot}")


if __name__ == "__main__":
    main()
