#!/usr/bin/env python3
"""
14_littoral_dataset_geodetic_path_preference.py

Run the geodetic orientation inverse over the full merged LITTORAL dataset,
using only geocoded records with reported elevations/depths in the -200 to
+200 m range, then plot normalized polar-path preference diagnostics.

Reads by default:
  outputs/merged/master_dataset.csv

Writes:
  outputs/geospatial_14/14_polar_path_preference_grid.csv
  outputs/geospatial_14/14_polar_path_preference_top_candidates.csv
  outputs/geospatial_14/14_polar_preference_north_polar.png
  outputs/geospatial_14/14_offset_azimuth_preference_heatmap.png
  outputs/geospatial_14/14_top_candidate_overview.png
  outputs/geospatial_14/14_polar_path_preference_summary.json
  outputs/geospatial_14/14_filtered_reported_records.csv
  outputs/geospatial_14/14_depth_bin_summary.csv

Run:
  python scripts/14_littoral_dataset_geodetic_path_preference.py

Example finer run:
  python scripts/14_littoral_dataset_geodetic_path_preference.py \
    --alpha-min 0.1 --alpha-max 45 --alpha-step 0.1 \
    --azimuth-step 0.25 --sigma-m 25
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "merged" / "master_dataset.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "geospatial_14"

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "littoral_matplotlib"),
)

EARTH_BULGE_M = 11000.0
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


def reported_depth_source_mask(df: pd.DataFrame) -> pd.Series:
    if "depth_source" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["depth_source"].map(
        lambda value: str(value).strip().lower() == "reported"
        if not pd.isna(value)
        else False
    )


def choose_reported_z(df: pd.DataFrame) -> pd.DataFrame:
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


def load_reported_records(path: Path, z_min: float, z_max: float) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df = choose_reported_z(df)

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["z_m"] = pd.to_numeric(df["z_m"], errors="coerce")

    out = df[
        reported_depth_source_mask(df)
        & df["latitude"].notna()
        & df["longitude"].notna()
        & df["z_m"].notna()
        & (df["z_m"] >= z_min)
        & (df["z_m"] <= z_max)
    ].copy()

    out["longitude"] = normalize_longitude(out["longitude"])
    return out.reset_index(drop=True)


def parse_depth_bins(value: str) -> list[float]:
    try:
        bins = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--depth-bins must be a comma-separated list of numeric bin edges"
        ) from exc
    if len(bins) < 2:
        raise argparse.ArgumentTypeError("--depth-bins must contain at least two edges")
    if any(b <= a for a, b in zip(bins, bins[1:])):
        raise argparse.ArgumentTypeError("--depth-bins must be strictly increasing")
    return bins


def normalize_depth_bin_argv(argv: list[str]) -> list[str]:
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == "--depth-bins" and i + 1 < len(argv):
            out.append(f"--depth-bins={argv[i + 1]}")
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def assign_record_weights(records: pd.DataFrame, weight_mode: str, depth_bins: list[float]) -> pd.DataFrame:
    out = records.copy()
    out["z_bin"] = pd.cut(out["z_m"], bins=depth_bins, include_lowest=True)

    if weight_mode == "uniform":
        out["weight"] = 1.0
        return out.reset_index(drop=True)

    if weight_mode != "depth-balanced":
        raise ValueError(f"Unsupported weight mode: {weight_mode}")

    out = out[out["z_bin"].notna()].copy()
    if out.empty:
        raise SystemExit("No reported records remain after applying depth bins.")

    bin_counts = out.groupby("z_bin", observed=True)["z_m"].transform("size").astype(float)
    out["weight"] = 1.0 / bin_counts
    mean_weight = float(out["weight"].mean())
    if mean_weight > 0:
        out["weight"] = out["weight"] / mean_weight

    return out.reset_index(drop=True)


def depth_bin_summary(records: pd.DataFrame) -> pd.DataFrame:
    if "z_bin" not in records.columns:
        return pd.DataFrame(
            columns=["z_bin", "n_records", "weight_sum", "z_min", "z_max", "z_mean", "z_median"]
        )

    grouped = (
        records[records["z_bin"].notna()]
        .groupby("z_bin", observed=True)
        .agg(
            n_records=("z_m", "size"),
            weight_sum=("weight", "sum"),
            z_min=("z_m", "min"),
            z_max=("z_m", "max"),
            z_mean=("z_m", "mean"),
            z_median=("z_m", "median"),
        )
        .reset_index()
    )
    grouped["z_bin"] = grouped["z_bin"].astype(str)
    return grouped


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
    w = records["weight"].to_numpy(float)
    weight_total = float(w.sum())
    if weight_total <= 0:
        raise SystemExit("Record weights must have positive total weight.")

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

        pref_sum[start:end] = (score * w[:, None]).sum(axis=0)
        hit_count[start:end] = ((abs_residual <= sigma_m) * w[:, None]).sum(axis=0)
        mae_sum[start:end] = (abs_residual * w[:, None]).sum(axis=0)
        rmse_sum[start:end] = (np.square(residual) * w[:, None]).sum(axis=0)

    out = pole_grid.copy()
    out["preference_raw"] = pref_sum
    out["preference_norm"] = pref_sum / weight_total
    out["hit_count_within_sigma"] = hit_count
    out["hit_fraction_within_sigma"] = hit_count / weight_total
    out["mae_m"] = mae_sum / weight_total
    out["rmse_m"] = np.sqrt(rmse_sum / weight_total)

    max_pref = float(out["preference_norm"].max())
    if max_pref > 0:
        out["preference_0_1"] = out["preference_norm"] / max_pref
    else:
        out["preference_0_1"] = 0.0

    return out


def spherical_centroid(lat, lon, weights=None):
    vectors = latlon_to_unit(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
    if weights is None:
        mean = vectors.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=float)
        mean = np.average(vectors, axis=0, weights=w)
    if np.linalg.norm(mean) == 0:
        return float(np.mean(lat)), float(np.mean(lon))
    out_lat, out_lon = unit_to_latlon(mean.reshape(1, 3))
    return float(out_lat[0]), float(out_lon[0])


def summarize_preference(grid, records, args, depth_bins, depth_bin_summary_csv):
    best = grid.sort_values("preference_0_1", ascending=False).iloc[0]
    q90 = grid[grid["preference_0_1"] >= grid["preference_0_1"].quantile(0.90)]
    q95 = grid[grid["preference_0_1"] >= grid["preference_0_1"].quantile(0.95)]
    q99 = grid[grid["preference_0_1"] >= grid["preference_0_1"].quantile(0.99)]

    c90 = spherical_centroid(q90["candidate_pole_lat"], q90["candidate_pole_lon"], q90["preference_0_1"])
    c95 = spherical_centroid(q95["candidate_pole_lat"], q95["candidate_pole_lon"], q95["preference_0_1"])
    c99 = spherical_centroid(q99["candidate_pole_lat"], q99["candidate_pole_lon"], q99["preference_0_1"])

    return {
        "input_csv": str(args.input),
        "n_reported_records": int(len(records)),
        "z_min_m": float(args.z_min),
        "z_max_m": float(args.z_max),
        "alpha_min_deg": float(args.alpha_min),
        "alpha_max_deg": float(args.alpha_max),
        "alpha_step_deg": float(args.alpha_step),
        "azimuth_step_deg": float(args.azimuth_step),
        "sigma_m": float(args.sigma_m),
        "bulge_m": float(args.bulge_m),
        "weight_mode": args.weight_mode,
        "depth_bins": [float(edge) for edge in depth_bins],
        "n_records_after_bin_filter": int(len(records)),
        "depth_bin_summary_csv": str(depth_bin_summary_csv) if depth_bin_summary_csv else None,
        "best_candidate": {
            key: float(best[key])
            for key in [
                "offset_deg",
                "orientation_azimuth_deg",
                "candidate_pole_lat",
                "candidate_pole_lon",
                "preference_norm",
                "preference_0_1",
                "hit_fraction_within_sigma",
                "mae_m",
                "rmse_m",
            ]
        },
        "top_10pct_weighted_centroid": {
            "lat": c90[0],
            "lon": c90[1],
        },
        "top_5pct_weighted_centroid": {
            "lat": c95[0],
            "lon": c95[1],
        },
        "top_1pct_weighted_centroid": {
            "lat": c99[0],
            "lon": c99[1],
        },
    }


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


def make_heatmap(grid, records, summary, out_path, projection_name):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
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
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.55, edgecolor="#202020")
    ax.add_feature(cfeature.LAKES.with_scale("110m"), facecolor="none", edgecolor="#444444", linewidth=0.2, alpha=0.35)

    sc = ax.scatter(
        grid["candidate_pole_lon"],
        grid["candidate_pole_lat"],
        c=grid["preference_0_1"],
        s=10,
        transform=data_crs,
        cmap="inferno",
        norm=Normalize(vmin=0, vmax=1),
        linewidths=0,
        alpha=0.88,
        rasterized=True,
        zorder=5,
    )

    best = summary["best_candidate"]
    ax.scatter(
        [best["candidate_pole_lon"]],
        [best["candidate_pole_lat"]],
        transform=data_crs,
        s=120,
        marker="*",
        c="cyan",
        edgecolors="black",
        linewidths=0.8,
        zorder=8,
        label="best pole-path preference",
    )

    c95 = summary["top_5pct_weighted_centroid"]
    ax.scatter(
        [c95["lon"]],
        [c95["lat"]],
        transform=data_crs,
        s=90,
        marker="X",
        c="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=8,
        label="top 5% weighted centroid",
    )

    if len(records) > 0:
        show = records
        if len(show) > 1500:
            show = show.sample(1500, random_state=42)
        ax.scatter(
            show["longitude"],
            show["latitude"],
            transform=data_crs,
            s=5,
            c="black",
            alpha=0.22,
            linewidths=0,
            zorder=6,
            label="reported records",
        )

    gl = ax.gridlines(
        crs=data_crs,
        draw_labels=True,
        linewidth=0.35,
        color="#555555",
        alpha=0.32,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.78, pad=0.06)
    cbar.set_label("Normalized polar path preference, 0–1")

    ax.legend(loc="lower left", frameon=True, framealpha=0.88, fontsize=8.5)

    ax.set_title(
        "14 LITTORAL normalized polar path preference\n"
        f"reported geocoded records: {summary['n_reported_records']}; "
        f"z range: {summary['z_min_m']:.0f} to {summary['z_max_m']:.0f} m; "
        f"sigma={summary['sigma_m']:.1f} m",
        fontsize=15,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close(fig)


def make_polar_preference_map(grid, summary, out_path, lat_min=35.0):
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    data_crs = ccrs.PlateCarree()
    proj = ccrs.NorthPolarStereo()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=proj)
    ax.set_extent([-180, 180, lat_min, 90], crs=data_crs)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#eeeeea", edgecolor="none")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f8fbff", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.45)

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

    best = summary["best_candidate"]
    c95 = summary["top_5pct_weighted_centroid"]

    ax.scatter(
        [best["candidate_pole_lon"]],
        [best["candidate_pole_lat"]],
        transform=data_crs,
        s=180,
        marker="*",
        c="cyan",
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
    )

    ax.scatter(
        [c95["lon"]],
        [c95["lat"]],
        transform=data_crs,
        s=120,
        marker="X",
        c="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.35, alpha=0.35, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.72, pad=0.06)
    cbar.set_label("Normalized polar path preference, gamma-stretched")

    ax.set_title(
        "LITTORAL 14 — candidate pole preference, north-polar view\n"
        f"best offset={best['offset_deg']:.2f}°, azimuth={best['orientation_azimuth_deg']:.2f}°",
        fontsize=13,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=260)
    plt.close(fig)


def make_offset_azimuth_heatmap(grid, out_path):
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
    ax.set_title("LITTORAL 14 — normalized preference in offset–azimuth parameter space")

    cbar = plt.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Normalized polar path preference, 0–1")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=260)
    plt.close(fig)


def make_top_candidate_overview(
    grid,
    records,
    summary,
    out_path,
    projection_name="robinson",
    top_quantile=0.95,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    data_crs = ccrs.PlateCarree()
    proj = projection(projection_name, ccrs)

    threshold = grid["preference_0_1"].quantile(top_quantile)
    top = grid[grid["preference_0_1"] >= threshold].copy()

    fig = plt.figure(figsize=(15, 8.5))
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.set_facecolor("#f7f7f4")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#eeeeea", edgecolor="none")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f8fbff", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.55)

    sc = ax.scatter(
        top["candidate_pole_lon"],
        top["candidate_pole_lat"],
        c=top["preference_0_1"],
        s=18,
        transform=data_crs,
        cmap="inferno",
        norm=Normalize(vmin=threshold, vmax=1),
        linewidths=0,
        alpha=0.85,
        rasterized=True,
        zorder=6,
    )

    if len(records) > 0:
        show = records if len(records) <= 1500 else records.sample(1500, random_state=42)
        ax.scatter(
            show["longitude"],
            show["latitude"],
            transform=data_crs,
            s=4,
            c="black",
            alpha=0.18,
            linewidths=0,
            zorder=5,
        )

    best = summary["best_candidate"]
    c95 = summary["top_5pct_weighted_centroid"]

    ax.scatter(
        [best["candidate_pole_lon"]],
        [best["candidate_pole_lat"]],
        transform=data_crs,
        s=130,
        marker="*",
        c="cyan",
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
        label="best candidate",
    )

    ax.scatter(
        [c95["lon"]],
        [c95["lat"]],
        transform=data_crs,
        s=95,
        marker="X",
        c="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
        label="top 5% centroid",
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.35, alpha=0.32, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.78, pad=0.06)
    cbar.set_label(f"Normalized preference, top {(1 - top_quantile) * 100:.1f}% only")

    ax.legend(loc="lower left", frameon=True, framealpha=0.88, fontsize=8.5)
    ax.set_title(
        "LITTORAL 14 — top-percentile polar path preference overview\n"
        f"showing candidates ≥ q{top_quantile:.2f}",
        fontsize=15,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=260)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--projection", default=DEFAULT_PROJECTION)

    parser.add_argument("--z-min", type=float, default=-200.0)
    parser.add_argument("--z-max", type=float, default=200.0)

    parser.add_argument("--alpha-min", type=float, default=0.1)
    parser.add_argument("--alpha-max", type=float, default=45.0)
    parser.add_argument("--alpha-step", type=float, default=0.25)
    parser.add_argument("--azimuth-step", type=float, default=0.5)

    parser.add_argument("--sigma-m", type=float, default=25.0)
    parser.add_argument("--bulge-m", type=float, default=EARTH_BULGE_M)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument("--plot-top-quantile", type=float, default=0.95)
    parser.add_argument("--polar-lat-min", type=float, default=35.0)
    parser.add_argument(
        "--weight-mode",
        choices=["uniform", "depth-balanced"],
        default="uniform",
    )
    parser.add_argument(
        "--depth-bins",
        default="-4000,-2000,-1000,-500,-200,200,500,1000,2000",
    )
    parser.add_argument("--write-bin-summary", action="store_true")

    args = parser.parse_args(normalize_depth_bin_argv(sys.argv[1:]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_reported_records(args.input, args.z_min, args.z_max)
    if records.empty:
        raise SystemExit("No reported geocoded records found in requested z range.")

    depth_bins = parse_depth_bins(args.depth_bins)
    records = assign_record_weights(records, args.weight_mode, depth_bins)

    pole_grid = build_candidate_pole_grid(
        args.alpha_min,
        args.alpha_max,
        args.alpha_step,
        args.azimuth_step,
    )

    scored = score_dataset(
        records=records,
        pole_grid=pole_grid,
        bulge_m=args.bulge_m,
        sigma_m=args.sigma_m,
        chunk_size=args.chunk_size,
    )

    grid_csv = OUT_DIR / "14_polar_path_preference_grid.csv"
    top_csv = OUT_DIR / "14_polar_path_preference_top_candidates.csv"
    records_csv = OUT_DIR / "14_filtered_reported_records.csv"
    summary_json = OUT_DIR / "14_polar_path_preference_summary.json"
    depth_bin_summary_csv = OUT_DIR / "14_depth_bin_summary.csv"

    bin_summary = depth_bin_summary(records)
    if args.write_bin_summary:
        bin_summary.to_csv(depth_bin_summary_csv, index=False)

    summary = summarize_preference(
        scored,
        records,
        args,
        depth_bins=depth_bins,
        depth_bin_summary_csv=depth_bin_summary_csv if args.write_bin_summary else None,
    )

    scored.to_csv(grid_csv, index=False)
    scored.sort_values("preference_0_1", ascending=False).head(args.top_n).to_csv(top_csv, index=False)
    records.to_csv(records_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_polar_preference_map(
        scored,
        summary,
        OUT_DIR / "14_polar_preference_north_polar.png",
        lat_min=args.polar_lat_min,
    )
    make_offset_azimuth_heatmap(scored, OUT_DIR / "14_offset_azimuth_preference_heatmap.png")
    make_top_candidate_overview(
        scored,
        records,
        summary,
        OUT_DIR / "14_top_candidate_overview.png",
        projection_name=args.projection,
        top_quantile=args.plot_top_quantile,
    )

    best = summary["best_candidate"]

    print("\nLITTORAL 14 polar path preference complete.")
    print(f"Input: {args.input}")
    print(f"Reported records used: {len(records)}")
    print(f"Weight mode: {args.weight_mode}")
    print(f"Occupied depth bins: {len(bin_summary)}")
    print(f"Effective weight total: {records['weight'].sum():.3f}")
    print(f"Output: {OUT_DIR}")
    print(
        "Best candidate: "
        f"offset={best['offset_deg']:.3f} deg, "
        f"azimuth={best['orientation_azimuth_deg']:.3f} deg, "
        f"pole=({best['candidate_pole_lat']:.3f}, {best['candidate_pole_lon']:.3f}), "
        f"preference={best['preference_0_1']:.4f}, "
        f"hit_fraction={best['hit_fraction_within_sigma']:.4f}, "
        f"MAE={best['mae_m']:.2f} m, "
        f"RMSE={best['rmse_m']:.2f} m"
    )


if __name__ == "__main__":
    main()
