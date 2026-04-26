#!/usr/bin/env python3
"""
15_littoral_depth_regime_inverse.py

Depth-regime inverse test for LITTORAL.

Purpose:
  Split reported geocoded shoreline / paleolittoral records into depth regimes,
  run the Script 14 geodetic pole-orientation inverse independently for each
  regime, and compare whether shallow, intermediate, and deep records prefer
  distinct polar-offset families.

Default regimes:
  shallow:       -200 to 200 m
  intermediate: -1000 to -200 m
  deep:          -4000 to -1000 m
  high:           200 to 2000 m
  full:         -4000 to 2000 m

Writes:
  outputs/geospatial_15/15_regime_summary.csv
  outputs/geospatial_15/15_regime_pairwise_angular_separation.csv
  outputs/geospatial_15/15_<regime>_preference_grid.csv
  outputs/geospatial_15/15_<regime>_top_candidates.csv
  outputs/geospatial_15/15_<regime>_offset_azimuth_heatmap.png
  outputs/geospatial_15/15_<regime>_polar_preference.png
  outputs/geospatial_15/15_regime_best_poles_map.png
  outputs/geospatial_15/15_depth_regime_composite.png
  outputs/geospatial_15/15_summary.json
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
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "merged" / "master_dataset.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "geospatial_15"

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


def load_records(path):
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
    ].copy()

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


def summarize_grid(regime_name, records, grid, z_min, z_max, args):
    best = grid.sort_values("preference_0_1", ascending=False).iloc[0]

    q90 = grid[grid["preference_0_1"] >= grid["preference_0_1"].quantile(0.90)]
    q95 = grid[grid["preference_0_1"] >= grid["preference_0_1"].quantile(0.95)]
    q99 = grid[grid["preference_0_1"] >= grid["preference_0_1"].quantile(0.99)]

    c90 = spherical_centroid(q90["candidate_pole_lat"], q90["candidate_pole_lon"], q90["preference_0_1"])
    c95 = spherical_centroid(q95["candidate_pole_lat"], q95["candidate_pole_lon"], q95["preference_0_1"])
    c99 = spherical_centroid(q99["candidate_pole_lat"], q99["candidate_pole_lon"], q99["preference_0_1"])

    return {
        "regime": regime_name,
        "z_min_m": float(z_min),
        "z_max_m": float(z_max),
        "n_records": int(len(records)),
        "weight_total": float(records["weight"].sum()),
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


def parse_regimes(text):
    regimes = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        name, bounds = part.split(":")
        z_min, z_max = bounds.split(",")
        regimes.append((name.strip(), float(z_min), float(z_max)))
    return regimes


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


def make_polar_preference_map(grid, summary, out_path, lat_min=30.0):
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

    gl = ax.gridlines(draw_labels=True, linewidth=0.35, alpha=0.35, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.72, pad=0.06)
    cbar.set_label("Normalized polar path preference, gamma-stretched")

    ax.set_title(
        f"LITTORAL 15 — {summary['regime']} polar preference\n"
        f"best offset={summary['best_offset_deg']:.2f}°, azimuth={summary['best_azimuth_deg']:.2f}°",
        fontsize=13,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=260)
    plt.close(fig)


def make_best_poles_map(summary_df, out_path, projection_name):
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
            f" {row['regime']} ({row['best_offset_deg']:.2f}°)",
            transform=data_crs,
            fontsize=8,
            zorder=11,
        )

    gl = ax.gridlines(draw_labels=True, linewidth=0.35, alpha=0.32, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.78, pad=0.06)
    cbar.set_label("Best polar offset by regime, degrees")

    ax.set_title("LITTORAL 15 — best candidate poles by depth regime", fontsize=15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=260)
    plt.close(fig)


def make_composite_plot(summary_df, grid_by_regime, pair_df, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import PowerNorm
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    regimes = summary_df["regime"].tolist()
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
        summary = summary_df.loc[summary_df["regime"] == regime].iloc[0]

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
        polar_ax.set_extent([-180, 180, 30.0, 90], crs=data_crs)
        polar_ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#eeeeea", edgecolor="none")
        polar_ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f8fbff", edgecolor="none")
        polar_ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.35)

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
        "LITTORAL 15 — depth-regime inverse composite\n"
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
        "--regimes",
        default="shallow:-200,200;intermediate:-1000,-200;deep:-4000,-1000;high:200,2000;full:-4000,2000",
        help='Semicolon-separated regimes like "name:zmin,zmax;name:zmin,zmax"',
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

    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records = load_records(args.input)
    regimes = parse_regimes(args.regimes)

    pole_grid = build_candidate_pole_grid(
        args.alpha_min,
        args.alpha_max,
        args.alpha_step,
        args.azimuth_step,
    )

    summaries = []
    grids_written = []
    grid_by_regime = {}

    for regime_name, z_min, z_max in regimes:
        records = all_records[
            (all_records["z_m"] >= z_min)
            & (all_records["z_m"] <= z_max)
        ].copy()

        if len(records) < args.min_records:
            print(f"Skipping {regime_name}: only {len(records)} records.")
            continue

        records["weight"] = 1.0

        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in regime_name)

        records_path = OUT_DIR / f"15_{safe}_records.csv"
        grid_path = OUT_DIR / f"15_{safe}_preference_grid.csv"
        top_path = OUT_DIR / f"15_{safe}_top_candidates.csv"
        heatmap_path = OUT_DIR / f"15_{safe}_offset_azimuth_heatmap.png"
        polar_path = OUT_DIR / f"15_{safe}_polar_preference.png"

        scored = score_dataset(
            records=records,
            pole_grid=pole_grid,
            bulge_m=args.bulge_m,
            sigma_m=args.sigma_m,
            chunk_size=args.chunk_size,
        )

        summary = summarize_grid(regime_name, records, scored, z_min, z_max, args)
        summaries.append(summary)
        grid_by_regime[regime_name] = scored

        records.to_csv(records_path, index=False)
        scored.to_csv(grid_path, index=False)
        scored.sort_values("preference_0_1", ascending=False).head(args.top_n).to_csv(top_path, index=False)

        make_offset_azimuth_heatmap(
            scored,
            heatmap_path,
            f"LITTORAL 15 — {regime_name} offset–azimuth preference",
        )

        make_polar_preference_map(
            scored,
            summary,
            polar_path,
            lat_min=30.0,
        )

        grids_written.append(
            {
                "regime": regime_name,
                "records_csv": str(records_path),
                "grid_csv": str(grid_path),
                "top_candidates_csv": str(top_path),
                "offset_azimuth_heatmap": str(heatmap_path),
                "polar_preference_png": str(polar_path),
            }
        )

        print(
            f"{regime_name}: n={len(records)}, "
            f"best offset={summary['best_offset_deg']:.3f}°, "
            f"azimuth={summary['best_azimuth_deg']:.3f}°, "
            f"pole=({summary['best_pole_lat']:.3f}, {summary['best_pole_lon']:.3f}), "
            f"hit={summary['hit_fraction_within_sigma']:.4f}, "
            f"MAE={summary['mae_m']:.2f} m"
        )

    if not summaries:
        raise SystemExit("No regimes had enough records to process.")

    summary_df = pd.DataFrame(summaries)
    summary_csv = OUT_DIR / "15_regime_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    pair_rows = []
    for i, a in summary_df.iterrows():
        for j, b in summary_df.iterrows():
            if j <= i:
                continue
            pair_rows.append(
                {
                    "regime_a": a["regime"],
                    "regime_b": b["regime"],
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
    pair_csv = OUT_DIR / "15_regime_pairwise_angular_separation.csv"
    pair_df.to_csv(pair_csv, index=False)

    best_map = OUT_DIR / "15_regime_best_poles_map.png"
    make_best_poles_map(summary_df, best_map, args.projection)

    composite_plot = OUT_DIR / "15_depth_regime_composite.png"
    make_composite_plot(summary_df, grid_by_regime, pair_df, composite_plot)

    summary_json = OUT_DIR / "15_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_csv": str(args.input),
                "output_dir": str(OUT_DIR),
                "alpha_min_deg": args.alpha_min,
                "alpha_max_deg": args.alpha_max,
                "alpha_step_deg": args.alpha_step,
                "azimuth_step_deg": args.azimuth_step,
                "sigma_m": args.sigma_m,
                "bulge_m": args.bulge_m,
                "regime_summary_csv": str(summary_csv),
                "pairwise_angular_separation_csv": str(pair_csv),
                "best_poles_map_png": str(best_map),
                "composite_plot_png": str(composite_plot),
                "regime_outputs": grids_written,
                "summaries": summaries,
            },
            f,
            indent=2,
        )

    print("\nLITTORAL 15 depth-regime inverse complete.")
    print(f"Output: {OUT_DIR}")
    print(f"Summary: {summary_csv}")
    print(f"Pairwise separations: {pair_csv}")
    print(f"Composite plot: {composite_plot}")


if __name__ == "__main__":
    main()
