#!/usr/bin/env python3
"""
12_littoral_heatmap.py

Render the full-range LITTORAL l=1 field gradient as a global heatmap over
continental reference outlines.

Reads by default:
  outputs/geospatial_05/05_dominant_gradient.csv

Writes by default:
  outputs/geospatial_12/12_littoral_gradient_heatmap.png
  outputs/geospatial_12/12_littoral_gradient_grid.csv
  outputs/geospatial_12/12_littoral_gradient_heatmap_summary.json

Run:
  python scripts/12_littoral_heatmap.py
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRADIENT_CSV = PROJECT_ROOT / "outputs" / "geospatial_05" / "05_dominant_gradient.csv"
DEFAULT_RECORDS_CSV = PROJECT_ROOT / "outputs" / "geospatial_05" / "05_littoral_records_with_predictions_and_residuals.csv"
DEFAULT_MACH_CSV = PROJECT_ROOT / "data" / "mach.csv"
OUTPUT_DIR_MACH_CSV = PROJECT_ROOT / "outputs" / "geospatial_12" / "mach.csv"
FALLBACK_MACH_CSV = PROJECT_ROOT / "outputs" / "mach.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "geospatial_12"

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "littoral_matplotlib"))


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


def load_gradient(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No gradient rows found in {path}")

    row = df.iloc[0]
    coeffs = _parse_coefficients(row["coefficients"])
    if coeffs.shape != (3,):
        raise ValueError(f"Expected three gradient coefficients, got {coeffs!r}")

    return {
        "r2": float(row.get("r2", np.nan)),
        "rmse": float(row.get("rmse", np.nan)),
        "mae": float(row.get("mae", np.nan)),
        "axis_lat": float(row["axis_lat"]),
        "axis_lon": float(row["axis_lon"]),
        "gradient_norm_m": float(row["gradient_norm_m"]),
        "intercept": float(row["intercept"]),
        "coefficients": coeffs,
    }


def _parse_coefficients(value) -> np.ndarray:
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
    else:
        parsed = value
    return np.asarray(parsed, dtype=float)


def build_field_grid(gradient: dict[str, object], resolution_deg: float) -> pd.DataFrame:
    half = resolution_deg / 2.0
    lats = np.arange(-90.0 + half, 90.0, resolution_deg)
    lons = np.arange(-180.0 + half, 180.0, resolution_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    xyz = latlon_to_unit(lat_grid.ravel(), lon_grid.ravel())
    coeffs = np.asarray(gradient["coefficients"], dtype=float)
    field = float(gradient["intercept"]) + xyz @ coeffs

    return pd.DataFrame(
        {
            "latitude": lat_grid.ravel(),
            "longitude": lon_grid.ravel(),
            "littoral_gradient_m": field,
        }
    )


def grid_to_arrays(grid: pd.DataFrame, value_col: str = "littoral_gradient_m") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_values = np.sort(grid["latitude"].unique())
    lon_values = np.sort(grid["longitude"].unique())
    field = (
        grid.pivot(index="latitude", columns="longitude", values=value_col)
        .reindex(index=lat_values, columns=lon_values)
        .to_numpy(float)
    )
    return lon_values, lat_values, field


def read_csv_flexible(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def infer_mach_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    lower = {c.lower().strip(): c for c in df.columns}

    lat_col = None
    lon_col = None
    m_col = None

    for key in ["lat", "latitude", "axis_lat", "euler_lat", "pole_lat"]:
        if key in lower:
            lat_col = lower[key]
            break

    for key in ["lon", "longitude", "axis_lon", "euler_lon", "pole_lon"]:
        if key in lower:
            lon_col = lower[key]
            break

    for key in ["m", "mach", "correlation", "compatibility", "compatibility_m", "anisotropic_variance"]:
        if key in lower:
            m_col = lower[key]
            break

    if lat_col is None or lon_col is None or m_col is None:
        raise ValueError(f"Could not infer Mach columns. Found columns: {list(df.columns)}")

    return lat_col, lon_col, m_col


def load_mach(path: Path) -> pd.DataFrame:
    df = read_csv_flexible(path)
    lat_col, lon_col, m_col = infer_mach_columns(df)

    out = pd.DataFrame(
        {
            "latitude": pd.to_numeric(df[lat_col], errors="coerce"),
            "longitude": pd.to_numeric(df[lon_col], errors="coerce"),
            "mach_m": pd.to_numeric(df[m_col], errors="coerce"),
        }
    ).dropna()

    out["longitude"] = normalize_longitude(out["longitude"])
    return out.drop_duplicates(subset=["latitude", "longitude"]).reset_index(drop=True)


def build_mach_interpolator(mach_grid: pd.DataFrame):
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    points = mach_grid[["longitude", "latitude"]].to_numpy(float)
    values = mach_grid["mach_m"].to_numpy(float)

    linear = LinearNDInterpolator(points, values)
    nearest = NearestNDInterpolator(points, values)

    def eval_m(lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = normalize_longitude(lon)
        vals = np.asarray(linear(lon, lat), dtype=float)
        missing = ~np.isfinite(vals)
        if np.any(missing):
            vals[missing] = nearest(lon[missing], lat[missing])
        return vals

    return eval_m


def add_mach_to_littoral_grid(littoral_grid: pd.DataFrame, mach_grid: pd.DataFrame) -> pd.DataFrame:
    eval_m = build_mach_interpolator(mach_grid)
    out = littoral_grid.copy()
    out["mach_m"] = eval_m(out["latitude"].to_numpy(float), out["longitude"].to_numpy(float))
    return out


def cell_edges(centers: np.ndarray, resolution_deg: float, low: float, high: float) -> np.ndarray:
    edges = np.concatenate([[centers[0] - resolution_deg / 2.0], centers + resolution_deg / 2.0])
    return np.clip(edges, low, high)


def make_heatmap(
    grid: pd.DataFrame,
    gradient: dict[str, object],
    out_path: Path,
    *,
    resolution_deg: float,
    projection_name: str,
    records_csv: Path | None,
    overlay_points: bool,
    point_limit: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    projection = _projection(projection_name, ccrs)
    data_crs = ccrs.PlateCarree()

    lons, lats, field = grid_to_arrays(grid)
    lon_edges = cell_edges(lons, resolution_deg, -180.0, 180.0)
    lat_edges = cell_edges(lats, resolution_deg, -90.0, 90.0)
    lon_edge_grid, lat_edge_grid = np.meshgrid(lon_edges, lat_edges)

    vmin = float(np.nanmin(field))
    vmax = float(np.nanmax(field))
    midpoint = (vmin + vmax) / 2.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(15, 8.5))
    ax = plt.axes(projection=projection)
    ax.set_global()
    ax.set_facecolor("#f7f7f4")

    mesh = ax.pcolormesh(
        lon_edge_grid,
        lat_edge_grid,
        field,
        transform=data_crs,
        cmap="coolwarm",
        norm=norm,
        shading="flat",
        alpha=0.92,
        rasterized=True,
    )

    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="none", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.55, edgecolor="#202020")
    ax.add_feature(cfeature.LAKES.with_scale("110m"), facecolor="none", edgecolor="#444444", linewidth=0.2, alpha=0.35)

    ax.contour(
        lons,
        lats,
        field,
        levels=[midpoint],
        transform=data_crs,
        colors="black",
        linewidths=1.35,
        linestyles="-",
        alpha=0.92,
    )

    gl = ax.gridlines(
        crs=data_crs,
        draw_labels=True,
        linewidth=0.35,
        color="#555555",
        alpha=0.35,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    axis_lat = float(gradient["axis_lat"])
    axis_lon = float(gradient["axis_lon"])
    anti_lat, anti_lon = unit_to_latlon(-latlon_to_unit([axis_lat], [axis_lon])[0].reshape(1, 3))
    ax.scatter([axis_lon], [axis_lat], transform=data_crs, s=70, marker="^", c="black", label="gradient high pole")
    ax.scatter([float(anti_lon[0])], [float(anti_lat[0])], transform=data_crs, s=70, marker="v", c="white", edgecolors="black", linewidths=1.0, label="gradient low pole")

    if overlay_points and records_csv is not None and records_csv.exists():
        records = pd.read_csv(records_csv)
        records = records.dropna(subset=["latitude", "longitude"])
        if point_limit > 0 and len(records) > point_limit:
            records = records.sample(point_limit, random_state=42)
        ax.scatter(
            records["longitude"],
            records["latitude"],
            transform=data_crs,
            s=10,
            c="black",
            alpha=0.28,
            linewidths=0,
            label="LITTORAL records",
        )

    cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", shrink=0.78, pad=0.06)
    cbar.set_label("Predicted z_m from LITTORAL l=1 gradient (m), full grid range")
    cbar.ax.axvline(midpoint, color="black", linewidth=1.1)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linewidth=1.35))
    labels.append(f"field midpoint ({midpoint:.2f} m)")
    ax.legend(handles, labels, loc="lower left", frameon=True, framealpha=0.86)
    ax.set_title(
        "12 LITTORAL full-range field gradient heatmap\n"
        f"axis=({axis_lat:.2f}, {axis_lon:.2f}), norm={float(gradient['gradient_norm_m']):.3f} m, R²={float(gradient['r2']):.4f}",
        fontsize=15,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close(fig)


def make_comparison_heatmap(
    grid: pd.DataFrame,
    mach_grid: pd.DataFrame,
    gradient: dict[str, object],
    out_path: Path,
    *,
    resolution_deg: float,
    projection_name: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    projection = _projection(projection_name, ccrs)
    data_crs = ccrs.PlateCarree()

    lons, lats, littoral_field = grid_to_arrays(grid, "littoral_gradient_m")
    _, _, mach_on_littoral_grid = grid_to_arrays(grid, "mach_m")
    mach_lons, mach_lats, mach_native = grid_to_arrays(mach_grid, "mach_m")

    lon_edges = cell_edges(lons, resolution_deg, -180.0, 180.0)
    lat_edges = cell_edges(lats, resolution_deg, -90.0, 90.0)
    lon_edge_grid, lat_edge_grid = np.meshgrid(lon_edges, lat_edges)

    mach_lon_step = _median_step(mach_lons)
    mach_lat_step = _median_step(mach_lats)
    mach_lon_edges = cell_edges(mach_lons, mach_lon_step, -180.0, 180.0)
    mach_lat_edges = cell_edges(mach_lats, mach_lat_step, -90.0, 90.0)
    mach_lon_edge_grid, mach_lat_edge_grid = np.meshgrid(mach_lon_edges, mach_lat_edges)

    littoral_min = float(np.nanmin(littoral_field))
    littoral_max = float(np.nanmax(littoral_field))
    littoral_mid = (littoral_min + littoral_max) / 2.0
    mach_min = float(np.nanmin(mach_native))
    mach_max = float(np.nanmax(mach_native))
    mach_mid = (mach_min + mach_max) / 2.0

    fig = plt.figure(figsize=(18, 8.5))
    axes = [
        plt.subplot(1, 2, 1, projection=projection),
        plt.subplot(1, 2, 2, projection=projection),
    ]

    for ax in axes:
        ax.set_global()
        ax.set_facecolor("#f7f7f4")
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="none", edgecolor="none")
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.48, edgecolor="#202020")
        ax.add_feature(cfeature.LAKES.with_scale("110m"), facecolor="none", edgecolor="#444444", linewidth=0.18, alpha=0.35)

    m0 = axes[0].pcolormesh(
        lon_edge_grid,
        lat_edge_grid,
        littoral_field,
        transform=data_crs,
        cmap="coolwarm",
        norm=Normalize(vmin=littoral_min, vmax=littoral_max),
        shading="flat",
        alpha=0.92,
        rasterized=True,
    )
    axes[0].contour(
        lons,
        lats,
        littoral_field,
        levels=[littoral_mid],
        transform=data_crs,
        colors="black",
        linewidths=1.25,
        alpha=0.92,
    )

    axis_lat = float(gradient["axis_lat"])
    axis_lon = float(gradient["axis_lon"])
    anti_lat, anti_lon = unit_to_latlon(-latlon_to_unit([axis_lat], [axis_lon])[0].reshape(1, 3))
    axes[0].scatter([axis_lon], [axis_lat], transform=data_crs, s=58, marker="^", c="black")
    axes[0].scatter([float(anti_lon[0])], [float(anti_lat[0])], transform=data_crs, s=58, marker="v", c="white", edgecolors="black", linewidths=0.9)
    axes[0].set_title("LITTORAL l=1 field gradient", fontsize=13)

    m1 = axes[1].pcolormesh(
        mach_lon_edge_grid,
        mach_lat_edge_grid,
        mach_native,
        transform=data_crs,
        cmap="viridis",
        norm=Normalize(vmin=mach_min, vmax=mach_max),
        shading="flat",
        alpha=0.92,
        rasterized=True,
    )
    axes[1].contour(
        lons,
        lats,
        mach_on_littoral_grid,
        levels=[mach_mid],
        transform=data_crs,
        colors="black",
        linewidths=1.25,
        alpha=0.92,
    )
    axes[1].set_title("Mach compatibility field", fontsize=13)

    for ax in axes:
        gl = ax.gridlines(crs=data_crs, draw_labels=False, linewidth=0.3, color="#555555", alpha=0.28, linestyle="--")
        gl.x_inline = False
        gl.y_inline = False

    cbar0 = fig.colorbar(m0, ax=axes[0], orientation="horizontal", shrink=0.82, pad=0.05)
    cbar0.set_label("Predicted z_m from LITTORAL gradient (m)")
    cbar0.ax.axvline(littoral_mid, color="black", linewidth=1.0)

    cbar1 = fig.colorbar(m1, ax=axes[1], orientation="horizontal", shrink=0.82, pad=0.05)
    cbar1.set_label("Mach compatibility M")
    cbar1.ax.axvline(mach_mid, color="black", linewidth=1.0)

    handles = [Line2D([0], [0], color="black", linewidth=1.25)]
    fig.legend(handles, ["field midpoint contour"], loc="lower center", frameon=True, framealpha=0.88)
    fig.suptitle("12 LITTORAL gradient compared with Mach field", fontsize=16)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=240)
    plt.close(fig)


def make_overlay_heatmap(
    grid: pd.DataFrame,
    mach_grid: pd.DataFrame,
    gradient: dict[str, object],
    out_path: Path,
    *,
    resolution_deg: float,
    projection_name: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import TwoSlopeNorm

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    projection = _projection(projection_name, ccrs)
    data_crs = ccrs.PlateCarree()

    lons, lats, littoral_field = grid_to_arrays(grid, "littoral_gradient_m")
    _, _, mach_field = grid_to_arrays(grid, "mach_m")

    lon_edges = cell_edges(lons, resolution_deg, -180.0, 180.0)
    lat_edges = cell_edges(lats, resolution_deg, -90.0, 90.0)
    lon_edge_grid, lat_edge_grid = np.meshgrid(lon_edges, lat_edges)

    littoral_min = float(np.nanmin(littoral_field))
    littoral_max = float(np.nanmax(littoral_field))
    littoral_mid = (littoral_min + littoral_max) / 2.0
    mach_min = float(np.nanmin(mach_field))
    mach_max = float(np.nanmax(mach_field))
    mach_mid = (mach_min + mach_max) / 2.0

    mach_low_levels = [float(v) for v in np.nanquantile(mach_field, [0.10, 0.20])]
    mach_high_levels = [float(v) for v in np.nanquantile(mach_field, [0.80, 0.90])]
    mach_geometry = field_geometry(mach_grid, "mach_m")

    fig = plt.figure(figsize=(15, 8.5))
    ax = plt.axes(projection=projection)
    ax.set_global()
    ax.set_facecolor("#f7f7f4")

    mesh = ax.pcolormesh(
        lon_edge_grid,
        lat_edge_grid,
        littoral_field,
        transform=data_crs,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=littoral_min, vcenter=littoral_mid, vmax=littoral_max),
        shading="flat",
        alpha=0.76,
        rasterized=True,
    )

    ax.contour(
        lons,
        lats,
        mach_field,
        levels=mach_low_levels,
        transform=data_crs,
        colors=["#2a6fbb", "#2a6fbb"],
        linewidths=[0.85, 1.05],
        linestyles=[":", "--"],
        alpha=0.78,
    )
    ax.contour(
        lons,
        lats,
        mach_field,
        levels=[mach_mid],
        transform=data_crs,
        colors="#7b3294",
        linewidths=1.45,
        linestyles="--",
        alpha=0.95,
    )
    ax.contour(
        lons,
        lats,
        mach_field,
        levels=mach_high_levels,
        transform=data_crs,
        colors=["#168a3a", "#168a3a"],
        linewidths=[1.05, 0.85],
        linestyles=["--", ":"],
        alpha=0.78,
    )
    ax.contour(
        lons,
        lats,
        littoral_field,
        levels=[littoral_mid],
        transform=data_crs,
        colors="black",
        linewidths=1.6,
        linestyles="-",
        alpha=0.96,
    )

    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="none", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.55, edgecolor="#202020")
    ax.add_feature(cfeature.LAKES.with_scale("110m"), facecolor="none", edgecolor="#444444", linewidth=0.2, alpha=0.35)

    axis_lat = float(gradient["axis_lat"])
    axis_lon = float(gradient["axis_lon"])
    anti_lat, anti_lon = unit_to_latlon(-latlon_to_unit([axis_lat], [axis_lon])[0].reshape(1, 3))
    ax.scatter([axis_lon], [axis_lat], transform=data_crs, s=82, marker="^", c="black", zorder=6)
    ax.scatter([float(anti_lon[0])], [float(anti_lat[0])], transform=data_crs, s=82, marker="v", c="white", edgecolors="black", linewidths=1.0, zorder=6)
    ax.scatter(
        [mach_geometry["high_center_lon"]],
        [mach_geometry["high_center_lat"]],
        transform=data_crs,
        s=110,
        marker="*",
        c="#168a3a",
        edgecolors="white",
        linewidths=0.8,
        zorder=7,
    )
    ax.scatter(
        [mach_geometry["low_center_lon"]],
        [mach_geometry["low_center_lat"]],
        transform=data_crs,
        s=90,
        marker="X",
        c="#2a6fbb",
        edgecolors="white",
        linewidths=0.8,
        zorder=7,
    )

    gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=0.35, color="#555555", alpha=0.32, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", shrink=0.78, pad=0.06)
    cbar.set_label("LITTORAL gradient field (m); Mach field shown as contours")
    cbar.ax.axvline(littoral_mid, color="black", linewidth=1.1)

    handles = [
        Line2D([0], [0], marker="^", color="black", linestyle="None", markersize=8),
        Line2D([0], [0], marker="v", color="black", markerfacecolor="white", linestyle="None", markersize=8),
        Line2D([0], [0], color="black", linewidth=1.6),
        Line2D([0], [0], color="#7b3294", linewidth=1.45, linestyle="--"),
        Line2D([0], [0], color="#168a3a", linewidth=1.0, linestyle="--"),
        Line2D([0], [0], color="#2a6fbb", linewidth=1.0, linestyle="--"),
        Line2D([0], [0], marker="*", color="#168a3a", markeredgecolor="white", linestyle="None", markersize=11),
        Line2D([0], [0], marker="X", color="#2a6fbb", markeredgecolor="white", linestyle="None", markersize=9),
    ]
    labels = [
        "LITTORAL high pole",
        "LITTORAL low pole",
        f"LITTORAL midpoint ({littoral_mid:.2f} m)",
        f"Mach midpoint ({mach_mid:.3f})",
        "Mach high contours (80/90%)",
        "Mach low contours (10/20%)",
        "Mach high-field center",
        "Mach low-field center",
    ]
    ax.legend(handles, labels, loc="lower left", frameon=True, framealpha=0.88, fontsize=8.5)
    ax.set_title("12 direct overlay of LITTORAL gradient and Mach field geometry", fontsize=15)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close(fig)


def field_geometry(grid: pd.DataFrame, value_col: str, quantile: float = 0.10) -> dict[str, float]:
    values = pd.to_numeric(grid[value_col], errors="coerce")
    valid = grid.loc[values.notna(), ["latitude", "longitude"]].copy()
    valid[value_col] = values.loc[values.notna()].to_numpy(float)
    if valid.empty:
        return {
            "high_center_lat": np.nan,
            "high_center_lon": np.nan,
            "low_center_lat": np.nan,
            "low_center_lon": np.nan,
            "max_lat": np.nan,
            "max_lon": np.nan,
            "min_lat": np.nan,
            "min_lon": np.nan,
        }

    high_threshold = valid[value_col].quantile(1.0 - quantile)
    low_threshold = valid[value_col].quantile(quantile)
    high = valid[valid[value_col] >= high_threshold]
    low = valid[valid[value_col] <= low_threshold]
    high_lat, high_lon = spherical_centroid(high["latitude"], high["longitude"])
    low_lat, low_lon = spherical_centroid(low["latitude"], low["longitude"])
    max_row = valid.loc[valid[value_col].idxmax()]
    min_row = valid.loc[valid[value_col].idxmin()]
    return {
        "high_center_lat": high_lat,
        "high_center_lon": high_lon,
        "low_center_lat": low_lat,
        "low_center_lon": low_lon,
        "max_lat": float(max_row["latitude"]),
        "max_lon": float(max_row["longitude"]),
        "min_lat": float(min_row["latitude"]),
        "min_lon": float(min_row["longitude"]),
    }


def spherical_centroid(lat, lon) -> tuple[float, float]:
    vectors = latlon_to_unit(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
    mean = vectors.mean(axis=0)
    if np.linalg.norm(mean) == 0:
        return float(np.mean(lat)), float(np.mean(lon))
    out_lat, out_lon = unit_to_latlon(mean.reshape(1, 3))
    return float(out_lat[0]), float(out_lon[0])


def _median_step(values: np.ndarray) -> float:
    diffs = np.diff(np.sort(np.asarray(values, dtype=float)))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    return float(np.median(diffs))


def _projection(name: str, ccrs):
    normalized = name.strip().lower()
    if normalized == "platecarree":
        return ccrs.PlateCarree()
    if normalized == "mollweide":
        return ccrs.Mollweide()
    if normalized == "equalearth":
        return ccrs.EqualEarth()
    if normalized == "robinson":
        return ccrs.Robinson()
    raise ValueError("Unsupported projection. Expected one of: robinson, equalearth, mollweide, platecarree")


def write_summary(
    path: Path,
    gradient_csv: Path,
    grid: pd.DataFrame,
    gradient: dict[str, object],
    args: argparse.Namespace,
    *,
    mach_csv: Path | None = None,
    mach_grid: pd.DataFrame | None = None,
) -> None:
    mach_payload = None
    if mach_grid is not None and "mach_m" in grid.columns:
        corr_df = grid[["littoral_gradient_m", "mach_m"]].dropna()
        pearson = float(corr_df["littoral_gradient_m"].corr(corr_df["mach_m"])) if len(corr_df) > 1 else None
        mach_geometry = field_geometry(mach_grid, "mach_m")
        mach_payload = {
            "mach_input": str(mach_csv) if mach_csv is not None else None,
            "n_native_points": int(len(mach_grid)),
            "min_m": float(mach_grid["mach_m"].min()),
            "max_m": float(mach_grid["mach_m"].max()),
            "midpoint_m": float((mach_grid["mach_m"].min() + mach_grid["mach_m"].max()) / 2.0),
            "mean_m": float(mach_grid["mach_m"].mean()),
            "median_m": float(mach_grid["mach_m"].median()),
            "pearson_with_littoral_gradient_on_grid": pearson,
            "geometry": mach_geometry,
        }

    anti_lat, anti_lon = unit_to_latlon(
        -latlon_to_unit([float(gradient["axis_lat"])], [float(gradient["axis_lon"])])[0].reshape(1, 3)
    )

    payload = {
        "metadata": {
            "gradient_input": str(gradient_csv),
            "mach_input": str(mach_csv) if mach_csv is not None else None,
            "grid_resolution_deg": float(args.resolution_deg),
            "projection": args.projection,
            "chronology_used": False,
            "origin_assumptions_used": False,
        },
        "gradient": {
            "r2": float(gradient["r2"]),
            "rmse": float(gradient["rmse"]),
            "mae": float(gradient["mae"]),
            "axis_lat": float(gradient["axis_lat"]),
            "axis_lon": float(gradient["axis_lon"]),
            "gradient_norm_m": float(gradient["gradient_norm_m"]),
            "intercept": float(gradient["intercept"]),
            "coefficients": [float(v) for v in np.asarray(gradient["coefficients"], dtype=float)],
            "geometry": {
                "high_pole_lat": float(gradient["axis_lat"]),
                "high_pole_lon": float(gradient["axis_lon"]),
                "low_pole_lat": float(anti_lat[0]),
                "low_pole_lon": float(anti_lon[0]),
            },
        },
        "field_range": {
            "n_grid_cells": int(len(grid)),
            "min_m": float(grid["littoral_gradient_m"].min()),
            "max_m": float(grid["littoral_gradient_m"].max()),
            "midpoint_m": float((grid["littoral_gradient_m"].min() + grid["littoral_gradient_m"].max()) / 2.0),
            "mean_m": float(grid["littoral_gradient_m"].mean()),
            "median_m": float(grid["littoral_gradient_m"].median()),
        },
        "mach_field": mach_payload,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_mach_path(path: Path | None) -> Path | None:
    candidates = []
    if path is not None:
        candidates.append(path)
    candidates.extend([DEFAULT_MACH_CSV, OUTPUT_DIR_MACH_CSV, FALLBACK_MACH_CSV])

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a full-range global heatmap of the LITTORAL l=1 field gradient.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gradient-csv", type=Path, default=DEFAULT_GRADIENT_CSV)
    parser.add_argument("--records-csv", type=Path, default=DEFAULT_RECORDS_CSV)
    parser.add_argument("--mach-csv", type=Path, default=DEFAULT_MACH_CSV, help="Mach field CSV. Defaults to data/mach.csv, falling back to outputs/mach.csv when absent.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--resolution-deg", type=float, default=1.0)
    parser.add_argument("--projection", choices=("robinson", "equalearth", "mollweide", "platecarree"), default="robinson")
    parser.add_argument("--overlay-points", action="store_true", help="Overlay LITTORAL source records from --records-csv.")
    parser.add_argument("--point-limit", type=int, default=1500, help="Maximum point overlay count; 0 disables sampling.")
    parser.add_argument("--no-mach", action="store_true", help="Skip Mach comparison outputs.")
    args = parser.parse_args(argv)

    if args.resolution_deg <= 0:
        raise ValueError("--resolution-deg must be positive")

    gradient_csv = args.gradient_csv if args.gradient_csv.is_absolute() else PROJECT_ROOT / args.gradient_csv
    records_csv = args.records_csv if args.records_csv.is_absolute() else PROJECT_ROOT / args.records_csv
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    mach_csv = None if args.no_mach else resolve_mach_path(args.mach_csv)

    gradient = load_gradient(gradient_csv)
    grid = build_field_grid(gradient, args.resolution_deg)
    mach_grid = None
    if mach_csv is not None:
        mach_grid = load_mach(mach_csv)
        grid = add_mach_to_littoral_grid(grid, mach_grid)

    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / "12_littoral_gradient_grid.csv"
    png_path = out_dir / "12_littoral_gradient_heatmap.png"
    comparison_png_path = out_dir / "12_littoral_mach_field_comparison.png"
    overlay_png_path = out_dir / "12_littoral_mach_field_overlay.png"
    summary_path = out_dir / "12_littoral_gradient_heatmap_summary.json"

    grid.to_csv(grid_path, index=False)
    make_heatmap(
        grid,
        gradient,
        png_path,
        resolution_deg=args.resolution_deg,
        projection_name=args.projection,
        records_csv=records_csv,
        overlay_points=args.overlay_points,
        point_limit=args.point_limit,
    )
    if mach_grid is not None:
        mach_grid.to_csv(out_dir / "12_mach_field_native_grid.csv", index=False)
        make_comparison_heatmap(
            grid,
            mach_grid,
            gradient,
            comparison_png_path,
            resolution_deg=args.resolution_deg,
            projection_name=args.projection,
        )
        make_overlay_heatmap(
            grid,
            mach_grid,
            gradient,
            overlay_png_path,
            resolution_deg=args.resolution_deg,
            projection_name=args.projection,
        )
    write_summary(summary_path, gradient_csv, grid, gradient, args, mach_csv=mach_csv, mach_grid=mach_grid)

    print("\nLITTORAL 12 heatmap complete.")
    print(f"Gradient input: {gradient_csv}")
    print(f"Output PNG: {png_path}")
    if mach_grid is not None:
        print(f"Mach input: {mach_csv}")
        print(f"Comparison PNG: {comparison_png_path}")
        print(f"Overlay PNG: {overlay_png_path}")
    print(f"Output grid: {grid_path}")
    print(f"Summary: {summary_path}")
    print(
        "Full field range: "
        f"{grid['littoral_gradient_m'].min():.6g} to {grid['littoral_gradient_m'].max():.6g} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
