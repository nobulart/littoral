#!/usr/bin/env python3
"""
04_littoral_axis_hypothesis_tests.py

Formal axis/plane hypothesis tests for the reported-source littoral subset.

Record selection: depth_source=reported, coordinates present, z_m <= 200 m.

Reads:
  outputs/geospatial_03/03_littoral_working_records_with_cells.csv
  or outputs/geospatial_02/02_littoral_working_records.csv

Writes:
  outputs/geospatial_04/

Purpose:
  Test whether the stable ~75–77E littoral gradient axis is:
    1. stronger than spatially constrained nulls,
    2. stable under one-per-cell resampling,
    3. distinguishable from named candidate axes/planes.

No dates. No origin assumptions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_03 = PROJECT_ROOT / "outputs" / "geospatial_03" / "03_littoral_working_records_with_cells.csv"
IN_02 = PROJECT_ROOT / "outputs" / "geospatial_02" / "02_littoral_working_records.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "geospatial_04"


def normalize_longitude(lon):
    return ((lon + 180.0) % 360.0) - 180.0


def latlon_to_unit(lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])


def angular_distance_deg(lat1, lon1, lat2, lon2):
    a = latlon_to_unit(np.array([lat1]), np.array([lon1]))[0]
    b = latlon_to_unit(np.array([lat2]), np.array([lon2]))[0]
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def axial_distance_deg(lat1, lon1, lat2, lon2):
    d = angular_distance_deg(lat1, lon1, lat2, lon2)
    return min(d, 180.0 - d)


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ beta
    resid = y - pred

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return beta, pred, resid, r2


def spherical_fit(df):
    r = latlon_to_unit(df["latitude"].to_numpy(float), df["longitude"].to_numpy(float))
    z = df["z_m"].to_numpy(float)

    beta, pred, resid, r2 = ols(r, z)
    gx, gy, gz = beta[1:]
    gnorm = float(np.sqrt(gx * gx + gy * gy + gz * gz))

    axis_lat = float(np.degrees(np.arcsin(gz / gnorm))) if gnorm else np.nan
    axis_lon = float(np.degrees(np.arctan2(gy, gx))) if gnorm else np.nan

    return {
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "axis_lat": axis_lat,
        "axis_lon": axis_lon,
        "gradient_norm_m": gnorm,
        "intercept": float(beta[0]),
        "coefficients": [float(x) for x in beta[1:]],
        "pred": pred,
        "resid": resid,
    }


def fixed_axis_score(df, axis_lat, axis_lon):
    r = latlon_to_unit(df["latitude"].to_numpy(float), df["longitude"].to_numpy(float))
    z = df["z_m"].to_numpy(float)
    axis = latlon_to_unit(np.array([axis_lat]), np.array([axis_lon]))[0]

    x = r @ axis
    beta, pred, resid, r2 = ols(x.reshape(-1, 1), z)

    return {
        "axis_lat": float(axis_lat),
        "axis_lon": float(axis_lon),
        "r2": float(r2),
        "slope_m_per_unit_vector": float(beta[1]),
        "intercept": float(beta[0]),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
    }


def fixed_plane_score(df, pole_lat, pole_lon):
    r = latlon_to_unit(df["latitude"].to_numpy(float), df["longitude"].to_numpy(float))
    z = df["z_m"].to_numpy(float)
    pole = latlon_to_unit(np.array([pole_lat]), np.array([pole_lon]))[0]

    signed_distance_deg = np.degrees(np.arcsin(np.clip(r @ pole, -1.0, 1.0)))
    beta, pred, resid, r2 = ols(signed_distance_deg.reshape(-1, 1), z)

    return {
        "pole_lat": float(pole_lat),
        "pole_lon": float(pole_lon),
        "r2": float(r2),
        "slope_m_per_degree": float(beta[1]),
        "intercept": float(beta[0]),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
    }


def plane_rescan(df, step=2):
    best = None

    for plat in np.arange(-88, 90, step):
        for plon in np.arange(-180, 180, step):
            score = fixed_plane_score(df, plat, plon)
            if best is None or score["r2"] > best["r2"]:
                best = score

    return best


def add_spatial_cells(df, lat_bin_deg, lon_bin_deg):
    out = df.copy()
    out["lat_bin"] = np.floor((out["latitude"] + 90.0) / lat_bin_deg).astype(int)
    out["lon_bin"] = np.floor((out["longitude"] + 180.0) / lon_bin_deg).astype(int)
    out["spatial_cell"] = out["lat_bin"].astype(str) + "_" + out["lon_bin"].astype(str)
    return out


def sample_one_per_cell(df, rng):
    parts = []
    for _, g in df.groupby("spatial_cell"):
        parts.append(g.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))))
    return pd.concat(parts, ignore_index=True)


def permute_within_cells(df, rng):
    out = df.copy()
    z_new = []

    for _, g in out.groupby("spatial_cell", sort=False):
        z_new.extend(rng.permutation(g["z_m"].to_numpy(float)))

    out["z_m"] = z_new
    return out


def global_permute(df, rng):
    out = df.copy()
    out["z_m"] = rng.permutation(out["z_m"].to_numpy(float))
    return out


def candidate_axes(empirical_lat, empirical_lon):
    """
    Axis candidates are gradient axes, not causal claims.
    """
    return [
        {"name": "empirical_littoral_axis", "lat": empirical_lat, "lon": empirical_lon},
        {"name": "empirical_antipode", "lat": -empirical_lat, "lon": normalize_longitude(empirical_lon + 180.0)},
        {"name": "equatorial_077E", "lat": 0.0, "lon": 77.0},
        {"name": "equatorial_075E", "lat": 0.0, "lon": 75.0},
        {"name": "ecdo_031E_equatorial", "lat": 0.0, "lon": 31.0},
        {"name": "ecdo_149W_equatorial", "lat": 0.0, "lon": -149.0},
        {"name": "excursion_170E_equatorial", "lat": 0.0, "lon": 170.0},
        {"name": "excursion_010W_equatorial", "lat": 0.0, "lon": -10.0},
        {"name": "null_000E_equatorial", "lat": 0.0, "lon": 0.0},
        {"name": "null_090E_equatorial", "lat": 0.0, "lon": 90.0},
        {"name": "null_180E_equatorial", "lat": 0.0, "lon": 180.0},
    ]


def candidate_planes(empirical_pole_lat, empirical_pole_lon):
    """
    Plane candidates are expressed as poles to great-circle planes.
    """
    return [
        {"name": "empirical_littoral_plane_pole", "lat": empirical_pole_lat, "lon": empirical_pole_lon},
        {"name": "empirical_antipodal_plane_pole", "lat": -empirical_pole_lat, "lon": normalize_longitude(empirical_pole_lon + 180.0)},
        {"name": "equatorial_077E_pole", "lat": 0.0, "lon": 77.0},
        {"name": "equatorial_075E_pole", "lat": 0.0, "lon": 75.0},
        {"name": "ecdo_031E_pole", "lat": 0.0, "lon": 31.0},
        {"name": "ecdo_149W_pole", "lat": 0.0, "lon": -149.0},
        {"name": "excursion_170E_pole", "lat": 0.0, "lon": 170.0},
        {"name": "excursion_010W_pole", "lat": 0.0, "lon": -10.0},
    ]


def bootstrap_axis_stability(df, n_boot, rng, min_n):
    rows = []

    for _ in range(n_boot):
        bdf = sample_one_per_cell(df, rng)
        if len(bdf) < min_n:
            continue

        sph = spherical_fit(bdf)
        plane = plane_rescan(bdf, step=4)

        rows.append({
            "n": int(len(bdf)),
            "spherical_r2": sph["r2"],
            "spherical_axis_lat": sph["axis_lat"],
            "spherical_axis_lon": sph["axis_lon"],
            "best_plane_r2": plane["r2"],
            "best_plane_pole_lat": plane["pole_lat"],
            "best_plane_pole_lon": plane["pole_lon"],
        })

    return rows


def null_distribution(df, n_perm, rng, mode, plane_step):
    rows = []

    for _ in range(n_perm):
        if mode == "global":
            pdf = global_permute(df, rng)
        elif mode == "within_cells":
            pdf = permute_within_cells(df, rng)
        else:
            raise ValueError(mode)

        sph = spherical_fit(pdf)
        plane = plane_rescan(pdf, step=plane_step)

        rows.append({
            "mode": mode,
            "spherical_r2": sph["r2"],
            "spherical_axis_lat": sph["axis_lat"],
            "spherical_axis_lon": sph["axis_lon"],
            "best_plane_r2": plane["r2"],
            "best_plane_pole_lat": plane["pole_lat"],
            "best_plane_pole_lon": plane["pole_lon"],
        })

    return rows


def summarize_numeric(rows):
    if not rows:
        return {}

    d = pd.DataFrame(rows)
    out = {}

    for col in d.columns:
        if pd.api.types.is_numeric_dtype(d[col]):
            out[col] = {
                "mean": float(d[col].mean()),
                "median": float(d[col].median()),
                "p05": float(d[col].quantile(0.05)),
                "p25": float(d[col].quantile(0.25)),
                "p75": float(d[col].quantile(0.75)),
                "p95": float(d[col].quantile(0.95)),
            }

    return out


def p_value_greater(observed, null_values):
    arr = np.asarray(null_values, dtype=float)
    return float((np.sum(arr >= observed) + 1) / (len(arr) + 1))


def grid_surface(df, out_dir, step=2):
    rows = []

    for lat in np.arange(-40, 42, step):
        for lon in np.arange(40, 112, step):
            score = fixed_axis_score(df, lat, lon)
            rows.append({
                "axis_lat": float(lat),
                "axis_lon": float(lon),
                "r2": score["r2"],
                "rmse": score["rmse"],
            })

    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "04_axis_r2_grid_surface.csv", index=False)
    return grid


def load_df(path, threshold, lat_bin_deg, lon_bin_deg):
    df = pd.read_csv(path)

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["z_m"] = pd.to_numeric(df["z_m"], errors="coerce")

    mask = df["latitude"].notna() & df["longitude"].notna() & df["z_m"].notna() & (df["z_m"] <= 200.0)
    if "depth_source" in df.columns:
        mask = mask & df["depth_source"].astype(str).str.strip().str.lower().eq("reported")
    df = df[mask].copy()
    df["longitude"] = normalize_longitude(df["longitude"])

    if "depth_regime" in df.columns:
        df = df[df["depth_regime"] == "littoral"].copy()

    df = df[df["z_m"] >= threshold].copy()

    if "spatial_cell" not in df.columns:
        df = add_spatial_cells(df, lat_bin_deg=lat_bin_deg, lon_bin_deg=lon_bin_deg)

    return df


def make_plots(df, boot_rows, null_rows, grid, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if boot_rows:
        b = pd.DataFrame(boot_rows)

        plt.figure(figsize=(8, 6))
        plt.scatter(b["spherical_axis_lon"], b["spherical_axis_lat"], s=18, alpha=0.45)
        plt.xlabel("Axis longitude")
        plt.ylabel("Axis latitude")
        plt.title("04 one-per-cell bootstrap axis stability")
        plt.tight_layout()
        plt.savefig(out_dir / "04_one_per_cell_axis_stability.png", dpi=220)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.hist(b["spherical_r2"], bins=25, alpha=0.8)
        plt.xlabel("Spherical R²")
        plt.ylabel("Bootstrap count")
        plt.title("04 one-per-cell spherical R²")
        plt.tight_layout()
        plt.savefig(out_dir / "04_one_per_cell_spherical_r2.png", dpi=220)
        plt.close()

    if null_rows:
        n = pd.DataFrame(null_rows)

        plt.figure(figsize=(8, 5))
        for mode, g in n.groupby("mode"):
            plt.hist(g["spherical_r2"], bins=25, alpha=0.45, label=mode)
        plt.xlabel("Null spherical R²")
        plt.ylabel("Permutation count")
        plt.title("04 spatial null distributions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "04_null_spherical_r2.png", dpi=220)
        plt.close()

    if grid is not None and len(grid):
        pivot = grid.pivot(index="axis_lat", columns="axis_lon", values="r2")

        plt.figure(figsize=(10, 6))
        plt.imshow(
            pivot.values,
            origin="lower",
            aspect="auto",
            extent=[
                float(grid["axis_lon"].min()),
                float(grid["axis_lon"].max()),
                float(grid["axis_lat"].min()),
                float(grid["axis_lat"].max()),
            ],
        )
        plt.colorbar(label="R²")
        plt.xlabel("Axis longitude")
        plt.ylabel("Axis latitude")
        plt.title("04 fixed-axis R² surface")
        plt.tight_layout()
        plt.savefig(out_dir / "04_axis_r2_grid_surface.png", dpi=220)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth-threshold", type=float, default=-200.0)
    parser.add_argument("--lat-bin-deg", type=float, default=10.0)
    parser.add_argument("--lon-bin-deg", type=float, default=20.0)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--min-n", type=int, default=12)
    parser.add_argument("--plane-step", type=int, default=4)
    parser.add_argument("--grid-step", type=int, default=2)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    input_path = Path(args.input) if args.input else (IN_03 if IN_03.exists() else IN_02)
    df = load_df(
        input_path,
        threshold=args.depth_threshold,
        lat_bin_deg=args.lat_bin_deg,
        lon_bin_deg=args.lon_bin_deg,
    )

    baseline_sph = spherical_fit(df)
    baseline_plane = plane_rescan(df, step=args.plane_step)

    axes = []
    for cand in candidate_axes(baseline_sph["axis_lat"], baseline_sph["axis_lon"]):
        score = fixed_axis_score(df, cand["lat"], cand["lon"])
        score["name"] = cand["name"]
        score["axial_distance_from_empirical_deg"] = axial_distance_deg(
            baseline_sph["axis_lat"], baseline_sph["axis_lon"], cand["lat"], cand["lon"]
        )
        axes.append(score)

    planes = []
    for cand in candidate_planes(baseline_plane["pole_lat"], baseline_plane["pole_lon"]):
        score = fixed_plane_score(df, cand["lat"], cand["lon"])
        score["name"] = cand["name"]
        score["axial_distance_from_empirical_pole_deg"] = axial_distance_deg(
            baseline_plane["pole_lat"], baseline_plane["pole_lon"], cand["lat"], cand["lon"]
        )
        planes.append(score)

    boot_rows = bootstrap_axis_stability(df, args.bootstraps, rng, args.min_n)

    null_rows = []
    null_rows.extend(null_distribution(df, args.permutations, rng, "global", args.plane_step))
    null_rows.extend(null_distribution(df, args.permutations, rng, "within_cells", args.plane_step))

    null_df = pd.DataFrame(null_rows)

    null_summary = {}
    for mode, g in null_df.groupby("mode"):
        null_summary[mode] = {
            "summary": summarize_numeric(g.to_dict("records")),
            "spherical_r2_p": p_value_greater(baseline_sph["r2"], g["spherical_r2"]),
            "best_plane_r2_p": p_value_greater(baseline_plane["r2"], g["best_plane_r2"]),
        }

    grid = grid_surface(df, OUT_DIR, step=args.grid_step)

    boot_summary = summarize_numeric(boot_rows)

    summary = {
        "metadata": {
            "input": str(input_path),
            "n": int(len(df)),
            "reported_depth_source_records_only": True,
            "selection_criteria": "depth_source == reported, latitude/longitude present, and z_m <= 200 m",
            "depth_threshold_m": float(args.depth_threshold),
            "lat_bin_deg": float(args.lat_bin_deg),
            "lon_bin_deg": float(args.lon_bin_deg),
            "bootstraps": int(args.bootstraps),
            "permutations_per_null": int(args.permutations),
            "plane_step_deg": int(args.plane_step),
            "grid_step_deg": int(args.grid_step),
            "chronology_used": False,
            "origin_assumptions_used": False,
        },
        "baseline": {
            "spherical": {k: v for k, v in baseline_sph.items() if k not in ["pred", "resid"]},
            "best_plane": baseline_plane,
        },
        "candidate_axis_scores": axes,
        "candidate_plane_scores": planes,
        "one_per_cell_bootstrap_summary": boot_summary,
        "null_summaries": null_summary,
    }

    with open(OUT_DIR / "04_littoral_axis_hypothesis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([summary["baseline"]["spherical"]]).to_csv(
        OUT_DIR / "04_baseline_spherical_axis.csv",
        index=False,
    )
    pd.DataFrame([baseline_plane]).to_csv(
        OUT_DIR / "04_baseline_best_plane.csv",
        index=False,
    )
    pd.DataFrame(axes).to_csv(
        OUT_DIR / "04_candidate_axis_scores.csv",
        index=False,
    )
    pd.DataFrame(planes).to_csv(
        OUT_DIR / "04_candidate_plane_scores.csv",
        index=False,
    )
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "04_one_per_cell_bootstrap_axes.csv",
        index=False,
    )
    null_df.to_csv(
        OUT_DIR / "04_null_distributions.csv",
        index=False,
    )
    df.to_csv(
        OUT_DIR / "04_littoral_working_records.csv",
        index=False,
    )

    make_plots(df, boot_rows, null_rows, grid, OUT_DIR)

    print("\nLITTORAL 04 littoral axis hypothesis tests complete.")
    print(f"Input: {input_path}")
    print(f"Output: {OUT_DIR}")
    print(f"Records: {len(df)}")

    print("\nBaseline spherical:")
    for k, v in summary["baseline"]["spherical"].items():
        print(f"  {k}: {v}")

    print("\nBaseline best plane:")
    for k, v in baseline_plane.items():
        print(f"  {k}: {v}")

    print("\nCandidate axes:")
    for row in sorted(axes, key=lambda r: r["r2"], reverse=True):
        print(
            f"  {row['name']}: r2={row['r2']:.4f}, "
            f"axis=({row['axis_lat']:.2f}, {row['axis_lon']:.2f}), "
            f"dist={row['axial_distance_from_empirical_deg']:.2f}°"
        )

    print("\nNull p-values:")
    for mode, values in null_summary.items():
        print(
            f"  {mode}: spherical_p={values['spherical_r2_p']}, "
            f"plane_p={values['best_plane_r2_p']}"
        )


if __name__ == "__main__":
    main()
