#!/usr/bin/env python3
"""
03_littoral_balanced_geometry.py

Spatial/source-balanced validation for the LITTORAL littoral subset.

Record selection: depth_source=reported, coordinates present, z_m <= 200 m.

Purpose:
  Test whether the reported-source littoral geometry survives after reducing
  regional clustering and source dominance.

Reads:
  outputs/geospatial_02/02_littoral_working_records.csv
  or outputs/geospatial_01/01_working_geocoded_elevation_records.csv

Writes:
  outputs/geospatial_03/

Tests:
  - littoral-only z_m >= -200 m
  - equal-area-ish lat/lon bin balancing
  - one-record-per-spatial-cell bootstrap
  - one-record-per-source bootstrap
  - combined spatial + source bootstrap
  - best spherical gradient
  - best great-circle plane
  - longitude harmonic scan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_02 = PROJECT_ROOT / "outputs" / "geospatial_02" / "02_littoral_working_records.csv"
IN_01 = PROJECT_ROOT / "outputs" / "geospatial_01" / "01_working_geocoded_elevation_records.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "geospatial_03"


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
    r = latlon_to_unit(
        df["latitude"].to_numpy(float),
        df["longitude"].to_numpy(float),
    )
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
    }


def plane_score(df, pole_lat, pole_lon):
    r = latlon_to_unit(
        df["latitude"].to_numpy(float),
        df["longitude"].to_numpy(float),
    )
    z = df["z_m"].to_numpy(float)

    nvec = latlon_to_unit(np.array([pole_lat]), np.array([pole_lon]))[0]
    d = np.degrees(np.arcsin(np.clip(r @ nvec, -1.0, 1.0)))

    beta, pred, resid, r2 = ols(d.reshape(-1, 1), z)

    return {
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "pole_lat": float(pole_lat),
        "pole_lon": float(pole_lon),
        "slope_m_per_degree": float(beta[1]),
        "intercept": float(beta[0]),
    }


def plane_rescan(df, step=4):
    best = None
    for plat in np.arange(-88, 90, step):
        for plon in np.arange(-180, 180, step):
            s = plane_score(df, plat, plon)
            if best is None or s["r2"] > best["r2"]:
                best = s
    return best


def harmonic_scan(df, kmax=8):
    lon = np.radians(df["longitude"].to_numpy(float))
    z = df["z_m"].to_numpy(float)

    rows = []
    for k in range(1, kmax + 1):
        X = np.column_stack([np.cos(k * lon), np.sin(k * lon)])
        beta, pred, resid, r2 = ols(X, z)
        amp = float(np.sqrt(beta[1] ** 2 + beta[2] ** 2))
        phase = float(np.degrees(np.arctan2(-beta[2], beta[1])))

        rows.append({
            "k": int(k),
            "r2": float(r2),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "amplitude_m": amp,
            "phase_deg": phase,
            "cos_coeff": float(beta[1]),
            "sin_coeff": float(beta[2]),
        })

    return rows


def best_harmonic(df, kmax=8):
    rows = harmonic_scan(df, kmax=kmax)
    return max(rows, key=lambda r: r["r2"])


def add_spatial_cells(df, lat_bin_deg=10.0, lon_bin_deg=20.0):
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


def sample_one_per_source(df, rng):
    if "source_id" not in df.columns:
        return df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).copy()

    parts = []
    for _, g in df.groupby("source_id"):
        parts.append(g.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))))
    return pd.concat(parts, ignore_index=True)


def sample_cell_then_source(df, rng):
    spatial = sample_one_per_cell(df, rng)

    if "source_id" not in spatial.columns:
        return spatial

    parts = []
    for _, g in spatial.groupby("source_id"):
        parts.append(g.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))))
    return pd.concat(parts, ignore_index=True)


def evaluate(df, plane_step=4, kmax=8):
    sph = spherical_fit(df)
    plane = plane_rescan(df, step=plane_step)
    harm = best_harmonic(df, kmax=kmax)

    return {
        "n": int(len(df)),
        "z_min": float(df["z_m"].min()),
        "z_max": float(df["z_m"].max()),
        "z_median": float(df["z_m"].median()),
        "spherical_r2": sph["r2"],
        "spherical_axis_lat": sph["axis_lat"],
        "spherical_axis_lon": sph["axis_lon"],
        "spherical_rmse": sph["rmse"],
        "best_plane_r2": plane["r2"],
        "best_plane_pole_lat": plane["pole_lat"],
        "best_plane_pole_lon": plane["pole_lon"],
        "best_plane_slope_m_per_degree": plane["slope_m_per_degree"],
        "best_harmonic_k": harm["k"],
        "best_harmonic_r2": harm["r2"],
        "best_harmonic_amplitude_m": harm["amplitude_m"],
        "best_harmonic_phase_deg": harm["phase_deg"],
    }


def bootstrap_scheme(df, scheme, n_boot, rng, min_n, plane_step, kmax):
    rows = []

    for _ in range(n_boot):
        if scheme == "one_per_cell":
            bdf = sample_one_per_cell(df, rng)
        elif scheme == "one_per_source":
            bdf = sample_one_per_source(df, rng)
        elif scheme == "cell_then_source":
            bdf = sample_cell_then_source(df, rng)
        elif scheme == "ordinary":
            bdf = df.sample(
                n=len(df),
                replace=True,
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).copy()
        else:
            raise ValueError(f"Unknown bootstrap scheme: {scheme}")

        if len(bdf) < min_n:
            continue

        row = evaluate(bdf, plane_step=plane_step, kmax=kmax)
        row["scheme"] = scheme
        rows.append(row)

    return rows


def summarize(rows):
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
    else:
        df = df[df["z_m"] >= threshold].copy()

    df = df[df["z_m"] >= threshold].copy()
    df = add_spatial_cells(df, lat_bin_deg=lat_bin_deg, lon_bin_deg=lon_bin_deg)

    return df


def make_plots(df, all_rows, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plt.figure(figsize=(11, 6))
    sc = plt.scatter(df["longitude"], df["latitude"], c=df["z_m"], s=28)
    plt.colorbar(sc, label="Reported elevation/depth z_m")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("LITTORAL littoral subset: reported depths")
    plt.tight_layout()
    plt.savefig(out_dir / "03_littoral_reported_depth_map.png", dpi=220)
    plt.close()

    cell_counts = (
        df.groupby("spatial_cell")
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    cell_counts.to_csv(out_dir / "03_spatial_cell_counts.csv", index=False)

    if all_rows:
        b = pd.DataFrame(all_rows)

        for metric in ["spherical_r2", "best_plane_r2", "best_harmonic_r2"]:
            plt.figure(figsize=(9, 5))
            for scheme, g in b.groupby("scheme"):
                plt.hist(g[metric], bins=25, alpha=0.45, label=scheme)
            plt.xlabel(metric)
            plt.ylabel("Bootstrap count")
            plt.title(f"Balanced bootstrap distribution: {metric}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"03_bootstrap_{metric}.png", dpi=220)
            plt.close()

        plt.figure(figsize=(8, 6))
        for scheme, g in b.groupby("scheme"):
            plt.scatter(g["spherical_axis_lon"], g["spherical_axis_lat"], s=14, label=scheme, alpha=0.55)
        plt.xlabel("Spherical axis longitude")
        plt.ylabel("Spherical axis latitude")
        plt.title("Bootstrap spherical-axis stability")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "03_bootstrap_axis_stability.png", dpi=220)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth-threshold", type=float, default=-200.0)
    parser.add_argument("--lat-bin-deg", type=float, default=10.0)
    parser.add_argument("--lon-bin-deg", type=float, default=20.0)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--min-n", type=int, default=12)
    parser.add_argument("--plane-step", type=int, default=4)
    parser.add_argument("--kmax", type=int, default=8)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    input_path = Path(args.input) if args.input else (IN_02 if IN_02.exists() else IN_01)
    df = load_df(
        input_path,
        threshold=args.depth_threshold,
        lat_bin_deg=args.lat_bin_deg,
        lon_bin_deg=args.lon_bin_deg,
    )

    baseline = evaluate(df, plane_step=args.plane_step, kmax=args.kmax)

    schemes = ["ordinary", "one_per_cell", "one_per_source", "cell_then_source"]

    all_rows = []
    summaries = {}

    for scheme in schemes:
        rows = bootstrap_scheme(
            df=df,
            scheme=scheme,
            n_boot=args.bootstraps,
            rng=rng,
            min_n=args.min_n,
            plane_step=args.plane_step,
            kmax=args.kmax,
        )
        all_rows.extend(rows)
        summaries[scheme] = summarize(rows)

        if rows:
            pd.DataFrame(rows).to_csv(
                OUT_DIR / f"03_bootstrap_{scheme}.csv",
                index=False,
            )

    summary = {
        "metadata": {
            "input": str(input_path),
            "n_littoral": int(len(df)),
            "reported_depth_source_records_only": True,
            "selection_criteria": "depth_source == reported, latitude/longitude present, and z_m <= 200 m",
            "depth_threshold_m": float(args.depth_threshold),
            "lat_bin_deg": float(args.lat_bin_deg),
            "lon_bin_deg": float(args.lon_bin_deg),
            "bootstraps": int(args.bootstraps),
            "min_n": int(args.min_n),
            "plane_step_deg": int(args.plane_step),
            "kmax": int(args.kmax),
            "chronology_used": False,
            "origin_assumptions_used": False,
        },
        "baseline_littoral": baseline,
        "bootstrap_summaries": summaries,
    }

    with open(OUT_DIR / "03_littoral_balanced_geometry_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([baseline]).to_csv(
        OUT_DIR / "03_baseline_littoral_metrics.csv",
        index=False,
    )

    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            OUT_DIR / "03_all_bootstrap_metrics.csv",
            index=False,
        )

    df.to_csv(OUT_DIR / "03_littoral_working_records_with_cells.csv", index=False)

    make_plots(df, all_rows, OUT_DIR)

    print("\nLITTORAL 03 balanced littoral geometry complete.")
    print(f"Input: {input_path}")
    print(f"Output: {OUT_DIR}")
    print(f"Records: {len(df)}")
    print(f"Spatial bins: {args.lat_bin_deg}° latitude × {args.lon_bin_deg}° longitude")

    print("\nBaseline:")
    for k, v in baseline.items():
        print(f"  {k}: {v}")

    print("\nBootstrap medians:")
    for scheme, s in summaries.items():
        print(f"\n  [{scheme}]")
        for metric in ["spherical_r2", "best_plane_r2", "best_harmonic_r2", "spherical_axis_lat", "spherical_axis_lon"]:
            if metric in s:
                print(f"    {metric}: median={s[metric]['median']}, p05={s[metric]['p05']}, p95={s[metric]['p95']}")


if __name__ == "__main__":
    main()
