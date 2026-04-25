#!/usr/bin/env python3
"""
02_geospatial_robust_validation.py

Robust validation layer for LITTORAL geospatial coherence.

Uses only source records with coordinates, depth_source=reported, and z_m <= 200 m.

Reads:
  outputs/geospatial_01/01_working_geocoded_elevation_records.csv
  or outputs/merged/master_dataset.csv

Writes:
  outputs/geospatial_02/

Adds:
  - robust outlier-trimmed fits
  - full plane-rescan permutation nulls
  - source-block bootstrap
  - leverage / Cook-style influence diagnostics
  - residual maps
  - split analyses:
      littoral:  z_m >= -200 m
      deepwater: z_m <  -200 m

Run:
  python scripts/02_geospatial_robust_validation.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_01 = PROJECT_ROOT / "outputs" / "geospatial_01" / "01_working_geocoded_elevation_records.csv"
MERGED = PROJECT_ROOT / "outputs" / "merged" / "master_dataset.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "geospatial_02"


def to_numeric(value):
    return pd.to_numeric(value, errors="coerce")


def is_reported_depth_source(value):
    if pd.isna(value):
        return False
    return str(value).strip().lower() == "reported"


def reported_depth_source_mask(df):
    if "depth_source" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["depth_source"].map(is_reported_depth_source)


def reported_observation_value(value, key):
    if pd.isna(value):
        return np.nan
    if isinstance(value, dict):
        payload = value
    else:
        try:
            payload = json.loads(str(value))
        except (TypeError, json.JSONDecodeError):
            return np.nan
    return to_numeric(payload.get(key))


def choose_reported_z(df):
    out = df.copy()

    for col in ["latitude", "longitude", "elevation_m", "z_m"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    reported_source = reported_depth_source_mask(out)

    z = pd.Series(np.nan, index=out.index, dtype=float)
    source = pd.Series("", index=out.index, dtype=object)

    if "z_m" in out.columns:
        z_mask = reported_source & out["z_m"].notna()
        z.loc[z_mask] = out.loc[z_mask, "z_m"]
        source.loc[z_mask] = out.get("z_source", "precomputed_z_m")

    if "elevation_m" in out.columns:
        elevation_m_mask = reported_source & z.isna() & out["elevation_m"].notna()
        z.loc[elevation_m_mask] = out.loc[elevation_m_mask, "elevation_m"]
        source.loc[elevation_m_mask] = "elevation_m"

    if "reported_observations" in out.columns:
        reported_depth = out["reported_observations"].map(
            lambda value: reported_observation_value(value, "reported_depth_m")
        )
        reported_elevation = out["reported_observations"].map(
            lambda value: reported_observation_value(value, "reported_elevation_m")
        )

        elevation_mask = reported_source & z.isna() & reported_elevation.notna()
        z.loc[elevation_mask] = reported_elevation.loc[elevation_mask]
        source.loc[elevation_mask] = "reported_observations.reported_elevation_m"

        depth_mask = reported_source & z.isna() & reported_depth.notna()
        z.loc[depth_mask] = -reported_depth.loc[depth_mask]
        source.loc[depth_mask] = "reported_observations.reported_depth_m"

    out["z_m"] = z
    out["z_source"] = source
    return out


def normalize_longitude(lon):
    return ((lon + 180.0) % 360.0) - 180.0


def add_depth_regime(df, threshold=-200.0):
    out = df.copy()
    out["depth_regime"] = np.where(out["z_m"] < threshold, "deepwater", "littoral")
    out["depth_regime_threshold_m"] = threshold
    return out


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
    lat = df["latitude"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    z = df["z_m"].to_numpy(float)
    r = latlon_to_unit(lat, lon)

    beta, pred, resid, r2 = ols(r, z)
    gx, gy, gz = beta[1:]
    gnorm = float(np.sqrt(gx * gx + gy * gy + gz * gz))

    axis_lat = float(np.degrees(np.arcsin(gz / gnorm))) if gnorm else np.nan
    axis_lon = float(np.degrees(np.arctan2(gy, gx))) if gnorm else np.nan

    return {
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "intercept": float(beta[0]),
        "coefficients": [float(x) for x in beta[1:]],
        "axis_lat": axis_lat,
        "axis_lon": axis_lon,
        "gradient_norm_m": gnorm,
        "prediction": pred,
        "residual": resid,
    }


def plane_score(df, pole_lat, pole_lon):
    lat = df["latitude"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    z = df["z_m"].to_numpy(float)
    r = latlon_to_unit(lat, lon)

    nvec = latlon_to_unit(np.array([pole_lat]), np.array([pole_lon]))[0]
    d = np.degrees(np.arcsin(np.clip(r @ nvec, -1.0, 1.0)))

    beta, pred, resid, r2 = ols(d.reshape(-1, 1), z)

    return {
        "pole_lat": float(pole_lat),
        "pole_lon": float(pole_lon),
        "slope_m_per_degree": float(beta[1]),
        "intercept": float(beta[0]),
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "prediction": pred,
        "residual": resid,
        "distance_deg": d,
    }


def plane_rescan(df, step=2):
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
            "k": k,
            "r2": float(r2),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "amplitude_m": amp,
            "phase_deg": phase,
            "cos_coeff": float(beta[1]),
            "sin_coeff": float(beta[2]),
        })

    return rows


def robust_trim(df, qlo=0.02, qhi=0.98):
    if len(df) < 5:
        return df.copy()
    lo, hi = df["z_m"].quantile([qlo, qhi])
    return df[(df["z_m"] >= lo) & (df["z_m"] <= hi)].copy()


def full_plane_null(df, observed_r2, n_perm, rng, step=4):
    z0 = df["z_m"].to_numpy(float).copy()
    null = []
    work = df.copy()

    for _ in range(n_perm):
        work["z_m"] = rng.permutation(z0)
        best = plane_rescan(work, step=step)
        null.append(best["r2"])

    arr = np.array(null)
    p = float((np.sum(arr >= observed_r2) + 1) / (len(arr) + 1))

    return {
        "n_perm": int(n_perm),
        "step_deg": int(step),
        "p": p,
        "null_mean": float(np.mean(arr)),
        "null_p95": float(np.quantile(arr, 0.95)),
        "null_p99": float(np.quantile(arr, 0.99)),
    }


def source_block_bootstrap(df, n_boot, rng, step=4, min_n=12):
    if "source_id" not in df.columns:
        return []

    sources = sorted(df["source_id"].dropna().unique())
    if len(sources) < 2:
        return []

    rows = []

    for _ in range(n_boot):
        sampled = rng.choice(sources, size=len(sources), replace=True)
        parts = [df[df["source_id"] == sid] for sid in sampled]
        bdf = pd.concat(parts, ignore_index=True)

        if len(bdf) < min_n:
            continue

        sph = spherical_fit(bdf)
        pl = plane_rescan(bdf, step=step)

        rows.append({
            "n": int(len(bdf)),
            "source_count": int(len(sampled)),
            "spherical_r2": sph["r2"],
            "spherical_axis_lat": sph["axis_lat"],
            "spherical_axis_lon": sph["axis_lon"],
            "best_plane_r2": pl["r2"],
            "best_plane_pole_lat": pl["pole_lat"],
            "best_plane_pole_lon": pl["pole_lon"],
            "best_plane_slope_m_per_degree": pl["slope_m_per_degree"],
        })

    return rows


def influence_table(df):
    lat = df["latitude"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    z = df["z_m"].to_numpy(float)

    X = latlon_to_unit(lat, lon)
    X1 = np.column_stack([np.ones(len(X)), X])

    beta, pred, resid, _ = ols(X, z)

    p = X1.shape[1]
    n = X1.shape[0]
    mse = np.sum(resid ** 2) / max(n - p, 1)

    hat = np.sum(X1 * (X1 @ np.linalg.pinv(X1.T @ X1)), axis=1)
    cooks = (resid ** 2 / (p * mse)) * (hat / np.maximum((1 - hat) ** 2, 1e-12))

    out = df.copy()
    out["spherical_pred_z"] = pred
    out["spherical_residual_z"] = resid
    out["hat_leverage"] = hat
    out["cooks_distance_proxy"] = cooks

    return out.sort_values("cooks_distance_proxy", ascending=False)


def summarize_boot(rows):
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
                "p95": float(d[col].quantile(0.95)),
            }

    return out


def prepare_df(path, threshold):
    df = pd.read_csv(path)

    df = choose_reported_z(df)

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["z_m"] = pd.to_numeric(df["z_m"], errors="coerce")

    df = df[
        reported_depth_source_mask(df)
        & df["latitude"].notna()
        & df["longitude"].notna()
        & df["z_m"].notna()
        & (df["z_m"] <= 200.0)
    ].copy()
    df["longitude"] = normalize_longitude(df["longitude"])
    df = add_depth_regime(df, threshold=threshold)

    return df


def run_one(name, df, args, rng):
    if len(df) < args.min_n:
        return {
            "name": name,
            "n": int(len(df)),
            "status": "too_few_records",
        }

    sph = spherical_fit(df)
    plane = plane_rescan(df, step=args.plane_step)
    harm = harmonic_scan(df, kmax=args.kmax)
    best_harm = max(harm, key=lambda x: x["r2"])

    trimmed = robust_trim(df, args.trim_low, args.trim_high)

    sph_trim = spherical_fit(trimmed) if len(trimmed) >= args.min_n else None
    plane_trim = plane_rescan(trimmed, step=args.plane_step) if len(trimmed) >= args.min_n else None

    null = None
    if args.full_plane_permutations > 0:
        null = full_plane_null(
            df=df,
            observed_r2=plane["r2"],
            n_perm=args.full_plane_permutations,
            rng=rng,
            step=args.null_plane_step,
        )

    return {
        "name": name,
        "n": int(len(df)),
        "z_min": float(df["z_m"].min()),
        "z_max": float(df["z_m"].max()),
        "z_median": float(df["z_m"].median()),
        "spherical": {k: v for k, v in sph.items() if k not in ["prediction", "residual"]},
        "best_plane": {k: v for k, v in plane.items() if k not in ["prediction", "residual", "distance_deg"]},
        "harmonics": harm,
        "best_harmonic": best_harm,
        "trimmed": {
            "n": int(len(trimmed)),
            "q_low": args.trim_low,
            "q_high": args.trim_high,
            "spherical": None if sph_trim is None else {
                k: v for k, v in sph_trim.items() if k not in ["prediction", "residual"]
            },
            "best_plane": None if plane_trim is None else {
                k: v for k, v in plane_trim.items() if k not in ["prediction", "residual", "distance_deg"]
            },
        },
        "full_plane_rescan_null": null,
    }


def flatten_headline(subset_name, result):
    row = {
        "subset": subset_name,
        "n": result.get("n"),
        "status": result.get("status", "ok"),
    }

    if result.get("status") == "too_few_records":
        return row

    row.update({
        "z_min": result["z_min"],
        "z_max": result["z_max"],
        "z_median": result["z_median"],
        "spherical_r2": result["spherical"]["r2"],
        "spherical_axis_lat": result["spherical"]["axis_lat"],
        "spherical_axis_lon": result["spherical"]["axis_lon"],
        "best_plane_r2": result["best_plane"]["r2"],
        "best_plane_pole_lat": result["best_plane"]["pole_lat"],
        "best_plane_pole_lon": result["best_plane"]["pole_lon"],
        "best_plane_slope_m_per_degree": result["best_plane"]["slope_m_per_degree"],
        "trimmed_n": result["trimmed"]["n"],
        "trimmed_spherical_r2": None if result["trimmed"]["spherical"] is None else result["trimmed"]["spherical"]["r2"],
        "trimmed_spherical_axis_lat": None if result["trimmed"]["spherical"] is None else result["trimmed"]["spherical"]["axis_lat"],
        "trimmed_spherical_axis_lon": None if result["trimmed"]["spherical"] is None else result["trimmed"]["spherical"]["axis_lon"],
        "trimmed_plane_r2": None if result["trimmed"]["best_plane"] is None else result["trimmed"]["best_plane"]["r2"],
        "trimmed_plane_pole_lat": None if result["trimmed"]["best_plane"] is None else result["trimmed"]["best_plane"]["pole_lat"],
        "trimmed_plane_pole_lon": None if result["trimmed"]["best_plane"] is None else result["trimmed"]["best_plane"]["pole_lon"],
        "full_plane_null_p": None if result["full_plane_rescan_null"] is None else result["full_plane_rescan_null"]["p"],
        "full_plane_null_p95": None if result["full_plane_rescan_null"] is None else result["full_plane_rescan_null"]["null_p95"],
        "full_plane_null_p99": None if result["full_plane_rescan_null"] is None else result["full_plane_rescan_null"]["null_p99"],
        "best_harmonic_k": result["best_harmonic"]["k"],
        "best_harmonic_r2": result["best_harmonic"]["r2"],
        "best_harmonic_amplitude_m": result["best_harmonic"]["amplitude_m"],
        "best_harmonic_phase_deg": result["best_harmonic"]["phase_deg"],
    })

    return row


def write_subset_outputs(subset_name, df, result, out_dir):
    safe = subset_name.replace(":", "_").replace("/", "_")

    df.to_csv(out_dir / f"02_{safe}_working_records.csv", index=False)

    if result.get("status") != "too_few_records":
        pd.DataFrame(result["harmonics"]).to_csv(
            out_dir / f"02_{safe}_longitude_harmonics.csv",
            index=False,
        )

        infl = influence_table(df)
        infl.to_csv(out_dir / f"02_{safe}_influence_ranked_records.csv", index=False)
        infl.head(25).to_csv(out_dir / f"02_{safe}_top25_influential_records.csv", index=False)

        return infl

    return None


def make_plots(subset_name, residual_df, out_dir):
    if residual_df is None or len(residual_df) < 2:
        return

    try:
        mpl_config_dir = out_dir / "_matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return

    safe = subset_name.replace(":", "_").replace("/", "_")

    plt.figure(figsize=(11, 6))
    sc = plt.scatter(
        residual_df["longitude"],
        residual_df["latitude"],
        c=residual_df["spherical_residual_z"],
        s=28,
    )
    plt.colorbar(sc, label="Spherical residual z_m")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Spherical-gradient residual map: {subset_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"02_{safe}_spherical_residual_map.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(residual_df["spherical_pred_z"], residual_df["z_m"], s=24)
    plt.xlabel("Predicted z_m")
    plt.ylabel("Observed reported z_m")
    plt.title(f"Observed vs predicted: spherical gradient: {subset_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"02_{safe}_observed_vs_predicted_spherical.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(residual_df["hat_leverage"], residual_df["cooks_distance_proxy"], s=24)
    plt.xlabel("Leverage")
    plt.ylabel("Cook's distance proxy")
    plt.title(f"Influence diagnostics: {subset_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"02_{safe}_influence_diagnostics.png", dpi=220)
    plt.close()
    plt.close("all")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plane-step", type=int, default=2)
    parser.add_argument("--null-plane-step", type=int, default=4)
    parser.add_argument("--full-plane-permutations", type=int, default=250)
    parser.add_argument("--source-bootstrap", type=int, default=500)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--min-n", type=int, default=12)
    parser.add_argument("--trim-low", type=float, default=0.02)
    parser.add_argument("--trim-high", type=float, default=0.98)
    parser.add_argument("--depth-threshold", type=float, default=-200.0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    input_path = Path(args.input) if args.input else (IN_01 if IN_01.exists() else MERGED)
    df = prepare_df(input_path, threshold=args.depth_threshold)

    subsets = {
        "all": df,
        "littoral": df[df["depth_regime"] == "littoral"].copy(),
        "deepwater": df[df["depth_regime"] == "deepwater"].copy(),
    }

    results = {
        "metadata": {
            "input": str(input_path),
            "n": int(len(df)),
            "reported_depth_source_records_only": True,
            "selection_criteria": "depth_source == reported, latitude/longitude present, and z_m <= 200 m",
            "depth_regime_threshold_m": float(args.depth_threshold),
            "littoral_definition": f"z_m >= {args.depth_threshold}",
            "deepwater_definition": f"z_m < {args.depth_threshold}",
            "seed": args.seed,
            "chronology_used": False,
            "origin_assumptions_used": False,
            "plane_step_deg": args.plane_step,
            "null_plane_step_deg": args.null_plane_step,
            "full_plane_permutations": args.full_plane_permutations,
            "source_bootstrap": args.source_bootstrap,
        },
        "subsets": {},
        "source_block_bootstrap_summary": {},
    }

    headline_rows = []

    for subset_name, sdf in subsets.items():
        res = run_one(subset_name, sdf, args, rng)
        results["subsets"][subset_name] = res
        headline_rows.append(flatten_headline(subset_name, res))

        residual_df = write_subset_outputs(subset_name, sdf, res, OUT_DIR)
        make_plots(subset_name, residual_df, OUT_DIR)

        boot_rows = source_block_bootstrap(
            sdf,
            args.source_bootstrap,
            rng,
            step=args.null_plane_step,
            min_n=args.min_n,
        )
        results["source_block_bootstrap_summary"][subset_name] = summarize_boot(boot_rows)

        if boot_rows:
            pd.DataFrame(boot_rows).to_csv(
                OUT_DIR / f"02_{subset_name}_source_block_bootstrap.csv",
                index=False,
            )

    # Group analyses within each depth regime.
    results["groups"] = {}

    for subset_name, sdf in subsets.items():
        results["groups"][subset_name] = {}

        for col in ["record_class", "indicator_type", "indicator_subtype", "source_id"]:
            if col not in sdf.columns:
                continue

            results["groups"][subset_name][col] = []

            for key, g in sdf.groupby(col, dropna=True):
                if len(g) >= args.min_n:
                    results["groups"][subset_name][col].append(
                        run_one(f"{subset_name}:{col}:{key}", g.copy(), args, rng)
                    )

    pd.DataFrame(headline_rows).to_csv(
        OUT_DIR / "02_headline_validation_metrics.csv",
        index=False,
    )

    regime_counts = (
        df["depth_regime"]
        .value_counts()
        .rename_axis("depth_regime")
        .reset_index(name="n")
    )
    regime_counts.to_csv(OUT_DIR / "02_depth_regime_counts.csv", index=False)

    with open(OUT_DIR / "02_robust_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nLITTORAL 02 robust validation complete.")
    print(f"Input: {input_path}")
    print(f"Output: {OUT_DIR}")
    print(f"Depth split: littoral z_m >= {args.depth_threshold}; deepwater z_m < {args.depth_threshold}")

    print("\nHeadline:")
    for row in headline_rows:
        print(f"\n  [{row['subset']}]")
        for k, v in row.items():
            if k != "subset":
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
