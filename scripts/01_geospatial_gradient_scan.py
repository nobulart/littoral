#!/usr/bin/env python3
"""
01_geospatial_gradient_scan.py

Chronology-blind first-pass geospatial coherence scan for LITTORAL merged outputs.

Uses only source records with coordinates, depth_source=reported, and z_m <= 200 m.

Reads:
  outputs/merged/master_dataset.csv
  or latest CSV in outputs/merged/

Writes:
  outputs/geospatial_01/

Core tests:
  - reported elevation/depth normalization
  - latitude / longitude gradients
  - spherical 3D gradient
  - great-circle plane sweep
  - longitude harmonic scan k=1..6
  - permutation nulls
  - leave-one-source-out sensitivity
  - split analyses:
      littoral:  z_m >= -200 m
      deepwater: z_m <  -200 m

Run:
  python scripts/01_geospatial_gradient_scan.py
  python scripts/01_geospatial_gradient_scan.py --permutations 5000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MERGED_DIR = PROJECT_ROOT / "outputs" / "merged"
OUT_DIR = PROJECT_ROOT / "outputs" / "geospatial_01"


def latest_csv(path: Path) -> Path:
    preferred = path / "master_dataset.csv"
    if preferred.exists():
        return preferred

    csvs = sorted(path.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {path}")
    return csvs[0]


def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def normalize_longitude(lon: pd.Series) -> pd.Series:
    return ((lon + 180.0) % 360.0) - 180.0


def latlon_to_unit(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])


def ols_fit(X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ beta
    resid = y - pred

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "intercept": float(beta[0]),
        "coefficients": [float(v) for v in beta[1:]],
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "n": int(len(y)),
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    if np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def simple_slope_score(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    fit = ols_fit(x.reshape(-1, 1), y)
    return {
        "slope": fit["coefficients"][0],
        "r2": fit["r2"],
        "rmse": fit["rmse"],
        "spearman": spearman(x, y),
    }


def _is_reported_depth_source(value: object) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() == "reported"


def _reported_depth_source_mask(df: pd.DataFrame) -> pd.Series:
    if "depth_source" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["depth_source"].map(_is_reported_depth_source)


def _reported_observation_value(value: object, key: str) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, dict):
        payload = value
    else:
        try:
            payload = json.loads(str(value))
        except (TypeError, json.JSONDecodeError):
            return np.nan
    return pd.to_numeric(payload.get(key), errors="coerce")


def choose_reported_elevation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the vertical value used by the scan.

    Preferred:
      elevation_m, which is the merged source-record elevation/depth value
      reported_observations.reported_elevation_m used as signed elevation fallback
      reported_observations.reported_depth_m converted to elevation fallback: z = -depth

    Values are only assigned for rows with depth_source == "reported".
    """
    out = df.copy()

    for col in ["latitude", "longitude", "elevation_m"]:
        if col in out.columns:
            out[col] = to_numeric_series(out[col])

    reported_source = _reported_depth_source_mask(out)

    z = pd.Series(np.nan, index=out.index, dtype=float)
    source = pd.Series("", index=out.index, dtype=object)

    if "elevation_m" in out.columns:
        elevation_m_mask = reported_source & out["elevation_m"].notna()
        z.loc[elevation_m_mask] = out.loc[elevation_m_mask, "elevation_m"]
        source.loc[elevation_m_mask] = "elevation_m"

    if "reported_observations" in out.columns:
        reported_depth = out["reported_observations"].map(
            lambda value: _reported_observation_value(value, "reported_depth_m")
        )
        reported_elevation = out["reported_observations"].map(
            lambda value: _reported_observation_value(value, "reported_elevation_m")
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


def add_depth_regime(df: pd.DataFrame, threshold: float = -200.0) -> pd.DataFrame:
    out = df.copy()
    out["depth_regime"] = np.where(out["z_m"] < threshold, "deepwater", "littoral")
    out["depth_regime_threshold_m"] = threshold
    return out


def analyze_subset(
    name: str,
    df: pd.DataFrame,
    permutations: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    d = df.dropna(subset=["latitude", "longitude", "z_m"]).copy()
    d["longitude"] = normalize_longitude(d["longitude"])

    if len(d) < 12:
        return {"name": name, "n": int(len(d)), "status": "too_few_records"}

    lat = d["latitude"].to_numpy(float)
    lon = d["longitude"].to_numpy(float)
    z = d["z_m"].to_numpy(float)
    r = latlon_to_unit(lat, lon)

    result: Dict[str, object] = {
        "name": name,
        "n": int(len(d)),
        "z_min": float(np.nanmin(z)),
        "z_max": float(np.nanmax(z)),
        "z_mean": float(np.nanmean(z)),
        "z_median": float(np.nanmedian(z)),
    }

    result["lat_gradient"] = simple_slope_score(lat, z)
    result["lon_gradient"] = simple_slope_score(lon, z)
    result["abs_lat_gradient"] = simple_slope_score(np.abs(lat), z)

    spherical = ols_fit(r, z)
    gx, gy, gz = spherical["coefficients"]
    gnorm = math.sqrt(gx * gx + gy * gy + gz * gz)

    if gnorm > 0:
        axis_lat = math.degrees(math.asin(gz / gnorm))
        axis_lon = math.degrees(math.atan2(gy, gx))
    else:
        axis_lat, axis_lon = np.nan, np.nan

    spherical["gradient_axis_lat"] = float(axis_lat)
    spherical["gradient_axis_lon"] = float(axis_lon)
    spherical["gradient_norm_m"] = float(gnorm)
    result["spherical_gradient"] = spherical

    best = None
    pole_lats = np.arange(-88, 90, 2)
    pole_lons = np.arange(-180, 180, 2)

    for plat in pole_lats:
        for plon in pole_lons:
            nvec = latlon_to_unit(np.array([plat]), np.array([plon]))[0]
            signed_distance_deg = np.degrees(np.arcsin(np.clip(r @ nvec, -1.0, 1.0)))
            score = simple_slope_score(signed_distance_deg, z)
            candidate = {
                "pole_lat": float(plat),
                "pole_lon": float(plon),
                "slope_m_per_degree": float(score["slope"]),
                "r2": float(score["r2"]),
                "rmse": float(score["rmse"]),
                "spearman": float(score["spearman"]),
            }
            if best is None or candidate["r2"] > best["r2"]:
                best = candidate

    result["best_great_circle_plane"] = best

    harmonics = []
    lon_rad = np.radians(lon)

    for k in range(1, 7):
        X = np.column_stack([np.cos(k * lon_rad), np.sin(k * lon_rad)])
        fit = ols_fit(X, z)
        a, b = fit["coefficients"]
        amp = math.sqrt(a * a + b * b)
        phase = math.degrees(math.atan2(-b, a))
        harmonics.append({
            "k": k,
            "r2": fit["r2"],
            "rmse": fit["rmse"],
            "amplitude_m": float(amp),
            "phase_deg": float(phase),
            "cos_coeff": float(a),
            "sin_coeff": float(b),
        })

    result["longitude_harmonics"] = harmonics
    result["best_longitude_harmonic"] = max(harmonics, key=lambda x: x["r2"])

    obs = {
        "spherical_r2": result["spherical_gradient"]["r2"],
        "plane_r2": result["best_great_circle_plane"]["r2"],
        "best_harmonic_r2": result["best_longitude_harmonic"]["r2"],
    }

    null_spherical = []
    null_plane_proxy = []
    null_harmonic = []

    nbest = latlon_to_unit(
        np.array([result["best_great_circle_plane"]["pole_lat"]]),
        np.array([result["best_great_circle_plane"]["pole_lon"]]),
    )[0]
    best_plane_distance = np.degrees(np.arcsin(np.clip(r @ nbest, -1.0, 1.0)))

    for _ in range(permutations):
        zp = rng.permutation(z)

        null_spherical.append(ols_fit(r, zp)["r2"])
        null_plane_proxy.append(simple_slope_score(best_plane_distance, zp)["r2"])

        best_hr2 = -np.inf
        for k in range(1, 7):
            X = np.column_stack([np.cos(k * lon_rad), np.sin(k * lon_rad)])
            best_hr2 = max(best_hr2, ols_fit(X, zp)["r2"])
        null_harmonic.append(best_hr2)

    def pval(observed: float, null_values: List[float]) -> float:
        arr = np.asarray(null_values)
        return float((np.sum(arr >= observed) + 1) / (len(arr) + 1))

    result["permutation_nulls"] = {
        "permutations": permutations,
        "spherical_r2_p": pval(obs["spherical_r2"], null_spherical),
        "plane_r2_fixed_best_plane_p": pval(obs["plane_r2"], null_plane_proxy),
        "best_harmonic_r2_p": pval(obs["best_harmonic_r2"], null_harmonic),
        "note": "Plane null is a fixed-best-plane proxy, not a full plane-rescan null.",
    }

    return result


def headline_from_result(prefix: str, result: Dict[str, object]) -> Dict[str, object]:
    if result.get("status") == "too_few_records":
        return {
            "subset": prefix,
            "n": result.get("n"),
            "status": result.get("status"),
        }

    return {
        "subset": prefix,
        "n": result["n"],
        "z_min": result["z_min"],
        "z_max": result["z_max"],
        "z_median": result["z_median"],
        "lat_r2": result["lat_gradient"]["r2"],
        "lon_r2": result["lon_gradient"]["r2"],
        "abs_lat_r2": result["abs_lat_gradient"]["r2"],
        "spherical_r2": result["spherical_gradient"]["r2"],
        "spherical_axis_lat": result["spherical_gradient"]["gradient_axis_lat"],
        "spherical_axis_lon": result["spherical_gradient"]["gradient_axis_lon"],
        "best_plane_r2": result["best_great_circle_plane"]["r2"],
        "best_plane_pole_lat": result["best_great_circle_plane"]["pole_lat"],
        "best_plane_pole_lon": result["best_great_circle_plane"]["pole_lon"],
        "best_plane_slope_m_per_degree": result["best_great_circle_plane"]["slope_m_per_degree"],
        "best_harmonic_k": result["best_longitude_harmonic"]["k"],
        "best_harmonic_r2": result["best_longitude_harmonic"]["r2"],
        "spherical_r2_p": result["permutation_nulls"]["spherical_r2_p"],
        "plane_r2_fixed_best_plane_p": result["permutation_nulls"]["plane_r2_fixed_best_plane_p"],
        "best_harmonic_r2_p": result["permutation_nulls"]["best_harmonic_r2_p"],
    }


def make_plots(df: pd.DataFrame, results: Dict[str, object], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    d = df.dropna(subset=["latitude", "longitude", "z_m"]).copy()
    d["longitude"] = normalize_longitude(d["longitude"])

    def plot_basic(subset_name: str, g: pd.DataFrame) -> None:
        if len(g) < 2:
            return

        safe_name = subset_name.replace(":", "_").replace("/", "_")

        plt.figure(figsize=(11, 6))
        sc = plt.scatter(g["longitude"], g["latitude"], c=g["z_m"], s=26)
        plt.colorbar(sc, label="Reported elevation/depth z_m")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"LITTORAL reported elevation/depth by location: {subset_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"01_{safe_name}_global_scatter_lon_lat_z.png", dpi=220)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.scatter(g["latitude"], g["z_m"], s=22)
        plt.xlabel("Latitude")
        plt.ylabel("Reported elevation/depth z_m")
        plt.title(f"Elevation/depth vs latitude: {subset_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"01_{safe_name}_z_vs_latitude.png", dpi=220)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.scatter(g["longitude"], g["z_m"], s=22)
        plt.xlabel("Longitude")
        plt.ylabel("Reported elevation/depth z_m")
        plt.title(f"Elevation/depth vs longitude: {subset_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"01_{safe_name}_z_vs_longitude.png", dpi=220)
        plt.close()

        if subset_name in results and results[subset_name].get("status") != "too_few_records":
            h = pd.DataFrame(results[subset_name]["longitude_harmonics"])
            plt.figure(figsize=(8, 5))
            plt.plot(h["k"], h["r2"], marker="o")
            plt.xlabel("Longitude harmonic k")
            plt.ylabel("R²")
            plt.title(f"Longitude harmonic coherence scan: {subset_name}")
            plt.tight_layout()
            plt.savefig(out_dir / f"01_{safe_name}_longitude_harmonic_r2.png", dpi=220)
            plt.close()

    plot_basic("all", d)

    for regime in ["littoral", "deepwater"]:
        g = d[d["depth_regime"] == regime].copy()
        plot_basic(regime, g)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--min-group-n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth-threshold", type=float, default=-200.0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_csv = Path(args.input) if args.input else latest_csv(MERGED_DIR)
    df_raw = pd.read_csv(input_csv)
    df = choose_reported_elevation(df_raw)

    usable = df[
        _reported_depth_source_mask(df)
        & df["latitude"].notna()
        & df["longitude"].notna()
        & df["z_m"].notna()
        & (df["z_m"] <= 200.0)
    ].copy()
    usable["longitude"] = normalize_longitude(usable["longitude"])
    usable = add_depth_regime(usable, threshold=args.depth_threshold)

    rng = np.random.default_rng(args.seed)

    results: Dict[str, object] = {
        "metadata": {
            "input_csv": str(input_csv),
            "total_records": int(len(df)),
            "usable_geocoded_reported_records": int(len(usable)),
            "reported_depth_source_records_only": True,
            "selection_criteria": "depth_source == reported, latitude/longitude present, and z_m <= 200 m",
            "depth_regime_threshold_m": float(args.depth_threshold),
            "littoral_definition": f"z_m >= {args.depth_threshold}",
            "deepwater_definition": f"z_m < {args.depth_threshold}",
            "permutations": int(args.permutations),
            "seed": int(args.seed),
            "chronology_used": False,
            "origin_assumptions_used": False,
        }
    }

    results["all"] = analyze_subset("all", usable, args.permutations, rng)

    for regime in ["littoral", "deepwater"]:
        g = usable[usable["depth_regime"] == regime].copy()
        results[regime] = analyze_subset(regime, g, args.permutations, rng)

    group_results = {}

    for group_col in ["depth_regime", "record_class", "indicator_type", "indicator_subtype", "source_id"]:
        if group_col not in usable.columns:
            continue

        group_results[group_col] = []
        for key, g in usable.groupby(group_col, dropna=True):
            if len(g) >= args.min_group_n:
                group_results[group_col].append(
                    analyze_subset(f"{group_col}:{key}", g, args.permutations, rng)
                )

    results["groups"] = group_results

    loo = []
    if "source_id" in usable.columns:
        counts = usable["source_id"].value_counts()
        for sid, n in counts.items():
            if n >= args.min_group_n:
                g = usable[usable["source_id"] != sid]
                res = analyze_subset(
                    f"leave_out_source:{sid}",
                    g,
                    max(100, args.permutations // 5),
                    rng,
                )
                loo.append({
                    "left_out_source_id": str(sid),
                    "left_out_n": int(n),
                    "remaining_n": int(len(g)),
                    "spherical_r2": res.get("spherical_gradient", {}).get("r2"),
                    "best_plane_r2": res.get("best_great_circle_plane", {}).get("r2"),
                    "best_plane_pole_lat": res.get("best_great_circle_plane", {}).get("pole_lat"),
                    "best_plane_pole_lon": res.get("best_great_circle_plane", {}).get("pole_lon"),
                    "best_harmonic_k": res.get("best_longitude_harmonic", {}).get("k"),
                    "best_harmonic_r2": res.get("best_longitude_harmonic", {}).get("r2"),
                })

    results["leave_one_source_out"] = loo

    usable.to_csv(OUT_DIR / "01_working_geocoded_elevation_records.csv", index=False)

    with open(OUT_DIR / "01_geospatial_gradient_scan_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    headline_rows = [
        headline_from_result("all", results["all"]),
        headline_from_result("littoral", results["littoral"]),
        headline_from_result("deepwater", results["deepwater"]),
    ]
    pd.DataFrame(headline_rows).to_csv(OUT_DIR / "01_headline_metrics.csv", index=False)

    for subset_name in ["all", "littoral", "deepwater"]:
        res = results[subset_name]
        if res.get("status") != "too_few_records":
            pd.DataFrame(res["longitude_harmonics"]).to_csv(
                OUT_DIR / f"01_{subset_name}_longitude_harmonics.csv",
                index=False,
            )

    if loo:
        pd.DataFrame(loo).to_csv(OUT_DIR / "01_leave_one_source_out.csv", index=False)

    regime_counts = (
        usable["depth_regime"]
        .value_counts()
        .rename_axis("depth_regime")
        .reset_index(name="n")
    )
    regime_counts.to_csv(OUT_DIR / "01_depth_regime_counts.csv", index=False)

    make_plots(usable, results, OUT_DIR)

    print("\nLITTORAL 01 geospatial gradient scan complete.")
    print(f"Input: {input_csv}")
    print(f"Usable records: {len(usable)} / {len(df)}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Depth split: littoral z_m >= {args.depth_threshold}; deepwater z_m < {args.depth_threshold}")

    print("\nHeadline:")
    for row in headline_rows:
        print(f"\n  [{row['subset']}]")
        for k, v in row.items():
            if k != "subset":
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
