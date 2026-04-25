#!/usr/bin/env python3
"""
11_littoral_conditioned_mach_discrimination.py

Tests whether the *best-performing* LITTORAL inverse axes are preferentially
Mach-admissible after conditioning on comparable geometry.

Adds:
  1. top-k candidate filtering by score/R²
  2. matched-random nulls using candidate angle distribution
  3. path morphology metrics:
       - ridge crossings
       - longest low-M run
       - longest high-M run
       - mean |dM/dtheta|

Reads:
  outputs/geospatial_10/10_littoral_candidate_mach_path_metrics.csv
  outputs/geospatial_07/07_top250_rotation_candidates.csv
  outputs/geospatial_09/09_constrained_joint_scan.csv
  outputs/mach.csv

Writes:
  outputs/geospatial_11/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "geospatial_11"

MACH_FILE = PROJECT_ROOT / "outputs" / "mach.csv"

CANDIDATE_FILES = [
    PROJECT_ROOT / "outputs" / "geospatial_07" / "07_top250_rotation_candidates.csv",
    PROJECT_ROOT / "outputs" / "geospatial_09" / "09_top250_constrained_joint_candidates.csv",
    PROJECT_ROOT / "outputs" / "geospatial_09" / "09_constrained_joint_scan.csv",
]


def normalize_longitude(lon):
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def latlon_to_unit(lat_deg, lon_deg):
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])


def unit_to_latlon(v):
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    lat = np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    return lat, normalize_longitude(lon)


def axis_vector(lat_deg, lon_deg):
    return latlon_to_unit([lat_deg], [lon_deg])[0]


def rotate_vectors(v, axis, angle_deg):
    v = np.asarray(v, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    theta = np.radians(np.asarray(angle_deg, dtype=float))
    if theta.ndim == 0:
        theta = np.full((v.shape[0],), float(theta))

    theta = theta.reshape(-1, 1)

    cross = np.cross(axis, v)
    dot = np.sum(v * axis, axis=-1, keepdims=True)

    return (
        v * np.cos(theta)
        + cross * np.sin(theta)
        + axis.reshape(1, 3) * dot * (1.0 - np.cos(theta))
    )


def read_csv_flexible(path):
    return pd.read_csv(path, sep=None, engine="python")


def load_mach(path):
    df = read_csv_flexible(path)
    lower = {c.lower().strip(): c for c in df.columns}

    lat_col = lower.get("latitude", lower.get("lat"))
    lon_col = lower.get("longitude", lower.get("lon"))
    m_col = lower.get("correlation", lower.get("m", lower.get("mach")))

    if lat_col is None or lon_col is None or m_col is None:
        raise ValueError(f"Could not infer Mach columns from {list(df.columns)}")

    out = pd.DataFrame({
        "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        "M": pd.to_numeric(df[m_col], errors="coerce"),
    }).dropna()

    out["lon"] = normalize_longitude(out["lon"])
    return out.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)


def build_interpolator(mach_df):
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    points = mach_df[["lon", "lat"]].to_numpy(float)
    values = mach_df["M"].to_numpy(float)

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


def load_candidates(paths, max_rows_per_file):
    frames = []

    for p in paths:
        if not p.exists():
            continue

        df = pd.read_csv(p)

        if not {"axis_lat", "axis_lon"}.issubset(df.columns):
            continue

        if "angle_deg" not in df.columns:
            df["angle_deg"] = np.nan

        if "score" not in df.columns:
            df["score"] = np.nan

        if "r2" not in df.columns:
            if "constrained_joint_r2" in df.columns:
                df["r2"] = df["constrained_joint_r2"]
            elif "joint_r2" in df.columns:
                df["r2"] = df["joint_r2"]
            else:
                df["r2"] = np.nan

        df["source_file"] = p.name
        frames.append(df.head(max_rows_per_file).copy())

    if not frames:
        raise FileNotFoundError("No candidate files found.")

    out = pd.concat(frames, ignore_index=True)
    out["axis_lat"] = pd.to_numeric(out["axis_lat"], errors="coerce")
    out["axis_lon"] = pd.to_numeric(out["axis_lon"], errors="coerce")
    out["angle_deg"] = pd.to_numeric(out["angle_deg"], errors="coerce")
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out["r2"] = pd.to_numeric(out["r2"], errors="coerce")

    out = out.dropna(subset=["axis_lat", "axis_lon"]).copy()
    out["axis_lon"] = normalize_longitude(out["axis_lon"])

    return out.drop_duplicates(
        subset=["source_file", "axis_lat", "axis_lon", "angle_deg"]
    ).reset_index(drop=True)


def choose_conditioned_subset(candidates, metric, top_fraction):
    df = candidates.copy()

    if metric not in df.columns or df[metric].notna().sum() == 0:
        raise ValueError(f"Cannot condition on unavailable metric: {metric}")

    df = df.dropna(subset=[metric]).copy()
    n = max(1, int(np.ceil(len(df) * top_fraction)))

    return df.sort_values(metric, ascending=False).head(n).reset_index(drop=True)


def longest_run(mask, value=True):
    best = 0
    cur = 0
    for x in mask:
        if bool(x) == value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def path_metrics(axis_lat, axis_lon, eval_m, basin_lat, basin_lon, threshold, sweep_step_deg):
    axis = axis_vector(axis_lat, axis_lon)
    start = axis_vector(basin_lat, basin_lon).reshape(1, 3)

    theta = np.arange(0.0, 360.0 + 0.001, sweep_step_deg)
    path = rotate_vectors(np.repeat(start, len(theta), axis=0), axis, theta)

    plat, plon = unit_to_latlon(path)
    m = eval_m(plat, plon)

    low = m < threshold
    high = m >= threshold

    crossings = int(np.sum(low[1:] != low[:-1]))
    dm = np.diff(m) / sweep_step_deg

    penalty = np.where(low, threshold - m, 0.0)

    return {
        "path_m_mean": float(np.mean(m)),
        "path_m_median": float(np.median(m)),
        "path_m_min": float(np.min(m)),
        "path_m_max": float(np.max(m)),
        "path_frac_below_threshold": float(np.mean(low)),
        "path_integrated_penalty": float(np.sum(penalty) * sweep_step_deg),
        "ridge_crossings": crossings,
        "longest_low_run_deg": float(longest_run(low, True) * sweep_step_deg),
        "longest_high_run_deg": float(longest_run(high, True) * sweep_step_deg),
        "mean_abs_dM_dtheta": float(np.mean(np.abs(dm))),
        "max_abs_dM_dtheta": float(np.max(np.abs(dm))) if len(dm) else np.nan,
    }


def infer_basin(mach_df, top_n):
    top = mach_df.sort_values("M", ascending=False).head(top_n)
    lat = float(top["lat"].mean())
    lon = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(top["lon"]))))))
    return lat, lon


def matched_random_axes(candidates, n_per_candidate, rng):
    rows = []

    for _, row in candidates.iterrows():
        angle = row.get("angle_deg", np.nan)

        for _ in range(n_per_candidate):
            z = rng.uniform(-1.0, 1.0)
            lat = np.degrees(np.arcsin(z))
            lon = rng.uniform(-180.0, 180.0)

            rows.append({
                "axis_lat": lat,
                "axis_lon": lon,
                "angle_deg": angle,
                "source_file": "matched_random",
                "r2": row.get("r2", np.nan),
                "score": row.get("score", np.nan),
            })

    return pd.DataFrame(rows)


def compute_metrics(df, eval_m, basin_lat, basin_lon, threshold, sweep_step_deg):
    rows = []

    for _, row in df.iterrows():
        m = path_metrics(
            axis_lat=float(row["axis_lat"]),
            axis_lon=float(row["axis_lon"]),
            eval_m=eval_m,
            basin_lat=basin_lat,
            basin_lon=basin_lon,
            threshold=threshold,
            sweep_step_deg=sweep_step_deg,
        )

        out = row.to_dict()
        out.update(m)
        rows.append(out)

    return pd.DataFrame(rows)


def summarize(obs, rnd):
    metrics = [
        "path_m_mean",
        "path_m_min",
        "path_frac_below_threshold",
        "path_integrated_penalty",
        "ridge_crossings",
        "longest_low_run_deg",
        "longest_high_run_deg",
        "mean_abs_dM_dtheta",
    ]

    rows = []

    for metric in metrics:
        o = obs[metric].dropna().to_numpy(float)
        r = rnd[metric].dropna().to_numpy(float)

        if len(o) == 0 or len(r) == 0:
            continue

        obs_med = float(np.median(o))

        rows.append({
            "metric": metric,
            "observed_median": obs_med,
            "observed_mean": float(np.mean(o)),
            "matched_random_median": float(np.median(r)),
            "matched_random_mean": float(np.mean(r)),
            "p_low": float((np.sum(r <= obs_med) + 1) / (len(r) + 1)),
            "p_high": float((np.sum(r >= obs_med) + 1) / (len(r) + 1)),
        })

    return pd.DataFrame(rows)


def make_plots(obs, rnd, summary, out_dir, label):
    import matplotlib.pyplot as plt

    for metric in [
        "path_integrated_penalty",
        "path_frac_below_threshold",
        "path_m_min",
        "ridge_crossings",
        "longest_low_run_deg",
        "mean_abs_dM_dtheta",
    ]:
        plt.figure(figsize=(9, 5))
        plt.hist(rnd[metric], bins=35, alpha=0.6, label="matched random")
        plt.axvline(obs[metric].median(), linestyle="--", label="observed top-candidate median")
        plt.xlabel(metric)
        plt.ylabel("Count")
        plt.title(f"11 conditioned Mach discrimination: {metric} ({label})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"11_{label}_{metric}.png", dpi=220)
        plt.close()

    plt.figure(figsize=(10, 5))
    x = np.arange(len(summary))
    plt.bar(x, summary["observed_median"] - summary["matched_random_median"])
    plt.xticks(x, summary["metric"], rotation=60, ha="right")
    plt.ylabel("Observed median - matched-random median")
    plt.title(f"11 conditioned effect sizes ({label})")
    plt.tight_layout()
    plt.savefig(out_dir / f"11_{label}_effect_sizes.png", dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mach", default=str(MACH_FILE))
    parser.add_argument("--condition-metric", choices=["score", "r2"], default="score")
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--max-rows-per-file", type=int, default=250)
    parser.add_argument("--matched-random-per-candidate", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep-step-deg", type=float, default=2.0)
    parser.add_argument("--low-m-quantile", type=float, default=0.20)
    parser.add_argument("--basin-top-n", type=int, default=5)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    mach_df = load_mach(Path(args.mach))
    eval_m = build_interpolator(mach_df)

    basin_lat, basin_lon = infer_basin(mach_df, args.basin_top_n)
    threshold = float(np.quantile(mach_df["M"], args.low_m_quantile))

    candidates = load_candidates(CANDIDATE_FILES, args.max_rows_per_file)
    top = choose_conditioned_subset(candidates, args.condition_metric, args.top_fraction)

    rnd = matched_random_axes(top, args.matched_random_per_candidate, rng)

    obs_metrics = compute_metrics(
        top, eval_m, basin_lat, basin_lon, threshold, args.sweep_step_deg
    )
    rnd_metrics = compute_metrics(
        rnd, eval_m, basin_lat, basin_lon, threshold, args.sweep_step_deg
    )

    summary = summarize(obs_metrics, rnd_metrics)

    label = f"top_{int(args.top_fraction * 100)}pct_by_{args.condition_metric}"

    obs_metrics.to_csv(OUT_DIR / f"11_observed_{label}_mach_metrics.csv", index=False)
    rnd_metrics.to_csv(OUT_DIR / f"11_matched_random_{label}_mach_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / f"11_summary_{label}.csv", index=False)

    with open(OUT_DIR / f"11_summary_{label}.json", "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "mach": str(args.mach),
                "condition_metric": args.condition_metric,
                "top_fraction": args.top_fraction,
                "n_candidates_total": int(len(candidates)),
                "n_observed_top": int(len(obs_metrics)),
                "n_matched_random": int(len(rnd_metrics)),
                "basin_lat": basin_lat,
                "basin_lon": basin_lon,
                "low_m_threshold": threshold,
                "sweep_step_deg": args.sweep_step_deg,
            },
            "summary": summary.to_dict("records"),
        }, f, indent=2)

    make_plots(obs_metrics, rnd_metrics, summary, OUT_DIR, label)

    print("\nLITTORAL 11 conditioned Mach discrimination complete.")
    print(f"Condition: top {args.top_fraction:.0%} by {args.condition_metric}")
    print(f"Observed top candidates: {len(obs_metrics)}")
    print(f"Matched random axes: {len(rnd_metrics)}")
    print(f"Output: {OUT_DIR}")

    print("\nSummary:")
    for _, row in summary.iterrows():
        print(
            f"  {row['metric']}: "
            f"obs_med={row['observed_median']:.6g}, "
            f"rnd_med={row['matched_random_median']:.6g}, "
            f"p_low={row['p_low']:.4f}, "
            f"p_high={row['p_high']:.4f}"
        )


if __name__ == "__main__":
    main()