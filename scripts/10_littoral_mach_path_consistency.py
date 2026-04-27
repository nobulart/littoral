#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "geospatial_10"

CANDIDATE_FILES = [
    PROJECT_ROOT / "outputs" / "geospatial_07" / "07_top250_rotation_candidates.csv",
    PROJECT_ROOT / "outputs" / "geospatial_09" / "09_top250_constrained_joint_candidates.csv",
    PROJECT_ROOT / "outputs" / "geospatial_09" / "09_constrained_joint_scan.csv",
]

MACH_CANDIDATES = [
    PROJECT_ROOT / "mach.csv",
    PROJECT_ROOT / "outputs" / "mach.csv",
    PROJECT_ROOT / "outputs" / "mach" / "mach.csv",
    PROJECT_ROOT / "outputs" / "mach" / "stage7_euler_likelihood_metrics.csv",
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


def angular_distance_deg(lat1, lon1, lat2, lon2):
    p1 = latlon_to_unit(lat1, lon1)
    p2 = latlon_to_unit(lat2, lon2)
    dot = np.sum(p1 * p2, axis=1)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def find_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def read_csv_flexible(path):
    return pd.read_csv(path, sep=None, engine="python")


def infer_columns(df):
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

    for key in [
        "m",
        "mach",
        "correlation",
        "compatibility",
        "compatibility_m",
        "anisotropic_variance",
    ]:
        if key in lower:
            m_col = lower[key]
            break

    if lat_col is None or lon_col is None or m_col is None:
        raise ValueError(f"Could not infer Mach columns. Found columns: {list(df.columns)}")

    return lat_col, lon_col, m_col


def load_mach(path):
    df = read_csv_flexible(path)
    lat_col, lon_col, m_col = infer_columns(df)

    out = pd.DataFrame({
        "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        "M": pd.to_numeric(df[m_col], errors="coerce"),
    }).dropna()

    out["lon"] = normalize_longitude(out["lon"])
    out = out.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    return out


def build_mach_interpolator(mach_df):
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


def load_candidates(paths, max_rows):
    frames = []

    for p in paths:
        if not p.exists():
            continue

        df = pd.read_csv(p)

        if not {"axis_lat", "axis_lon"}.issubset(df.columns):
            continue

        if "angle_deg" not in df.columns:
            df["angle_deg"] = np.nan

        df["source_file"] = p.name
        frames.append(df.head(max_rows).copy())

    if not frames:
        raise FileNotFoundError("No candidate files found with axis_lat/axis_lon columns.")

    out = pd.concat(frames, ignore_index=True)
    out["axis_lat"] = pd.to_numeric(out["axis_lat"], errors="coerce")
    out["axis_lon"] = pd.to_numeric(out["axis_lon"], errors="coerce")
    out["angle_deg"] = pd.to_numeric(out["angle_deg"], errors="coerce")
    out = out.dropna(subset=["axis_lat", "axis_lon"]).copy()
    out["axis_lon"] = normalize_longitude(out["axis_lon"])

    return out.drop_duplicates(
        subset=["source_file", "axis_lat", "axis_lon", "angle_deg"]
    ).reset_index(drop=True)


def path_metrics_for_axis(axis_lat, axis_lon, eval_m, basin_lat, basin_lon, m_threshold, sweep_step_deg):
    axis = axis_vector(axis_lat, axis_lon)
    start = axis_vector(basin_lat, basin_lon).reshape(1, 3)

    angles = np.arange(0.0, 360.0 + 0.001, sweep_step_deg)
    path = rotate_vectors(np.repeat(start, len(angles), axis=0), axis, angles)

    plat, plon = unit_to_latlon(path)
    mvals = eval_m(plat, plon)

    below = mvals < m_threshold
    penalty = np.where(below, m_threshold - mvals, 0.0)

    return {
        "path_m_mean": float(np.mean(mvals)),
        "path_m_median": float(np.median(mvals)),
        "path_m_min": float(np.min(mvals)),
        "path_m_max": float(np.max(mvals)),
        "path_m_std": float(np.std(mvals)),
        "path_frac_below_threshold": float(np.mean(below)),
        "path_integrated_penalty": float(np.sum(penalty) * sweep_step_deg),
        "path_threshold": float(m_threshold),
        "path_n": int(len(angles)),
    }


def candidate_metrics(candidates, mach_df, eval_m, args):
    rows = []

    basin_lat = args.basin_lat
    basin_lon = args.basin_lon

    if basin_lat is None or basin_lon is None:
        top = mach_df.sort_values("M", ascending=False).head(args.basin_top_n)
        basin_lat = float(top["lat"].mean())
        basin_lon = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(top["lon"]))))))

    m_threshold = float(np.quantile(mach_df["M"], args.low_m_quantile))

    for _, row in candidates.iterrows():
        alat = float(row["axis_lat"])
        alon = float(row["axis_lon"])

        point_m = float(eval_m(np.array([alat]), np.array([alon]))[0])
        dist_to_basin = float(angular_distance_deg([alat], [alon], [basin_lat], [basin_lon])[0])

        metrics = path_metrics_for_axis(
            axis_lat=alat,
            axis_lon=alon,
            eval_m=eval_m,
            basin_lat=basin_lat,
            basin_lon=basin_lon,
            m_threshold=m_threshold,
            sweep_step_deg=args.sweep_step_deg,
        )

        out = row.to_dict()
        out.update({
            "axis_point_M": point_m,
            "distance_to_basin_deg": dist_to_basin,
            "basin_lat": basin_lat,
            "basin_lon": basin_lon,
        })
        out.update(metrics)
        rows.append(out)

    return pd.DataFrame(rows)


def random_axes(n, rng):
    z = rng.uniform(-1.0, 1.0, n)
    lon = rng.uniform(-180.0, 180.0, n)
    lat = np.degrees(np.arcsin(z))
    return pd.DataFrame({"axis_lat": lat, "axis_lon": lon, "source_file": "random_axes"})


def summarize_against_random(obs_df, random_df):
    metrics = [
        "axis_point_M",
        "distance_to_basin_deg",
        "path_m_mean",
        "path_m_min",
        "path_frac_below_threshold",
        "path_integrated_penalty",
    ]

    rows = []

    for metric in metrics:
        obs = obs_df[metric].dropna().to_numpy(float)
        rnd = random_df[metric].dropna().to_numpy(float)

        if len(obs) == 0 or len(rnd) == 0:
            continue

        rows.append({
            "metric": metric,
            "observed_median": float(np.median(obs)),
            "observed_mean": float(np.mean(obs)),
            "random_median": float(np.median(rnd)),
            "random_mean": float(np.mean(rnd)),
            "observed_p10": float(np.quantile(obs, 0.10)),
            "observed_p90": float(np.quantile(obs, 0.90)),
            "random_p10": float(np.quantile(rnd, 0.10)),
            "random_p90": float(np.quantile(rnd, 0.90)),
            "empirical_p_high": float((np.sum(rnd >= np.median(obs)) + 1) / (len(rnd) + 1)),
            "empirical_p_low": float((np.sum(rnd <= np.median(obs)) + 1) / (len(rnd) + 1)),
        })

    return pd.DataFrame(rows)


def make_plots(obs, rnd, mach_df, out_dir):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 6))
    sc = plt.scatter(mach_df["lon"], mach_df["lat"], c=mach_df["M"], s=18, alpha=0.75)
    plt.scatter(obs["axis_lon"], obs["axis_lat"], s=24, marker="x", c="black")
    plt.colorbar(sc, label="Mach compatibility M")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("10 LITTORAL candidate axes over Mach compatibility field")
    plt.tight_layout()
    plt.savefig(out_dir / "10_candidate_axes_over_mach_field.png", dpi=220)
    plt.close()

    for metric, label in [
        ("path_m_mean", "Path mean M"),
        ("path_m_min", "Path minimum M"),
        ("path_frac_below_threshold", "Fraction below low-M threshold"),
        ("path_integrated_penalty", "Integrated low-M penalty"),
        ("distance_to_basin_deg", "Distance to dominant basin"),
    ]:
        plt.figure(figsize=(9, 5))
        plt.hist(rnd[metric], bins=35, alpha=0.55, label="random axes")
        plt.axvline(obs[metric].median(), linestyle="--", label="candidate median")
        plt.xlabel(label)
        plt.ylabel("Count")
        plt.title(f"10 candidate-vs-random: {label}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"10_candidate_vs_random_{metric}.png", dpi=220)
        plt.close()

    if "score" in obs.columns:
        plt.figure(figsize=(9, 5))
        plt.scatter(obs["score"], obs["path_integrated_penalty"], s=18)
        plt.xlabel("LITTORAL inverse score")
        plt.ylabel("Mach path integrated penalty")
        plt.title("10 inverse score vs Mach ridge-avoidance penalty")
        plt.tight_layout()
        plt.savefig(out_dir / "10_inverse_score_vs_mach_penalty.png", dpi=220)
        plt.close()

    if "r2" in obs.columns:
        plt.figure(figsize=(9, 5))
        plt.scatter(obs["r2"], obs["path_integrated_penalty"], s=18)
        plt.xlabel("LITTORAL inverse R²")
        plt.ylabel("Mach path integrated penalty")
        plt.title("10 inverse R² vs Mach ridge-avoidance penalty")
        plt.tight_layout()
        plt.savefig(out_dir / "10_inverse_r2_vs_mach_penalty.png", dpi=220)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mach", default=None)
    parser.add_argument("--max-candidates-per-file", type=int, default=250)
    parser.add_argument("--random-axes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep-step-deg", type=float, default=2.0)
    parser.add_argument("--low-m-quantile", type=float, default=0.20)
    parser.add_argument("--basin-top-n", type=int, default=5)
    parser.add_argument("--basin-lat", type=float, default=None)
    parser.add_argument("--basin-lon", type=float, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    mach_path = Path(args.mach) if args.mach else find_existing(MACH_CANDIDATES)
    if mach_path is None:
        raise FileNotFoundError("Could not find mach.csv. Use --mach /path/to/mach.csv")

    mach_df = load_mach(mach_path)
    eval_m = build_mach_interpolator(mach_df)

    candidates = load_candidates(CANDIDATE_FILES, args.max_candidates_per_file)
    obs_metrics = candidate_metrics(candidates, mach_df, eval_m, args)

    rnd_axes = random_axes(args.random_axes, rng)
    rnd_metrics = candidate_metrics(rnd_axes, mach_df, eval_m, args)

    summary_df = summarize_against_random(obs_metrics, rnd_metrics)

    obs_metrics.to_csv(OUT_DIR / "10_littoral_candidate_mach_path_metrics.csv", index=False)
    rnd_metrics.to_csv(OUT_DIR / "10_random_axis_mach_path_metrics.csv", index=False)
    summary_df.to_csv(OUT_DIR / "10_candidate_vs_random_summary.csv", index=False)

    best_low_penalty = obs_metrics.sort_values("path_integrated_penalty", ascending=True).head(50)
    best_high_mmean = obs_metrics.sort_values("path_m_mean", ascending=False).head(50)

    best_low_penalty.to_csv(OUT_DIR / "10_top50_low_penalty_candidates.csv", index=False)
    best_high_mmean.to_csv(OUT_DIR / "10_top50_high_path_m_candidates.csv", index=False)

    summary = {
        "metadata": {
            "mach_input": str(mach_path),
            "candidate_files": [str(p) for p in CANDIDATE_FILES if p.exists()],
            "n_candidates": int(len(obs_metrics)),
            "n_random_axes": int(len(rnd_metrics)),
            "sweep_step_deg": float(args.sweep_step_deg),
            "low_m_quantile": float(args.low_m_quantile),
            "chronology_used": False,
            "origin_assumptions_used": False,
        },
        "mach_field": {
            "n": int(len(mach_df)),
            "m_min": float(mach_df["M"].min()),
            "m_max": float(mach_df["M"].max()),
            "m_mean": float(mach_df["M"].mean()),
            "m_threshold": float(np.quantile(mach_df["M"], args.low_m_quantile)),
        },
        "candidate_vs_random_summary": summary_df.to_dict("records"),
        "best_low_penalty_candidate": best_low_penalty.head(1).to_dict("records"),
        "best_high_path_m_candidate": best_high_mmean.head(1).to_dict("records"),
    }

    with open(OUT_DIR / "10_littoral_mach_path_consistency_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_plots(obs_metrics, rnd_metrics, mach_df, OUT_DIR)

    print("\nLITTORAL 10 Mach path-consistency test complete.")
    print(f"Mach input: {mach_path}")
    print(f"Output: {OUT_DIR}")
    print(f"Candidate axes: {len(obs_metrics)}")
    print(f"Random axes: {len(rnd_metrics)}")

    print("\nCandidate vs random summary:")
    for _, row in summary_df.iterrows():
        print(
            f"  {row['metric']}: "
            f"obs_med={row['observed_median']:.6g}, "
            f"rand_med={row['random_median']:.6g}, "
            f"p_low={row['empirical_p_low']:.4f}, "
            f"p_high={row['empirical_p_high']:.4f}"
        )

    print("\nBest low-penalty candidate:")
    if len(best_low_penalty):
        r = best_low_penalty.iloc[0]
        print(
            f"  axis=({r['axis_lat']:.2f}, {r['axis_lon']:.2f}), "
            f"angle={r.get('angle_deg', np.nan):.2f}, "
            f"penalty={r['path_integrated_penalty']:.6g}, "
            f"path_mean_M={r['path_m_mean']:.6g}, "
            f"source={r.get('source_file', '')}"
        )


if __name__ == "__main__":
    main()
