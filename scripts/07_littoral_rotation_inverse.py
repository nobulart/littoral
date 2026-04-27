#!/usr/bin/env python3
"""
07_littoral_rotation_inverse.py

Inverse search for the rotation/displacement geometry whose centrifugal
sea-surface anomaly best matches reported LITTORAL littoral depth gradients.

Record selection: depth_source=reported, coordinates present, z_m <= 200 m.

Compares relative gradients only:
  z_m ~ a + b * modeled_anomaly

No dates. No absolute sea-level assumptions.

Reads:
  outputs/geospatial_06/06_littoral_working_records.csv
  or outputs/geospatial_04/04_littoral_working_records.csv

Writes:
  outputs/geospatial_07/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_06 = PROJECT_ROOT / "outputs" / "geospatial_06" / "06_littoral_working_records.csv"
IN_04 = PROJECT_ROOT / "outputs" / "geospatial_04" / "04_littoral_working_records.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "geospatial_07"

EARTH_RADIUS_M = 6_371_000.0
OMEGA = 7.2921159e-5


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


def axis_vector(lat_deg, lon_deg):
    return latlon_to_unit(np.array([lat_deg]), np.array([lon_deg]))[0]


def rotate_vector(v, axis, angle_deg):
    """
    Rodrigues rotation.
    Rotates vector(s) v about axis by angle_deg.
    """
    theta = np.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)

    return (
        v * np.cos(theta)
        + np.cross(axis, v) * np.sin(theta)
        + axis * np.sum(v * axis, axis=-1, keepdims=True) * (1.0 - np.cos(theta))
    )


def ols_score(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "intercept": float(beta[0]),
        "scale": float(beta[1]),
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "prediction": pred,
        "residual": resid,
    }


def rankdata(a):
    return pd.Series(a).rank(method="average").to_numpy(float)


def spearman(x, y):
    xr = rankdata(x)
    yr = rankdata(y)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def kendall_tau_simple(x, y):
    """
    O(n^2), fine for n ~ dozens.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(x)
    concordant = 0
    discordant = 0

    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        prod = dx * dy

        concordant += int(np.sum(prod > 0))
        discordant += int(np.sum(prod < 0))

    denom = concordant + discordant
    return float((concordant - discordant) / denom) if denom else np.nan


def centrifugal_potential_proxy(site_vectors, pole_vector):
    """
    Centrifugal potential is proportional to distance^2 from rotation axis:
      Phi_c = -0.5 * omega^2 * R^2 * sin^2(theta)
    where theta is angular distance from pole.

    Since sin^2(theta) = 1 - (r dot p)^2.

    Absolute constants are included for physical scaling, but later normalized.
    """
    mu = np.clip(site_vectors @ pole_vector, -1.0, 1.0)
    return -0.5 * OMEGA**2 * EARTH_RADIUS_M**2 * (1.0 - mu**2)


def modeled_anomaly(site_vectors, candidate_axis, angle_deg, relaxation=1.0):
    """
    Candidate TPW geometry.

    Present pole is geographic north: [0,0,1].

    A candidate rotation is represented by rotating the present spin pole
    around candidate_axis by angle_deg. This gives the displaced/reoriented
    pole relative to the present crust.

    The anomaly is the difference between rotated and present centrifugal
    potential. Relaxation is a linear amplitude scalar only; rank tests are
    insensitive to it.
    """
    present_pole = np.array([0.0, 0.0, 1.0])
    rotated_pole = rotate_vector(present_pole.reshape(1, 3), candidate_axis, angle_deg)[0]
    rotated_pole = rotated_pole / np.linalg.norm(rotated_pole)

    phi_present = centrifugal_potential_proxy(site_vectors, present_pole)
    phi_rotated = centrifugal_potential_proxy(site_vectors, rotated_pole)

    return relaxation * (phi_rotated - phi_present), rotated_pole


def pole_latlon(vec):
    vec = vec / np.linalg.norm(vec)
    lat = float(np.degrees(np.arcsin(np.clip(vec[2], -1.0, 1.0))))
    lon = float(np.degrees(np.arctan2(vec[1], vec[0])))
    return lat, lon


def add_spatial_cells(df, lat_bin_deg=10.0, lon_bin_deg=20.0):
    out = df.copy()
    out["lat_bin"] = np.floor((out["latitude"] + 90.0) / lat_bin_deg).astype(int)
    out["lon_bin"] = np.floor((out["longitude"] + 180.0) / lon_bin_deg).astype(int)
    out["spatial_cell"] = out["lat_bin"].astype(str) + "_" + out["lon_bin"].astype(str)
    return out


def scan_orientations(df, args):
    site_vecs = latlon_to_unit(
        df["latitude"].to_numpy(float),
        df["longitude"].to_numpy(float),
    )
    z = df["z_m"].to_numpy(float)

    rows = []

    axis_lats = np.arange(args.axis_lat_min, args.axis_lat_max + 0.001, args.axis_lat_step)
    axis_lons = np.arange(-180.0, 180.0, args.axis_lon_step)
    angles = np.arange(args.angle_min, args.angle_max + 0.001, args.angle_step)

    total = len(axis_lats) * len(axis_lons) * len(angles)
    done = 0

    for alat in axis_lats:
        for alon in axis_lons:
            ax = axis_vector(alat, alon)

            for angle in angles:
                anomaly, rotated_pole = modeled_anomaly(
                    site_vectors=site_vecs,
                    candidate_axis=ax,
                    angle_deg=angle,
                    relaxation=args.relaxation,
                )

                if np.std(anomaly) == 0:
                    continue

                # Normalize anomaly for numerical stability.
                a_norm = (anomaly - np.mean(anomaly)) / np.std(anomaly)

                fit = ols_score(a_norm, z)
                rho = spearman(a_norm, z)
                tau = kendall_tau_simple(a_norm, z)

                rp_lat, rp_lon = pole_latlon(rotated_pole)

                rows.append({
                    "axis_lat": float(alat),
                    "axis_lon": float(alon),
                    "angle_deg": float(angle),
                    "rotated_pole_lat": rp_lat,
                    "rotated_pole_lon": rp_lon,
                    "r2": fit["r2"],
                    "rmse": fit["rmse"],
                    "mae": fit["mae"],
                    "scale_m_per_sigma": fit["scale"],
                    "intercept_m": fit["intercept"],
                    "spearman": rho,
                    "abs_spearman": abs(rho),
                    "kendall_tau": tau,
                    "abs_kendall_tau": abs(tau),
                    "score": fit["r2"] * abs(rho),
                })

                done += 1

        if args.verbose:
            print(f"  scanned axis_lat={alat:.1f}; progress {done}/{total}")

    return pd.DataFrame(rows)


def evaluate_fixed_candidates(df):
    site_vecs = latlon_to_unit(df["latitude"].to_numpy(float), df["longitude"].to_numpy(float))
    z = df["z_m"].to_numpy(float)

    candidates = [
        {
            "name": "ECDO_104_axis_31E_equatorial",
            "axis_lat": 0.0,
            "axis_lon": 31.0,
            "angle_deg": 104.0,
        },
        {
            "name": "ECDO_104_axis_149W_equatorial",
            "axis_lat": 0.0,
            "axis_lon": -149.0,
            "angle_deg": 104.0,
        },
        {
            "name": "observed_l1_axis_77E_104deg",
            "axis_lat": 0.0,
            "axis_lon": 77.0,
            "angle_deg": 104.0,
        },
        {
            "name": "observed_l1_axis_77E_52deg",
            "axis_lat": 0.0,
            "axis_lon": 77.0,
            "angle_deg": 52.0,
        },
        {
            "name": "observed_l1_axis_77E_26deg",
            "axis_lat": 0.0,
            "axis_lon": 77.0,
            "angle_deg": 26.0,
        },
        {
            "name": "return_half_52_axis_31E",
            "axis_lat": 0.0,
            "axis_lon": 31.0,
            "angle_deg": 52.0,
        },
        {
            "name": "return_quarter_26_axis_31E",
            "axis_lat": 0.0,
            "axis_lon": 31.0,
            "angle_deg": 26.0,
        },
        {
            "name": "return_eighth_13_axis_31E",
            "axis_lat": 0.0,
            "axis_lon": 31.0,
            "angle_deg": 13.0,
        },
    ]

    rows = []

    for c in candidates:
        ax = axis_vector(c["axis_lat"], c["axis_lon"])
        anomaly, rotated_pole = modeled_anomaly(site_vecs, ax, c["angle_deg"])
        a_norm = (anomaly - np.mean(anomaly)) / np.std(anomaly)

        fit = ols_score(a_norm, z)
        rho = spearman(a_norm, z)
        tau = kendall_tau_simple(a_norm, z)
        rp_lat, rp_lon = pole_latlon(rotated_pole)

        rows.append({
            **c,
            "rotated_pole_lat": rp_lat,
            "rotated_pole_lon": rp_lon,
            "r2": fit["r2"],
            "rmse": fit["rmse"],
            "mae": fit["mae"],
            "scale_m_per_sigma": fit["scale"],
            "intercept_m": fit["intercept"],
            "spearman": rho,
            "abs_spearman": abs(rho),
            "kendall_tau": tau,
            "abs_kendall_tau": abs(tau),
            "score": fit["r2"] * abs(rho),
        })

    return pd.DataFrame(rows)


def permutation_test_best(df, best_row, args, rng):
    """
    Tests whether the best model score survives permutation of z values.

    Two nulls:
      global: permute all depths
      within_cells: permute depths only within spatial cells
    """
    if args.permutations <= 0:
        return {}

    site_vecs = latlon_to_unit(df["latitude"].to_numpy(float), df["longitude"].to_numpy(float))

    ax = axis_vector(best_row["axis_lat"], best_row["axis_lon"])
    anomaly, _ = modeled_anomaly(site_vecs, ax, best_row["angle_deg"])
    a_norm = (anomaly - np.mean(anomaly)) / np.std(anomaly)

    obs_score = float(best_row["score"])
    obs_r2 = float(best_row["r2"])
    obs_abs_spearman = float(best_row["abs_spearman"])

    results = {}

    for mode in ["global", "within_cells"]:
        null_scores = []
        null_r2 = []
        null_abs_spearman = []

        for _ in range(args.permutations):
            if mode == "global":
                z_perm = rng.permutation(df["z_m"].to_numpy(float))
            else:
                z_perm = df["z_m"].to_numpy(float).copy()
                for _, idx in df.groupby("spatial_cell").groups.items():
                    idx = np.array(list(idx), dtype=int)
                    z_perm[idx] = rng.permutation(z_perm[idx])

            fit = ols_score(a_norm, z_perm)
            rho = spearman(a_norm, z_perm)

            null_r2.append(fit["r2"])
            null_abs_spearman.append(abs(rho))
            null_scores.append(fit["r2"] * abs(rho))

        ns = np.asarray(null_scores)
        nr = np.asarray(null_r2)
        nh = np.asarray(null_abs_spearman)

        results[mode] = {
            "n_perm": int(args.permutations),
            "score_p": float((np.sum(ns >= obs_score) + 1) / (len(ns) + 1)),
            "r2_p": float((np.sum(nr >= obs_r2) + 1) / (len(nr) + 1)),
            "abs_spearman_p": float((np.sum(nh >= obs_abs_spearman) + 1) / (len(nh) + 1)),
            "score_null_median": float(np.median(ns)),
            "score_null_p95": float(np.quantile(ns, 0.95)),
            "r2_null_median": float(np.median(nr)),
            "r2_null_p95": float(np.quantile(nr, 0.95)),
            "abs_spearman_null_median": float(np.median(nh)),
            "abs_spearman_null_p95": float(np.quantile(nh, 0.95)),
        }

    return results


def predict_for_best(df, best_row):
    site_vecs = latlon_to_unit(df["latitude"].to_numpy(float), df["longitude"].to_numpy(float))
    z = df["z_m"].to_numpy(float)

    ax = axis_vector(best_row["axis_lat"], best_row["axis_lon"])
    anomaly, rotated_pole = modeled_anomaly(site_vecs, ax, best_row["angle_deg"])

    a_norm = (anomaly - np.mean(anomaly)) / np.std(anomaly)
    fit = ols_score(a_norm, z)

    out = df.copy()
    out["rotation_model_anomaly_raw"] = anomaly
    out["rotation_model_anomaly_zscore"] = a_norm
    out["rotation_model_prediction_z_m"] = fit["prediction"]
    out["rotation_model_residual_z_m"] = fit["residual"]

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

    df = df[df["z_m"] >= threshold].copy()
    df = add_spatial_cells(df, lat_bin_deg=lat_bin_deg, lon_bin_deg=lon_bin_deg)

    return df


def make_plots(scan, pred_df, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    top = scan.sort_values("score", ascending=False).head(5000)

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        top["axis_lon"],
        top["axis_lat"],
        c=top["score"],
        s=16,
    )
    plt.colorbar(sc, label="score = R² × |Spearman|")
    plt.xlabel("Candidate rotation-axis longitude")
    plt.ylabel("Candidate rotation-axis latitude")
    plt.title("07 inverse rotation search: top candidate axes")
    plt.tight_layout()
    plt.savefig(out_dir / "07_top_candidate_axes.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.scatter(top["angle_deg"], top["score"], s=12)
    plt.xlabel("Displacement angle")
    plt.ylabel("score")
    plt.title("07 score vs displacement angle")
    plt.tight_layout()
    plt.savefig(out_dir / "07_score_vs_angle.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.scatter(
        pred_df["rotation_model_prediction_z_m"],
        pred_df["z_m"],
        s=28,
    )
    plt.xlabel("Predicted z_m from scaled rotation anomaly")
    plt.ylabel("Observed reported z_m")
    plt.title("07 observed vs best rotation-gradient prediction")
    plt.tight_layout()
    plt.savefig(out_dir / "07_observed_vs_best_rotation_prediction.png", dpi=220)
    plt.close()

    plt.figure(figsize=(11, 6))
    sc = plt.scatter(
        pred_df["longitude"],
        pred_df["latitude"],
        c=pred_df["rotation_model_residual_z_m"],
        s=30,
    )
    plt.colorbar(sc, label="Residual z_m")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("07 residual map after best rotation-gradient fit")
    plt.tight_layout()
    plt.savefig(out_dir / "07_best_rotation_residual_map.png", dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--depth-threshold", type=float, default=-200.0)
    parser.add_argument("--lat-bin-deg", type=float, default=10.0)
    parser.add_argument("--lon-bin-deg", type=float, default=20.0)

    parser.add_argument("--axis-lat-min", type=float, default=-60.0)
    parser.add_argument("--axis-lat-max", type=float, default=60.0)
    parser.add_argument("--axis-lat-step", type=float, default=5.0)
    parser.add_argument("--axis-lon-step", type=float, default=5.0)

    parser.add_argument("--angle-min", type=float, default=1.0)
    parser.add_argument("--angle-max", type=float, default=120.0)
    parser.add_argument("--angle-step", type=float, default=2.0)

    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    input_path = Path(args.input) if args.input else (IN_06 if IN_06.exists() else IN_04)

    df = load_df(
        input_path,
        threshold=args.depth_threshold,
        lat_bin_deg=args.lat_bin_deg,
        lon_bin_deg=args.lon_bin_deg,
    )

    scan = scan_orientations(df, args)
    scan = scan.sort_values("score", ascending=False).reset_index(drop=True)

    fixed = evaluate_fixed_candidates(df).sort_values("score", ascending=False).reset_index(drop=True)

    best = scan.iloc[0].to_dict()
    pred_df = predict_for_best(df, best)

    nulls = permutation_test_best(df, best, args, rng)

    scan.to_csv(OUT_DIR / "07_rotation_inverse_scan.csv", index=False)
    scan.head(250).to_csv(OUT_DIR / "07_top250_rotation_candidates.csv", index=False)
    fixed.to_csv(OUT_DIR / "07_fixed_reference_candidates.csv", index=False)
    pred_df.to_csv(OUT_DIR / "07_littoral_records_with_best_rotation_prediction.csv", index=False)

    summary = {
        "metadata": {
            "input": str(input_path),
            "n": int(len(df)),
            "reported_depth_source_records_only": True,
            "selection_criteria": "depth_source == reported, latitude/longitude present, and z_m <= 200 m",
            "relative_gradients_only": True,
            "absolute_depths_interpreted": False,
            "depth_threshold_m": float(args.depth_threshold),
            "axis_lat_min": float(args.axis_lat_min),
            "axis_lat_max": float(args.axis_lat_max),
            "axis_lat_step": float(args.axis_lat_step),
            "axis_lon_step": float(args.axis_lon_step),
            "angle_min": float(args.angle_min),
            "angle_max": float(args.angle_max),
            "angle_step": float(args.angle_step),
            "permutations": int(args.permutations),
            "seed": int(args.seed),
            "chronology_used": False,
            "origin_assumptions_used": False,
        },
        "best_candidate": best,
        "fixed_reference_candidates": fixed.to_dict("records"),
        "permutation_tests_for_best_candidate": nulls,
    }

    with open(OUT_DIR / "07_rotation_inverse_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_plots(scan, pred_df, OUT_DIR)

    print("\nLITTORAL 07 rotation inverse search complete.")
    print(f"Input: {input_path}")
    print(f"Output: {OUT_DIR}")
    print(f"Records: {len(df)}")

    print("\nBest candidate:")
    for k, v in best.items():
        print(f"  {k}: {v}")

    print("\nFixed reference candidates:")
    for _, row in fixed.head(12).iterrows():
        print(
            f"  {row['name']}: "
            f"axis=({row['axis_lat']:.1f}, {row['axis_lon']:.1f}), "
            f"angle={row['angle_deg']:.1f}, "
            f"r2={row['r2']:.4f}, "
            f"spearman={row['spearman']:.4f}, "
            f"score={row['score']:.4f}"
        )

    print("\nPermutation tests for best candidate:")
    for mode, vals in nulls.items():
        print(
            f"  {mode}: "
            f"score_p={vals['score_p']}, "
            f"r2_p={vals['r2_p']}, "
            f"abs_spearman_p={vals['abs_spearman_p']}"
        )


if __name__ == "__main__":
    main()
