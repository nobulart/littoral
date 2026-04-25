#!/usr/bin/env python3
"""
08_littoral_joint_rotation_quadrupole_inverse.py

Joint inverse model for reported-source littoral gradients:

Record selection: depth_source=reported, coordinates present, z_m <= 200 m.

  z_m ~ a
        + b1 * centrifugal_rotation_anomaly
        + b2..b6 * l=2 Earth-fixed quadrupole basis

Purpose:
  Test whether a constrained rotation geometry, especially ECDO-like
  31E / ~104°, becomes competitive once the quadrupole component is included.

Reads:
  outputs/geospatial_07/07_littoral_records_with_best_rotation_prediction.csv
  or outputs/geospatial_06/06_littoral_working_records.csv

Writes:
  outputs/geospatial_08/

No dates. No absolute-depth assumptions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_07 = PROJECT_ROOT / "outputs" / "geospatial_07" / "07_littoral_records_with_best_rotation_prediction.csv"
IN_06 = PROJECT_ROOT / "outputs" / "geospatial_06" / "06_littoral_working_records.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "geospatial_08"

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
    theta = np.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)

    return (
        v * np.cos(theta)
        + np.cross(axis, v) * np.sin(theta)
        + axis * np.sum(v * axis, axis=-1, keepdims=True) * (1.0 - np.cos(theta))
    )


def centrifugal_potential_proxy(site_vectors, pole_vector):
    mu = np.clip(site_vectors @ pole_vector, -1.0, 1.0)
    return -0.5 * OMEGA**2 * EARTH_RADIUS_M**2 * (1.0 - mu**2)


def modeled_rotation_anomaly(site_vectors, axis_lat, axis_lon, angle_deg):
    present_pole = np.array([0.0, 0.0, 1.0])
    axis = axis_vector(axis_lat, axis_lon)

    rotated_pole = rotate_vector(present_pole.reshape(1, 3), axis, angle_deg)[0]
    rotated_pole = rotated_pole / np.linalg.norm(rotated_pole)

    phi_present = centrifugal_potential_proxy(site_vectors, present_pole)
    phi_rotated = centrifugal_potential_proxy(site_vectors, rotated_pole)

    anomaly = phi_rotated - phi_present

    if np.std(anomaly) > 0:
        anomaly = (anomaly - np.mean(anomaly)) / np.std(anomaly)

    return anomaly, rotated_pole


def pole_latlon(vec):
    vec = vec / np.linalg.norm(vec)
    return (
        float(np.degrees(np.arcsin(np.clip(vec[2], -1.0, 1.0)))),
        float(np.degrees(np.arctan2(vec[1], vec[0]))),
    )


def quadrupole_basis(site_vectors):
    x, y, z = site_vectors[:, 0], site_vectors[:, 1], site_vectors[:, 2]

    X2 = np.column_stack([
        x * y,
        x * z,
        y * z,
        x * x - y * y,
        3.0 * z * z - 1.0,
    ])

    names = [
        "q_xy",
        "q_xz",
        "q_yz",
        "q_x2_minus_y2",
        "q_3z2_minus_1",
    ]

    # Standardize columns to prevent numerical domination by basis scale.
    X2s = X2.copy()
    for i in range(X2s.shape[1]):
        sd = np.std(X2s[:, i])
        if sd > 0:
            X2s[:, i] = (X2s[:, i] - np.mean(X2s[:, i])) / sd

    return X2s, names


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ beta
    resid = y - pred

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return beta, pred, resid, r2


def rankdata(a):
    return pd.Series(a).rank(method="average").to_numpy(float)


def spearman(x, y):
    xr = rankdata(x)
    yr = rankdata(y)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def model_score(df, axis_lat, axis_lon, angle_deg, include_quadrupole=True):
    site_vecs = latlon_to_unit(df["latitude"].to_numpy(float), df["longitude"].to_numpy(float))
    z = df["z_m"].to_numpy(float)

    rot, rotated_pole = modeled_rotation_anomaly(site_vecs, axis_lat, axis_lon, angle_deg)

    if include_quadrupole:
        Q, qnames = quadrupole_basis(site_vecs)
        X = np.column_stack([rot, Q])
        names = ["rotation_anomaly"] + qnames
    else:
        X = rot.reshape(-1, 1)
        names = ["rotation_anomaly"]

    beta, pred, resid, r2 = ols(X, z)

    rp_lat, rp_lon = pole_latlon(rotated_pole)

    return {
        "axis_lat": float(axis_lat),
        "axis_lon": float(axis_lon),
        "angle_deg": float(angle_deg),
        "rotated_pole_lat": rp_lat,
        "rotated_pole_lon": rp_lon,
        "include_quadrupole": bool(include_quadrupole),
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "spearman_pred_obs": spearman(pred, z),
        "rotation_only_spearman": spearman(rot, z),
        "intercept": float(beta[0]),
        "coefficients": {name: float(value) for name, value in zip(names, beta[1:])},
        "prediction": pred,
        "residual": resid,
        "rotation_anomaly": rot,
    }


def scan_joint(df, args):
    rows = []

    axis_lats = np.arange(args.axis_lat_min, args.axis_lat_max + 0.001, args.axis_lat_step)
    axis_lons = np.arange(-180.0, 180.0, args.axis_lon_step)
    angles = np.arange(args.angle_min, args.angle_max + 0.001, args.angle_step)

    for alat in axis_lats:
        for alon in axis_lons:
            for angle in angles:
                fit = model_score(
                    df=df,
                    axis_lat=alat,
                    axis_lon=alon,
                    angle_deg=angle,
                    include_quadrupole=True,
                )
                rows.append({k: v for k, v in fit.items() if k not in ["prediction", "residual", "rotation_anomaly"]})

    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)


def evaluate_references(df):
    refs = [
        ("ECDO_31E_104", 0.0, 31.0, 104.0),
        ("ECDO_31E_107", 0.0, 31.0, 107.0),
        ("ECDO_149W_104", 0.0, -149.0, 104.0),
        ("ECDO_31E_52", 0.0, 31.0, 52.0),
        ("ECDO_31E_26", 0.0, 31.0, 26.0),
        ("observed_77E_104", 0.0, 77.0, 104.0),
        ("observed_77E_107", 0.0, 77.0, 107.0),
        ("best_07_40N_110E_107", 40.0, 110.0, 107.0),
    ]

    rows = []

    for name, alat, alon, angle in refs:
        rot_only = model_score(df, alat, alon, angle, include_quadrupole=False)
        joint = model_score(df, alat, alon, angle, include_quadrupole=True)

        rows.append({
            "name": name,
            "axis_lat": alat,
            "axis_lon": alon,
            "angle_deg": angle,
            "rotated_pole_lat": joint["rotated_pole_lat"],
            "rotated_pole_lon": joint["rotated_pole_lon"],
            "rotation_only_r2": rot_only["r2"],
            "joint_r2": joint["r2"],
            "quadrupole_gain_r2": joint["r2"] - rot_only["r2"],
            "rotation_only_rmse": rot_only["rmse"],
            "joint_rmse": joint["rmse"],
            "joint_spearman_pred_obs": joint["spearman_pred_obs"],
            "rotation_only_spearman": rot_only["rotation_only_spearman"],
            "joint_rotation_coeff": joint["coefficients"].get("rotation_anomaly"),
        })

    return pd.DataFrame(rows).sort_values("joint_r2", ascending=False).reset_index(drop=True)


def add_spatial_cells(df, lat_bin_deg=10.0, lon_bin_deg=20.0):
    out = df.copy()
    out["lat_bin"] = np.floor((out["latitude"] + 90.0) / lat_bin_deg).astype(int)
    out["lon_bin"] = np.floor((out["longitude"] + 180.0) / lon_bin_deg).astype(int)
    out["spatial_cell"] = out["lat_bin"].astype(str) + "_" + out["lon_bin"].astype(str)
    return out


def permute_within_cells(df, rng):
    out = df.copy()
    out["z_perm"] = out["z_m"]

    for _, idx in out.groupby("spatial_cell").groups.items():
        idx = np.array(list(idx), dtype=int)
        out.loc[idx, "z_perm"] = rng.permutation(out.loc[idx, "z_m"].to_numpy(float))

    out["z_m"] = out["z_perm"]
    return out.drop(columns=["z_perm"])


def permute_global(df, rng):
    out = df.copy()
    out["z_m"] = rng.permutation(out["z_m"].to_numpy(float))
    return out


def permutation_test_fixed(df, reference_rows, args, rng):
    rows = []

    for _, ref in reference_rows.iterrows():
        name = ref["name"]
        alat = float(ref["axis_lat"])
        alon = float(ref["axis_lon"])
        angle = float(ref["angle_deg"])
        observed = float(ref["joint_r2"])

        for mode in ["global", "within_cells"]:
            null = []

            for _ in range(args.permutations):
                if mode == "global":
                    pdf = permute_global(df, rng)
                else:
                    pdf = permute_within_cells(df, rng)

                fit = model_score(pdf, alat, alon, angle, include_quadrupole=True)
                null.append(fit["r2"])

            arr = np.asarray(null)

            rows.append({
                "name": name,
                "mode": mode,
                "observed_joint_r2": observed,
                "p": float((np.sum(arr >= observed) + 1) / (len(arr) + 1)),
                "null_median": float(np.median(arr)),
                "null_p95": float(np.quantile(arr, 0.95)),
                "null_p99": float(np.quantile(arr, 0.99)),
            })

    return pd.DataFrame(rows)


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


def write_best_prediction(df, best_row):
    fit = model_score(
        df,
        axis_lat=float(best_row["axis_lat"]),
        axis_lon=float(best_row["axis_lon"]),
        angle_deg=float(best_row["angle_deg"]),
        include_quadrupole=True,
    )

    out = df.copy()
    out["joint_prediction_z_m"] = fit["prediction"]
    out["joint_residual_z_m"] = fit["residual"]
    out["joint_rotation_anomaly"] = fit["rotation_anomaly"]

    return out, fit


def make_plots(scan, refs, pred_df, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    top = scan.head(5000)

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(top["axis_lon"], top["axis_lat"], c=top["r2"], s=16)
    plt.colorbar(sc, label="Joint model R²")
    plt.xlabel("Candidate rotation-axis longitude")
    plt.ylabel("Candidate rotation-axis latitude")
    plt.title("08 joint rotation + quadrupole: top candidate axes")
    plt.tight_layout()
    plt.savefig(out_dir / "08_top_joint_candidate_axes.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.scatter(top["angle_deg"], top["r2"], s=12)
    plt.xlabel("Displacement angle")
    plt.ylabel("Joint model R²")
    plt.title("08 joint R² vs displacement angle")
    plt.tight_layout()
    plt.savefig(out_dir / "08_joint_r2_vs_angle.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(range(len(refs)), refs["joint_r2"])
    plt.xticks(range(len(refs)), refs["name"], rotation=60, ha="right")
    plt.ylabel("Joint model R²")
    plt.title("08 fixed reference joint-model scores")
    plt.tight_layout()
    plt.savefig(out_dir / "08_fixed_reference_joint_scores.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(pred_df["joint_prediction_z_m"], pred_df["z_m"], s=28)
    plt.xlabel("Predicted z_m from joint model")
    plt.ylabel("Observed reported z_m")
    plt.title("08 observed vs joint prediction")
    plt.tight_layout()
    plt.savefig(out_dir / "08_observed_vs_joint_prediction.png", dpi=220)
    plt.close()

    plt.figure(figsize=(11, 6))
    sc = plt.scatter(pred_df["longitude"], pred_df["latitude"], c=pred_df["joint_residual_z_m"], s=30)
    plt.colorbar(sc, label="Joint residual z_m")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("08 residual map after joint model")
    plt.tight_layout()
    plt.savefig(out_dir / "08_joint_residual_map.png", dpi=220)
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

    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    input_path = Path(args.input) if args.input else (IN_07 if IN_07.exists() else IN_06)

    df = load_df(
        input_path,
        threshold=args.depth_threshold,
        lat_bin_deg=args.lat_bin_deg,
        lon_bin_deg=args.lon_bin_deg,
    )

    scan = scan_joint(df, args)
    refs = evaluate_references(df)

    best_row = scan.iloc[0].to_dict()
    pred_df, best_fit = write_best_prediction(df, best_row)

    perm = permutation_test_fixed(df, refs, args, rng)

    scan.to_csv(OUT_DIR / "08_joint_rotation_quadrupole_scan.csv", index=False)
    scan.head(250).to_csv(OUT_DIR / "08_top250_joint_candidates.csv", index=False)
    refs.to_csv(OUT_DIR / "08_fixed_reference_joint_candidates.csv", index=False)
    perm.to_csv(OUT_DIR / "08_fixed_reference_joint_permutation_tests.csv", index=False)
    pred_df.to_csv(OUT_DIR / "08_littoral_records_with_joint_prediction.csv", index=False)

    summary = {
        "metadata": {
            "input": str(input_path),
            "n": int(len(df)),
            "reported_depth_source_records_only": True,
            "selection_criteria": "depth_source == reported, latitude/longitude present, and z_m <= 200 m",
            "relative_gradients_only": True,
            "absolute_depths_interpreted": False,
            "depth_threshold_m": float(args.depth_threshold),
            "axis_lat_step": float(args.axis_lat_step),
            "axis_lon_step": float(args.axis_lon_step),
            "angle_step": float(args.angle_step),
            "permutations": int(args.permutations),
            "seed": int(args.seed),
            "chronology_used": False,
            "origin_assumptions_used": False,
        },
        "best_joint_candidate": {
            k: v for k, v in best_fit.items()
            if k not in ["prediction", "residual", "rotation_anomaly"]
        },
        "fixed_reference_candidates": refs.to_dict("records"),
        "fixed_reference_permutation_tests": perm.to_dict("records"),
    }

    with open(OUT_DIR / "08_joint_rotation_quadrupole_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_plots(scan, refs, pred_df, OUT_DIR)

    print("\nLITTORAL 08 joint rotation + quadrupole inverse complete.")
    print(f"Input: {input_path}")
    print(f"Output: {OUT_DIR}")
    print(f"Records: {len(df)}")

    print("\nBest joint candidate:")
    for k, v in summary["best_joint_candidate"].items():
        print(f"  {k}: {v}")

    print("\nFixed reference candidates:")
    for _, row in refs.iterrows():
        print(
            f"  {row['name']}: "
            f"rotation_only_r2={row['rotation_only_r2']:.4f}, "
            f"joint_r2={row['joint_r2']:.4f}, "
            f"gain={row['quadrupole_gain_r2']:.4f}, "
            f"rot_coeff={row['joint_rotation_coeff']:.3f}"
        )

    print("\nPermutation tests:")
    for _, row in perm.iterrows():
        print(
            f"  {row['name']} [{row['mode']}]: "
            f"p={row['p']:.4f}, "
            f"obs={row['observed_joint_r2']:.4f}, "
            f"null95={row['null_p95']:.4f}"
        )


if __name__ == "__main__":
    main()
