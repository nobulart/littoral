#!/usr/bin/env python3
"""
09_littoral_constrained_rotation_quadrupole_inverse.py

Regularized joint inverse:

  z_m ~ a + b1 * rotation_anomaly + ridge(l=2 quadrupole terms)

Record selection: depth_source=reported, coordinates present, z_m <= 200 m.

Purpose:
  Prevent free l=2 terms from swallowing the rotation signal.

Reads:
  outputs/geospatial_08/08_littoral_records_with_joint_prediction.csv
  or outputs/geospatial_06/06_littoral_working_records.csv

Writes:
  outputs/geospatial_09/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_08 = PROJECT_ROOT / "outputs" / "geospatial_08" / "08_littoral_records_with_joint_prediction.csv"
IN_06 = PROJECT_ROOT / "outputs" / "geospatial_06" / "06_littoral_working_records.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "geospatial_09"

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


def pole_latlon(vec):
    vec = vec / np.linalg.norm(vec)
    return (
        float(np.degrees(np.arcsin(np.clip(vec[2], -1.0, 1.0)))),
        float(np.degrees(np.arctan2(vec[1], vec[0]))),
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
    anomaly = zscore(anomaly)

    return anomaly, rotated_pole


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

    for i in range(X2.shape[1]):
        X2[:, i] = zscore(X2[:, i])

    return X2, names


def zscore(x):
    x = np.asarray(x, dtype=float)
    sd = np.std(x)
    if sd == 0:
        return x * 0.0
    return (x - np.mean(x)) / sd


def rankdata(a):
    return pd.Series(a).rank(method="average").to_numpy(float)


def spearman(x, y):
    xr = rankdata(x)
    yr = rankdata(y)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def ridge_fit(X, y, penalty_vector):
    """
    Ridge with unpenalized intercept.

    penalty_vector length must equal number of columns in X.
    Use penalty 0 for unpenalized rotation term and lambda for q terms.
    """
    X1 = np.column_stack([np.ones(len(X)), X])
    P = np.diag(np.concatenate([[0.0], penalty_vector]))

    beta = np.linalg.solve(X1.T @ X1 + P, X1.T @ y)
    pred = X1 @ beta
    resid = y - pred

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return beta, pred, resid, r2


def model_score(df, axis_lat, axis_lon, angle_deg, lambda_q):
    site_vecs = latlon_to_unit(df["latitude"].to_numpy(float), df["longitude"].to_numpy(float))
    y = df["z_m"].to_numpy(float)

    rot, rotated_pole = modeled_rotation_anomaly(site_vecs, axis_lat, axis_lon, angle_deg)
    Q, qnames = quadrupole_basis(site_vecs)

    X = np.column_stack([rot, Q])
    names = ["rotation_anomaly"] + qnames

    penalty = np.array([0.0] + [lambda_q] * Q.shape[1], dtype=float)

    beta, pred, resid, r2 = ridge_fit(X, y, penalty)

    rp_lat, rp_lon = pole_latlon(rotated_pole)

    q_beta = beta[2:]
    q_norm = float(np.sqrt(np.sum(q_beta ** 2)))

    return {
        "axis_lat": float(axis_lat),
        "axis_lon": float(axis_lon),
        "angle_deg": float(angle_deg),
        "rotated_pole_lat": rp_lat,
        "rotated_pole_lon": rp_lon,
        "lambda_q": float(lambda_q),
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "spearman_pred_obs": spearman(pred, y),
        "rotation_only_spearman": spearman(rot, y),
        "intercept": float(beta[0]),
        "rotation_coeff": float(beta[1]),
        "quadrupole_coeff_norm": q_norm,
        "coefficients": {name: float(value) for name, value in zip(names, beta[1:])},
        "prediction": pred,
        "residual": resid,
        "rotation_anomaly": rot,
    }


def rotation_only_score(df, axis_lat, axis_lon, angle_deg):
    site_vecs = latlon_to_unit(df["latitude"].to_numpy(float), df["longitude"].to_numpy(float))
    y = df["z_m"].to_numpy(float)

    rot, rotated_pole = modeled_rotation_anomaly(site_vecs, axis_lat, axis_lon, angle_deg)

    X = rot.reshape(-1, 1)
    beta, pred, resid, r2 = ridge_fit(X, y, np.array([0.0]))
    rp_lat, rp_lon = pole_latlon(rotated_pole)

    return {
        "axis_lat": float(axis_lat),
        "axis_lon": float(axis_lon),
        "angle_deg": float(angle_deg),
        "rotated_pole_lat": rp_lat,
        "rotated_pole_lon": rp_lon,
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "spearman_pred_obs": spearman(pred, y),
        "rotation_coeff": float(beta[1]),
    }


def scan(df, args):
    rows = []

    axis_lats = np.arange(args.axis_lat_min, args.axis_lat_max + 0.001, args.axis_lat_step)
    axis_lons = np.arange(-180.0, 180.0, args.axis_lon_step)
    angles = np.arange(args.angle_min, args.angle_max + 0.001, args.angle_step)

    for lambda_q in args.lambda_q:
        for alat in axis_lats:
            for alon in axis_lons:
                for angle in angles:
                    fit = model_score(df, alat, alon, angle, lambda_q)
                    rows.append({
                        k: v for k, v in fit.items()
                        if k not in ["prediction", "residual", "rotation_anomaly", "coefficients"]
                    })

    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)


def evaluate_references(df, lambda_values):
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
        rot = rotation_only_score(df, alat, alon, angle)

        for lam in lambda_values:
            joint = model_score(df, alat, alon, angle, lam)

            rows.append({
                "name": name,
                "axis_lat": alat,
                "axis_lon": alon,
                "angle_deg": angle,
                "lambda_q": lam,
                "rotated_pole_lat": joint["rotated_pole_lat"],
                "rotated_pole_lon": joint["rotated_pole_lon"],
                "rotation_only_r2": rot["r2"],
                "constrained_joint_r2": joint["r2"],
                "quadrupole_gain_r2": joint["r2"] - rot["r2"],
                "rotation_only_rmse": rot["rmse"],
                "constrained_joint_rmse": joint["rmse"],
                "spearman_pred_obs": joint["spearman_pred_obs"],
                "rotation_coeff": joint["rotation_coeff"],
                "quadrupole_coeff_norm": joint["quadrupole_coeff_norm"],
            })

    return pd.DataFrame(rows).sort_values(
        ["lambda_q", "constrained_joint_r2"],
        ascending=[True, False],
    ).reset_index(drop=True)


def add_spatial_cells(df, lat_bin_deg=10.0, lon_bin_deg=20.0):
    out = df.copy()
    out["lat_bin"] = np.floor((out["latitude"] + 90.0) / lat_bin_deg).astype(int)
    out["lon_bin"] = np.floor((out["longitude"] + 180.0) / lon_bin_deg).astype(int)
    out["spatial_cell"] = out["lat_bin"].astype(str) + "_" + out["lon_bin"].astype(str)
    return out


def permute_global(df, rng):
    out = df.copy()
    out["z_m"] = rng.permutation(out["z_m"].to_numpy(float))
    return out


def permute_within_cells(df, rng):
    out = df.copy()
    out["z_perm"] = out["z_m"]

    for _, idx in out.groupby("spatial_cell").groups.items():
        idx = np.array(list(idx), dtype=int)
        out.loc[idx, "z_perm"] = rng.permutation(out.loc[idx, "z_m"].to_numpy(float))

    out["z_m"] = out["z_perm"]
    return out.drop(columns=["z_perm"])


def permutation_test(df, candidates, args, rng):
    rows = []

    test_rows = candidates.copy()
    if len(test_rows) > args.max_perm_candidates:
        test_rows = test_rows.head(args.max_perm_candidates)

    for _, cand in test_rows.iterrows():
        name = cand.get("name", "scan_candidate")
        alat = float(cand["axis_lat"])
        alon = float(cand["axis_lon"])
        angle = float(cand["angle_deg"])
        lam = float(cand["lambda_q"])
        obs = float(cand["constrained_joint_r2"] if "constrained_joint_r2" in cand else cand["r2"])

        for mode in ["global", "within_cells"]:
            null = []

            for _ in range(args.permutations):
                pdf = permute_global(df, rng) if mode == "global" else permute_within_cells(df, rng)
                fit = model_score(pdf, alat, alon, angle, lam)
                null.append(fit["r2"])

            arr = np.asarray(null)

            rows.append({
                "name": name,
                "mode": mode,
                "axis_lat": alat,
                "axis_lon": alon,
                "angle_deg": angle,
                "lambda_q": lam,
                "observed_r2": obs,
                "p": float((np.sum(arr >= obs) + 1) / (len(arr) + 1)),
                "null_median": float(np.median(arr)),
                "null_p95": float(np.quantile(arr, 0.95)),
                "null_p99": float(np.quantile(arr, 0.99)),
            })

    return pd.DataFrame(rows)


def write_prediction(df, row):
    fit = model_score(
        df,
        float(row["axis_lat"]),
        float(row["axis_lon"]),
        float(row["angle_deg"]),
        float(row["lambda_q"]),
    )

    out = df.copy()
    out["constrained_prediction_z_m"] = fit["prediction"]
    out["constrained_residual_z_m"] = fit["residual"]
    out["rotation_anomaly"] = fit["rotation_anomaly"]

    return out, fit


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


def make_plots(scan_df, refs, pred_df, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    best_by_lambda = (
        scan_df.sort_values("r2", ascending=False)
        .groupby("lambda_q", as_index=False)
        .first()
        .sort_values("lambda_q")
    )

    plt.figure(figsize=(9, 5))
    plt.plot(best_by_lambda["lambda_q"], best_by_lambda["r2"], marker="o")
    plt.xscale("log")
    plt.xlabel("Quadrupole ridge penalty λ")
    plt.ylabel("Best constrained joint R²")
    plt.title("09 best R² vs quadrupole penalty")
    plt.tight_layout()
    plt.savefig(out_dir / "09_best_r2_vs_lambda.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(best_by_lambda["lambda_q"], best_by_lambda["angle_deg"], marker="o")
    plt.xscale("log")
    plt.xlabel("Quadrupole ridge penalty λ")
    plt.ylabel("Best displacement angle")
    plt.title("09 best angle vs quadrupole penalty")
    plt.tight_layout()
    plt.savefig(out_dir / "09_best_angle_vs_lambda.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        best_by_lambda["axis_lon"],
        best_by_lambda["axis_lat"],
        c=best_by_lambda["lambda_q"],
        s=60,
    )
    plt.colorbar(sc, label="λ")
    plt.xlabel("Best axis longitude")
    plt.ylabel("Best axis latitude")
    plt.title("09 best axis path as quadrupole is constrained")
    plt.tight_layout()
    plt.savefig(out_dir / "09_best_axis_path_vs_lambda.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    lambda_values = np.sort(refs["lambda_q"].unique())
    mid_lambda = lambda_values[len(lambda_values) // 2]

    subset = refs[np.isclose(refs["lambda_q"], mid_lambda)].copy()
    subset = subset.sort_values("constrained_joint_r2", ascending=False)

    if len(subset) > 0:
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(subset)), subset["constrained_joint_r2"])
        plt.xticks(range(len(subset)), subset["name"], rotation=60, ha="right")
        plt.ylabel("Constrained joint R²")
        plt.title(f"09 fixed references at λ={mid_lambda:g}")
        plt.tight_layout()
        plt.savefig(out_dir / "09_fixed_reference_scores_mid_lambda.png", dpi=220)
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(pred_df["constrained_prediction_z_m"], pred_df["z_m"], s=28)
    plt.xlabel("Predicted z_m")
    plt.ylabel("Observed reported z_m")
    plt.title("09 observed vs constrained joint prediction")
    plt.tight_layout()
    plt.savefig(out_dir / "09_observed_vs_constrained_prediction.png", dpi=220)
    plt.close()

    plt.figure(figsize=(11, 6))
    sc = plt.scatter(
        pred_df["longitude"],
        pred_df["latitude"],
        c=pred_df["constrained_residual_z_m"],
        s=30,
    )
    plt.colorbar(sc, label="Residual z_m")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("09 constrained joint residual map")
    plt.tight_layout()
    plt.savefig(out_dir / "09_constrained_joint_residual_map.png", dpi=220)
    plt.close()


def parse_lambda_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


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

    parser.add_argument(
        "--lambda-q",
        type=parse_lambda_list,
        default=parse_lambda_list("0.1,1,3,10,30,100,300,1000,3000,10000,30000,100000"),
    )

    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--max-perm-candidates", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    input_path = Path(args.input) if args.input else (IN_08 if IN_08.exists() else IN_06)

    df = load_df(
        input_path,
        threshold=args.depth_threshold,
        lat_bin_deg=args.lat_bin_deg,
        lon_bin_deg=args.lon_bin_deg,
    )

    scan_df = scan(df, args)
    refs = evaluate_references(df, args.lambda_q)

    # Choose a conservative best: highest R² among λ >= 100 unless none.
    conservative_pool = scan_df[scan_df["lambda_q"] >= 100].copy()
    best_row = conservative_pool.iloc[0].to_dict() if len(conservative_pool) else scan_df.iloc[0].to_dict()

    pred_df, best_fit = write_prediction(df, best_row)

    # Permute fixed references plus best-by-lambda.
    best_by_lambda = (
        scan_df.sort_values("r2", ascending=False)
        .groupby("lambda_q", as_index=False)
        .first()
        .sort_values("r2", ascending=False)
    )
    best_by_lambda["name"] = best_by_lambda["lambda_q"].map(lambda x: f"best_scan_lambda_{x:g}")

    ref_perm_input = refs.copy()
    ref_perm_input["r2"] = ref_perm_input["constrained_joint_r2"]

    perm_input = pd.concat(
        [
            ref_perm_input[["name", "axis_lat", "axis_lon", "angle_deg", "lambda_q", "r2"]],
            best_by_lambda[["name", "axis_lat", "axis_lon", "angle_deg", "lambda_q", "r2"]],
        ],
        ignore_index=True,
    )

    perm = permutation_test(df, perm_input, args, rng)

    scan_df.to_csv(OUT_DIR / "09_constrained_joint_scan.csv", index=False)
    scan_df.head(250).to_csv(OUT_DIR / "09_top250_constrained_joint_candidates.csv", index=False)
    refs.to_csv(OUT_DIR / "09_fixed_reference_constrained_scores.csv", index=False)
    perm.to_csv(OUT_DIR / "09_permutation_tests.csv", index=False)
    pred_df.to_csv(OUT_DIR / "09_littoral_records_with_constrained_prediction.csv", index=False)

    summary = {
        "metadata": {
            "input": str(input_path),
            "n": int(len(df)),
            "reported_depth_source_records_only": True,
            "selection_criteria": "depth_source == reported, latitude/longitude present, and z_m <= 200 m",
            "relative_gradients_only": True,
            "absolute_depths_interpreted": False,
            "lambda_q": args.lambda_q,
            "conservative_best_rule": "highest R2 among lambda_q >= 100",
            "permutations": int(args.permutations),
            "seed": int(args.seed),
            "chronology_used": False,
            "origin_assumptions_used": False,
        },
        "best_conservative_candidate": {
            k: v for k, v in best_fit.items()
            if k not in ["prediction", "residual", "rotation_anomaly"]
        },
        "best_by_lambda": best_by_lambda.to_dict("records"),
        "fixed_reference_scores": refs.to_dict("records"),
        "permutation_tests": perm.to_dict("records"),
    }

    with open(OUT_DIR / "09_constrained_rotation_quadrupole_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_plots(scan_df, refs, pred_df, OUT_DIR)

    print("\nLITTORAL 09 constrained rotation + quadrupole inverse complete.")
    print(f"Input: {input_path}")
    print(f"Output: {OUT_DIR}")
    print(f"Records: {len(df)}")

    print("\nBest conservative candidate:")
    for k, v in summary["best_conservative_candidate"].items():
        print(f"  {k}: {v}")

    print("\nBest by lambda:")
    for _, row in best_by_lambda.iterrows():
        print(
            f"  λ={row['lambda_q']:g}: "
            f"axis=({row['axis_lat']:.1f}, {row['axis_lon']:.1f}), "
            f"angle={row['angle_deg']:.1f}, "
            f"r2={row['r2']:.4f}, "
            f"rot_coeff={row['rotation_coeff']:.3f}, "
            f"q_norm={row['quadrupole_coeff_norm']:.3f}"
        )

    print("\nFixed references, strongest λ:")
    lam_max = max(args.lambda_q)
    sub = refs[refs["lambda_q"] == lam_max].sort_values("constrained_joint_r2", ascending=False)
    for _, row in sub.iterrows():
        print(
            f"  {row['name']}: "
            f"rot_only={row['rotation_only_r2']:.4f}, "
            f"joint={row['constrained_joint_r2']:.4f}, "
            f"gain={row['quadrupole_gain_r2']:.4f}, "
            f"rot_coeff={row['rotation_coeff']:.3f}, "
            f"q_norm={row['quadrupole_coeff_norm']:.3f}"
        )


if __name__ == "__main__":
    main()
