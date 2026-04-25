#!/usr/bin/env python3
"""
06_littoral_quadrupole_validation.py

Focused validation of the l=2 / quadrupole-like component in reported-source
littoral records.

Record selection: depth_source=reported, coordinates present, z_m <= 200 m.

Reads:
  outputs/geospatial_05/05_littoral_records_with_predictions_and_residuals.csv
  or outputs/geospatial_04/04_littoral_working_records.csv

Writes:
  outputs/geospatial_06/

Purpose:
  Test whether the l=2 component survives:
    - l=1 removal
    - one-cell leave-out
    - source leave-out
    - one-per-cell bootstrap
    - within-cell nulls

No dates. No origin assumptions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_05 = PROJECT_ROOT / "outputs" / "geospatial_05" / "05_littoral_records_with_predictions_and_residuals.csv"
IN_04 = PROJECT_ROOT / "outputs" / "geospatial_04" / "04_littoral_working_records.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "geospatial_06"


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


def design_l1_l2(df):
    r = latlon_to_unit(df["latitude"].to_numpy(float), df["longitude"].to_numpy(float))
    x, y, z = r[:, 0], r[:, 1], r[:, 2]

    X1 = np.column_stack([x, y, z])
    X2 = np.column_stack([
        x * y,
        x * z,
        y * z,
        x * x - y * y,
        3.0 * z * z - 1.0,
    ])

    names_l1 = ["l1_x", "l1_y", "l1_z"]
    names_l2 = ["l2_xy", "l2_xz", "l2_yz", "l2_x2_minus_y2", "l2_3z2_minus_1"]

    return X1, X2, names_l1, names_l2


def fit_l1_l2(df):
    z = df["z_m"].to_numpy(float)
    X1, X2, names_l1, names_l2 = design_l1_l2(df)

    beta1, pred1, resid1, r2_1 = ols(X1, z)
    beta2, pred2, resid2, r2_2 = ols(np.column_stack([X1, X2]), z)

    # Fit l2 directly to l1 residuals.
    beta_resid_l2, pred_resid_l2, resid_after_l2, r2_resid_l2 = ols(X2, resid1)

    l1_vec = beta1[1:]
    l1_norm = float(np.sqrt(np.sum(l1_vec ** 2)))
    l1_axis_lat = float(np.degrees(np.arcsin(l1_vec[2] / l1_norm))) if l1_norm else np.nan
    l1_axis_lon = float(np.degrees(np.arctan2(l1_vec[1], l1_vec[0]))) if l1_norm else np.nan

    l2_coeffs_full = beta2[1 + len(names_l1):]
    l2_coeffs_resid = beta_resid_l2[1:]

    return {
        "n": int(len(df)),
        "l1_r2": float(r2_1),
        "l1_rmse": float(np.sqrt(np.mean(resid1 ** 2))),
        "l1_axis_lat": l1_axis_lat,
        "l1_axis_lon": l1_axis_lon,
        "l1_norm": l1_norm,
        "l1_l2_r2": float(r2_2),
        "l1_l2_rmse": float(np.sqrt(np.mean(resid2 ** 2))),
        "l2_delta_r2": float(r2_2 - r2_1),
        "residual_l2_r2": float(r2_resid_l2),
        "residual_l2_rmse": float(np.sqrt(np.mean(resid_after_l2 ** 2))),
        "l2_norm_full_model": float(np.sqrt(np.sum(l2_coeffs_full ** 2))),
        "l2_norm_residual_model": float(np.sqrt(np.sum(l2_coeffs_resid ** 2))),
        "l1_coefficients": {
            name: float(value) for name, value in zip(names_l1, beta1[1:])
        },
        "l2_coefficients_full_model": {
            name: float(value) for name, value in zip(names_l2, l2_coeffs_full)
        },
        "l2_coefficients_residual_model": {
            name: float(value) for name, value in zip(names_l2, l2_coeffs_resid)
        },
        "pred_l1": pred1,
        "resid_l1": resid1,
        "pred_l1_l2": pred2,
        "resid_l1_l2": resid2,
        "pred_l2_on_l1_resid": pred_resid_l2,
        "resid_after_l2_on_l1_resid": resid_after_l2,
    }


def l2_tensor_from_coeffs(coeffs):
    """
    Convert the five l=2 coefficients into a symmetric trace-free tensor Q
    such that approximately f2 = r.T @ Q @ r.

    Basis:
      a*xy + b*xz + c*yz + d*(x^2-y^2) + e*(3z^2-1)

    Since x^2+y^2+z^2=1:
      e*(3z^2-1) = e*(2z^2 - x^2 - y^2)

    Q:
      Qxx = d - e
      Qyy = -d - e
      Qzz = 2e
      Qxy = a/2
      Qxz = b/2
      Qyz = c/2
    """
    a = coeffs["l2_xy"]
    b = coeffs["l2_xz"]
    c = coeffs["l2_yz"]
    d = coeffs["l2_x2_minus_y2"]
    e = coeffs["l2_3z2_minus_1"]

    Q = np.array([
        [d - e, a / 2.0, b / 2.0],
        [a / 2.0, -d - e, c / 2.0],
        [b / 2.0, c / 2.0, 2.0 * e],
    ], dtype=float)

    return Q


def tensor_eigensystem(Q):
    vals, vecs = np.linalg.eigh(Q)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    rows = []
    for i in range(3):
        v = vecs[:, i]
        lat = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
        lon = float(np.degrees(np.arctan2(v[1], v[0])))
        rows.append({
            "axis_rank": int(i + 1),
            "eigenvalue": float(vals[i]),
            "axis_lat": lat,
            "axis_lon": lon,
        })

    return rows


def add_spatial_cells(df, lat_bin_deg=10.0, lon_bin_deg=20.0):
    out = df.copy()

    if "spatial_cell" not in out.columns:
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
    out["z_perm"] = out["z_m"]

    for _, idx in out.groupby("spatial_cell").groups.items():
        idx = list(idx)
        out.loc[idx, "z_perm"] = rng.permutation(out.loc[idx, "z_m"].to_numpy(float))

    out["z_m"] = out["z_perm"]
    return out.drop(columns=["z_perm"])


def permute_global(df, rng):
    out = df.copy()
    out["z_m"] = rng.permutation(out["z_m"].to_numpy(float))
    return out


def leave_one_cell(df, min_n):
    rows = []

    for cell, g in df.groupby("spatial_cell"):
        sub = df[df["spatial_cell"] != cell].copy()
        if len(sub) < min_n:
            continue

        fit = fit_l1_l2(sub)
        rows.append({
            "left_out_cell": str(cell),
            "left_out_n": int(len(g)),
            "remaining_n": int(len(sub)),
            "l1_r2": fit["l1_r2"],
            "l1_l2_r2": fit["l1_l2_r2"],
            "l2_delta_r2": fit["l2_delta_r2"],
            "residual_l2_r2": fit["residual_l2_r2"],
            "l1_axis_lat": fit["l1_axis_lat"],
            "l1_axis_lon": fit["l1_axis_lon"],
            "l2_norm_residual_model": fit["l2_norm_residual_model"],
        })

    return rows


def leave_one_source(df, min_n):
    if "source_id" not in df.columns:
        return []

    rows = []

    for source, g in df.groupby("source_id"):
        sub = df[df["source_id"] != source].copy()
        if len(sub) < min_n:
            continue

        fit = fit_l1_l2(sub)
        rows.append({
            "left_out_source": str(source),
            "left_out_n": int(len(g)),
            "remaining_n": int(len(sub)),
            "l1_r2": fit["l1_r2"],
            "l1_l2_r2": fit["l1_l2_r2"],
            "l2_delta_r2": fit["l2_delta_r2"],
            "residual_l2_r2": fit["residual_l2_r2"],
            "l1_axis_lat": fit["l1_axis_lat"],
            "l1_axis_lon": fit["l1_axis_lon"],
            "l2_norm_residual_model": fit["l2_norm_residual_model"],
        })

    return rows


def bootstrap_one_per_cell(df, n_boot, rng, min_n):
    rows = []

    for _ in range(n_boot):
        bdf = sample_one_per_cell(df, rng)
        if len(bdf) < min_n:
            continue

        fit = fit_l1_l2(bdf)
        Q = l2_tensor_from_coeffs(fit["l2_coefficients_residual_model"])
        eig = tensor_eigensystem(Q)

        row = {
            "n": int(len(bdf)),
            "l1_r2": fit["l1_r2"],
            "l1_l2_r2": fit["l1_l2_r2"],
            "l2_delta_r2": fit["l2_delta_r2"],
            "residual_l2_r2": fit["residual_l2_r2"],
            "l1_axis_lat": fit["l1_axis_lat"],
            "l1_axis_lon": fit["l1_axis_lon"],
            "l2_norm_residual_model": fit["l2_norm_residual_model"],
        }

        for e in eig:
            prefix = f"q_axis{e['axis_rank']}"
            row[f"{prefix}_eigenvalue"] = e["eigenvalue"]
            row[f"{prefix}_lat"] = e["axis_lat"]
            row[f"{prefix}_lon"] = e["axis_lon"]

        rows.append(row)

    return rows


def null_l2(df, n_perm, rng, mode):
    rows = []

    for _ in range(n_perm):
        if mode == "within_cells":
            pdf = permute_within_cells(df, rng)
        elif mode == "global":
            pdf = permute_global(df, rng)
        else:
            raise ValueError(mode)

        fit = fit_l1_l2(pdf)

        rows.append({
            "mode": mode,
            "n": int(len(pdf)),
            "l1_r2": fit["l1_r2"],
            "l1_l2_r2": fit["l1_l2_r2"],
            "l2_delta_r2": fit["l2_delta_r2"],
            "residual_l2_r2": fit["residual_l2_r2"],
            "l2_norm_residual_model": fit["l2_norm_residual_model"],
        })

    return rows


def p_greater(obs, null_values):
    arr = np.asarray(null_values, dtype=float)
    return float((np.sum(arr >= obs) + 1) / (len(arr) + 1))


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


def make_plots(df, fit, eig_rows, boot_rows, null_rows, loo_cell_rows, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_df = df.copy()
    plot_df["l1_prediction"] = fit["pred_l1"]
    plot_df["l1_residual"] = fit["resid_l1"]
    plot_df["l1_l2_prediction"] = fit["pred_l1_l2"]
    plot_df["l1_l2_residual"] = fit["resid_l1_l2"]

    plt.figure(figsize=(11, 6))
    sc = plt.scatter(plot_df["longitude"], plot_df["latitude"], c=plot_df["l1_residual"], s=30)
    plt.colorbar(sc, label="Residual after l=1")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("06 residual after l=1 gradient")
    plt.tight_layout()
    plt.savefig(out_dir / "06_residual_after_l1_map.png", dpi=220)
    plt.close()

    plt.figure(figsize=(11, 6))
    sc = plt.scatter(plot_df["longitude"], plot_df["latitude"], c=plot_df["l1_l2_residual"], s=30)
    plt.colorbar(sc, label="Residual after l=1+l=2")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("06 residual after l=1+l=2")
    plt.tight_layout()
    plt.savefig(out_dir / "06_residual_after_l1_l2_map.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(plot_df["l1_l2_prediction"], plot_df["z_m"], s=28)
    plt.xlabel("Predicted z_m from l=1+l=2")
    plt.ylabel("Observed reported z_m")
    plt.title("06 observed vs l=1+l=2 prediction")
    plt.tight_layout()
    plt.savefig(out_dir / "06_observed_vs_l1_l2_prediction.png", dpi=220)
    plt.close()

    if boot_rows:
        b = pd.DataFrame(boot_rows)

        plt.figure(figsize=(8, 5))
        plt.hist(b["l2_delta_r2"], bins=25, alpha=0.8)
        plt.xlabel("l=2 ΔR² after l=1")
        plt.ylabel("Bootstrap count")
        plt.title("06 one-per-cell l=2 ΔR²")
        plt.tight_layout()
        plt.savefig(out_dir / "06_bootstrap_l2_delta_r2.png", dpi=220)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(b["q_axis1_lon"], b["q_axis1_lat"], s=15, alpha=0.45)
        plt.xlabel("Quadrupole leading-axis longitude")
        plt.ylabel("Quadrupole leading-axis latitude")
        plt.title("06 bootstrap quadrupole leading-axis stability")
        plt.tight_layout()
        plt.savefig(out_dir / "06_bootstrap_quadrupole_axis1_stability.png", dpi=220)
        plt.close()

    if null_rows:
        n = pd.DataFrame(null_rows)

        plt.figure(figsize=(8, 5))
        for mode, g in n.groupby("mode"):
            plt.hist(g["l2_delta_r2"], bins=25, alpha=0.45, label=mode)
        plt.xlabel("Null l=2 ΔR²")
        plt.ylabel("Permutation count")
        plt.title("06 l=2 null distributions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "06_null_l2_delta_r2.png", dpi=220)
        plt.close()

    if loo_cell_rows:
        c = pd.DataFrame(loo_cell_rows).sort_values("l2_delta_r2")

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(c)), c["l2_delta_r2"])
        plt.xlabel("Leave-one-cell case")
        plt.ylabel("l=2 ΔR²")
        plt.title("06 leave-one-cell l=2 stability")
        plt.tight_layout()
        plt.savefig(out_dir / "06_leave_one_cell_l2_delta_r2.png", dpi=220)
        plt.close()

    plot_df.to_csv(out_dir / "06_littoral_records_with_l1_l2_predictions.csv", index=False)


def strip_arrays(fit):
    return {k: v for k, v in fit.items() if not isinstance(v, np.ndarray)}


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
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    input_path = Path(args.input) if args.input else (IN_05 if IN_05.exists() else IN_04)

    df = load_df(
        input_path,
        threshold=args.depth_threshold,
        lat_bin_deg=args.lat_bin_deg,
        lon_bin_deg=args.lon_bin_deg,
    )

    fit = fit_l1_l2(df)

    Q_full = l2_tensor_from_coeffs(fit["l2_coefficients_full_model"])
    Q_resid = l2_tensor_from_coeffs(fit["l2_coefficients_residual_model"])

    eig_full = tensor_eigensystem(Q_full)
    eig_resid = tensor_eigensystem(Q_resid)

    boot_rows = bootstrap_one_per_cell(df, args.bootstraps, rng, args.min_n)

    null_rows = []
    null_rows.extend(null_l2(df, args.permutations, rng, "global"))
    null_rows.extend(null_l2(df, args.permutations, rng, "within_cells"))

    null_df = pd.DataFrame(null_rows)

    null_summaries = {}
    for mode, g in null_df.groupby("mode"):
        null_summaries[mode] = {
            "summary": summarize_numeric(g.to_dict("records")),
            "l2_delta_r2_p": p_greater(fit["l2_delta_r2"], g["l2_delta_r2"]),
            "residual_l2_r2_p": p_greater(fit["residual_l2_r2"], g["residual_l2_r2"]),
            "l2_norm_residual_model_p": p_greater(
                fit["l2_norm_residual_model"],
                g["l2_norm_residual_model"],
            ),
        }

    loo_cell_rows = leave_one_cell(df, args.min_n)
    loo_source_rows = leave_one_source(df, args.min_n)

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
            "seed": int(args.seed),
            "chronology_used": False,
            "origin_assumptions_used": False,
        },
        "baseline_l1_l2": strip_arrays(fit),
        "quadrupole_tensor_full_model": Q_full.tolist(),
        "quadrupole_tensor_residual_model": Q_resid.tolist(),
        "quadrupole_eigensystem_full_model": eig_full,
        "quadrupole_eigensystem_residual_model": eig_resid,
        "one_per_cell_bootstrap_summary": summarize_numeric(boot_rows),
        "leave_one_cell_summary": summarize_numeric(loo_cell_rows),
        "leave_one_source_summary": summarize_numeric(loo_source_rows),
        "null_summaries": null_summaries,
    }

    with open(OUT_DIR / "06_littoral_quadrupole_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([strip_arrays(fit)]).to_csv(
        OUT_DIR / "06_baseline_l1_l2_metrics.csv",
        index=False,
    )
    pd.DataFrame(eig_full).to_csv(
        OUT_DIR / "06_quadrupole_eigensystem_full_model.csv",
        index=False,
    )
    pd.DataFrame(eig_resid).to_csv(
        OUT_DIR / "06_quadrupole_eigensystem_residual_model.csv",
        index=False,
    )
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "06_one_per_cell_bootstrap_l2.csv",
        index=False,
    )
    null_df.to_csv(
        OUT_DIR / "06_l2_null_distributions.csv",
        index=False,
    )
    pd.DataFrame(loo_cell_rows).to_csv(
        OUT_DIR / "06_leave_one_cell_l2.csv",
        index=False,
    )
    pd.DataFrame(loo_source_rows).to_csv(
        OUT_DIR / "06_leave_one_source_l2.csv",
        index=False,
    )
    df.to_csv(
        OUT_DIR / "06_littoral_working_records.csv",
        index=False,
    )

    make_plots(df, fit, eig_resid, boot_rows, null_rows, loo_cell_rows, OUT_DIR)

    print("\nLITTORAL 06 quadrupole validation complete.")
    print(f"Input: {input_path}")
    print(f"Output: {OUT_DIR}")
    print(f"Records: {len(df)}")

    print("\nBaseline:")
    for k in [
        "l1_r2",
        "l1_l2_r2",
        "l2_delta_r2",
        "residual_l2_r2",
        "l1_axis_lat",
        "l1_axis_lon",
        "l2_norm_residual_model",
    ]:
        print(f"  {k}: {fit[k]}")

    print("\nQuadrupole residual-model eigensystem:")
    for row in eig_resid:
        print(
            f"  axis {row['axis_rank']}: "
            f"eig={row['eigenvalue']:.6f}, "
            f"lat={row['axis_lat']:.3f}, lon={row['axis_lon']:.3f}"
        )

    print("\nNull p-values:")
    for mode, values in null_summaries.items():
        print(
            f"  {mode}: "
            f"l2_delta_r2_p={values['l2_delta_r2_p']}, "
            f"residual_l2_r2_p={values['residual_l2_r2_p']}, "
            f"l2_norm_p={values['l2_norm_residual_model_p']}"
        )


if __name__ == "__main__":
    main()
