#!/usr/bin/env python3
"""
05_littoral_spectral_decomposition.py

Spectral decomposition of reported-source littoral geometry.

Record selection: depth_source=reported, coordinates present, z_m <= 200 m.

Reads:
  outputs/geospatial_04/04_littoral_working_records.csv
  or outputs/geospatial_03/03_littoral_working_records_with_cells.csv

Writes:
  outputs/geospatial_05/

Purpose:
  Decompose the reported-source littoral signal into low-degree spherical
  components and test whether residual longitude harmonics survive after
  removing the dominant dipole/gradient.

No dates. No origin assumptions.

Run:
  python scripts/05_littoral_spectral_decomposition.py

Stronger run:
  python scripts/05_littoral_spectral_decomposition.py --permutations 5000 --bootstraps 5000
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
IN_04 = PROJECT_ROOT / "outputs" / "geospatial_04" / "04_littoral_working_records.csv"
IN_03 = PROJECT_ROOT / "outputs" / "geospatial_03" / "03_littoral_working_records_with_cells.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "geospatial_05"


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


def spherical_gradient_fit(df):
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
        "prediction": pred,
        "residual": resid,
    }


def real_spherical_design(lat_deg, lon_deg, lmax=3):
    """
    Real low-degree spherical-harmonic-like basis using elementary forms.

    l=0:
      constant handled by intercept in OLS.

    l=1:
      x, y, z

    l=2:
      xy, xz, yz, x^2-y^2, 3z^2-1

    l=3:
      z(5z^2-3), x(5z^2-1), y(5z^2-1),
      z(x^2-y^2), 2xyz, x(x^2-3y^2), y(3x^2-y^2)

    These are not normalized geodesy harmonics, but they span the same
    low-degree real polynomial subspaces on the unit sphere.
    """
    r = latlon_to_unit(lat_deg, lon_deg)
    x, y, z = r[:, 0], r[:, 1], r[:, 2]

    cols = []
    names = []
    degrees = []

    if lmax >= 1:
        cols += [x, y, z]
        names += ["l1_x", "l1_y", "l1_z"]
        degrees += [1, 1, 1]

    if lmax >= 2:
        cols += [
            x * y,
            x * z,
            y * z,
            x * x - y * y,
            3.0 * z * z - 1.0,
        ]
        names += ["l2_xy", "l2_xz", "l2_yz", "l2_x2_minus_y2", "l2_3z2_minus_1"]
        degrees += [2, 2, 2, 2, 2]

    if lmax >= 3:
        cols += [
            z * (5.0 * z * z - 3.0),
            x * (5.0 * z * z - 1.0),
            y * (5.0 * z * z - 1.0),
            z * (x * x - y * y),
            2.0 * x * y * z,
            x * (x * x - 3.0 * y * y),
            y * (3.0 * x * x - y * y),
        ]
        names += [
            "l3_z_5z2_minus_3",
            "l3_x_5z2_minus_1",
            "l3_y_5z2_minus_1",
            "l3_z_x2_minus_y2",
            "l3_2xyz",
            "l3_x_x2_minus_3y2",
            "l3_y_3x2_minus_y2",
        ]
        degrees += [3, 3, 3, 3, 3, 3, 3]

    X = np.column_stack(cols) if cols else np.empty((len(lat_deg), 0))
    return X, names, degrees


def spectral_fit(df, lmax):
    lat = df["latitude"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    y = df["z_m"].to_numpy(float)

    X, names, degrees = real_spherical_design(lat, lon, lmax=lmax)
    beta, pred, resid, r2 = ols(X, y)

    coeffs = []
    for name, degree, value in zip(names, degrees, beta[1:]):
        coeffs.append({
            "term": name,
            "degree": int(degree),
            "coefficient": float(value),
        })

    degree_power = {}
    for degree in sorted(set(degrees)):
        vals = np.array([c["coefficient"] for c in coeffs if c["degree"] == degree], dtype=float)
        degree_power[f"l{degree}_coefficient_norm"] = float(np.sqrt(np.sum(vals ** 2)))

    return {
        "lmax": int(lmax),
        "n_terms": int(len(names)),
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "intercept": float(beta[0]),
        "coefficients": coeffs,
        "degree_power": degree_power,
        "prediction": pred,
        "residual": resid,
    }


def incremental_spectral_table(df, max_l=3):
    rows = []
    prev_r2 = 0.0

    for lmax in range(1, max_l + 1):
        fit = spectral_fit(df, lmax)
        row = {
            "lmax": lmax,
            "n_terms": fit["n_terms"],
            "r2": fit["r2"],
            "delta_r2": fit["r2"] - prev_r2,
            "rmse": fit["rmse"],
            "mae": fit["mae"],
        }
        row.update(fit["degree_power"])
        rows.append(row)
        prev_r2 = fit["r2"]

    return rows


def longitude_harmonic_scan(values, lon_deg, kmax=8):
    lon = np.radians(lon_deg)
    y = np.asarray(values, dtype=float)

    rows = []
    for k in range(1, kmax + 1):
        X = np.column_stack([np.cos(k * lon), np.sin(k * lon)])
        beta, pred, resid, r2 = ols(X, y)

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


def best_longitude_harmonic(values, lon_deg, kmax=8):
    rows = longitude_harmonic_scan(values, lon_deg, kmax=kmax)
    return max(rows, key=lambda r: r["r2"])


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


def permute_global(df, rng):
    out = df.copy()
    out["z_m"] = rng.permutation(out["z_m"].to_numpy(float))
    return out


def permute_within_cells(df, rng):
    out = df.copy()
    out["z_m_permuted"] = out["z_m"].to_numpy(float)

    for _, idx in out.groupby("spatial_cell").groups.items():
        idx = list(idx)
        out.loc[idx, "z_m_permuted"] = rng.permutation(out.loc[idx, "z_m"].to_numpy(float))

    out["z_m"] = out["z_m_permuted"]
    out = out.drop(columns=["z_m_permuted"])
    return out


def permutation_nulls(df, n_perm, rng, kmax, max_l):
    rows = []

    for mode in ["global", "within_cells"]:
        for _ in range(n_perm):
            if mode == "global":
                pdf = permute_global(df, rng)
            else:
                pdf = permute_within_cells(df, rng)

            grad = spherical_gradient_fit(pdf)
            spec_rows = incremental_spectral_table(pdf, max_l=max_l)
            spec_l3 = spectral_fit(pdf, max_l)

            residual_after_l1 = pdf["z_m"].to_numpy(float) - spherical_gradient_fit(pdf)["prediction"]
            best_resid_harm = best_longitude_harmonic(
                residual_after_l1,
                pdf["longitude"].to_numpy(float),
                kmax=kmax,
            )

            rows.append({
                "mode": mode,
                "gradient_r2": grad["r2"],
                "l1_r2": spec_rows[0]["r2"],
                "l2_r2": spec_rows[1]["r2"] if max_l >= 2 else np.nan,
                "l3_r2": spec_rows[2]["r2"] if max_l >= 3 else np.nan,
                "l2_delta_r2": spec_rows[1]["delta_r2"] if max_l >= 2 else np.nan,
                "l3_delta_r2": spec_rows[2]["delta_r2"] if max_l >= 3 else np.nan,
                "residual_best_harmonic_k": best_resid_harm["k"],
                "residual_best_harmonic_r2": best_resid_harm["r2"],
            })

    return rows


def bootstrap_one_per_cell(df, n_boot, rng, kmax, max_l, min_n):
    rows = []

    for _ in range(n_boot):
        bdf = sample_one_per_cell(df, rng)
        if len(bdf) < min_n:
            continue

        grad = spherical_gradient_fit(bdf)
        spec_rows = incremental_spectral_table(bdf, max_l=max_l)

        residual_after_l1 = bdf["z_m"].to_numpy(float) - grad["prediction"]
        raw_harm = best_longitude_harmonic(
            bdf["z_m"].to_numpy(float),
            bdf["longitude"].to_numpy(float),
            kmax=kmax,
        )
        resid_harm = best_longitude_harmonic(
            residual_after_l1,
            bdf["longitude"].to_numpy(float),
            kmax=kmax,
        )

        row = {
            "n": int(len(bdf)),
            "gradient_r2": grad["r2"],
            "gradient_axis_lat": grad["axis_lat"],
            "gradient_axis_lon": grad["axis_lon"],
            "raw_best_harmonic_k": raw_harm["k"],
            "raw_best_harmonic_r2": raw_harm["r2"],
            "residual_best_harmonic_k": resid_harm["k"],
            "residual_best_harmonic_r2": resid_harm["r2"],
        }

        for srow in spec_rows:
            row[f"l{srow['lmax']}_r2"] = srow["r2"]
            row[f"l{srow['lmax']}_delta_r2"] = srow["delta_r2"]

        rows.append(row)

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


def make_plots(df, residual_df, spec_table, boot_rows, null_rows, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(spec_table["lmax"], spec_table["r2"], marker="o")
    plt.xlabel("Maximum spherical degree")
    plt.ylabel("R²")
    plt.title("05 incremental low-degree spectral fit")
    plt.tight_layout()
    plt.savefig(out_dir / "05_incremental_spectral_r2.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(spec_table["lmax"], spec_table["delta_r2"])
    plt.xlabel("Added degree")
    plt.ylabel("ΔR²")
    plt.title("05 incremental spectral contribution")
    plt.tight_layout()
    plt.savefig(out_dir / "05_incremental_spectral_delta_r2.png", dpi=220)
    plt.close()

    plt.figure(figsize=(11, 6))
    sc = plt.scatter(
        residual_df["longitude"],
        residual_df["latitude"],
        c=residual_df["residual_after_gradient"],
        s=30,
    )
    plt.colorbar(sc, label="Residual after l=1 gradient")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("05 residual map after dominant gradient")
    plt.tight_layout()
    plt.savefig(out_dir / "05_residual_after_gradient_map.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(residual_df["prediction_gradient"], residual_df["z_m"], s=26)
    plt.xlabel("Predicted z_m from l=1 gradient")
    plt.ylabel("Observed reported z_m")
    plt.title("05 observed vs gradient prediction")
    plt.tight_layout()
    plt.savefig(out_dir / "05_observed_vs_gradient_prediction.png", dpi=220)
    plt.close()

    if boot_rows:
        b = pd.DataFrame(boot_rows)

        plt.figure(figsize=(8, 5))
        plt.hist(b["gradient_r2"], bins=25, alpha=0.8)
        plt.xlabel("Gradient R²")
        plt.ylabel("Bootstrap count")
        plt.title("05 one-per-cell gradient R²")
        plt.tight_layout()
        plt.savefig(out_dir / "05_bootstrap_gradient_r2.png", dpi=220)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(b["gradient_axis_lon"], b["gradient_axis_lat"], s=16, alpha=0.45)
        plt.xlabel("Axis longitude")
        plt.ylabel("Axis latitude")
        plt.title("05 one-per-cell gradient-axis stability")
        plt.tight_layout()
        plt.savefig(out_dir / "05_bootstrap_axis_stability.png", dpi=220)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.hist(b["residual_best_harmonic_r2"], bins=25, alpha=0.8)
        plt.xlabel("Residual best harmonic R²")
        plt.ylabel("Bootstrap count")
        plt.title("05 residual longitude harmonic after gradient removal")
        plt.tight_layout()
        plt.savefig(out_dir / "05_bootstrap_residual_harmonic_r2.png", dpi=220)
        plt.close()

    if null_rows:
        n = pd.DataFrame(null_rows)

        plt.figure(figsize=(8, 5))
        for mode, g in n.groupby("mode"):
            plt.hist(g["l1_r2"], bins=25, alpha=0.45, label=mode)
        plt.xlabel("Null l=1 R²")
        plt.ylabel("Permutation count")
        plt.title("05 null l=1 distributions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "05_null_l1_r2.png", dpi=220)
        plt.close()

        plt.figure(figsize=(8, 5))
        for mode, g in n.groupby("mode"):
            plt.hist(g["residual_best_harmonic_r2"], bins=25, alpha=0.45, label=mode)
        plt.xlabel("Null residual best harmonic R²")
        plt.ylabel("Permutation count")
        plt.title("05 null residual harmonic distributions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "05_null_residual_harmonic_r2.png", dpi=220)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth-threshold", type=float, default=-200.0)
    parser.add_argument("--lat-bin-deg", type=float, default=10.0)
    parser.add_argument("--lon-bin-deg", type=float, default=20.0)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--max-l", type=int, default=3)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--min-n", type=int, default=12)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    input_path = Path(args.input) if args.input else (IN_04 if IN_04.exists() else IN_03)

    df = load_df(
        input_path,
        threshold=args.depth_threshold,
        lat_bin_deg=args.lat_bin_deg,
        lon_bin_deg=args.lon_bin_deg,
    )

    gradient = spherical_gradient_fit(df)
    spec_rows = incremental_spectral_table(df, max_l=args.max_l)
    spec_lmax = spectral_fit(df, args.max_l)

    raw_harmonics = longitude_harmonic_scan(
        df["z_m"].to_numpy(float),
        df["longitude"].to_numpy(float),
        kmax=args.kmax,
    )
    raw_best_harm = max(raw_harmonics, key=lambda r: r["r2"])

    residual_after_gradient = df["z_m"].to_numpy(float) - gradient["prediction"]

    residual_harmonics = longitude_harmonic_scan(
        residual_after_gradient,
        df["longitude"].to_numpy(float),
        kmax=args.kmax,
    )
    residual_best_harm = max(residual_harmonics, key=lambda r: r["r2"])

    residual_df = df.copy()
    residual_df["prediction_gradient"] = gradient["prediction"]
    residual_df["residual_after_gradient"] = residual_after_gradient
    residual_df["prediction_lmax"] = spec_lmax["prediction"]
    residual_df["residual_after_lmax"] = spec_lmax["residual"]

    null_rows = permutation_nulls(
        df=df,
        n_perm=args.permutations,
        rng=rng,
        kmax=args.kmax,
        max_l=args.max_l,
    )

    null_df = pd.DataFrame(null_rows)

    null_summaries = {}
    for mode, g in null_df.groupby("mode"):
        null_summaries[mode] = {
            "summary": summarize_numeric(g.to_dict("records")),
            "gradient_r2_p": p_greater(gradient["r2"], g["gradient_r2"]),
            "l1_r2_p": p_greater(spec_rows[0]["r2"], g["l1_r2"]),
            "l2_r2_p": p_greater(spec_rows[1]["r2"], g["l2_r2"]) if args.max_l >= 2 else None,
            "l3_r2_p": p_greater(spec_rows[2]["r2"], g["l3_r2"]) if args.max_l >= 3 else None,
            "residual_best_harmonic_r2_p": p_greater(
                residual_best_harm["r2"],
                g["residual_best_harmonic_r2"],
            ),
        }

    boot_rows = bootstrap_one_per_cell(
        df=df,
        n_boot=args.bootstraps,
        rng=rng,
        kmax=args.kmax,
        max_l=args.max_l,
        min_n=args.min_n,
    )

    summary = {
        "metadata": {
            "input": str(input_path),
            "n": int(len(df)),
            "reported_depth_source_records_only": True,
            "selection_criteria": "depth_source == reported, latitude/longitude present, and z_m <= 200 m",
            "depth_threshold_m": float(args.depth_threshold),
            "lat_bin_deg": float(args.lat_bin_deg),
            "lon_bin_deg": float(args.lon_bin_deg),
            "kmax": int(args.kmax),
            "max_l": int(args.max_l),
            "permutations_per_null": int(args.permutations),
            "bootstraps": int(args.bootstraps),
            "seed": int(args.seed),
            "chronology_used": False,
            "origin_assumptions_used": False,
        },
        "dominant_gradient": {
            k: v for k, v in gradient.items() if k not in ["prediction", "residual"]
        },
        "incremental_spectral_fit": spec_rows,
        "lmax_fit": {
            k: v for k, v in spec_lmax.items() if k not in ["prediction", "residual"]
        },
        "raw_longitude_harmonics": raw_harmonics,
        "raw_best_longitude_harmonic": raw_best_harm,
        "residual_after_gradient_longitude_harmonics": residual_harmonics,
        "residual_after_gradient_best_longitude_harmonic": residual_best_harm,
        "null_summaries": null_summaries,
        "one_per_cell_bootstrap_summary": summarize_numeric(boot_rows),
    }

    with open(OUT_DIR / "05_littoral_spectral_decomposition_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([summary["dominant_gradient"]]).to_csv(
        OUT_DIR / "05_dominant_gradient.csv",
        index=False,
    )
    pd.DataFrame(spec_rows).to_csv(
        OUT_DIR / "05_incremental_spectral_fit.csv",
        index=False,
    )
    pd.DataFrame(spec_lmax["coefficients"]).to_csv(
        OUT_DIR / "05_lmax_coefficients.csv",
        index=False,
    )
    pd.DataFrame(raw_harmonics).to_csv(
        OUT_DIR / "05_raw_longitude_harmonics.csv",
        index=False,
    )
    pd.DataFrame(residual_harmonics).to_csv(
        OUT_DIR / "05_residual_after_gradient_longitude_harmonics.csv",
        index=False,
    )
    residual_df.to_csv(
        OUT_DIR / "05_littoral_records_with_predictions_and_residuals.csv",
        index=False,
    )
    null_df.to_csv(
        OUT_DIR / "05_null_distributions.csv",
        index=False,
    )
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "05_one_per_cell_bootstrap_spectral_metrics.csv",
        index=False,
    )

    make_plots(
        df=df,
        residual_df=residual_df,
        spec_table=pd.DataFrame(spec_rows),
        boot_rows=boot_rows,
        null_rows=null_rows,
        out_dir=OUT_DIR,
    )

    print("\nLITTORAL 05 spectral decomposition complete.")
    print(f"Input: {input_path}")
    print(f"Output: {OUT_DIR}")
    print(f"Records: {len(df)}")

    print("\nDominant gradient:")
    for k, v in summary["dominant_gradient"].items():
        print(f"  {k}: {v}")

    print("\nIncremental spectral fit:")
    for row in spec_rows:
        print(
            f"  lmax={row['lmax']}: "
            f"r2={row['r2']:.6f}, delta_r2={row['delta_r2']:.6f}, rmse={row['rmse']:.3f}"
        )

    print("\nRaw best longitude harmonic:")
    for k, v in raw_best_harm.items():
        print(f"  {k}: {v}")

    print("\nResidual best longitude harmonic after gradient removal:")
    for k, v in residual_best_harm.items():
        print(f"  {k}: {v}")

    print("\nNull p-values:")
    for mode, values in null_summaries.items():
        print(
            f"  {mode}: "
            f"gradient_p={values['gradient_r2_p']}, "
            f"l1_p={values['l1_r2_p']}, "
            f"residual_harmonic_p={values['residual_best_harmonic_r2_p']}"
        )


if __name__ == "__main__":
    main()
