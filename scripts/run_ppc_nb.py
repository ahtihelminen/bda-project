#!/usr/bin/env python
"""
Posterior predictive checks for the 2-class NB MVP model.

- Inputs:
    * idata.nc from the PyMC fit, containing posterior samples of:
        - mu0, mu1, phi0, phi1
          (or log_mu0, log_mu1, log_phi0, log_phi1)
    * counts parquet: sequence, patch_id, time_bin, y
    * labels csv:     sequence, patch_id, time_bin, z (0/1)

- Output:
    * PNG file with histograms + ECDFs for background (z=0)
      and drone (z=1): <output-prefix>_ppc_nb.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _get_param(
    idata: az.InferenceData,
    name: str,
    log_name: str | None = None,
) -> np.ndarray:
    """
    Extract a 1D array of posterior samples for a parameter.

    Tries:
        - posterior[name]
        - exp(posterior[log_name]) if log_name is given
    """
    posterior = idata.posterior

    if name in posterior:
        arr = posterior[name].values  # (chain, draw, ...)
        return arr.reshape(-1)

    if log_name is not None and log_name in posterior:
        arr = np.exp(posterior[log_name].values)
        return arr.reshape(-1)

    raise KeyError(
        f"Neither '{name}' nor '{log_name}' found in idata.posterior"
    )


def _simulate_yrep_class(
    rng: np.random.Generator,
    mu_samples: np.ndarray,
    phi_samples: np.ndarray,
    n_obs: int,
    max_draws: int | None = None,
) -> np.ndarray:
    """
    Simulate replicated counts for one class using NB(mu, phi).

    Returns an array of shape (n_draws, n_obs).
    """
    if max_draws is not None and max_draws < mu_samples.size:
        idx = rng.choice(mu_samples.size, size=max_draws, replace=False)
        mu_samples = mu_samples[idx]
        phi_samples = phi_samples[idx]

    n_draws = mu_samples.size
    yrep = np.empty((n_draws, n_obs), dtype=int)

    for i in range(n_draws):
        mu = float(mu_samples[i])
        phi = float(phi_samples[i])

        # NB2 parameterisation:
        # mean = mu, var = mu + mu^2 / phi
        # numpy uses (n, p) with:
        #   n = phi, p = phi / (phi + mu)
        if mu <= 0 or phi <= 0:
            # very defensive; skip weird draws
            yrep[i, :] = 0
            continue

        n = phi
        p = phi / (phi + mu)
        yrep[i, :] = rng.negative_binomial(n, p, size=n_obs)

    return yrep


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simple empirical CDF."""
    x = np.asarray(x, dtype=float)
    x_sorted = np.sort(x)
    n = x_sorted.size
    y = np.arange(1, n + 1) / n
    return x_sorted, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Posterior predictive checks for NB MVP model."
    )
    parser.add_argument(
        "idata_path",
        type=Path,
        help="Path to idata.nc (InferenceData saved with arviz.to_netcdf).",
    )
    parser.add_argument(
        "--counts-parquet",
        type=str,
        required=True,
        help="Parquet with columns: sequence, patch_id, time_bin, y",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        required=True,
        help="CSV with columns: sequence, patch_id, time_bin, z (0/1)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="ppc_nb",
        help="Prefix for output PNG file (default: ppc_nb).",
    )
    parser.add_argument(
        "--max-draws",
        type=int,
        default=1000,
        help="Maximum number of posterior draws to use (default: 1000).",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=100_000,
        help="Maximum number of rows from counts/labels to use for PPC.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for posterior predictive sampling.",
    )

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # --- Load inference data (for parameters) ---
    print(f"Loading inference data from {args.idata_path} ...")
    idata = az.from_netcdf(args.idata_path)

    # --- Load per-patch counts + labels like in the comparison script ---
    print(f"Loading counts from {args.counts_parquet} ...")
    counts = pd.read_parquet(args.counts_parquet)

    print(f"Loading labels from {args.labels_csv} ...")
    labels = pd.read_csv(args.labels_csv)

    required_counts = {"sequence", "patch_id", "time_bin", "y"}
    required_labels = {"sequence", "patch_id", "time_bin", "z"}

    if not required_counts.issubset(counts.columns):
        raise RuntimeError(
            f"Counts parquet must have columns {sorted(required_counts)}; "
            f"got {sorted(counts.columns)}"
        )
    if not required_labels.issubset(labels.columns):
        raise RuntimeError(
            f"Labels csv must have columns {sorted(required_labels)}; "
            f"got {sorted(labels.columns)}"
        )

    df = counts.merge(
        labels[["sequence", "patch_id", "time_bin", "z"]],
        on=["sequence", "patch_id", "time_bin"],
        how="left",
    )
    df["z"] = df["z"].fillna(0).astype(int)

    # Keep only {0,1}
    df = df[df["z"].isin([0, 1])].copy()
    if df.empty:
        raise RuntimeError("No rows with z in {0,1} after merge.")

    # Subsample if needed
    if len(df) > args.max_eval_rows:
        df_eval = df.sample(n=args.max_eval_rows, random_state=1337)
        print(f"Using subset of {args.max_eval_rows} rows out of {len(df)} for PPC.")
    else:
        df_eval = df
        print(f"Using all {len(df_eval)} rows for PPC.")

    # Observed counts and labels
    y = df_eval["y"].to_numpy(dtype=int)
    z = df_eval["z"].to_numpy(dtype=int)

    # --- Extract posterior parameters (mu0, mu1, phi0, phi1) ---
    print("Extracting posterior samples for mu0, mu1, phi0, phi1 ...")
    mu0 = _get_param(idata, "mu0", log_name="log_mu0")
    mu1 = _get_param(idata, "mu1", log_name="log_mu1")
    phi0 = _get_param(idata, "phi0", log_name="log_phi0")
    phi1 = _get_param(idata, "phi1", log_name="log_phi1")

    print(
        f"Using {mu0.size} posterior draws "
        f"(will subsample to at most {args.max_draws})."
    )

    # --- Split observed counts by class ---
    mask_bg = z == 0
    mask_drone = z == 1

    y_bg = y[mask_bg]
    y_drone = y[mask_drone]

    if y_bg.size == 0 or y_drone.size == 0:
        raise SystemExit(
            f"Need both classes present. "
            f"Found n_bg={y_bg.size}, n_drone={y_drone.size}."
        )

    print(f"n_bg = {y_bg.size}, n_drone = {y_drone.size}")

    # --- Simulate posterior predictive replicated counts ---
    print("Simulating posterior predictive replicated counts ...")

    yrep_bg = _simulate_yrep_class(
        rng,
        mu_samples=mu0,
        phi_samples=phi0,
        n_obs=y_bg.size,
        max_draws=args.max_draws,
    )
    yrep_drone = _simulate_yrep_class(
        rng,
        mu_samples=mu1,
        phi_samples=phi1,
        n_obs=y_drone.size,
        max_draws=args.max_draws,
    )

    # For histograms we can just flatten over draws
    yrep_bg_flat = yrep_bg.ravel()
    yrep_drone_flat = yrep_drone.ravel()

    # --- Build plots ---
    print("Generating PPC plots ...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    def _auto_bins(obs: np.ndarray, rep: np.ndarray, max_bins: int = 40) -> np.ndarray:
        both = np.concatenate([obs, rep])
        q_low = np.percentile(both, 0)
        q_high = np.percentile(both, 99.5)
        q_low = max(q_low, 0)
        if q_high <= q_low:
            q_high = q_low + 1.0
        n_bins = min(max_bins, int(q_high - q_low + 1))
        n_bins = max(n_bins, 5)
        return np.linspace(q_low, q_high, n_bins)

    # ---------------------------
    # HISTOGRAM: BACKGROUND
    # ---------------------------
    bins_bg = _auto_bins(y_bg, yrep_bg_flat)
    ax = axes[0, 0]
    ax.hist(
        yrep_bg_flat,
        bins=bins_bg,
        density=True,
        alpha=0.4,
        label="replicated (z=0)",
    )
    ax.hist(
        y_bg,
        bins=bins_bg,
        density=True,
        histtype="step",
        linewidth=2.0,
        label="observed (z=0)",
    )
    ax.set_title("Background counts (histogram)")
    ax.set_xlabel("y")
    ax.set_ylabel("density (log)")
    ax.set_yscale("log")
    ax.legend()

    # ---------------------------
    # HISTOGRAM: DRONE
    # ---------------------------
    bins_drone = _auto_bins(y_drone, yrep_drone_flat)
    ax = axes[0, 1]
    ax.hist(
        yrep_drone_flat,
        bins=bins_drone,
        density=True,
        alpha=0.4,
        label="replicated (z=1)",
    )
    ax.hist(
        y_drone,
        bins=bins_drone,
        density=True,
        histtype="step",
        linewidth=2.0,
        label="observed (z=1)",
    )
    ax.set_title("Drone counts (histogram)")
    ax.set_xlabel("y")
    ax.set_ylabel("density (log)")
    ax.set_yscale("log")
    ax.legend()

    # ---------------------------
    # ECDF: BACKGROUND
    # ---------------------------
    ax = axes[1, 0]
    x_obs_bg, F_obs_bg = _ecdf(y_bg)
    x_rep_bg, F_rep_bg = _ecdf(yrep_bg_flat)
    ax.step(x_rep_bg, F_rep_bg, where="post", alpha=0.6, label="replicated (z=0)")
    ax.step(x_obs_bg, F_obs_bg, where="post", linewidth=2.0, label="observed (z=0)")
    ax.set_title("Background counts (ECDF)")
    ax.set_xlabel("y")
    ax.set_ylabel("F(y)")
    ax.legend()

    # ---------------------------
    # ECDF: DRONE
    # ---------------------------
    ax = axes[1, 1]
    x_obs_drone, F_obs_drone = _ecdf(y_drone)
    x_rep_drone, F_rep_drone = _ecdf(yrep_drone_flat)
    ax.step(x_rep_drone, F_rep_drone, where="post", alpha=0.6, label="replicated (z=1)")
    ax.step(x_obs_drone, F_obs_drone, where="post", linewidth=2.0, label="observed (z=1)")
    ax.set_title("Drone counts (ECDF)")
    ax.set_xlabel("y")
    ax.set_ylabel("F(y)")
    ax.legend()


    fig.suptitle("Posterior predictive checks for NB model", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    out_path = Path(f".data/{args.output_prefix}_ppc_nb.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved PPC figure to {out_path}")


if __name__ == "__main__":
    main()
