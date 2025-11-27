#!/usr/bin/env python
"""
Posterior predictive checks for the hierarchical 2-class NB model.

Model:
    y_n | z_n = c, patch_id = i ~ NegBinom(mu_{c,i}, phi_c)
    log mu_{c,i} = log_mu_c + b_i
    b_i ~ Normal(0, sigma_b)

Inputs:
    * hier_idata.nc: ArviZ InferenceData from the Stan fit
        - posterior variables: log_mu0, log_mu1, b[K], alpha0, alpha1, ...
    * counts parquet: sequence, patch_id, time_bin, y
    * labels csv:     sequence, patch_id, time_bin, z (0/1)
    * patch csv:      patch_id, mu0_patch, mu1_patch
      (rows correspond to the same K patches as b[1..K])

Output:
    * <output-prefix>_ppc_hier_nb.png with hist + ECDF PPC, log y-axis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x_sorted = np.sort(x)
    n = x_sorted.size
    y = np.arange(1, n + 1) / n
    return x_sorted, y


def _simulate_yrep_hier_class(
    rng: np.random.Generator,
    mu_patch_samples: np.ndarray,   # (n_draws, n_patches)
    phi_samples: np.ndarray,        # (n_draws,)
    patch_index_obs: np.ndarray,    # (n_obs,), indices into patch axis
) -> np.ndarray:
    """
    Simulate replicated counts for one class using patch-specific NB(mu, phi).

    Returns yrep with shape (n_draws, n_obs).
    """
    n_draws, n_patches = mu_patch_samples.shape
    if phi_samples.shape[0] != n_draws:
        raise ValueError(
            f"phi_samples has {phi_samples.shape[0]} draws, "
            f"but mu_patch_samples has {n_draws} draws."
        )

    n_obs = patch_index_obs.size
    yrep = np.empty((n_draws, n_obs), dtype=int)

    for s in range(n_draws):
        mu_obs = mu_patch_samples[s, patch_index_obs]  # (n_obs,)
        phi = float(phi_samples[s])

        # NB2 parameterisation:
        # mean = mu, var = mu + mu^2 / phi
        # numpy uses (n, p) with:
        #   n = phi, p = phi / (phi + mu)
        mask_bad = (mu_obs <= 0) | (phi <= 0)
        if np.any(mask_bad):
            mu_obs = np.where(mu_obs <= 0, 1e-6, mu_obs)
            if phi <= 0:
                phi = 1e-6

        n_param = phi
        p_param = phi / (phi + mu_obs)
        yrep[s, :] = rng.negative_binomial(n_param, p_param)

    return yrep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Posterior predictive checks for hierarchical NB model."
    )
    parser.add_argument(
        "idata_path",
        type=Path,
        help="Path to hierarchical idata.nc (InferenceData).",
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
        "--patch-csv",
        type=str,
        required=True,
        help="CSV with columns: patch_id, mu0_patch, mu1_patch "
             "(same K patches/order as Stan random effect b).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="ppc_hier_nb",
        help="Prefix for output PNG file (default: ppc_hier_nb).",
    )
    parser.add_argument(
        "--max-draws",
        type=int,
        default=500,
        help="Maximum number of posterior draws to use (default: 500).",
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

    # --- Load inference data ---
    print(f"Loading hierarchical inference data from {args.idata_path} ...")
    idata = az.from_netcdf(args.idata_path)
    post = idata.posterior  # type: ignore # xarray.Dataset

    # --- Extract scalar chains ---
    log_mu0 = post["log_mu0"].values.reshape(-1)
    log_mu1 = post["log_mu1"].values.reshape(-1)

    # Dispersions: use alpha0/alpha1 as phi0/phi1
    phi0 = post["alpha0"].values.reshape(-1)
    phi1 = post["alpha1"].values.reshape(-1)

    # Patch effects b: shape (chain, draw, K)
    b_arr = post["b"].values
    # Flatten chains/draws → (n_draws, K)
    b_flat = b_arr.reshape(-1, b_arr.shape[-1])
    n_draws_total, n_patches = b_flat.shape

    # Keep same number of draws for all params
    n_draws = min(args.max_draws, n_draws_total, log_mu0.size, log_mu1.size,
                  phi0.size, phi1.size)
    log_mu0 = log_mu0[:n_draws]
    log_mu1 = log_mu1[:n_draws]
    phi0 = phi0[:n_draws]
    phi1 = phi1[:n_draws]
    b_flat = b_flat[:n_draws, :]

    print(f"Using {n_draws} posterior draws (K={n_patches} patches in Stan model).")

    # Reconstruct patch-level means per draw
    # mu_{c,i} = exp(log_mu_c + b_i)
    mu0_patch_samples = np.exp(log_mu0[:, None] + b_flat)  # (n_draws, K)
    mu1_patch_samples = np.exp(log_mu1[:, None] + b_flat)  # (n_draws, K)

    # --- Load patch mapping from patch CSV ---
    print(f"Loading patch mapping from {args.patch_csv} ...")
    patch_df = pd.read_csv(args.patch_csv)
    if "patch_id" not in patch_df.columns:
        raise RuntimeError(f"{args.patch_csv} must contain 'patch_id' column.")
    patch_ids_fit = patch_df["patch_id"].to_numpy()
    if patch_ids_fit.size != n_patches:
        print(
            f"WARNING: patch_csv has {patch_ids_fit.size} patches but Stan 'b' has {n_patches}. "
            "Assuming order is consistent and using min(K_csv, K_stan)."
        )
        K = min(patch_ids_fit.size, n_patches)
        patch_ids_fit = patch_ids_fit[:K]
        mu0_patch_samples = mu0_patch_samples[:, :K]
        mu1_patch_samples = mu1_patch_samples[:, :K]
        b_flat = b_flat[:, :K]
        n_patches = K

    patch_id_to_idx = {int(pid): idx for idx, pid in enumerate(patch_ids_fit)}

    # --- Load counts + labels and merge, like other scripts ---
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
    df = df[df["z"].isin([0, 1])].copy()
    if df.empty:
        raise RuntimeError("No rows with z in {0,1} after merge.")

    # Restrict to patches that were actually in the Stan fit
    mask_fit_patches = df["patch_id"].isin(patch_ids_fit)
    n_before = len(df)
    df = df[mask_fit_patches].copy()
    n_after = len(df)
    if n_after == 0:
        raise RuntimeError(
            "After restricting to patches present in hierarchical model, no rows remain."
        )
    if n_after < n_before:
        print(
            f"Filtered out {n_before - n_after} rows with patches not used in Stan fit."
        )

    # Subsample for PPC
    if len(df) > args.max_eval_rows:
        df_eval = df.sample(n=args.max_eval_rows, random_state=1337)
        print(f"Using subset of {args.max_eval_rows} rows out of {len(df)} for PPC.")
    else:
        df_eval = df
        print(f"Using all {len(df_eval)} rows for PPC.")

    y_all = df_eval["y"].to_numpy(dtype=int)
    z_all = df_eval["z"].to_numpy(dtype=int)
    patch_id_all = df_eval["patch_id"].to_numpy()

    # Map patch_id → column index in mu*_patch_samples
    patch_index_all = np.array(
        [patch_id_to_idx[int(pid)] for pid in patch_id_all],
        dtype=int,
    )

    # Split by class
    mask_bg = z_all == 0
    mask_drone = z_all == 1

    y_bg = y_all[mask_bg]
    y_drone = y_all[mask_drone]
    patch_idx_bg = patch_index_all[mask_bg]
    patch_idx_drone = patch_index_all[mask_drone]

    if y_bg.size == 0 or y_drone.size == 0:
        raise SystemExit(
            f"Need both classes present. "
            f"Found n_bg={y_bg.size}, n_drone={y_drone.size}."
        )

    print(f"n_bg = {y_bg.size}, n_drone = {y_drone.size}")

    # --- Simulate posterior predictive replicated counts ---
    print("Simulating posterior predictive replicated counts ...")

    yrep_bg = _simulate_yrep_hier_class(
        rng,
        mu_patch_samples=mu0_patch_samples,
        phi_samples=phi0,
        patch_index_obs=patch_idx_bg,
    )
    yrep_drone = _simulate_yrep_hier_class(
        rng,
        mu_patch_samples=mu1_patch_samples,
        phi_samples=phi1,
        patch_index_obs=patch_idx_drone,
    )

    yrep_bg_flat = yrep_bg.ravel()
    yrep_drone_flat = yrep_drone.ravel()

    # --- Build plots (log y-axis on all panels) ---
    print("Generating PPC plots for hierarchical model ...")
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

    # Histograms: background
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
    ax.set_title("Hier NB: background counts (histogram)")
    ax.set_xlabel("y")
    ax.set_ylabel("density (log)")
    ax.set_yscale("log")
    ax.legend()

    # Histograms: drone
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
    ax.set_title("Hier NB: drone counts (histogram)")
    ax.set_xlabel("y")
    ax.set_ylabel("density (log)")
    ax.set_yscale("log")
    ax.legend()

    # ECDF: background
    ax = axes[1, 0]
    x_obs_bg, F_obs_bg = _ecdf(y_bg)
    x_rep_bg, F_rep_bg = _ecdf(yrep_bg_flat)
    ax.step(x_rep_bg, F_rep_bg, where="post", alpha=0.6, label="replicated (z=0)")
    ax.step(x_obs_bg, F_obs_bg, where="post", linewidth=2.0, label="observed (z=0)")
    ax.set_title("Hier NB: background counts (ECDF)")
    ax.set_xlabel("y")
    ax.set_ylabel("F(y)")
    ax.legend()

    # ECDF: drone
    ax = axes[1, 1]
    x_obs_drone, F_obs_drone = _ecdf(y_drone)
    x_rep_drone, F_rep_drone = _ecdf(yrep_drone_flat)
    ax.step(x_rep_drone, F_rep_drone, where="post", alpha=0.6, label="replicated (z=1)")
    ax.step(x_obs_drone, F_obs_drone, where="post", linewidth=2.0, label="observed (z=1)")
    ax.set_title("Hier NB: drone counts (ECDF)")
    ax.set_xlabel("y")
    ax.set_ylabel("F(y)")
    ax.legend()

    fig.suptitle("Posterior predictive checks – hierarchical NB model", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # type: ignore

    plt.show()
    out_path = Path(f"{args.output_prefix}_ppc_hier_nb.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved PPC figure to {out_path}")


if __name__ == "__main__":
    main()
