from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import pymc as pm

from evio.source.dat_file import DatFileSource

from bda.counts import build_counts_table

def fit_hierarchical_nb_model(
    df: pd.DataFrame,
    *,
    max_samples: int = 3000,
    random_state: int = 1337,
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.9,
    chains: int = 4,
    cores: int | None = None,
    idata_path: str = "./data/hierarchical_nb_idata.nc",
    patch_csv: str | None = "./data/hierarchical_patch_means.csv",
    global_csv: str | None = "./data/hierarchical_global_means.csv",
) -> dict[str, float]:
    """
    Fit a hierarchical 2-class Negative Binomial model using MCMC on a random subset.

    Observation model:
        y_n | z_n = c, patch_id = i ~ NegBinom(mu_{c,i}, phi_c)

    with
        log mu_{c,i} = log_mu_c + b_i
        b_i ~ Normal(0, sigma_b)
        sigma_b ~ half-Normal(0, 1)

    Expects df to have columns:
        y : event counts (int)
        z : labels in {0,1}  (0=background, 1=drone)
        patch_id : patch index in {0, ..., num_patches-1}
    """
    required_cols = {"y", "z", "patch_id"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"DataFrame must contain columns {sorted(required_cols)}; missing {sorted(missing)}."
        raise ValueError(msg)

    # Subsample for faster inference
    if len(df) > max_samples:
        df_fit = df.sample(n=max_samples, random_state=random_state)
        print(f"Using subset of {max_samples} rows out of {len(df)} for MCMC.")
    else:
        df_fit = df
        print(f"Using all {len(df)} rows for MCMC.")

    y = df_fit["y"].to_numpy()
    z = df_fit["z"].to_numpy().astype(np.int32)
    patch_ids = df_fit["patch_id"].to_numpy().astype(np.int32)

    # Use only patches that appear in the subsample
    unique_patches = np.unique(patch_ids)
    n_patches = unique_patches.size
    print(f"Detected {n_patches} patches in subsample.")

    # Remap patch_ids to 0..n_patches-1 for compact indexing
    patch_id_map = {pid: i for i, pid in enumerate(unique_patches)}
    patch_idx = np.array([patch_id_map[pid] for pid in patch_ids], dtype=np.int32)

    if cores is None:
        cores = min(chains, 4)

    with pm.Model() as model:  # noqa: F841
        # Global class-specific means (log-scale)
        log_mu0 = pm.Normal("log_mu0", mu=0.0, sigma=2.0)
        log_mu1 = pm.Normal("log_mu1", mu=0.0, sigma=2.0)

        # Dispersion parameters (log-scale)
        log_phi0 = pm.Normal("log_phi0", mu=0.0, sigma=1.0)
        log_phi1 = pm.Normal("log_phi1", mu=0.0, sigma=1.0)

        # Non-centred patch-level random effects
        sigma_b = pm.HalfNormal("sigma_b", sigma=1.0)
        b_raw = pm.Normal("b_raw", mu=0.0, sigma=1.0, shape=n_patches)
        b = pm.Deterministic("b", b_raw * sigma_b)

        # Deterministic transforms
        mu0 = pm.Deterministic("mu0", pm.math.exp(log_mu0))  # type: ignore[arg-type]  # noqa: F841
        mu1 = pm.Deterministic("mu1", pm.math.exp(log_mu1))  # type: ignore[arg-type]  # noqa: F841
        alpha0 = pm.Deterministic("alpha0", pm.math.exp(log_phi0))  # type: ignore[arg-type]
        alpha1 = pm.Deterministic("alpha1", pm.math.exp(log_phi1))  # type: ignore[arg-type]

        # Class- and patch-specific log-mean for each observation
        log_mu_obs = pm.math.where(  # type: ignore[arg-type]
            z == 1,
            log_mu1 + b[patch_idx],
            log_mu0 + b[patch_idx],
        )
        mu_obs = pm.math.exp(log_mu_obs) # type: ignore

        # Class-specific dispersion for each observation
        alpha_obs = pm.math.where(z == 1, alpha1, alpha0)  # type: ignore[arg-type]

        pm.NegativeBinomial("y_obs", mu=mu_obs, alpha=alpha_obs, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            chains=chains,
            cores=cores,
            return_inferencedata=True,
        )
    
    # Save InferenceData
    # Save InferenceData
    idata_out = Path(idata_path)
    idata_out.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(idata_out) # type: ignore
    print(f"Saved NB InferenceData to {idata_out}")

    post = idata.posterior # type: ignore

    # Helper: posterior mean as scalar
    def mean_scalar(name: str) -> float:
        return float(post[name].mean().values)

    # Global means
    log_mu0_mean = mean_scalar("log_mu0")
    log_mu1_mean = mean_scalar("log_mu1")
    log_phi0_mean = mean_scalar("log_phi0")
    log_phi1_mean = mean_scalar("log_phi1")
    sigma_b_mean = mean_scalar("sigma_b")
    mu0_mean = mean_scalar("mu0")
    mu1_mean = mean_scalar("mu1")
    phi0_mean = mean_scalar("alpha0")
    phi1_mean = mean_scalar("alpha1")

    # Patch-level means: E[b_i] and corresponding mu0_i, mu1_i
    # b has shape (chain, draw, patch)
    b_mean = post["b"].mean(dim=("chain", "draw")).values  # shape (n_patches,)

    mu0_patch = np.exp(log_mu0_mean + b_mean)
    mu1_patch = np.exp(log_mu1_mean + b_mean)

    # Save per-patch means to CSV if requested
    if patch_csv is not None:
        patch_df = pd.DataFrame(
            {
                "patch_id": unique_patches,
                "mu0_patch": mu0_patch,
                "mu1_patch": mu1_patch,
            }
        )
        patch_df.to_csv(patch_csv, index=False)
        print(f"Saved patch-level means to {patch_csv}")

    # Optionally save global means to CSV
    if global_csv is not None:
        global_df = pd.DataFrame(
            {
                "log_mu0": [log_mu0_mean],
                "log_mu1": [log_mu1_mean],
                "log_phi0": [log_phi0_mean],
                "log_phi1": [log_phi1_mean],
                "mu0": [mu0_mean],
                "mu1": [mu1_mean],
                "phi0": [phi0_mean],
                "phi1": [phi1_mean],
                "sigma_b": [sigma_b_mean],
            }
        )
        global_df.to_csv(global_csv, index=False)
        print(f"Saved global means to {global_csv}")

    return {
        "mu0": mu0_mean,
        "mu1": mu1_mean,
        "phi0": phi0_mean,
        "phi1": phi1_mean,
        "log_mu0": log_mu0_mean,
        "log_mu1": log_mu1_mean,
        "log_phi0": log_phi0_mean,
        "log_phi1": log_phi1_mean,
        "sigma_b": sigma_b_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--window",
        type=float,
        default=10.0,
        help="Window duration in ms",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=9,
        help="Number of patch rows for MVP model grid",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=16,
        help="Number of patch cols for MVP model grid",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        help="CSV with labels (sequence,patch_id,time_bin,z).",
    )
    parser.add_argument(
        "--counts-parquet",
        type=str,
        default=None,
        help=(
            "Parquet file with per-patch per-bin event counts "
            "(sequence,patch_id,time_bin,y). "
            "If provided, skips building counts from .dat."
        ),
    )
    parser.add_argument(
        "--dat",
        type=str,
        default=None,
        help="Path to .dat file. Not used if --counts-parquet is provided.",
    )
    parser.add_argument(
        "--idata-out",
        type=str,
        default="./data/hier_idata.nc",
        help="Where to save the InferenceData NetCDF file.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum number of rows to use for MAP fitting.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=1337,
        help="Random seed for subsampling.",
    )

    args = parser.parse_args()

    if args.counts_parquet is not None:
        counts = pd.read_parquet(args.counts_parquet)
        print(f"Loaded counts from {args.counts_parquet} with {len(counts)} rows.")
    else:
        if args.dat is None:
            msg = (
                "Either --counts-parquet must be provided or "
                "--dat must be set to a valid .dat file path."
            )
            raise RuntimeError(msg)

        src = DatFileSource(
            args.dat,
            width=1280,
            height=720,
            window_length_us=int(args.window * 1000),
        )
        counts = build_counts_table(
            src,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            width=1280,
            height=720,
            sequence_id=0,
        )
        labels_path = Path(args.labels_csv) if args.labels_csv is not None else None
        if labels_path is not None:
            labels_file_name = labels_path.stem
            parquet_file_name = f"counts_{labels_file_name.split('_', 1)[-1]}.parquet"
        else:
            parquet_file_name = "counts_unlabeled.parquet"

        out_path = Path("./data") / parquet_file_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        counts.to_parquet(out_path, index=False)
        print(f"Exported counts table with {len(counts)} rows to {out_path}")

    if args.labels_csv is not None:
        labels = pd.read_csv(args.labels_csv)
        # Expect labels to have: sequence, patch_id, time_bin, z
        counts = counts.merge(
            labels,
            on=["sequence", "patch_id", "time_bin"],
            how="left",
        )
        counts["z"] = counts["z"].fillna(0).astype(int)

    if "z" not in counts.columns:
        raise RuntimeError(
            "Cannot fit model: no 'z' column found. Provide --labels-csv with labels."
        )

    params = fit_hierarchical_nb_model(
        counts,
        max_samples=args.max_samples,
        random_state=args.random_state,
    )

    print("Fitted hierarchical Negative Binomial model parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:.4f}")

    return


if __name__ == "__main__":
    main()
