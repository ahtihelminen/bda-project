from pathlib import Path
import argparse

import arviz as az
import pandas as pd
from cmdstanpy import CmdStanModel

from evio.source.dat_file import DatFileSource
from bda.counts import build_counts_table


def fit_nb_model(
    df: pd.DataFrame,
    *,
    max_samples: int = 30000,
    random_state: int = 1337,
    idata_path: str = "./data/nb_idata.nc",
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int | None = None,
    stan_file: str = "./stan/nb_2class.stan",
) -> dict[str, float]:
    """
    Fit the 2-class Negative Binomial MVP model using Stan on a random subset.

    Expects df to have columns:
        y : event counts (int)
        z : labels in {0,1}  (0=background, 1=drone)

    Returns
    -------
    dict with posterior means:
        mu0, mu1, phi0, phi1, log_mu0, log_mu1, log_phi0, log_phi1

    Here phi0, phi1 correspond to the overdispersion parameters alpha0, alpha1
    from the Stan model (matching the PyMC version).
    """
    if "y" not in df.columns or "z" not in df.columns:
        msg = "DataFrame must contain columns 'y' and 'z'."
        raise ValueError(msg)

    # Subsample for faster MCMC
    if len(df) > max_samples:
        df_fit = df.sample(n=max_samples, random_state=random_state)
        print(f"Using subset of {max_samples} rows out of {len(df)} for MCMC.")
    else:
        df_fit = df
        print(f"Using all {len(df)} rows for MCMC ({len(df_fit)} rows).")

    y = df_fit["y"].to_numpy()
    z = df_fit["z"].to_numpy().astype(int)

    if cores is None:
        cores = min(chains, 4)

    # Stan data
    stan_data = {
        "N": len(y),
        "y": y,
        "z": z,
    }

    print(f"Compiling Stan model from {stan_file}...")
    model = CmdStanModel(stan_file=stan_file)

    print(
        f"Running Stan for MVP model (draws={draws}, tune={tune}, "
        f"chains={chains}, parallel_chains={cores})..."
    )
    fit = model.sample(
        data=stan_data,
        chains=chains,
        parallel_chains=cores,
        iter_warmup=tune,
        iter_sampling=draws,
        seed=random_state,
    )

    # Convert to ArviZ InferenceData, including log-likelihood for LOO
    idata = az.from_cmdstanpy(
        posterior=fit,
        log_likelihood="log_lik",  # name from generated quantities in Stan
    )

    # Save InferenceData
    idata_out = Path(idata_path)
    idata_out.parent.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(idata, idata_out)
    print(f"Saved MVP InferenceData to {idata_out}")

    post = idata.posterior # type: ignore

    def mean_scalar(name: str) -> float:
        return float(post[name].mean().values)

    # Stan model defines transformed parameters:
    #   mu0, mu1, alpha0, alpha1, log_mu0, log_mu1, log_phi0, log_phi1, ...
    params = {
        "mu0": mean_scalar("mu0"),
        "mu1": mean_scalar("mu1"),
        "phi0": mean_scalar("alpha0"),  # keep same naming as PyMC version
        "phi1": mean_scalar("alpha1"),
        "log_mu0": mean_scalar("log_mu0"),
        "log_mu1": mean_scalar("log_mu1"),
        "log_phi0": mean_scalar("log_phi0"),
        "log_phi1": mean_scalar("log_phi1"),
    }

    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--window", type=float, default=10, help="Window duration in ms"
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
        help=("CSV with labels (sequence,patch_id,time_bin,z). "),
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
        "--output-csv",
        type=str,
        default="./data/nb_global_means.csv",
        help="Where to save posterior mean NB parameters as CSV.",
    )
    parser.add_argument(
        "--idata-out",
        type=str,
        default="./data/nb_idata.nc",
        help="Where to save the InferenceData NetCDF file.",
    )
    parser.add_argument(
        "--stan-file",
        type=str,
        default="./models/nb.stan",
        help="Path to Stan model file.",
    )

    args = parser.parse_args()

    # Load or build counts
    if args.counts_parquet is not None:
        counts = pd.read_parquet(args.counts_parquet)
        print(f"Loaded counts from {args.counts_parquet} with {len(counts)} rows.")
    else:
        src = DatFileSource(
            args.dat, width=1280, height=720, window_length_us=args.window * 1000
        )
        counts = build_counts_table(
            src,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            width=1280,
            height=720,
            sequence_id=0,
        )
        labels_path = Path(args.labels_csv)
        labels_file_name = labels_path.stem
        parquet_file_name = f"counts_{labels_file_name.split('_', 1)[-1]}.parquet"
        parquet_out = Path("./data") / parquet_file_name
        parquet_out.parent.mkdir(parents=True, exist_ok=True)
        counts.to_parquet(parquet_out, index=False)
        print(f"Exported MVP table with {len(counts)} rows to {parquet_out}")

    # Merge labels
    if args.labels_csv is not None:
        labels = pd.read_csv(args.labels_csv)
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

    # Fit MVP model via Stan
    params = fit_nb_model(
        counts,
        idata_path=args.idata_out,
        stan_file=args.stan_file,
    )

    print("Fitted MVP Negative Binomial model parameters (posterior means):")
    for k, v in params.items():
        print(f"  {k}: {v:.4f}")

    # Save CSV with posterior means
    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame([params])
    df_out.to_csv(output_csv_path, index=False)

    print(f"Saved MVP NB posterior means to {output_csv_path}")


if __name__ == "__main__":
    main()
