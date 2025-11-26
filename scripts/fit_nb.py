from pathlib import Path
import argparse

import pandas as pd
import pymc as pm
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
) -> dict[str, float]:
    """
    Fit the 2-class Negative Binomial MVP model using MCMC on a random subset.

    Expects df to have columns:
        y : event counts (int)
        z : labels in {0,1}  (0=background, 1=drone)

    Returns
    -------
    dict with posterior means:
        mu0, mu1, phi0, phi1, log_mu0, log_mu1, log_phi0, log_phi1
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

    with pm.Model() as model:  # noqa: F841
        # Priors on log-scale
        log_mu0 = pm.Normal("log_mu0", mu=0.0, sigma=2.0)
        log_mu1 = pm.Normal("log_mu1", mu=0.0, sigma=2.0)
        log_phi0 = pm.Normal("log_phi0", mu=0.0, sigma=1.0)
        log_phi1 = pm.Normal("log_phi1", mu=0.0, sigma=1.0)

        mu0 = pm.Deterministic("mu0", pm.math.exp(log_mu0))  # type: ignore[arg-type]
        mu1 = pm.Deterministic("mu1", pm.math.exp(log_mu1))  # type: ignore[arg-type]
        alpha0 = pm.Deterministic("alpha0", pm.math.exp(log_phi0))  # type: ignore[arg-type]
        alpha1 = pm.Deterministic("alpha1", pm.math.exp(log_phi1))  # type: ignore[arg-type]

        mu = pm.math.where(z == 1, mu1, mu0)  # type: ignore[arg-type]
        alpha = pm.math.where(z == 1, alpha1, alpha0)  # type: ignore[arg-type]

        pm.NegativeBinomial("y_obs", mu=mu, alpha=alpha, observed=y)

        print(
            f"Running MCMC for MVP model (draws={draws}, tune={tune}, "
            f"chains={chains}, cores={cores})..."
        )
        idata = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=0.9,
            chains=chains,
            cores=cores,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )

    # Save InferenceData
    idata_out = Path(idata_path)
    idata_out.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(idata_out) # type: ignore
    print(f"Saved MVP InferenceData to {idata_out}")

    post = idata.posterior # type: ignore

    def mean_scalar(name: str) -> float:
        return float(post[name].mean().values)

    params = {
        "mu0": mean_scalar("mu0"),
        "mu1": mean_scalar("mu1"),
        "phi0": mean_scalar("alpha0"),
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

    # Fit MVP model via MCMC
    params = fit_nb_model(counts, idata_path=args.idata_out)

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
