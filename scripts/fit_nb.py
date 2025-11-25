from pathlib import Path
import pandas as pd
import argparse
import pymc as pm
from evio.source.dat_file import DatFileSource

from bda.counts import build_counts_table

def fit_nb_model(
    df: pd.DataFrame,
    *,
    max_samples: int = 30000,
    random_state: int = 1337,
) -> dict[str, float]:
    """
    Fit the 2-class Negative Binomial MVP model using MAP on a random subset.

    Expects df to have columns:
        y : event counts (int)
        z : labels in {0,1}  (0=background, 1=drone)

    Parameters
    ----------
    max_samples :
        Maximum number of rows to use for MAP. If df is larger,
        a random subset of this size is taken.
    random_state :
        Seed for the random subset.

    Returns
    -------
    dict with MAP estimates:
        mu0, mu1, phi0, phi1, log_mu0, log_mu1, log_phi0, log_phi1
    """
    if "y" not in df.columns or "z" not in df.columns:
        msg = "DataFrame must contain columns 'y' and 'z'."
        raise ValueError(msg)

    # Subsample for fast optimization
    if len(df) > max_samples:
        df_fit = df.sample(n=max_samples, random_state=random_state)
        print(f"Using subset of {max_samples} rows out of {len(df)} for MAP.")
    else:
        df_fit = df
        print(f"Using all {len(df)} rows for MAP.")

    y = df_fit["y"].to_numpy()
    z = df_fit["z"].to_numpy().astype(int)

    with pm.Model() as model:  # noqa: F841
        # Priors on log-scale
        log_mu0 = pm.Normal("log_mu0", mu=0.0, sigma=2.0)
        log_mu1 = pm.Normal("log_mu1", mu=0.0, sigma=2.0)
        log_phi0 = pm.Normal("log_phi0", mu=0.0, sigma=1.0)
        log_phi1 = pm.Normal("log_phi1", mu=0.0, sigma=1.0)

        mu0 = pm.Deterministic("mu0", pm.math.exp(log_mu0))  # type: ignore
        mu1 = pm.Deterministic("mu1", pm.math.exp(log_mu1))  # type: ignore
        alpha0 = pm.Deterministic("alpha0", pm.math.exp(log_phi0))  # type: ignore
        alpha1 = pm.Deterministic("alpha1", pm.math.exp(log_phi1))  # type: ignore

        mu = pm.math.where(z == 1, mu1, mu0)  # type: ignore
        alpha = pm.math.where(z == 1, alpha1, alpha0)  # type: ignore

        pm.NegativeBinomial("y_obs", mu=mu, alpha=alpha, observed=y)

        map_est = pm.find_MAP(method="L-BFGS-B", progressbar=True)

    return {
        "mu0": float(map_est["mu0"]),  # type: ignore
        "mu1": float(map_est["mu1"]),  # type: ignore
        "phi0": float(map_est["alpha0"]),  # type: ignore
        "phi1": float(map_est["alpha1"]),  # type: ignore
        "log_mu0": float(map_est["log_mu0"]),  # type: ignore
        "log_mu1": float(map_est["log_mu1"]),  # type: ignore
        "log_phi0": float(map_est["log_phi0"]),  # type: ignore
        "log_phi1": float(map_est["log_phi1"]),  # type: ignore
    }


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
            "Parquet file with per-patch per-bin event counts (sequence,patch_id,time_bin,y)."
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
        help="Where to save the MAP NB parameter CSV.",
    )

    args = parser.parse_args()

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
        counts.to_parquet(f"./data/{parquet_file_name}", index=False)
        print(
            f"Exported MVP table with {len(counts)} rows to ./data/{parquet_file_name}"
        )
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
    params = fit_nb_model(counts)
    print("Fitted MVP Negative Binomial model parameters (MAP):")
    for k, v in params.items():
        print(f"  {k}: {v:.4f}")

    # Save CSV
    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame([params])
    df_out.to_csv(output_csv_path, index=False)

    print(f"Saved MVP NB parameters to {output_csv_path}")



if __name__ == "__main__":
    main()
