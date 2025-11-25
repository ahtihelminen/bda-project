import argparse  # noqa: INP001
import math

import cv2
import numpy as np
import pandas as pd

from evio.core.pacer import Pacer
from evio.source.dat_file import DatFileSource
from bda.counts import compute_patch_counts_for_window
from bda.rendering import (
    get_window,
    get_frame,
    draw_hud,
    draw_grid,  # noqa: F401
    draw_prob_heatmap,
)
# ---------- NB classifier helpers ----------


def log_nb_pmf_scalar(y: int, mu: float, phi: float) -> float:
    """Negative Binomial pmf in log-space, parameterised by mean mu and alpha=phi."""
    r = float(phi)
    p = r / (r + float(mu))
    return (
        math.lgamma(y + r)
        - math.lgamma(r)
        - math.lgamma(y + 1.0)
        + r * math.log(p)
        + y * math.log(1.0 - p)
    )


def compute_patch_probs_global(
    counts: np.ndarray,
    mu0: float,
    mu1: float,
    phi0: float,
    phi1: float,
    prior_drone: float,
) -> np.ndarray:
    """
    Compute P(drone | y) per patch using NB likelihoods and a Bernoulli prior,
    assuming global (non-hierarchical) parameters shared across patches.
    """
    eps = 1e-9
    pi1 = float(np.clip(prior_drone, eps, 1.0 - eps))
    pi0 = 1.0 - pi1

    counts_flat = counts.astype(np.int64, copy=False).ravel()
    log_p0 = np.array(
        [log_nb_pmf_scalar(int(y), mu0, phi0) for y in counts_flat],
        dtype=np.float64,
    ) + math.log(pi0)
    log_p1 = np.array(
        [log_nb_pmf_scalar(int(y), mu1, phi1) for y in counts_flat],
        dtype=np.float64,
    ) + math.log(pi1)

    m = np.maximum(log_p0, log_p1)
    log_denom = m + np.log(np.exp(log_p0 - m) + np.exp(log_p1 - m))
    p1 = np.exp(log_p1 - log_denom)

    return p1.reshape(counts.shape)


def compute_patch_probs_hierarchical(
    counts: np.ndarray,
    mu0_patch_flat: np.ndarray,
    mu1_patch_flat: np.ndarray,
    phi0: float,
    phi1: float,
    prior_drone: float,
    rows: int,
    cols: int,
    fallback_mu0: float | None = None,
    fallback_mu1: float | None = None,
) -> np.ndarray:
    """
    Compute P(drone | y) per patch using patch-specific means (hierarchical model).

    mu0_patch_flat, mu1_patch_flat:
        1D arrays of length rows*cols with per-patch means.
        If an entry is NaN and fallback_mu* is provided, fall back to global mean.
    """
    probs = np.zeros((rows, cols), dtype=float)

    eps = 1e-9
    pi1 = float(np.clip(prior_drone, eps, 1.0 - eps))
    pi0 = 1.0 - pi1

    for r in range(rows):
        for c in range(cols):
            y = int(counts[r, c])
            pid = r * cols + c

            mu0 = float(mu0_patch_flat[pid])
            mu1 = float(mu1_patch_flat[pid])

            if (np.isnan(mu0) or np.isnan(mu1)) and fallback_mu0 is not None:
                # fall back to global MVP if this patch wasn't in the subsample
                mu0 = float(fallback_mu0)
                mu1 = float(fallback_mu1) # type: ignore

            # If both are still NaN (no fallback), skip colouring (prob stays 0)
            if np.isnan(mu0) or np.isnan(mu1):
                continue

            log_p0 = log_nb_pmf_scalar(y, mu0, phi0) + math.log(pi0)
            log_p1 = log_nb_pmf_scalar(y, mu1, phi1) + math.log(pi1)

            m = max(log_p0, log_p1)
            log_denom = m + math.log(math.exp(log_p0 - m) + math.exp(log_p1 - m))
            probs[r, c] = math.exp(log_p1 - log_denom)

    return probs


# ---------- Main CLI ----------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument(
        "--window", type=float, default=10, help="Window duration in ms"
    )
    parser.add_argument(
        "--speed", type=float, default=1, help="Playback speed (1 is real time)"
    )
    parser.add_argument(
        "--force-speed",
        action="store_true",
        help="Force the playback speed by dropping windows",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=9,
        help="Number of grid rows to display over the frame",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=16,
        help="Number of grid columns to display over the frame",
    )
    parser.add_argument(
        "--prior-drone",
        type=float,
        default=0.05,
        help="Prior probability P(z=1) used in the classifier.",
    )
    # Hierarchical extras
    parser.add_argument(
        "--patch-means-csv",
        type=str,
        required=True,
        help=(
            "CSV with columns: patch_id, mu0_patch, mu1_patch. "
            "If given, hierarchical mode is enabled."
        ),
    )
    parser.add_argument(
        "--global-means-csv",
        type=str,
        required=True,
        help=(
            "Optional CSV with global means/dispersion: mu0,mu1,phi0,phi1, "
            "used as defaults/fallback if CLI values are missing."
        ),
    )
    parser.add_argument(
        "--heatmap-alpha",
        type=float,
        default=0.6,
        help="Alpha blending for the probability heatmap.",
    )

    
    args = parser.parse_args()


    
    global_df = pd.read_csv(args.global_means_csv)
    row = global_df.iloc[0]

    mu0 = float(row["mu0"])
    mu1 = float(row["mu1"])
    phi0 = float(row["phi0"])
    phi1 = float(row["phi1"])

    # Hierarchical patch-level means
    mu0_patch_flat: np.ndarray | None = None
    mu1_patch_flat: np.ndarray | None = None

    patch_df = pd.read_csv(args.patch_means_csv)
    n_patches = args.grid_rows * args.grid_cols
    mu0_patch_flat = np.full(n_patches, np.nan, dtype=float)
    mu1_patch_flat = np.full(n_patches, np.nan, dtype=float)

    for _, row in patch_df.iterrows():
        pid = int(row["patch_id"])
        if 0 <= pid < n_patches:
            mu0_patch_flat[pid] = float(row["mu0_patch"])
            mu1_patch_flat[pid] = float(row["mu1_patch"])


    src = DatFileSource(
        args.dat, width=1280, height=720, window_length_us=int(args.window * 1000)
    )

    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)

    cv2.namedWindow("Evio Player", cv2.WINDOW_NORMAL)
    for batch_range in pacer.pace(src.ranges()):
        window = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )
        x_coords, y_coords, polarities_on = window
        frame = get_frame((x_coords, y_coords, polarities_on))

        # HUD
        draw_hud(frame, pacer, batch_range)

        # Heatmap from NB model, if enabled
        counts = compute_patch_counts_for_window(
            x_coords,
            y_coords,
            rows=args.grid_rows,
            cols=args.grid_cols,
            width=frame.shape[1],
            height=frame.shape[0],
        )

        probs = compute_patch_probs_hierarchical(
            counts,
            mu0_patch_flat=mu0_patch_flat,
            mu1_patch_flat=mu1_patch_flat,
            phi0=float(phi0),
            phi1=float(phi1),
            prior_drone=float(args.prior_drone),
            rows=args.grid_rows,
            cols=args.grid_cols,
            fallback_mu0=mu0,
            fallback_mu1=mu1,
        )
        
        draw_prob_heatmap(
            frame,
            probs,
            rows=args.grid_rows,
            cols=args.grid_cols,
            alpha=float(args.heatmap_alpha),
        )

        # Optional grid overlay on top
        # draw_grid(frame, rows=args.grid_rows, cols=args.grid_cols)

        cv2.imshow("Evio Player", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
