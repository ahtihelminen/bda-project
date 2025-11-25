import argparse  # noqa: INP001
import math

import cv2
import numpy as np

from evio.core.pacer import Pacer
from evio.source.dat_file import DatFileSource
from bda.counts import compute_patch_counts_for_window
from bda.neg_binomial import log_nb_pmf_scalar
from bda.rendering import (
    get_window, 
    get_frame, 
    draw_hud,
    draw_grid,  # noqa: F401
    draw_prob_heatmap
)

def compute_patch_probs(
    counts: np.ndarray,
    mu0: float,
    mu1: float,
    phi0: float, 
    phi1: float,
    prior_drone: float,
) -> np.ndarray:
    """
    Compute P(drone | y) per patch using NB likelihoods and a Bernoulli prior.
    """
    eps = 1e-9
    pi1 = float(np.clip(prior_drone, eps, 1.0 - eps))
    pi0 = 1.0 - pi1

    # vectorized over counts
    counts_flat = counts.astype(np.int64, copy=False).ravel()
    log_p0 = np.array(
        [log_nb_pmf_scalar(int(y), mu0, phi0) for y in counts_flat],
        dtype=np.float64,
    ) + math.log(pi0)
    log_p1 = np.array(
        [log_nb_pmf_scalar(int(y), mu1, phi1) for y in counts_flat],
        dtype=np.float64,
    ) + math.log(pi1)

    # log-sum-exp
    m = np.maximum(log_p0, log_p1)
    log_denom = m + np.log(np.exp(log_p0 - m) + np.exp(log_p1 - m))
    p1 = np.exp(log_p1 - log_denom)

    return p1.reshape(counts.shape)


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

    # NB MAP params for heatmap
    parser.add_argument(
        "--mu0",
        type=float,
        default=None,
        help="NB mean for background (class 0). Enable heatmap if all NB params are set.",
    )
    parser.add_argument(
        "--mu1",
        type=float,
        default=None,
        help="NB mean for drone (class 1).",
    )
    parser.add_argument(
        "--phi0",
        type=float,
        default=None,
        help="NB dispersion for background (class 0).",
    )
    parser.add_argument(
        "--phi1",
        type=float,
        default=None,
        help="NB dispersion for drone (class 1).",
    )
    parser.add_argument(
        "--prior-drone",
        type=float,
        default=0.5,
        help="Prior probability P(z=1) used in the classifier.",
    )
    parser.add_argument(
        "--heatmap-alpha",
        type=float,
        default=0.4,
        help="Alpha blending for the probability heatmap.",
    )

    args = parser.parse_args()

    use_heatmap = (
        args.mu0 is not None
        and args.mu1 is not None
        and args.phi0 is not None
        and args.phi1 is not None
    )

    src = DatFileSource(
        args.dat, width=1280, height=720, window_length_us=args.window * 1000
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
        if use_heatmap:
            counts = compute_patch_counts_for_window(
                x_coords,
                y_coords,
                rows=args.grid_rows,
                cols=args.grid_cols,
                width=frame.shape[1],
                height=frame.shape[0],
            )
            probs = compute_patch_probs(
                counts,
                mu0=float(args.mu0),
                mu1=float(args.mu1),
                phi0=float(args.phi0),
                phi1=float(args.phi1),
                prior_drone=float(args.prior_drone),
            )
            draw_prob_heatmap(
                frame,
                probs,
                rows=args.grid_rows,
                cols=args.grid_cols,
                alpha=float(args.heatmap_alpha),
            )

        # Grid overlay on top
        # draw_grid(frame, rows=args.grid_rows, cols=args.grid_cols)

        cv2.imshow("Evio Player", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
