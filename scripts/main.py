import argparse  # noqa: INP001
import math
import time

import cv2
import numpy as np

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource


def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # get indexes corresponding to events within the window
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 28) & 0xF) > 0

    return x_coords, y_coords, pixel_polarity


def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int = 1280,
    height: int = 720,
    *,
    base_color: tuple[int, int, int] = (127, 127, 127),  # gray
    on_color: tuple[int, int, int] = (255, 255, 255),  # white
    off_color: tuple[int, int, int] = (0, 0, 0),  # black
) -> np.ndarray:
    x_coords, y_coords, polarities_on = window
    frame = np.full((height, width, 3), base_color, np.uint8)
    frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color

    return frame


def draw_hud(
    frame: np.ndarray,
    pacer: Pacer,
    batch_range: BatchRange,
    *,
    color: tuple[int, int, int] = (0, 0, 0),  # black by default
) -> None:
    """Overlay timing info: wall time, recording time, and playback speed."""
    if pacer._t_start is None or pacer._e_start is None:
        return

    wall_time_s = time.perf_counter() - pacer._t_start
    rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)

    if pacer.force_speed:
        first_row_str = (
            f"speed={pacer.speed:.2f}x"
            f"  drops/ms={pacer.instantaneous_drop_rate:.2f}"
            f"  avg(drops/ms)={pacer.average_drop_rate:.2f}"
        )
    else:
        first_row_str = (
            f"(target) speed={pacer.speed:.2f}x  force_speed = False, no drops"
        )

    second_row_str = f"wall={wall_time_s:7.3f}s  rec={rec_time_s:7.3f}s"

    # first row
    cv2.putText(
        frame,
        first_row_str,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    # second row
    cv2.putText(
        frame,
        second_row_str,
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_grid(
    frame: np.ndarray,
    rows: int,
    cols: int,
    color: tuple[int, int, int] = (127, 127, 127),
    thickness: int = 1,
) -> None:
    """Draw a regular grid over the frame for manual patch labelling."""
    h, w = frame.shape[:2]

    # Vertical lines
    step_x = w / cols
    for c in range(1, cols):
        x = int(round(c * step_x))
        cv2.line(frame, (x, 0), (x, h), color, thickness, cv2.LINE_AA)

    # Horizontal lines
    step_y = h / rows
    for r in range(1, rows):
        y = int(round(r * step_y))
        cv2.line(frame, (0, y), (w, y), color, thickness, cv2.LINE_AA)


# ---------- NB classifier helpers ----------


def _log_nb_pmf(y: np.ndarray, mu: float, phi: float) -> np.ndarray:
    """
    Negative binomial log pmf with mean mu and dispersion phi:

        Var(Y) = mu + mu^2 / phi

    p(y | mu, phi) = NB(y; r=phi, p = phi / (phi + mu))
    """
    # Ensure float array
    y = y.astype(np.float64, copy=False)

    r = phi
    p = phi / (phi + mu)  # success prob
    # log C(y + r - 1, y) = lgamma(y+r) - lgamma(r) - lgamma(y+1)
    return (
        math.lgamma(r + 0.0)
        - math.lgamma(r)
        + 0.0 * y  # broadcast shape hack if needed
    ) + (
        np.vectorize(math.lgamma)(y + r)
        - math.lgamma(r)
        - np.vectorize(math.lgamma)(y + 1.0)
        + r * math.log(p)
        + y * math.log(1.0 - p)
    )


def _log_nb_pmf_scalar(y: int, mu: float, phi: float) -> float:
    r = phi
    p = phi / (phi + mu)
    return (
        math.lgamma(y + r)
        - math.lgamma(r)
        - math.lgamma(y + 1.0)
        + r * math.log(p)
        + y * math.log(1.0 - p)
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
        [_log_nb_pmf_scalar(int(y), mu0, phi0) for y in counts_flat],
        dtype=np.float64,
    ) + math.log(pi0)
    log_p1 = np.array(
        [_log_nb_pmf_scalar(int(y), mu1, phi1) for y in counts_flat],
        dtype=np.float64,
    ) + math.log(pi1)

    # log-sum-exp
    m = np.maximum(log_p0, log_p1)
    log_denom = m + np.log(np.exp(log_p0 - m) + np.exp(log_p1 - m))
    p1 = np.exp(log_p1 - log_denom)

    return p1.reshape(counts.shape)


def draw_prob_heatmap(
    frame: np.ndarray,
    probs: np.ndarray,
    rows: int,
    cols: int,
    alpha: float = 0.5,
    p_vis: float = 0.2,  # minimum probability to start visualizing
) -> None:
    h, w = frame.shape[:2]
    patch_w = w / cols
    patch_h = h / rows

    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return

    for r in range(rows):
        for c in range(cols):
            p = float(np.clip(probs[r, c], 0.0, 1.0))

            # below threshold -> do not color this patch
            if p < p_vis:
                continue

            # normalize into [0,1] range for gradient
            p_scaled = (p - p_vis) / (1.0 - p_vis)
            p_scaled = float(np.clip(p_scaled, 0.0, 1.0))

            # BLUE → RED gradient in BGR:
            #   blue = 255*(1 - p_scaled)
            #   red  = 255*p_scaled
            blue = int(round(255 * (1.0 - p_scaled)))
            red = int(round(255 * p_scaled))
            green = 0
            color = (blue, green, red)  # BGR

            x0 = int(round(c * patch_w))
            x1 = int(round((c + 1) * patch_w))
            y0 = int(round(r * patch_h))
            y1 = int(round((r + 1) * patch_h))

            roi = frame[y0:y1, x0:x1]
            if roi.size == 0:
                continue

            overlay = np.full_like(roi, color, dtype=np.uint8)
            cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def compute_patch_counts_for_window(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    rows: int,
    cols: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Bin events of the current window into a rows x cols grid."""
    counts = np.zeros((rows, cols), dtype=np.int32)
    if x_coords.size == 0:
        return counts

    # map pixels -> patch indices
    col_idx = (x_coords.astype(np.int64) * cols) // width
    row_idx = (y_coords.astype(np.int64) * rows) // height

    col_idx = np.clip(col_idx, 0, cols - 1)
    row_idx = np.clip(row_idx, 0, rows - 1)

    np.add.at(counts, (row_idx, col_idx), 1)
    return counts


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
        default=0.8,
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
