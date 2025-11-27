import numpy as np
import time
from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange
import cv2

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
def draw_prob_heatmap(
    frame: np.ndarray,
    probs: np.ndarray,
    rows: int,
    cols: int,
    alpha: float = 1,   # overall scale, 1.0 → use full 1.0→0.2 range
    p_vis: float = 0.85,   # minimum probability to start visualizing
) -> None:
    h, w = frame.shape[:2]
    patch_w = w / cols
    patch_h = h / rows

    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return

    # base opacity range: 1.0 (fully opaque) down to 0.2
    base_max = 1
    base_min = 0.4

    for r in range(rows):
        for c in range(cols):
            p = float(np.clip(probs[r, c], 0.0, 1.0))
            # print(f"Patch ({r}, {c}): p={p:.3f}")
            # below threshold -> do not color this patch
            if p < p_vis:
                continue

            # normalize into [0,1] for scaling
            p_scaled = (p - p_vis) / (1.0 - p_vis)
            p_scaled = float(np.clip(p_scaled, 0.0, 1.0))

            # opacity mapping:
            #   p_scaled = 0  -> opacity = 1.0 (fully opaque)
            #   p_scaled = 1  -> opacity = 0.2 (mostly transparent)
            # "increasingly" decreasing: use quadratic to drop faster near high p
            raw_opacity = base_max - (p_scaled ** 2) * (base_max - base_min)
            patch_alpha = alpha * raw_opacity  # global scale

            # fixed color, no gradient
            color = (0, 0, 255)  # red in BGR

            x0 = int(round(c * patch_w))
            x1 = int(round((c + 1) * patch_w))
            y0 = int(round(r * patch_h))
            y1 = int(round((r + 1) * patch_h))

            roi = frame[y0:y1, x0:x1]
            if roi.size == 0:
                continue

            overlay = np.full_like(roi, color, dtype=np.uint8)
            cv2.addWeighted(overlay, 1-patch_alpha, roi, patch_alpha, 0, dst=roi)
