import numpy as np
from evio.source.dat_file import DatFileSource
from bda.rendering import get_window
import pandas as pd


def build_counts_table(
    src: DatFileSource,
    grid_rows: int,
    grid_cols: int,
    *,
    width: int = 1280,
    height: int = 720,
    sequence_id: int = 0,
) -> pd.DataFrame:
    """
    Build the MVP table:
        (sequence, patch_id, time_bin, y)

    y = event count in patch over one window (bin).
    """
    patch_w = width // grid_cols
    patch_h = height // grid_rows
    num_patches = grid_rows * grid_cols

    rows: list[tuple[int, int, int, int]] = []

    for time_bin, batch_range in enumerate(src.ranges()):
        x_coords, y_coords, _ = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )
        if x_coords.size == 0:
            counts = np.zeros(num_patches, dtype=np.int32)
        else:
            # map pixels -> patch indices
            patch_x = np.clip(x_coords // patch_w, 0, grid_cols - 1)
            patch_y = np.clip(y_coords // patch_h, 0, grid_rows - 1)
            patch_id = patch_y * grid_cols + patch_x

            counts = np.bincount(patch_id, minlength=num_patches)

        for pid in range(num_patches):
            y_count = int(counts[pid])
            rows.append((sequence_id, pid, time_bin, y_count))

    counts = pd.DataFrame(rows, columns=["sequence", "patch_id", "time_bin", "y"])
    return counts


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