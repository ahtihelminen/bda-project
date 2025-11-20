import argparse
import numpy as np
import pandas as pd

from evio.source.dat_file import DatFileSource


def load_boxes(txt_path: str) -> pd.DataFrame:
    """
    TXT format (one per line):

        timestamp_s: x0, y0, x1, y1, drone_id

    Example:
        13.533198: 1245.77, 410.62, 1281.63, 462.98, 1
    """
    timestamps = []
    x0s = []
    y0s = []
    x1s = []
    y1s = []
    ids = []

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            ts_part, coords_part = line.split(":")
            timestamp_s = float(ts_part.strip())

            parts = [p.strip() for p in coords_part.split(",")]
            if len(parts) != 5:
                raise ValueError(f"Line has wrong format: {line}")

            x0, y0, x1, y1 = map(float, parts[:4])
            drone_id = int(float(parts[4]))

            timestamps.append(timestamp_s)
            x0s.append(x0)
            y0s.append(y0)
            x1s.append(x1)
            y1s.append(y1)
            ids.append(drone_id)

    df = pd.DataFrame(
        {
            "timestamp_s": timestamps,
            "x0": x0s,
            "y0": y0s,
            "x1": x1s,
            "y1": y1s,
            "drone_id": ids,
        }
    )
    return df


def boxes_to_patch_labels(
    df_boxes: pd.DataFrame,
    src: DatFileSource,
    grid_rows: int,
    grid_cols: int,
    window_ms: float,
) -> pd.DataFrame:
    width, height = src.width, src.height

    patch_w = width / grid_cols
    patch_h = height / grid_rows

    # work in seconds to avoid µs confusion
    window_s = window_ms / 1000.0

    # normalize timestamps so first annotation is at t = 0
    df_boxes = df_boxes.copy()

    # how many bins do we actually have? -> from DatFileSource windows
    total_bins = len(src)

    # map boxes -> bin indices in [0, total_bins-1]
    df_boxes["time_bin"] = np.floor(df_boxes["timestamp_s"] / window_s).astype(int)
    df_boxes = df_boxes[
        (df_boxes["time_bin"] >= 0) & (df_boxes["time_bin"] < total_bins)
    ]

    rows = []
    sequence = 0  # single recording

    for _, row in df_boxes.iterrows():
        time_bin = int(row["time_bin"])

        x0 = int(row["x0"])
        y0 = int(row["y0"])
        x1 = int(row["x1"])
        y1 = int(row["y1"])

        # clamp to frame
        x0 = max(0, min(width - 1, x0))
        x1 = max(0, min(width - 1, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(0, min(height - 1, y1))

        col_start = int(x0 // patch_w)
        col_end = int(x1 // patch_w)
        row_start = int(y0 // patch_h)
        row_end = int(y1 // patch_h)

        col_start = max(0, min(grid_cols - 1, col_start))
        col_end = max(0, min(grid_cols - 1, col_end))
        row_start = max(0, min(grid_rows - 1, row_start))
        row_end = max(0, min(grid_rows - 1, row_end))

        for r in range(row_start, row_end + 1):
            for c in range(col_start, col_end + 1):
                patch_id = r * grid_cols + c
                rows.append((sequence, patch_id, time_bin, 1))

    if rows:
        df_pos = pd.DataFrame(rows, columns=["sequence", "patch_id", "time_bin", "z"])
        df_pos = (
            df_pos.groupby(["sequence", "patch_id", "time_bin"]).max().reset_index()
        )
    else:
        print("WARNING: no boxes mapped into any time_bin; all labels will be 0")
        df_pos = pd.DataFrame(columns=["sequence", "patch_id", "time_bin", "z"])

    print(
        "time_bin range in final label table:",
        int(df_pos["time_bin"].min()),
        "→",
        int(df_pos["time_bin"].max()),
        f"(total_bins={total_bins})",
    )

    return df_pos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-txt", required=True, help="interpolated_coordinates.txt"
    )
    parser.add_argument("--dat", required=True, help="Recording .dat file")
    parser.add_argument("--window-ms", type=float, default=10.0)
    parser.add_argument("--grid-rows", type=int, default=9)
    parser.add_argument("--grid-cols", type=int, default=16)
    parser.add_argument("--out", required=True, help="Output CSV")
    args = parser.parse_args()

    # DatFileSource takes window_length_us, width, height
    src = DatFileSource(
        args.dat,
        window_length_us=int(args.window_ms * 1000),
        width=1280,
        height=720,
    )

    df_boxes = load_boxes(args.labels_txt)
    df = boxes_to_patch_labels(
        df_boxes=df_boxes,
        src=src,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        window_ms=args.window_ms,
    )

    df.to_csv(args.out, index=False)
    print(f"Saved label CSV → {args.out} (rows={len(df)})")


if __name__ == "__main__":
    main()
