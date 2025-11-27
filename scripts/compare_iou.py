#!/usr/bin/env python
"""
Patch-level IoU comparison between NB and hierarchical NB models.

We treat, for each (sequence, time_bin):

- Ground truth box: set of patches with z = 1
- Predicted box: set of patches with p(z=1|y) >= threshold

Then compute IoU = |GT ∩ Pred| / |GT ∪ Pred| per frame,
and summarize IoU for NB vs hierarchical models.

Inputs:
  * counts parquet: sequence, patch_id, time_bin, y
  * labels csv:     sequence, patch_id, time_bin, z
  * nb globals:     mu0, mu1, phi0, phi1
  * hier globals:   mu0, mu1, phi0, phi1
  * hier patch:     patch_id, mu0_patch, mu1_patch
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bda.neg_binomial import posterior_prob_drone


def iou_sets(gt: set[int], pred: set[int]) -> float:
    """IoU between two sets of patch_ids."""
    if not gt and not pred:
        return 1.0
    union = gt | pred
    if not union:
        return 0.0
    inter = gt & pred
    return len(inter) / len(union)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--counts-parquet",
        type=str,
        required=True,
        help="Parquet with columns: sequence, patch_id, time_bin, y",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        required=True,
        help="CSV with columns: sequence, patch_id, time_bin, z (0/1)",
    )
    parser.add_argument(
        "--nb-globals-csv",
        type=str,
        required=True,
        help="CSV with columns: mu0, mu1, phi0, phi1 (nb model).",
    )
    parser.add_argument(
        "--hier-global-csv",
        type=str,
        required=True,
        help="CSV with columns: mu0, mu1, phi0, phi1, sigma_b (hierarchical model).",
    )
    parser.add_argument(
        "--hier-patch-csv",
        type=str,
        required=True,
        help="CSV with columns: patch_id, mu0_patch, mu1_patch.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold on P(z=1|y) for defining predicted-positive patches.",
    )
    parser.add_argument(
        "--prior-drone",
        type=float,
        default=None,
        help="Prior P(z=1). If not set, use empirical prevalence from labels.",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=50.0,
        help="Window length in milliseconds (for skip-seconds).",
    )
    parser.add_argument(
        "--skip-seconds",
        type=float,
        default=0.0,
        help="Skip first N seconds based on time_bin and window-ms.",
    )

    args = parser.parse_args()

    # Load data
    counts = pd.read_parquet(args.counts_parquet)
    labels = pd.read_csv(args.labels_csv)

    required_counts = {"sequence", "patch_id", "time_bin", "y"}
    required_labels = {"sequence", "patch_id", "time_bin", "z"}

    if not required_counts.issubset(counts.columns):
        raise RuntimeError(
            f"Counts parquet must have columns {sorted(required_counts)}; "
            f"got {sorted(counts.columns)}"
        )
    if not required_labels.issubset(labels.columns):
        raise RuntimeError(
            f"Labels csv must have columns {sorted(required_labels)}; "
            f"got {sorted(labels.columns)}"
        )

    df = counts.merge(
        labels[["sequence", "patch_id", "time_bin", "z"]],
        on=["sequence", "patch_id", "time_bin"],
        how="left",
    )
    df["z"] = df["z"].fillna(0).astype(int)
    df = df[df["z"].isin([0, 1])].copy()
    if df.empty:
        raise RuntimeError("No rows with z in {0,1} after merge.")

    # Optional: skip first N seconds
    if args.skip_seconds > 0:
        bins_to_skip = int((args.skip_seconds * 1000.0) / args.window_ms)
        before = len(df)
        df = df[df["time_bin"] >= bins_to_skip].copy()
        print(
            f"Skipping first {args.skip_seconds:.1f}s "
            f"(>= {bins_to_skip} bins); kept {len(df)} of {before} rows."
        )

    # Restrict to frames (sequence, time_bin) where there's at least one positive patch
    pos_frames = (
        df.loc[df["z"] == 1, ["sequence", "time_bin"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if pos_frames.empty:
        raise RuntimeError("No frames with any z=1 found; cannot compute IoU.")

    df = df.merge(pos_frames, on=["sequence", "time_bin"], how="inner")
    print(
        f"Evaluating IoU on {len(pos_frames)} frames "
        f"with at least one positive patch."
    )

    y_counts = df["y"].to_numpy()
    z_true = df["z"].to_numpy().astype(int)
    patch_ids = df["patch_id"].to_numpy().astype(int)

    # Prior
    if args.prior_drone is not None:
        prior_drone = float(args.prior_drone)
        print(f"Using user-specified prior P(z=1) = {prior_drone:.6f}")
    else:
        # prevalence over these frames
        prior_drone = float(np.mean(z_true))
        print(f"Using empirical prior P(z=1) over IoU frames = {prior_drone:.6f}")

    # NB globals
    nb_globals = pd.read_csv(args.nb_globals_csv).iloc[0]
    nb_mu0 = float(nb_globals["mu0"])
    nb_mu1 = float(nb_globals["mu1"])
    nb_phi0 = float(nb_globals["phi0"])
    nb_phi1 = float(nb_globals["phi1"])

    # Hierarchical globals
    hier_globals = pd.read_csv(args.hier_global_csv).iloc[0]
    hier_mu0_global = float(hier_globals["mu0"])
    hier_mu1_global = float(hier_globals["mu1"])
    hier_phi0 = float(hier_globals["phi0"])
    hier_phi1 = float(hier_globals["phi1"])

    # Hierarchical patch-level means
    hier_patch_df = pd.read_csv(args.hier_patch_csv)
    if not {"patch_id", "mu0_patch", "mu1_patch"}.issubset(hier_patch_df.columns):
        raise RuntimeError(
            f"{args.hier_patch_csv} must contain 'patch_id', 'mu0_patch', 'mu1_patch'."
        )

    patch_to_mu0: dict[int, float] = {}
    patch_to_mu1: dict[int, float] = {}
    for _, row in hier_patch_df.iterrows():
        pid = int(row["patch_id"])
        patch_to_mu0[pid] = float(row["mu0_patch"])
        patch_to_mu1[pid] = float(row["mu1_patch"])

    mu0_hier = np.empty_like(y_counts, dtype=float)
    mu1_hier = np.empty_like(y_counts, dtype=float)

    for i, pid in enumerate(patch_ids):
        m0 = patch_to_mu0.get(pid, float("nan"))
        m1 = patch_to_mu1.get(pid, float("nan"))
        if not np.isfinite(m0):
            m0 = hier_mu0_global
        if not np.isfinite(m1):
            m1 = hier_mu1_global
        mu0_hier[i] = m0
        mu1_hier[i] = m1

    # Global NB probabilities
    mu0_nb_arr = np.full_like(y_counts, nb_mu0, dtype=float)
    mu1_nb_arr = np.full_like(y_counts, nb_mu1, dtype=float)
    p_drone_nb = posterior_prob_drone(
        y=y_counts,
        mu0=mu0_nb_arr,
        mu1=mu1_nb_arr,
        phi0=nb_phi0,
        phi1=nb_phi1,
        prior_drone=prior_drone,
    )

    # Hierarchical probabilities
    p_drone_hier = posterior_prob_drone(
        y=y_counts,
        mu0=mu0_hier,
        mu1=mu1_hier,
        phi0=hier_phi0,
        phi1=hier_phi1,
        prior_drone=prior_drone,
    )

    df = df.copy()
    df["p_nb"] = p_drone_nb
    df["p_hier"] = p_drone_hier

    thr = float(args.threshold)
    print(f"Using threshold {thr:.3f} on P(z=1|y) for predicted patches.\n")

    # Compute IoU per frame (sequence, time_bin)
    ious_nb: list[float] = []
    ious_hier: list[float] = []

    for (seq, tbin), g in df.groupby(["sequence", "time_bin"]):
        gt_patches = set(g.loc[g["z"] == 1, "patch_id"].tolist())
        if not gt_patches:
            # shouldn't happen because we filtered to pos_frames
            continue

        pred_nb = set(g.loc[g["p_nb"] >= thr, "patch_id"].tolist())
        pred_hier = set(g.loc[g["p_hier"] >= thr, "patch_id"].tolist())

        ious_nb.append(iou_sets(gt_patches, pred_nb))
        ious_hier.append(iou_sets(gt_patches, pred_hier))

    ious_nb_arr = np.array(ious_nb, dtype=float)
    ious_hier_arr = np.array(ious_hier, dtype=float)

    def summarize(name: str, arr: np.ndarray) -> None:
        if arr.size == 0:
            print(f"{name}: no IoU values.")
            return
        print(f"=== {name} ===")
        print(f"Frames: {arr.size}")
        print(f"Mean IoU:   {arr.mean():.4f}")
        print(f"Median IoU: {np.median(arr):.4f}")
        for thr_iou in (0.1, 0.3, 0.5):
            frac = float(np.mean(arr >= thr_iou))
            print(f"Frac IoU >= {thr_iou:.1f}: {frac:.4f}")
        print()

    summarize("NB model", ious_nb_arr)
    summarize("Hierarchical model", ious_hier_arr)


if __name__ == "__main__":
    main()
