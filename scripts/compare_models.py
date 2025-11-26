#!/usr/bin/env python
"""
Compare nb and hierarchical NB models on the same labelled data.

- Loads:
    * counts parquet: sequence, patch_id, time_bin, y
    * labels csv:     sequence, patch_id, time_bin, z
    * nb globals:    mu0, mu1, phi0, phi1
    * Hier globals:   mu0, mu1, phi0, phi1, sigma_b (optional)
    * Hier patch:     patch_id, mu0_patch, mu1_patch

- Computes:
    * P(z=1 | y) under nb
    * P(z=1 | y) under hierarchical
    * ROC AUC, accuracy, precision, recall for both
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bda.neg_binomial import posterior_prob_drone

# ---------- Metrics ----------

def binary_classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Return accuracy, precision, recall for a given threshold."""
    y_true = y_true.astype(int)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute ROC AUC for binary classification from scratch (no sklearn).

    Assumes y_true in {0,1}, y_score are continuous scores (higher = more positive).
    """
    y_true = y_true.astype(int)
    y_score = y_score.astype(float)

    # Sort by decreasing score
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    # Count positives/negatives
    P = np.sum(y_true_sorted == 1)
    N = np.sum(y_true_sorted == 0)
    if P == 0 or N == 0:
        return float("nan")

    # TPR/FPR stepwise
    tp = 0.0
    fp = 0.0
    prev_score = None
    tpr_list: list[float] = [0.0]
    fpr_list: list[float] = [0.0]

    for idx in range(len(y_true_sorted)):
        if prev_score is None or y_score[order[idx]] != prev_score:
            tpr_list.append(tp / P)
            fpr_list.append(fp / N)
            prev_score = y_score[order[idx]]

        if y_true_sorted[idx] == 1:
            tp += 1.0
        else:
            fp += 1.0

    tpr_list.append(tp / P)
    fpr_list.append(fp / N)

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)

    # Trapezoidal area
    # Sort by FPR (should already be non-decreasing)
    order2 = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[order2]
    tpr_arr = tpr_arr[order2]

    auc = float(np.trapezoid(tpr_arr, fpr_arr))
    return auc


# ---------- Main comparison ----------


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
        "--max-eval-rows",
        type=int,
        default=50000,
        help="Max number of rows to use for evaluation (subsample if larger).",
    )
    parser.add_argument(
        "--prior-drone",
        type=float,
        default=None,
        help="Prior P(z=1). If not set, use empirical prevalence from labels.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on P(z=1|y) for accuracy/precision/recall.",
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
        labels[['sequence','patch_id','time_bin','z']],
        on=["sequence", "patch_id", "time_bin"],
        how="left",
    )
    df['z'] = df['z'].fillna(0).astype(int)

    # Ensure binary labels
    df = df[df["z"].isin([0, 1])].copy()
    if df.empty:
        raise RuntimeError("No rows with z in {0,1} after merge.")

    # Subsample for evaluation
    if len(df) > args.max_eval_rows:
        df_eval = df.sample(n=args.max_eval_rows, random_state=1337)
        print(f"Using subset of {args.max_eval_rows} rows out of {len(df)} for eval.")
    else:
        df_eval = df
        print(f"Using all {len(df)} rows for eval ({len(df_eval)}).")

    y_counts = df_eval["y"].to_numpy()
    z_true = df_eval["z"].to_numpy().astype(int)
    patch_ids = df_eval["patch_id"].to_numpy().astype(int)

    # Prior
    if args.prior_drone is not None:
        prior_drone = float(args.prior_drone)
        print(f"Using user-specified prior P(z=1) = {prior_drone:.6f}")
    else:
        prior_drone = float(np.mean(z_true))
        print(f"Using empirical prior P(z=1) = {prior_drone:.6f}")

    # nb globals
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

    # Build arrays of per-row mu0/mu1 for hierarchical model
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
        # fallback to global if missing or nan
        if not np.isfinite(m0):
            m0 = hier_mu0_global
        if not np.isfinite(m1):
            m1 = hier_mu1_global
        mu0_hier[i] = m0
        mu1_hier[i] = m1

    # ---------- Compute probabilities ----------

    # nb (global)
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

    # Hierarchical (patch-specific)
    p_drone_hier = posterior_prob_drone(
        y=y_counts,
        mu0=mu0_hier,
        mu1=mu1_hier,
        phi0=hier_phi0,
        phi1=hier_phi1,
        prior_drone=prior_drone,
    )

    # ---------- Metrics ----------

    # ROC AUC
    auc_nb = roc_auc_binary(z_true, p_drone_nb)
    auc_hier = roc_auc_binary(z_true, p_drone_hier)

    # Classification metrics at threshold
    metrics_nb = binary_classification_metrics(z_true, p_drone_nb, threshold=args.threshold)
    metrics_hier = binary_classification_metrics(z_true, p_drone_hier, threshold=args.threshold)

    print("\n=== NB model ===")
    print(f"ROC AUC: {auc_nb:.4f}")
    print(
        f"Accuracy: {metrics_nb['accuracy']:.4f}, "
        f"Precision: {metrics_nb['precision']:.4f}, "
        f"Recall: {metrics_nb['recall']:.4f}"
    )
    print(
        f"TP={metrics_nb['tp']:.0f}, TN={metrics_nb['tn']:.0f}, "
        f"FP={metrics_nb['fp']:.0f}, FN={metrics_nb['fn']:.0f}"
    )

    print("\n=== Hierarchical NB model ===")
    print(f"ROC AUC: {auc_hier:.4f}")
    print(
        f"Accuracy: {metrics_hier['accuracy']:.4f}, "
        f"Precision: {metrics_hier['precision']:.4f}, "
        f"Recall: {metrics_hier['recall']:.4f}"
    )
    print(
        f"TP={metrics_hier['tp']:.0f}, TN={metrics_hier['tn']:.0f}, "
        f"FP={metrics_hier['fp']:.0f}, FN={metrics_hier['fn']:.0f}"
    )


if __name__ == "__main__":
    main()
