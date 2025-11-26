#!/usr/bin/env python
"""
Compare two Bayesian models using PSIS-LOO.

Usage example:

uv run scripts/loo_compare.py \
  --nb-idata ./data/nb_idata.nc \
  --hier-idata ./data/hierarchical_idata.nc
"""

from __future__ import annotations

import argparse

import arviz as az


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nb-idata",
        type=str,
        required=True,
        help="Path to NetCDF file with InferenceData for nb model.",
    )
    parser.add_argument(
        "--hier-idata",
        type=str,
        required=True,
        help="Path to NetCDF file with InferenceData for hierarchical model.",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="log",
        choices=["log", "deviance", "negative_log"],
        help="Scale for LOO / elpd (default: log).",
    )
    args = parser.parse_args()

    print("Loading InferenceData...")
    idata_nb = az.from_netcdf(args.nb_idata)
    idata_hier = az.from_netcdf(args.hier_idata)

    print("\nComputing PSIS-LOO for nb model...")
    loo_nb = az.loo(idata_nb, scale=args.scale)
    print(loo_nb)

    print("\nComputing PSIS-LOO for hierarchical model...")
    loo_hier = az.loo(idata_hier, scale=args.scale)
    print(loo_hier)

    print("\nModel comparison (higher elpd_loo is better):")
    comp = az.compare(
        {"nb": idata_nb, "hierarchical": idata_hier},
        scale=args.scale,
        method="BB-pseudo-BMA",
    )
    print(comp)


if __name__ == "__main__":
    main()
