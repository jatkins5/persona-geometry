"""Overlay Exp-3 DAS panel-dimension sweeps for comparison (no GPU), averaging over splits.

--results groups runs per model and averages across them (e.g. several random train/test splits):
    "qwen2.5-7b=dirA,dirB,dirC;llama3.1-8b=dirX,dirY,dirZ"
A single dir per model also works ("qwen=dir1;llama=dir2").

Plots, side by side:
  (left)  mean raw held-out interchange CE vs k (band = spread across splits), with each model's
          no-intervention and full-difference reference lines (averaged);
  (right) fraction of the achievable full-difference effect recovered on UNSEEN personas vs k --
          the generalization test (1.0 = matches the full A->B difference; 0 = no better than not
          intervening). Band = std across splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True,
                    help="'label=dir1,dir2;label2=dirA,dirB' -- dirs each hold an exp3_results.json")
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def load_group(dirs):
    """Return (k, learned[ndirs,nk], recovered[ndirs,nk], no_int_mean, ceil_mean)."""
    k = None
    learned, recovered, no_ints, ceils = [], [], [], []
    for d in dirs:
        p = json.loads((Path(d) / "exp3_results.json").read_text())["panel_dimension"]
        k = np.array(p["k"]); lm = np.array(p["learned_mean"])
        no_int, ceil = p["no_intervention"], p["full_diff_ref"]
        learned.append(lm)
        recovered.append((no_int - lm) / (no_int - ceil))
        no_ints.append(no_int); ceils.append(ceil)
    return k, np.array(learned), np.array(recovered), float(np.mean(no_ints)), float(np.mean(ceils))


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = []
    for part in args.results.split(";"):
        label, dirs = part.split("=")
        groups.append((label, dirs.split(",")))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for (label, dirs), c in zip(groups, colors):
        k, learned, recovered, no_int, ceil = load_group(dirs)
        n = len(dirs)
        lm, ls = learned.mean(0), learned.std(0)
        rm, rs = recovered.mean(0), recovered.std(0)
        tag = f"{label} (n={n})" if n > 1 else label
        ax1.plot(k, lm, "-o", lw=2, color=c, label=tag)
        ax1.fill_between(k, lm - ls, lm + ls, color=c, alpha=0.15)
        ax1.axhline(no_int, ls=":", lw=1, color=c, alpha=0.5)
        ax1.axhline(ceil, ls="--", lw=1, color=c, alpha=0.6)
        ax2.plot(k, rm, "-o", lw=2, color=c, label=tag)
        ax2.fill_between(k, rm - rs, rm + rs, color=c, alpha=0.15)

    ax1.set_xlabel("subspace dimension k"); ax1.set_ylabel("held-out CE of B's responses")
    ax1.set_title("Raw interchange CE vs k  (dotted = no-interv., dashed = full-difference)")
    ax1.legend()
    ax2.axhline(1.0, ls="--", color="grey", lw=1); ax2.axhline(0, ls=":", color="grey", lw=1)
    ax2.set_xlabel("subspace dimension k")
    ax2.set_ylabel("fraction of full-difference effect recovered")
    ax2.set_title("Does the learned subspace generalize to UNSEEN personas?  (band = across splits)")
    ax2.legend()
    fig.suptitle("Exp 3 DAS panel dimension: Qwen vs Llama (disjoint train/test, multi-split)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
