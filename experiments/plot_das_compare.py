"""Overlay the Exp-3 DAS panel-dimension sweeps from several models for comparison (no GPU).

Reads each run's exp3_results.json and plots, side by side:
  (left)  raw held-out interchange CE vs k, with each model's full-difference reference;
  (right) fraction of each model's own (no-intervention -> best) CE reduction achieved by dim k,
          which exposes how front-loaded vs distributed the causal interface is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True,
                    help="comma-separated result dirs, each containing exp3_results.json")
    ap.add_argument("--out", type=Path, required=True, help="output PNG path")
    return ap.parse_args()


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for d in args.results.split(","):
        data = json.loads((Path(d) / "exp3_results.json").read_text())
        p = data["panel_dimension"]
        runs.append((data["params"]["model_key"], p))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for (label, p), c in zip(runs, colors):
        k = np.array(p["k"]); learned = np.array(p["learned_mean"]); sd = np.array(p["learned_sd"])
        no_int, best = p["no_intervention"], min(p["learned_mean"])
        ax1.errorbar(k, learned, yerr=sd, marker="o", lw=2, capsize=3, color=c, label=label)
        ax1.axhline(p["full_diff_ref"], ls="--", lw=1, color=c, alpha=0.6)
        frac = (no_int - learned) / (no_int - best)
        ax2.plot(k, frac, "-o", lw=2, color=c, label=label)

    ax1.set_xlabel("subspace dimension k"); ax1.set_ylabel("held-out CE of B's responses")
    ax1.set_title("Raw interchange CE vs k  (dashed = each model's full-difference)")
    ax1.legend()
    ax2.axhline(0.9, ls=":", color="grey", lw=1)
    ax2.set_xlabel("subspace dimension k")
    ax2.set_ylabel("fraction of own (no-interv. → best) CE reduction")
    ax2.set_title("How front-loaded is the causal interface?")
    ax2.legend()
    fig.suptitle("Exp 3 DAS panel dimension: Qwen vs Llama")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
