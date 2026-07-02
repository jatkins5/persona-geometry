"""Aggregate the Exp-3 persona-space check across random splits: is the learned DAS panel a
DISTINCT causal interface, or just the top-k PCA of the persona vectors?

    plot_das_pspace.py --results "qwen2.5-7b=dir0,dir1,dir2;llama3.1-8b=dirA,dirB,dirC" --out out.png

Left panel: learned DAS-D vs PCA-k of persona space (mean over splits, band = std) per model.
Right panel: the decisive gap (learned - PCA) per model with a std band -- above 0 => DAS beats PCA
(distinct interface); straddling/below 0 => the panel is just the persona-variance subspace.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="'label=dir0,dir1;label2=dirA,dirB'")
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def load_group(dirs):
    k = None
    learned, pca, rand = [], [], []
    for d in dirs:
        r = json.loads((Path(d) / "exp3_persona_space_check.json").read_text())
        k = r["k"] if k is None else k
        assert r["k"] == k, "k-lists differ across splits"
        learned.append(r["learned_rec"]); pca.append(r["pca_rec"]); rand.append(r["random_rec"])
    return np.array(k), np.array(learned), np.array(pca), np.array(rand)


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [(part.split("=")[0], part.split("=")[1].split(",")) for part in args.results.split(";")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for (label, dirs), c in zip(groups, colors):
        k, learned, pca, rand = load_group(dirs)
        n = len(dirs)
        lm, ls = learned.mean(0), learned.std(0)
        pm, ps = pca.mean(0), pca.std(0)
        gap = learned - pca                                   # per-split gap
        gm, gs = gap.mean(0), gap.std(0)
        print(f"{label} (n={n} splits): learned-minus-PCA gap  "
              + "  ".join(f"k{int(kk)}={g:+.2f}±{s:.2f}" for kk, g, s in zip(k, gm, gs)))
        ax1.plot(k, lm, "-o", lw=2, color=c, label=f"{label} learned")
        ax1.fill_between(k, lm - ls, lm + ls, color=c, alpha=0.12)
        ax1.plot(k, pm, "--s", lw=1.8, color=c, alpha=0.8, label=f"{label} PCA-k")
        ax2.plot(k, gm, "-o", lw=2, color=c, label=f"{label} (n={n})")
        ax2.fill_between(k, gm - gs, gm + gs, color=c, alpha=0.15)

    ax1.axhline(1.0, ls="--", color="k", lw=0.8, alpha=0.5)
    ax1.set_xscale("log", base=2); ax1.set_xlabel("subspace dimension k")
    ax1.set_ylabel("fraction of full-difference effect recovered")
    ax1.set_title("Held-out recovery: learned DAS-D (solid) vs PCA-k (dashed)"); ax1.legend()
    ax2.axhline(0, ls=":", color="grey", lw=1)
    ax2.set_xscale("log", base=2); ax2.set_xlabel("subspace dimension k")
    ax2.set_ylabel("learned − PCA-k recovery gap")
    ax2.set_title("Is the panel distinct from persona space?  (>0 = DAS beats PCA)"); ax2.legend()
    fig.suptitle("Exp 3: is the emergent panel a distinct interface or just persona-space PCA? (multi-split)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
