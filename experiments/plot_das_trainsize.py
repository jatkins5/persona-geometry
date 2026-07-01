"""Plot the Exp-3 train-size sweep, merging one or more exp3_train_size_sweep.json runs.

Merging lets us add a new train size (e.g. N=150 run separately) to an earlier sweep without
re-running the smaller sizes -- as long as every run used the SAME test split (same --split-seed,
--n-test, --candidate-pool), so their no-intervention / full-difference / random baselines match.

    plot_das_trainsize.py --results dirA,dirB --out out.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="comma-separated dirs with exp3_train_size_sweep.json")
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    learned, learned_sd, k_ref, random_rec, model_key = {}, {}, None, None, None
    for d in args.results.split(","):
        r = json.loads((Path(d) / "exp3_train_size_sweep.json").read_text())
        k_ref = r["k"] if k_ref is None else k_ref
        assert r["k"] == k_ref, "runs use different k-lists; cannot merge"
        random_rec = r["random_recovered"] if random_rec is None else random_rec
        model_key = r.get("params", {}).get("model_key", "model")
        for N, vals in r["learned"].items():
            learned[int(N)] = vals
            learned_sd[int(N)] = r.get("learned_sd", {}).get(N, [0] * len(vals))

    k = np.array(k_ref)
    sizes = sorted(learned)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    cmap = plt.cm.viridis(np.linspace(0.12, 0.88, len(sizes)))
    for N, c in zip(sizes, cmap):
        y = np.array(learned[N]); sd = np.array(learned_sd[N])
        ax.plot(k, y, "-o", lw=2, color=c, label=f"train N={N}")
        ax.fill_between(k, y - sd, y + sd, color=c, alpha=0.12)
    ax.plot(k, random_rec, "--s", color="grey", lw=1.5, label="random subspace")
    ax.axhline(1.0, ls="--", color="k", lw=0.8, alpha=0.5); ax.axhline(0, ls=":", color="grey", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("subspace dimension k"); ax.set_ylabel("fraction of full-difference effect recovered")
    ax.set_title(f"Exp 3 ({model_key}): panel dimension vs training-persona count")
    ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"saved -> {args.out}  (train sizes: {sizes})")


if __name__ == "__main__":
    main()
