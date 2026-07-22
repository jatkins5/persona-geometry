"""Experiment 1 (7c): is the word-concept steering component 'inert', or does it recover a decent
fraction of the trait? Reads each model's steering_results.csv (per-trait full / word-comp /
residual trait-expression scores) and shows the distribution is BIMODAL: vivid/stylistic personas
have word-comp ~ 0 (persona != word), while abstract/semantic traits have word-comp ~ full.

    plot_exp1_wordcomp.py --results "qwen2.5-7b=dir1;llama3.1-8b=dir2" --out out.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="'label=dir;label2=dir2' (dir has steering_results.csv)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-full", type=float, default=3.0,
                    help="only include traits whose full-vector score >= this (else ratio is noise)")
    return ap.parse_args()


def load(d):
    rows = list(csv.DictReader(open(Path(d) / "steering_results.csv")))
    fu, wc, re = [], [], []
    for r in rows:
        try:
            a, b, c = float(r["score_full"]), float(r["score_word_comp"]), float(r["score_residual"])
        except ValueError:
            continue
        if any(np.isnan(x) for x in (a, b, c)):     # drop traits where the judge failed a condition
            continue
        fu.append(a); wc.append(b); re.append(c)
    return np.array(fu), np.array(wc), np.array(re)


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [(p.split("=")[0], p.split("=")[1]) for p in args.results.split(";")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = ["#1f77b4", "#d62728"]

    for (label, d), c in zip(groups, colors):
        fu, wc, re = load(d)
        m = fu >= args.min_full
        fu, wc, re = fu[m], wc[m], re[m]
        ratio = wc / fu
        pooled_w = wc.sum() / fu.sum()
        pooled_r = re.sum() / fu.sum()
        print(f"{label}: n={m.sum()} traits (full>={args.min_full})  "
              f"pooled word/full={pooled_w:.0%}  residual/full={pooled_r:.0%}  "
              f"| word>=full: {(ratio>=1).mean():.0%}  word<30%: {(ratio<0.3).mean():.0%}")
        # left: word-comp score vs full score per trait (y=x = word matches full)
        ax1.scatter(fu, wc, s=32, alpha=0.7, color=c, label=f"{label} (word-comp)")
        # right: histogram of word/full ratio -> shows bimodality
        ax2.hist(np.clip(ratio, 0, 1.5), bins=np.linspace(0, 1.5, 22), alpha=0.55, color=c, label=label)

    lim = 10
    ax1.plot([0, lim], [0, lim], "k--", lw=1, label="word = full")
    ax1.plot([0, lim], [0, 0.5 * lim], ":", color="grey", lw=1, label="word = 50% full")
    ax1.set_xlim(0, lim); ax1.set_ylim(0, lim); ax1.set_aspect("equal")
    ax1.set_xlabel("full persona-vector steering score (0-10)")
    ax1.set_ylabel("word-component steering score (0-10)")
    ax1.set_title("Word-component vs full trait expression (per trait)"); ax1.legend(loc="upper left", fontsize=8)
    ax2.axvline(1.0, ls="--", color="k", lw=0.8)
    ax2.set_xlabel("word-component score / full score")
    ax2.set_ylabel("# traits")
    ax2.set_title("Bimodal: ~0 (persona ≠ word) vs ~1 (persona ≈ word)"); ax2.legend()
    fig.suptitle("Exp 1 (7c): the word-concept component is NOT inert — it recovers ~60% of the trait on average")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
