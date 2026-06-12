"""Experiment 1 (word-vs-persona decomposition) at scale + a frequency analysis.

Two questions, one cheap forward-pass-only run (no generation, no judge):

  1. REPLICATION. The notebook's 7b decomposed only the 6 strongest persona vectors into
     a lexical part (the component along the trait WORD's meaning direction) and a
     behavioural residual, finding personas are ~2-9% lexical. Here we run that same
     decomposition over the full analysed pool (default 220 adjectives).

  2. NEW. Is a persona's lexical fraction related to how common the trait word is in
     ordinary English? We pair each trait's lexical fraction with its Zipf word frequency
     and correlate, controlling for the obvious confound (rare words split into more
     subword tokens, which changes how the word-concept vector is pooled).

Method, per trait word:
    persona vector  pv  = cached mean response activation, centered on the 220-pool mean
    word concept    wu  = unit activation at the trait-word token in neutral carriers
                          (also centered into the same frame)
    lexical fraction    = |cos(pv, wu)|          (component of the persona along the word)
    residual fraction   = sqrt(1 - lexical^2)    (the orthogonal behavioural part)

All per-trait numbers go to logs.csv; run-level settings to global_params.csv; a few
human-readable highlights + the correlation stats to summary.txt; and a scatter to PNG.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t
from scipy import stats
from tqdm import tqdm

import persona_lib as pl


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=pl.DEFAULT_MODEL, choices=list(pl.MODELS),
                    help="which registry model to decompose (default: qwen2.5-7b)")
    ap.add_argument("--n-traits", type=int, default=pl.N_TRAITS,
                    help="size of the analysed adjective pool (default: 220, the notebook's pool)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output dir (default: results/exp1_decomposition[<model suffix>])")
    return ap.parse_args()


def decompose_pool(model, tokenizer, names, Vc, centroid, layer) -> pd.DataFrame:
    """Run the lexical/residual decomposition for every trait in the pool."""
    import wordfreq  # imported here so --help works without the package installed

    rows = []
    for idx, trait in enumerate(tqdm(names, desc="decompose")):
        # behavioural side: the persona vector, already centered on the pool mean
        pv = Vc[idx]
        pv_norm = float(pv.norm())

        # lexical side: the trait word's meaning direction, centered into the same frame
        word_raw, n_word_tokens = pl.word_concept_vector(model, tokenizer, trait, layer)
        wv = word_raw - centroid
        wu = wv / wv.norm()

        # decomposition: how much of the persona points along the word's meaning
        signed_cos = float((pv @ wu) / pv_norm)
        lexical_fraction = abs(signed_cos)
        residual_fraction = float(np.sqrt(max(0.0, 1.0 - lexical_fraction ** 2)))

        # how common the trait word is in ordinary English (Zipf scale: ~1 rare ... ~7 common;
        # 0.0 means out-of-vocabulary for the frequency list)
        zipf = wordfreq.zipf_frequency(trait, "en")

        rows.append({
            "trait": trait,
            "zipf_frequency": zipf,
            "is_oov": zipf == 0.0,
            "n_word_tokens": n_word_tokens,        # confound: subword pieces the word splits into
            "word_length": len(trait),
            "persona_norm": pv_norm,
            "word_concept_norm": float(wv.norm()),
            "signed_cos": signed_cos,
            "lexical_fraction": lexical_fraction,
            "residual_fraction": residual_fraction,
        })
    return pd.DataFrame(rows)


def correlation_report(df: pd.DataFrame) -> dict:
    """Correlate lexical fraction with word frequency, raw and controlling for token count.

    Spearman (rank) is the headline because the relationship may be monotonic-but-nonlinear.
    The partial correlation removes n_word_tokens from BOTH variables (rarer words tokenise
    into more pieces, which could drive any apparent frequency effect on its own).
    """
    in_vocab = df[~df["is_oov"]].copy()       # OOV words have no real frequency to correlate
    x, y = in_vocab["zipf_frequency"].to_numpy(), in_vocab["lexical_fraction"].to_numpy()

    spearman_r, spearman_p = stats.spearmanr(x, y)
    pearson_r, pearson_p = stats.pearsonr(x, y)

    # partial Spearman of (freq, lexical) controlling for token count: rank-residualise both
    def rank_residuals(target: np.ndarray, control: np.ndarray) -> np.ndarray:
        tr, cr = stats.rankdata(target), stats.rankdata(control)
        slope = np.polyfit(cr, tr, 1)[0]
        return tr - slope * cr

    ctrl = in_vocab["n_word_tokens"].to_numpy().astype(float)
    partial_r, partial_p = stats.pearsonr(rank_residuals(x, ctrl), rank_residuals(y, ctrl))

    # sanity: does frequency itself track token count? (if so the control matters)
    freq_vs_tokens_r, _ = stats.spearmanr(x, ctrl)

    return {
        "n_in_vocab": int(len(in_vocab)),
        "n_oov": int(df["is_oov"].sum()),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "partial_spearman_r_control_tokens": float(partial_r), "partial_p": float(partial_p),
        "freq_vs_tokencount_spearman_r": float(freq_vs_tokens_r),
    }


def write_outputs(df: pd.DataFrame, report: dict, params: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df_sorted = df.sort_values("lexical_fraction", ascending=False).reset_index(drop=True)
    df_sorted.to_csv(out_dir / "logs.csv", index=False)
    pd.DataFrame([params]).to_csv(out_dir / "global_params.csv", index=False)
    (out_dir / "correlation_report.json").write_text(json.dumps(report, indent=2))

    # human-readable summary -- the artifact to actually READ before trusting the numbers
    lex = df["lexical_fraction"]
    lines = [
        "EXPERIMENT 1 -- word-vs-persona decomposition, scaled",
        f"  pool size: {len(df)} adjectives   layer: {params['persona_layer']}   model: {params['model']}",
        "",
        "LEXICAL FRACTION (|cos(persona, word-concept)|) across the pool:",
        f"  mean {lex.mean():.1%}   median {lex.median():.1%}   "
        f"min {lex.min():.1%}   max {lex.max():.1%}",
        f"  (notebook's 6-trait finding was ~2-9% lexical; behavioural residual dominates)",
        "",
        "MOST lexical personas (word leaks most into the vector):",
    ]
    top = df.sort_values("lexical_fraction", ascending=False).head(12)
    bot = df.sort_values("lexical_fraction").head(12)
    for r in top.itertuples():
        lines.append(f"  {r.trait:>16}  lexical {r.lexical_fraction:5.1%}  "
                     f"zipf {r.zipf_frequency:4.1f}  tokens {r.n_word_tokens}")
    lines.append("LEAST lexical personas (almost purely behavioural):")
    for r in bot.itertuples():
        lines.append(f"  {r.trait:>16}  lexical {r.lexical_fraction:5.1%}  "
                     f"zipf {r.zipf_frequency:4.1f}  tokens {r.n_word_tokens}")
    lines += [
        "",
        "FREQUENCY vs LEXICAL CONTENT:",
        f"  in-vocab traits: {report['n_in_vocab']}   out-of-vocab (no freq): {report['n_oov']}",
        f"  Spearman r = {report['spearman_r']:+.3f}  (p = {report['spearman_p']:.3g})",
        f"  Pearson  r = {report['pearson_r']:+.3f}  (p = {report['pearson_p']:.3g})",
        f"  partial Spearman (control token count) r = "
        f"{report['partial_spearman_r_control_tokens']:+.3f}  (p = {report['partial_p']:.3g})",
        f"  [freq vs token-count Spearman r = {report['freq_vs_tokencount_spearman_r']:+.3f} "
        f"-- how much the confound bites]",
        "",
        "  Read: r > 0 => commoner words carry MORE lexical content; r < 0 => LESS.",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    _save_scatter(df, report, out_dir / "scatter_frequency_vs_lexical.png")


def _save_scatter(df: pd.DataFrame, report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iv = df[~df["is_oov"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(iv["zipf_frequency"], iv["lexical_fraction"],
                    c=iv["n_word_tokens"], cmap="viridis", s=28, alpha=0.8)
    # least-squares guide line over the in-vocab points
    a, b = np.polyfit(iv["zipf_frequency"], iv["lexical_fraction"], 1)
    xs = np.linspace(iv["zipf_frequency"].min(), iv["zipf_frequency"].max(), 50)
    ax.plot(xs, a * xs + b, "r--", lw=1.5,
            label=f"Spearman r={report['spearman_r']:+.2f} (p={report['spearman_p']:.2g})")
    ax.set_xlabel("Zipf word frequency in English (higher = more common)")
    ax.set_ylabel("lexical fraction  |cos(persona, word)|")
    ax.set_title("Does a trait word's commonness predict its persona's lexical content?")
    ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, label="# subword tokens in word")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"saved scatter -> {path}")


def main() -> None:
    args = parse_args()
    cfg = pl.get_config(args.model)
    out_dir = args.out_dir or (Path(__file__).resolve().parent / "results"
                               / f"exp1_decomposition{cfg.cache_suffix}")
    print(f"loading {cfg.hf_id} + reconstructing the {args.n_traits}-trait pool ...")
    model, tokenizer, device = pl.load_model(args.model)

    pool_traits = pl.load_trait_pool(n_traits=args.n_traits)
    names, Vp, Vc, centroid = pl.load_persona_frame(pool_traits, args.model)
    assert len(names) == args.n_traits, f"expected {args.n_traits} traits, got {len(names)}"
    print(f"device={device}  model={cfg.key}  layer={cfg.persona_layer}  "
          f"pool={len(names)}  d_model={Vc.shape[1]}")

    df = decompose_pool(model, tokenizer, names, Vc, centroid, cfg.persona_layer)
    report = correlation_report(df)

    params = {
        "experiment": "exp1_decomposition",
        "model_key": cfg.key,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": cfg.hf_id,
        "persona_layer": cfg.persona_layer,
        "n_traits": len(names),
        "rel_threshold": pl.REL_THRESHOLD,
        "tie_seed": pl.TIE_SEED,
        "carriers": "; ".join(__import__("prompts").WORD_CONCEPT_CARRIERS),
        "frequency_source": f"wordfreq=={_pkg_version('wordfreq')} zipf_frequency(en)",
        "torch": t.__version__,
        "python": platform.python_version(),
    }
    write_outputs(df, report, params, out_dir)
    print(f"\ndone -> {out_dir}")


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
