"""Experiment 1 (surface-word contamination + residual-steering) at scale.

Replicates notebook 7a and 7c over the full analysed pool instead of the 6 strongest
traits. (The word-vs-persona decomposition, 7b, is its own quick job:
exp1_decomposition.py.) Two parts, per trait:

  7a CONTAMINATION. Does merely naming the trait WORD (neutrally, or with the opposite
     stance) push activations along that trait's persona axis? If the axis fired on mere
     mention it would not be a clean abstraction. We project both the prompt-side and the
     response-side activations onto the trait axis under four conditions
     (baseline / mention-neutral / mention-opposite / persona) and report
        contamination ratio = (mention - baseline) / (persona - baseline)
     where 0 = clean and ~1 = the word alone fully activates the axis.

  7c RESIDUAL STEERING. Split each persona vector into its word-concept component and
     its behavioural residual, steer with full / word-component / residual at MATCHED
     norm, and have the autorater score how strongly the generation shows the trait. If
     the residual still elicits the trait, the persona is genuinely behavioural, not lexical.

Heavy: ~thousands of local generations + ~n_traits*12 judge calls. Everything is logged
incrementally (one row per LLM call) so a crash mid-run keeps the completed traits.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch as t
from tqdm import tqdm

import persona_lib as pl
import prompts

CONTAM_LOG_FIELDS = ["trait", "condition", "question", "response", "prompt_proj", "resp_proj"]
STEER_LOG_FIELDS = ["trait", "condition", "question", "response", "judge_score", "judge_raw"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=pl.DEFAULT_MODEL, choices=list(pl.MODELS),
                    help="which registry model to run (default: qwen2.5-7b)")
    ap.add_argument("--n-traits", type=int, default=pl.N_TRAITS,
                    help="pool size to build the frame/centroid from (default 220)")
    ap.add_argument("--start", type=int, default=0,
                    help="index of the first pool trait to process (for chunking across jobs)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only this many traits starting at --start (for a smoke test or "
                         "chunk); the centroid still uses the full pool so numbers stay comparable")
    ap.add_argument("--steer-coeff", type=float, default=1.5,
                    help="7c steering coefficient (notebook's coherent operating point)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output dir (default: results/exp1_contamination_steer[<model suffix>])")
    return ap.parse_args()


def parse_judge_score(raw: str) -> int | None:
    """Pull the 0-10 integer out of the autorater's reply (notebook 7c)."""
    m = re.search(r"\d+", raw or "")
    return min(int(m.group()), 10) if m else None


def run_contamination(model, tokenizer, trait, axis_unit, centroid, persona_system,
                      log_writer, max_new_tokens, persona_layer) -> dict[str, float]:
    """7a for one trait: projections under each condition, returning the two contamination ratios."""
    eval_qs = prompts.EVAL_QUESTIONS[:3]
    conditions = {
        "baseline": [("", q) for q in eval_qs],
        "mention-neutral": [("", q) for q in prompts.mention_neutral_questions(trait)],
        "mention-opposite": [("", q) for q in prompts.mention_opposite_questions(trait)],
        "persona": [(persona_system, q) for q in eval_qs],
    }
    proj = {c: {"prompt": [], "resp": []} for c in conditions}
    for cond, items in conditions.items():
        for system, q in items:
            response = pl.generate_persona_response(model, tokenizer, system, q, max_new_tokens)
            resp_act = pl.extract_response_activations(
                model, tokenizer, [system], [q], [response], persona_layer)[0].float()
            prompt_mean, _ = pl.prompt_hidden(model, tokenizer, system, q, persona_layer)
            r_proj = float((resp_act - centroid) @ axis_unit)
            p_proj = float((prompt_mean - centroid) @ axis_unit)
            proj[cond]["prompt"].append(p_proj)
            proj[cond]["resp"].append(r_proj)
            log_writer("contam", {"trait": trait, "condition": cond, "question": q,
                                  "response": response, "prompt_proj": p_proj, "resp_proj": r_proj})

    def ratio(cond: str, side: str) -> float:
        base = np.mean(proj["baseline"][side])
        on = np.mean(proj["persona"][side])
        return (np.mean(proj[cond][side]) - base) / (on - base + 1e-9)

    return {
        "trait": trait,
        "prompt_ratio_neutral": ratio("mention-neutral", "prompt"),
        "prompt_ratio_opposite": ratio("mention-opposite", "prompt"),
        "resp_ratio_neutral": ratio("mention-neutral", "resp"),
        "resp_ratio_opposite": ratio("mention-opposite", "resp"),
        "persona_minus_baseline_resp": float(np.mean(proj["persona"]["resp"]) - np.mean(proj["baseline"]["resp"])),
    }


def run_steering(model, tokenizer, client, trait, persona_vec, word_unit,
                 steer_coeff, log_writer, max_new_tokens, steer_layer) -> dict[str, float]:
    """7c for one trait: steer full/word-component/residual at matched norm, judge the trait."""
    norm = persona_vec.norm()
    word_component = (persona_vec @ word_unit) * word_unit
    residual = persona_vec - word_component
    directions = {
        "full": persona_vec,
        "word-comp": word_component / word_component.norm() * norm,
        "residual": residual / residual.norm() * norm,
    }
    steer_qs = prompts.EVAL_QUESTIONS[:4]

    # generate every steered text first, then judge them all in one parallel batch
    texts = {cond: [pl.generate_with_steerer(model, tokenizer, q, vec, steer_layer,
                                             steer_coeff, max_new_tokens) for q in steer_qs]
             for cond, vec in directions.items()}
    flat = [(cond, q, txt) for cond in directions for q, txt in zip(steer_qs, texts[cond])]
    judge_msgs = [prompts.trait_expression_judge(trait, txt) for _, _, txt in flat]
    raw_scores = pl.judge_calls_parallel(client, judge_msgs)

    per_cond: dict[str, list[int]] = {c: [] for c in directions}
    for (cond, q, txt), raw in zip(flat, raw_scores):
        score = parse_judge_score(raw)
        if score is not None:
            per_cond[cond].append(score)
        log_writer("steer", {"trait": trait, "condition": cond, "question": q, "response": txt,
                             "judge_score": "" if score is None else score, "judge_raw": raw})

    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return {"trait": trait, "score_full": mean(per_cond["full"]),
            "score_word_comp": mean(per_cond["word-comp"]), "score_residual": mean(per_cond["residual"])}


def summarize(contam_rows, steer_rows, out_dir, params):
    """Aggregate both parts, write summary.txt + figures, and print the headline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cr = {k: np.array([r[k] for r in contam_rows]) for k in
          ["resp_ratio_neutral", "resp_ratio_opposite", "prompt_ratio_neutral", "prompt_ratio_opposite"]}
    sf = np.array([r["score_full"] for r in steer_rows])
    sw = np.array([r["score_word_comp"] for r in steer_rows])
    sr = np.array([r["score_residual"] for r in steer_rows])

    # The contamination RATIO divides by (persona - baseline); weak personas have a tiny
    # denominator and produce wild ratios, so we report the robust MEDIAN over all traits
    # and the MEAN over the strong-persona half (where the axis is actually well-defined).
    strength = np.abs([r["persona_minus_baseline_resp"] for r in contam_rows])
    strong = strength >= np.median(strength)
    nanmean = lambda a: float(np.nanmean(a))
    nanmed = lambda a: float(np.nanmedian(a))

    lines = [
        "EXPERIMENT 1 -- contamination + residual-steering, scaled",
        f"  model: {params['model']}   traits processed: {len(contam_rows)}   "
        f"layer: {params['persona_layer']}   steer_coeff: {params['steer_coeff']}   "
        f"judge: {pl.AUTORATER_MODEL}",
        "",
        "7a CONTAMINATION ratio (0 = clean axis, ~1 = the word alone activates it):",
        "  [median over all traits | mean over the strong-persona half]",
        f"  response-side: mention-neutral  {nanmed(cr['resp_ratio_neutral']):+.2f} | "
        f"{nanmean(cr['resp_ratio_neutral'][strong]):+.2f}    "
        f"mention-opposite {nanmed(cr['resp_ratio_opposite']):+.2f} | "
        f"{nanmean(cr['resp_ratio_opposite'][strong]):+.2f}",
        f"  prompt-side:   mention-neutral  {nanmed(cr['prompt_ratio_neutral']):+.2f} | "
        f"{nanmean(cr['prompt_ratio_neutral'][strong]):+.2f}    "
        f"mention-opposite {nanmed(cr['prompt_ratio_opposite']):+.2f} | "
        f"{nanmean(cr['prompt_ratio_opposite'][strong]):+.2f}",
        "  (low response-side ratios => the persona axis is a clean abstraction at scale;",
        "   positive prompt-side is the literal word token sitting on the axis, per 7a-ii)",
        "",
        "7c RESIDUAL STEERING -- trait expression 0-10 (norm-matched directions):",
        f"  mean:   full {nanmean(sf):.2f}   word-component {nanmean(sw):.2f}   residual {nanmean(sr):.2f}",
        f"  median: full {nanmed(sf):.2f}   word-component {nanmed(sw):.2f}   residual {nanmed(sr):.2f}",
        f"  residual keeps {nanmean(sr) / max(nanmean(sf), 1e-9):.0%} of full's trait expression; "
        f"word-component keeps {nanmean(sw) / max(nanmean(sf), 1e-9):.0%}",
        "  (residual ~ full and >> word-component => persona is behavioural, not the word)",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    # figure 1: distribution of response-side contamination ratios (clipped to [-2, 2] so
    # the weak-persona outliers don't flatten the histogram)
    clip = lambda a: np.clip(a[~np.isnan(a)], -2, 2)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(clip(cr["resp_ratio_neutral"]), bins=30, range=(-2, 2), alpha=0.7, label="mention-neutral")
    ax.hist(clip(cr["resp_ratio_opposite"]), bins=30, range=(-2, 2), alpha=0.7, label="mention-opposite")
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("response-side contamination ratio (clipped to ±2)")
    ax.set_ylabel("# traits"); ax.set_title("7a: contamination ratio across the pool (0 = clean)")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "contamination_ratios.png", dpi=140)

    # figure 2: steering scores full vs word vs residual
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([0, 1, 2], [nanmean(sf), nanmean(sw), nanmean(sr)],
           yerr=[np.nanstd(sf), np.nanstd(sw), np.nanstd(sr)], capsize=5,
           color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["full", "word-component", "residual"])
    ax.set_ylabel("trait expression (0-10)")
    ax.set_title("7c: steering with full vs word-component vs residual (norm-matched)")
    fig.tight_layout(); fig.savefig(out_dir / "steering_scores.png", dpi=140)
    print(f"saved figures -> {out_dir}")


def main() -> None:
    args = parse_args()
    cfg = pl.get_config(args.model)
    out_dir = args.out_dir or (Path(__file__).resolve().parent / "results"
                               / f"exp1_contamination_steer{cfg.cache_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {cfg.hf_id} + reconstructing the {args.n_traits}-trait pool ...")
    model, tokenizer, device = pl.load_model(args.model)
    client = pl.make_openrouter_client()

    pool_traits = pl.load_trait_pool(n_traits=args.n_traits)
    names, Vp, Vc, centroid = pl.load_persona_frame(pool_traits, args.model)
    assert len(names) == args.n_traits, f"expected {args.n_traits} traits, got {len(names)}"
    persona_systems = json.loads((pl.DATA_DIR / "persona_pool_system_prompts.json").read_text())

    start = args.start
    end = min(start + args.limit, len(names)) if args.limit else len(names)
    indices = list(range(start, end))
    process = [names[i] for i in indices]
    print(f"device={device}  model={cfg.key}  layer={cfg.persona_layer}/{cfg.steer_layer}  "
          f"pool={len(names)}  processing[{start}:{end}]={len(process)}  coeff={args.steer_coeff}")

    # incremental, crash-robust logging: open both per-call CSVs up front
    contam_fp = open(out_dir / "contamination_logs.csv", "w", newline="")
    steer_fp = open(out_dir / "steering_logs.csv", "w", newline="")
    contam_w = csv.DictWriter(contam_fp, fieldnames=CONTAM_LOG_FIELDS); contam_w.writeheader()
    steer_w = csv.DictWriter(steer_fp, fieldnames=STEER_LOG_FIELDS); steer_w.writeheader()

    def log_writer(kind: str, row: dict) -> None:
        (contam_w if kind == "contam" else steer_w).writerow(row)
        (contam_fp if kind == "contam" else steer_fp).flush()

    contam_rows, steer_rows = [], []
    for idx in tqdm(indices, desc="7a+7c"):
        trait = names[idx]
        persona_vec = Vc[idx]
        axis_unit = persona_vec / persona_vec.norm()
        word_raw, _ = pl.word_concept_vector(model, tokenizer, trait, cfg.persona_layer)
        word_unit = (word_raw - centroid)
        word_unit = word_unit / word_unit.norm()

        contam_rows.append(run_contamination(
            model, tokenizer, trait, axis_unit, centroid, persona_systems[trait],
            log_writer, args.max_new_tokens, cfg.persona_layer))
        steer_rows.append(run_steering(
            model, tokenizer, client, trait, persona_vec, word_unit,
            args.steer_coeff, log_writer, args.max_new_tokens, cfg.steer_layer))

    contam_fp.close(); steer_fp.close()

    # per-trait result tables
    _write_csv(out_dir / "contamination_results.csv", contam_rows)
    _write_csv(out_dir / "steering_results.csv", steer_rows)

    # a few raw transcripts to actually READ before trusting the aggregates
    _dump_transcripts(out_dir, process, steer_rows)

    params = {
        "experiment": "exp1_contamination_steer",
        "model_key": cfg.key,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": cfg.hf_id, "judge_model": pl.AUTORATER_MODEL,
        "persona_layer": cfg.persona_layer, "steer_layer": cfg.steer_layer,
        "n_traits_pool": len(names), "n_traits_processed": len(process),
        "steer_coeff": args.steer_coeff, "max_new_tokens": args.max_new_tokens,
        "torch": t.__version__, "python": platform.python_version(),
    }
    _write_csv(out_dir / "global_params.csv", [params])
    summarize(contam_rows, steer_rows, out_dir, params)
    print(f"\ndone -> {out_dir}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _dump_transcripts(out_dir: Path, traits, steer_rows, k: int = 8) -> None:
    """Read steering_logs.csv back and print sample residual-steered generations per trait."""
    rows = list(csv.DictReader(open(out_dir / "steering_logs.csv")))
    by_trait_residual = {}
    for r in rows:
        if r["condition"] == "residual":
            by_trait_residual.setdefault(r["trait"], r)
    lines = ["SAMPLE RESIDUAL-STEERED GENERATIONS (one per trait)\n"]
    for tr in list(traits)[:k]:
        ex = by_trait_residual.get(tr)
        if ex:
            lines.append(f"[{tr}] (judge={ex['judge_score']})  {ex['response'][:280]}\n")
    (out_dir / "transcripts.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
