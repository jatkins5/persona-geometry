"""Re-extract persona vectors for a new model (port of notebook 6d), for cross-model replication.

The persona pool (the 220 adjectives and their system prompts) is model-agnostic and already
cached, so we REUSE those. What must be redone per model is the activations: we have the new
model role-play each persona (fresh generations -- the vector must reflect THIS model acting,
not reading another model's text) and take the mean response activation. We capture EVERY layer
in one pass so the persona layer can be chosen offline.

Two modes:
  (default) extract: generate + all-layer activations -> save all-layer cache + responses, print a
                     per-layer separation diagnostic, and write a provisional single-layer file at
                     the registry's persona_layer.
  --write-layer L:   no GPU; slice the all-layer cache at layer L -> the single-layer persona-vectors
                     file the Exp-1 scripts read. Use after eyeballing the diagnostic.

Heavy (generation): ~n_traits * pool_n_q completions. Run on a GPU node via sbatch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch as t
from tqdm import tqdm

import persona_lib as pl
import prompts

POOL_N_Q = 6   # notebook 6d uses the first 6 eval questions per persona


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama3.1-8b", choices=list(pl.MODELS),
                    help="which registry model to extract vectors for")
    ap.add_argument("--n-traits", type=int, default=pl.N_TRAITS)
    ap.add_argument("--pool-n-q", type=int, default=POOL_N_Q)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--write-layer", type=int, default=None,
                    help="CPU-only: slice the existing all-layer cache at this layer into the "
                         "single-layer persona-vectors file, then exit")
    return ap.parse_args()


def all_layer_cache_path(model_key: str) -> Path:
    return pl.DATA_DIR / f"persona_pool_alllayer{pl.get_config(model_key).cache_suffix}.pt"


def responses_cache_path(model_key: str) -> Path:
    return pl.DATA_DIR / f"persona_pool_responses{pl.get_config(model_key).cache_suffix}.json"


def write_single_layer(model_key: str, layer: int) -> Path:
    """Slice the all-layer cache at `layer` into the single-layer file the Exp-1 scripts load."""
    all_layer = t.load(all_layer_cache_path(model_key))     # name -> (num_hidden, d)
    n_hidden = next(iter(all_layer.values())).shape[0]
    if not 0 <= layer < n_hidden:
        raise ValueError(f"layer {layer} out of range [0, {n_hidden})")
    single = {name: vecs[layer].clone() for name, vecs in all_layer.items()}
    out = pl.persona_vectors_path(model_key)
    t.save(single, out)
    print(f"wrote {len(single)} persona vectors at layer {layer} -> {out.name}")
    return out


def layer_diagnostic(all_layer: dict[str, t.Tensor], names: list[str]) -> None:
    """Per-layer structure of the persona space, to choose the persona layer for a new model.

    For each layer we center the persona vectors and report: how far personas spread from the
    centroid (mean norm), how much real structure there is beyond chance orthogonality (fraction
    of |cos| above 3x the random baseline), and the effective dimensionality (PCA participation
    ratio). A good persona layer has high structure and a moderate-to-high spread, usually in the
    middle-to-late band. The notebook's Qwen choice sits at ~71% depth.
    """
    A = t.stack([all_layer[n].float() for n in names])      # (N, num_hidden, d)
    N, n_hidden, d = A.shape
    chance_sd = 1.0 / np.sqrt(d - 1)
    print(f"\nlayer diagnostic ({N} personas, d={d}, {n_hidden} hidden layers):")
    print(f"  {'layer':>5} {'depth':>6} {'mean_norm':>10} {'struct>3sd':>11} {'eff_dim':>8}")
    best = []
    for L in range(n_hidden):
        Vc = A[:, L, :] - A[:, L, :].mean(0)
        norms = Vc.norm(dim=1)
        Vn = Vc / norms.clamp_min(1e-9)[:, None]
        cos = Vn @ Vn.T
        off = cos[~t.eye(N, dtype=bool)]
        struct = float((off.abs() > 3 * chance_sd).float().mean())
        sv = t.linalg.svdvals(Vc)                            # participation ratio of variance
        eff_dim = float((sv.pow(2).sum() ** 2) / sv.pow(4).sum())
        depth = L / (n_hidden - 1)
        if 0.45 <= depth <= 0.85:
            best.append((struct, L))
        print(f"  {L:>5} {depth:>6.2f} {float(norms.mean()):>10.2f} {struct:>11.1%} {eff_dim:>8.1f}")
    if best:
        suggested = max(best)[1]
        print(f"\n  suggested persona layer (max structure in 0.45-0.85 depth band): {suggested}")


def main() -> None:
    args = parse_args()
    cfg = pl.get_config(args.model)

    # CPU-only mode: just (re)write the single-layer file from an existing all-layer cache
    if args.write_layer is not None:
        write_single_layer(args.model, args.write_layer)
        return

    print(f"loading {cfg.hf_id} for persona-vector extraction ...")
    model, tokenizer, device = pl.load_model(args.model)
    pool_traits = pl.load_trait_pool(n_traits=args.n_traits)
    persona_systems = json.loads((pl.DATA_DIR / "persona_pool_system_prompts.json").read_text())
    questions = prompts.EVAL_QUESTIONS[: args.pool_n_q]
    print(f"device={device}  model={cfg.key}  traits={len(pool_traits)}  q/persona={len(questions)}")

    # resume support: skip traits already in the caches
    all_layer = t.load(all_layer_cache_path(args.model)) if all_layer_cache_path(args.model).exists() else {}
    responses = json.loads(responses_cache_path(args.model).read_text()) \
        if responses_cache_path(args.model).exists() else {}

    todo = [w for w in pool_traits if w not in all_layer]
    print(f"{len(all_layer)} already cached; extracting {len(todo)} new")
    for trait in tqdm(todo, desc="extract"):
        system = persona_systems[trait]
        per_q = []
        texts = []
        for q in questions:
            resp = pl.generate_persona_response(model, tokenizer, system, q, args.max_new_tokens)
            texts.append(resp)
            per_q.append(pl.extract_response_activations_all_layers(model, tokenizer, system, q, resp))
        all_layer[trait] = t.stack(per_q).mean(0).half()    # (num_hidden, d), averaged over questions
        responses[trait] = texts
        # checkpoint every few personas so a crash keeps progress
        if len(all_layer) % 10 == 0:
            t.save(all_layer, all_layer_cache_path(args.model))
            responses_cache_path(args.model).write_text(json.dumps(responses))

    t.save(all_layer, all_layer_cache_path(args.model))
    responses_cache_path(args.model).write_text(json.dumps(responses, indent=2))
    print(f"saved all-layer vectors -> {all_layer_cache_path(args.model).name}")

    names = [w for w in pool_traits if w in all_layer]
    layer_diagnostic(all_layer, names)

    # provisional single-layer file at the registry layer so downstream 'just runs'
    write_single_layer(args.model, cfg.persona_layer)
    print(f"\n(provisional layer {cfg.persona_layer}; re-pick with --write-layer L after the diagnostic)")


if __name__ == "__main__":
    main()
