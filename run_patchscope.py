"""Self-contained runner for the Patchscope experiment (spare GPU).

    python run_patchscope.py --vectors persona_pool_vectors.pt

Loads Qwen2.5-7B-Instruct, validates the Patchscope read-out on the known personas, then
searches the persona PCA box for blind spots the read-out can't name. Tune the read-out
(prompt / inject_layer) until validate() recovery is high BEFORE trusting the blind spots.
"""
import argparse

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from patchscope import PatchscopeReader, adversarial_search, pca_box, validate

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="persona_pool_vectors.pt", help="name -> (d,) tensor")
    ap.add_argument("--inject-layer", type=int, default=19)
    ap.add_argument("--n-pc", type=int, default=10)
    ap.add_argument("--n-random", type=int, default=400)
    ap.add_argument("--n-refine", type=int, default=200)
    ap.add_argument("--max-personas", type=int, default=0, help="cap validate() set for a quick look")
    args = ap.parse_args()

    device = "cuda" if t.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=t.bfloat16).to(device).eval()

    pool = t.load(args.vectors)
    print(f"loaded {len(pool)} persona vectors (d={next(iter(pool.values())).shape[-1]})")

    reader = PatchscopeReader(model, tok, inject_layer=args.inject_layer)

    print("\n=== 1. read-out validation (tune prompt/inject_layer until this is high) ===")
    subset = pool if not args.max_personas else dict(list(pool.items())[: args.max_personas])
    validate(reader, subset)

    print("\n=== 2. adversarial blind-spot search ===")
    box = pca_box(pool, n_pc=args.n_pc)
    blind = adversarial_search(reader, box, words=list(pool),
                               n_random=args.n_random, n_refine=args.n_refine)
    print(f"\n{'conf':>7}  {'best-guess':>20}  read-out")
    for b in blind:
        print(f"{b['confidence']:7.2f}  {b['best_guess']:>20}  {b['readout']}")


if __name__ == "__main__":
    main()
