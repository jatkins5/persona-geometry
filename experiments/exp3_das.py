"""Experiment 3 -- is the persona causal interface low-dimensional? (DAS), model-parameterized.

Port of notebook Section 8, judge-free core. Learns a shared k-dim subspace D (a rotation via
gradient descent) such that swapping ONLY the D-component of the residual stream (persona A -> B,
same question, all positions, layer = steer_layer) makes A behave like B. The smallest k that
reproduces the full-difference interchange is the persona "panel" dimension.

Three measurements, all judge-free (teacher-forced cross-entropy of B's cached responses):
  8g  PANEL DIMENSION  -- held-out CE vs k, multi-seed, learned vs random vs full-difference (P=I).
                          The knee where learned CE stops improving = panel dimension.
  8h  CLEAN-RUN CONTROL -- does between-persona variation on clean (unintervened) runs live in the
                          learned D? (variance captured + leave-one-out persona-ID). Confirms D is
                          USED, not a dormant pathway the intervention merely exploits.
  8j  DISTINCT INTERFACE -- at BIG_N personas (high-dim persona span), retrain D at k and check the
                          swap still works (held-out CE) while D keeps only a small slice of the
                          persona difference (<< a variance-optimal PCA-k). => narrow causal interface
                          distinct from the (high-dim) persona representation.

Motivation for the cross-model run: Llama's persona space is ~2x higher-dimensional than Qwen's
(participation ratio ~9 vs ~4). If persona is more distributed, the panel dimension should come out
LARGER (or the interface less clean) than Qwen's k~5-6.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch as t
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tqdm import tqdm

import persona_lib as pl
import prompts

POOL_N_Q = 6   # questions each persona has cached responses for (notebook POOL_QUESTIONS)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama3.1-8b", choices=list(pl.MODELS))
    ap.add_argument("--n-personas", type=int, default=8, help="personas for the 8h clean-run control")
    ap.add_argument("--n-train", type=int, default=12, help="8g: personas to TRAIN the projector on")
    ap.add_argument("--n-test", type=int, default=12,
                    help="8g: DISJOINT personas to measure held-out CE on (never seen in training)")
    ap.add_argument("--n-test-pairs", type=int, default=12, help="8g: held-out test pairs to average CE over")
    ap.add_argument("--split-seed", type=int, default=-1,
                    help="8g: if >=0, RANDOMLY assign train/test personas (drawn from the strongest "
                         "--candidate-pool) with this seed, instead of the strongest-vs-next split")
    ap.add_argument("--candidate-pool", type=int, default=40,
                    help="8g: pool of strongest personas to randomly draw train/test from (with --split-seed)")
    ap.add_argument("--k-list", default="1,2,3,4,6,8,12")
    ap.add_argument("--seeds", default="0,1,2", help="training seeds to average the held-out CE over")
    ap.add_argument("--steps", type=int, default=150, help="SGD steps per (k, seed)")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--coeff", type=float, default=1.0, help="interchange strength (norm-matched to ||Vc[B]-Vc[A]||)")
    ap.add_argument("--resp-tokens", type=int, default=48, help="teacher-forced response length cap")
    ap.add_argument("--big-n", type=int, default=32, help="persona count for the distinct-interface test (8j)")
    ap.add_argument("--big-k", type=int, default=6)
    ap.add_argument("--big-steps", type=int, default=800)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--panel-only", action="store_true",
                    help="run only the 8g panel-dimension sweep (skip 8h/8j); for fast k-sweeps")
    ap.add_argument("--bisect", action="store_true",
                    help="binary-search the smallest k that recovers --bisect-target of the "
                         "full-difference (P=I) effect on the DISJOINT test personas")
    ap.add_argument("--bisect-target", type=float, default=0.9,
                    help="target fraction of the full-difference effect for --bisect (default 0.9)")
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap.parse_args()


# ---- DAS primitives (notebook 8d/8f) -------------------------------------------------
def projector(W: t.Tensor, eps: float) -> t.Tensor:
    """Orthogonal projector onto colspace(W): P = W (W^T W + eps I)^-1 W^T."""
    return W @ t.linalg.inv(W.T @ W + eps * t.eye(W.shape[1], device=W.device)) @ W.T


def nm_offset(P: t.Tensor, diff: t.Tensor, coeff: float) -> t.Tensor:
    """Interchange offset projected onto P, rescaled to ||diff|| -- equal strength at every k."""
    pd = P @ diff
    return coeff * diff.norm() * pd / (pd.norm() + 1e-6)


class Exp3:
    """Holds the model + cached frame and the teacher-forced interchange loss at the DAS site."""

    def __init__(self, model, tokenizer, cfg, Vc, centroid, pca, names, systems, responses,
                 questions, resp_tokens, eps):
        self.model, self.tok, self.cfg = model, tokenizer, cfg
        self.Vc, self.centroid, self.pca = Vc, centroid, pca
        self.names, self.systems, self.responses = names, systems, responses
        self.questions, self.resp_tokens, self.eps = questions, resp_tokens, eps
        self.layer = cfg.steer_layer
        self.dev = model.device

    def idx(self, name: str) -> int:
        return self.names.index(name)

    def diff(self, a: str, b: str) -> t.Tensor:
        """A->B persona difference (centered vectors), on device."""
        return (self.Vc[self.idx(b)] - self.Vc[self.idx(a)]).float().to(self.dev)

    def teacher_forced_loss(self, system: str, question: str, response: str, offset: t.Tensor) -> t.Tensor:
        """CE of `response` given (system, question) with `offset` added at all positions, layer = site."""
        formatted = self.tok.apply_chat_template(
            pl._normalize_messages([{"role": "system", "content": system},
                                    {"role": "user", "content": question}]),
            tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tok(formatted, return_tensors="pt").input_ids.to(self.dev)
        resp_ids = self.tok(response, return_tensors="pt", add_special_tokens=False
                            ).input_ids[:, : self.resp_tokens].to(self.dev)
        input_ids = t.cat([prompt_ids, resp_ids], dim=1)
        labels = input_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100               # supervise only B's response tokens

        def add_offset(module, layer_input, layer_output):
            hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output
            hidden = hidden + offset.to(hidden.dtype)
            return (hidden,) + layer_output[1:] if isinstance(layer_output, tuple) else hidden

        handle = pl._return_layers(self.model)[self.layer].register_forward_hook(add_offset)
        try:
            logits = self.model(input_ids, use_cache=False).logits
        finally:
            handle.remove()
        return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                               labels[:, 1:].reshape(-1), ignore_index=-100)

    def train_subspace(self, k: int, seed: int, steps: int, lr: float, coeff: float,
                       order, diff_of) -> t.Tensor:
        """Train a k-dim norm-matched interchange projector; return P.detach()."""
        t.manual_seed(seed)
        W = t.nn.Parameter(t.randn(self.Vc.shape[1], k, device=self.dev) * 0.02)
        opt = t.optim.Adam([W], lr=lr)
        for step in range(steps):
            (a, b) = order[step % len(order)][0]
            q = order[(step * 7) % len(order)][1]
            response = self.responses[b][self.questions.index(q)]
            with t.enable_grad():                              # load_model set grad off globally
                offset = nm_offset(projector(W, self.eps), diff_of[(a, b)], coeff)
                loss = self.teacher_forced_loss(self.systems[a], q, response, offset)
                opt.zero_grad(); loss.backward()
            opt.step()
        return projector(W, self.eps).detach()

    def heldout_ce(self, P: t.Tensor, pairs, diff_of, coeff: float) -> float:
        """Mean teacher-forced CE of B's responses over held-out pairs under the norm-matched A->B swap."""
        vals = []
        for a, b in pairs:
            for qi, q in enumerate(self.questions):
                offset = nm_offset(P, diff_of[(a, b)], coeff).detach()
                vals.append(self.teacher_forced_loss(self.systems[a], q, self.responses[b][qi], offset).item())
        return float(np.mean(vals))


# ---- the three measurements ----------------------------------------------------------
def panel_dimension(E: Exp3, train_personas, test_personas, n_test_pairs, k_list, seeds,
                    steps, lr, coeff, eps) -> dict:
    """8g: held-out interchange CE vs k, training P on `train_personas` and measuring CE on a
    DISJOINT `test_personas` set (so the projector never saw the test personas' directions)."""
    d_model = E.Vc.shape[1]
    train_pairs = [(a, b) for a in train_personas for b in train_personas if a != b]
    test_all = [(a, b) for a in test_personas for b in test_personas if a != b]
    stride = max(1, len(test_all) // n_test_pairs)
    test_pairs = test_all[::stride][:n_test_pairs]
    diff_train = {(a, b): E.diff(a, b) for a, b in train_pairs}   # used only for training
    diff_test = {(a, b): E.diff(a, b) for a, b in test_pairs}     # used only for evaluation
    order = [(p, q) for p in train_pairs for q in E.questions]

    eye = t.eye(d_model, device=E.dev)
    ref = E.heldout_ce(eye, test_pairs, diff_test, coeff)         # full-difference reference (P=I)
    zero = E.heldout_ce(t.zeros(d_model, d_model, device=E.dev), test_pairs, diff_test, coeff)

    print(f"\n8g panel dimension: train on {len(train_personas)} personas ({len(train_pairs)} pairs) "
          f"-> test on {len(test_personas)} DISJOINT personas ({len(test_pairs)} pairs):")
    print(f"  no-intervention CE (P=0): {zero:.3f}   full-difference CE (P=I): {ref:.3f}")
    print(f"  {'k':>3} | {'learned (mean +/- sd)':>24} | {'random':>8}")
    out = {"k": k_list, "learned_mean": [], "learned_sd": [], "random_mean": [],
           "full_diff_ref": ref, "no_intervention": zero, "n_train_personas": len(train_personas),
           "n_test_personas": len(test_personas), "n_test_pairs": len(test_pairs)}
    for k in k_list:
        learned = [E.heldout_ce(E.train_subspace(k, s, steps, lr, coeff, order, diff_train),
                                test_pairs, diff_test, coeff) for s in seeds]
        randoms = []
        for s in seeds:
            t.manual_seed(500 + s)
            randoms.append(E.heldout_ce(projector(t.randn(d_model, k, device=E.dev), eps).detach(),
                                        test_pairs, diff_test, coeff))
        out["learned_mean"].append(float(np.mean(learned)))
        out["learned_sd"].append(float(np.std(learned)))
        out["random_mean"].append(float(np.mean(randoms)))
        print(f"  {k:>3} | {np.mean(learned):8.3f} +/- {np.std(learned):.3f}        | {np.mean(randoms):8.3f}")
    return out


def bisect_dimension(E: Exp3, train_personas, test_personas, n_test_pairs, target, k_max,
                     seeds, steps, lr, coeff, eps) -> dict:
    """Binary-search the smallest k whose learned subspace recovers `target` of the full-difference
    (P=I) effect on the DISJOINT test personas. Held-out CE is (assumed) monotone decreasing in k;
    the full trajectory is printed so non-monotone noise near the boundary is visible. f(k_max)=P=I
    by construction (a full-rank projector is the identity), so the target is always reachable."""
    d_model = E.Vc.shape[1]
    train_pairs = [(a, b) for a in train_personas for b in train_personas if a != b]
    test_all = [(a, b) for a in test_personas for b in test_personas if a != b]
    stride = max(1, len(test_all) // n_test_pairs)
    test_pairs = test_all[::stride][:n_test_pairs]
    diff_train = {p: E.diff(*p) for p in train_pairs}
    diff_test = {p: E.diff(*p) for p in test_pairs}
    order = [(p, q) for p in train_pairs for q in E.questions]

    eye = t.eye(d_model, device=E.dev)
    ceil = E.heldout_ce(eye, test_pairs, diff_test, coeff)
    no_int = E.heldout_ce(t.zeros(d_model, d_model, device=E.dev), test_pairs, diff_test, coeff)
    target_ce = no_int - target * (no_int - ceil)
    print(f"\nbisect: no-intervention CE {no_int:.3f}  full-difference (P=I) CE {ceil:.3f}")
    print(f"  target = {target:.0%} of effect  ->  CE <= {target_ce:.3f}   (k in [1, {k_max}])")

    cache = {}

    def recovered_ce(k):
        if k not in cache:
            ces = [E.heldout_ce(E.train_subspace(k, s, steps, lr, coeff, order, diff_train),
                                test_pairs, diff_test, coeff) for s in seeds]
            cache[k] = float(np.mean(ces))
            rec = (no_int - cache[k]) / (no_int - ceil) if no_int > ceil else float("nan")
            print(f"  probe k={k:>5}  CE {cache[k]:.3f}  recovered {rec:+.0%}")
        return cache[k]

    lo, hi = 1, k_max
    while lo < hi:
        mid = (lo + hi) // 2
        if recovered_ce(mid) <= target_ce:
            hi = mid
        else:
            lo = mid + 1
    final_rec = (no_int - cache.get(lo, recovered_ce(lo))) / (no_int - ceil)
    print(f"\n=> smallest k reaching {target:.0%} of the full-difference effect on UNSEEN personas: "
          f"k = {lo}  (recovered {final_rec:+.0%}, CE {cache[lo]:.3f})")
    return {"target": target, "no_intervention": no_int, "full_diff_ce": ceil, "target_ce": target_ce,
            "k_max": k_max, "k_star": lo, "k_star_recovered": float(final_rec),
            "probes": {str(k): v for k, v in sorted(cache.items())},
            "n_train_personas": len(train_personas), "n_test_personas": len(test_personas),
            "n_test_pairs": len(test_pairs)}


def clean_run_control(E: Exp3, personas, panel_k, steps, lr, coeff, eps) -> dict:
    """8h: does clean-run between-persona variation live in the learned D? (variance + persona-ID)."""
    d_model = E.Vc.shape[1]

    @t.no_grad()
    def clean_summary(system, question):
        formatted = E.tok.apply_chat_template(
            pl._normalize_messages([{"role": "system", "content": system},
                                    {"role": "user", "content": question}]),
            tokenize=False, add_generation_prompt=True)
        inputs = E.tok(formatted, return_tensors="pt").to(E.dev)
        cap = {}

        def grab(module, layer_in, layer_out):
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            cap["h"] = hidden[0].mean(0).float().cpu()        # mean over prompt positions
        h = pl._return_layers(E.model)[E.layer].register_forward_hook(grab)
        try:
            E.model(**inputs)
        finally:
            h.remove()
        return cap["h"]

    S, labels = [], []
    for pi, name in enumerate(personas):
        for q in E.questions:
            S.append(clean_summary(E.systems[name], q)); labels.append(pi)
    S = t.stack(S); labels = np.array(labels)
    Sc = S - S.mean(0)
    cents = t.stack([Sc[labels == i].mean(0) for i in range(len(personas))])
    cents = cents - cents.mean(0)

    all_pairs = [(a, b) for a in personas for b in personas if a != b]
    diff_of = {(a, b): E.diff(a, b) for a, b in all_pairs}
    order = [(p, q) for p in all_pairs for q in E.questions]
    P = E.train_subspace(panel_k, 0, steps, lr, coeff, order, diff_of).cpu()

    def variance_in(Pm):
        return float((cents @ Pm).pow(2).sum() / cents.pow(2).sum())

    def persona_id_acc(Pm):
        X = (Sc @ Pm).numpy()
        correct = 0
        for i in range(len(X)):
            c2 = {c: X[[j for j in range(len(X)) if labels[j] == c and j != i]].mean(0) for c in set(labels)}
            correct += min(c2, key=lambda c: np.linalg.norm(X[i] - c2[c])) == labels[i]
        return correct / len(X)

    rv, ra = [], []
    for s in range(3):
        t.manual_seed(200 + s)
        Pr = projector(t.randn(d_model, panel_k), eps).cpu()
        rv.append(variance_in(Pr)); ra.append(persona_id_acc(Pr))
    res = {"panel_k": panel_k, "learned_var": variance_in(P), "random_var": float(np.mean(rv)),
           "learned_id_acc": persona_id_acc(P), "random_id_acc": float(np.mean(ra)),
           "chance_id_acc": 1 / len(personas)}
    print(f"\n8h clean-run control (k={panel_k}):  between-persona var  learned {res['learned_var']:.2f} "
          f"vs random {res['random_var']:.3f};  persona-ID acc learned {res['learned_id_acc']:.0%} "
          f"vs random {res['random_id_acc']:.0%} (chance {res['chance_id_acc']:.0%})")
    return res


def distinct_interface(E: Exp3, big_n, big_k, steps, lr, coeff, eps) -> dict:
    """8j: at BIG_N personas, retrain D at k and check swap still works while D keeps << PCA-k of the diff."""
    d_model = E.Vc.shape[1]
    big_idx = E.Vc.norm(dim=1).argsort(descending=True)[:big_n].tolist()
    big = [E.names[i] for i in big_idx]
    Vbig = t.stack([E.Vc[i].float() for i in big_idx])
    span_rank = int(t.linalg.matrix_rank(Vbig - Vbig.mean(0), tol=1e-3))

    big_pairs = [(a, b) for a in big for b in big if a != b]
    stride = max(1, len(big_pairs) // 6)
    held = big_pairs[::stride][:6]
    train = [p for p in big_pairs if p not in held]
    diff_of = {(a, b): E.diff(a, b) for a, b in big_pairs}
    order = [(p, q) for p in train for q in E.questions]

    print(f"\n8j distinct interface: {big_n} personas span {span_rank}-dim; retraining D at k={big_k} "
          f"({steps} steps)...")
    P = E.train_subspace(big_k, 0, steps, lr, coeff, order, diff_of)
    eye = t.eye(d_model, device=E.dev)
    learned_ce = E.heldout_ce(P, held, diff_of, coeff)
    full_ce = E.heldout_ce(eye, held, diff_of, coeff)
    t.manual_seed(0)
    rand_ce = E.heldout_ce(projector(t.randn(d_model, big_k, device=E.dev), eps).detach(), held, diff_of, coeff)

    # geometry: fraction of each persona difference kept by the learned D vs variance-optimal PCA-k vs random
    Pl = P.cpu()
    t.manual_seed(0)
    Pr = projector(t.randn(d_model, big_k), eps).cpu()
    C = t.tensor(E.pca.components_[:big_k], dtype=t.float32)
    Ppca = C.T @ C

    def cap(Pm, v):
        proj = v @ Pm
        return float((proj @ proj) / (v @ v))

    capL, capP, capR = [], [], []
    for a, b in big_pairs:
        diff = (E.Vc[E.idx(b)] - E.Vc[E.idx(a)]).float()
        capL.append(cap(Pl, diff)); capP.append(cap(Ppca, diff)); capR.append(cap(Pr, diff))
    res = {"big_n": big_n, "big_k": big_k, "span_rank": span_rank,
           "heldout_ce_learned": learned_ce, "heldout_ce_full": full_ce, "heldout_ce_random": rand_ce,
           "captured_learned": float(np.mean(capL)), "captured_pca": float(np.mean(capP)),
           "captured_random": float(np.mean(capR))}
    print(f"  held-out CE: learned k={big_k} {learned_ce:.3f}  full {full_ce:.3f}  random {rand_ce:.3f}")
    print(f"  persona-difference captured: learned {res['captured_learned']:.2f}  "
          f"(cos {res['captured_learned']**0.5:.2f})  |  PCA-{big_k} {res['captured_pca']:.2f}  "
          f"|  random {res['captured_random']:.3f}")
    return res


def make_plots(panel, out_dir, model_key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = panel["k"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(k, panel["learned_mean"], yerr=panel["learned_sd"], marker="o", lw=2, capsize=4,
                label="learned k-subspace")
    ax.plot(k, panel["random_mean"], "-s", color="darkorange", label="random k-subspace")
    ax.axhline(panel["full_diff_ref"], ls="--", color="gray", label="full-difference (P=I)")
    ax.axhline(panel["no_intervention"], ls=":", color="lightgray", label="no intervention (P=0)")
    ax.set_xlabel("subspace dimension k"); ax.set_ylabel("held-out CE of B's responses")
    ax.set_title(f"Exp 3 panel dimension ({model_key}): knee where learned CE plateaus")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "panel_dimension.png", dpi=140); plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = pl.get_config(args.model)
    out_dir = args.out_dir or (Path(__file__).resolve().parent / "results" / f"exp3_das{cfg.cache_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    k_list = [int(x) for x in args.k_list.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    print(f"loading {cfg.hf_id} for DAS at layer {cfg.steer_layer} ...")
    model, tokenizer, device = pl.load_model(args.model)
    for p in model.parameters():           # freeze: backward only reaches the learned W
        p.requires_grad_(False)

    pool = pl.load_trait_pool()
    names, Vp, Vc, centroid = pl.load_persona_frame(pool, args.model)
    pca = PCA(n_components=min(20, len(names))).fit(Vc.numpy())
    systems = json.loads((pl.DATA_DIR / "persona_pool_system_prompts.json").read_text())
    responses = json.loads((pl.DATA_DIR
                            / f"persona_pool_responses{cfg.cache_suffix}.json").read_text())
    questions = prompts.EVAL_QUESTIONS[:POOL_N_Q]

    E = Exp3(model, tokenizer, cfg, Vc, centroid, pca, names, systems, responses,
             questions, args.resp_tokens, args.eps)
    # disjoint train/test persona split, so the projector never sees the test personas' directions
    # -- a true generalization test of the shared causal subspace. --split-seed randomizes the
    # assignment within the strongest pool (removes the strong-vs-weak confound of the ranked split).
    ranked = [names[i] for i in Vc.norm(dim=1).argsort(descending=True).tolist()]
    if args.split_seed >= 0:
        import random as _random
        pick = _random.Random(args.split_seed).sample(ranked[: args.candidate_pool],
                                                       args.n_train + args.n_test)
        train_personas, test_personas = pick[: args.n_train], pick[args.n_train:]
    else:
        train_personas = ranked[: args.n_train]
        test_personas = ranked[args.n_train: args.n_train + args.n_test]
    print(f"device={device}  model={cfg.key}  site=layer {cfg.steer_layer}  split_seed={args.split_seed}")
    print(f"  train personas ({len(train_personas)}): {', '.join(train_personas)}")
    print(f"  test  personas ({len(test_personas)}): {', '.join(test_personas)}")

    if args.bisect:
        bres = bisect_dimension(E, train_personas, test_personas, args.n_test_pairs,
                                args.bisect_target, Vc.shape[1], seeds, args.steps, args.lr,
                                args.coeff, args.eps)
        out = {"params": {"experiment": "exp3_das_bisect", "model_key": cfg.key, "model": cfg.hf_id,
                          "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "das_layer": cfg.steer_layer, "n_train": args.n_train, "n_test": args.n_test,
                          "n_test_pairs": args.n_test_pairs, "split_seed": args.split_seed,
                          "seeds": seeds, "steps": args.steps, "coeff": args.coeff,
                          "train_personas": train_personas, "test_personas": test_personas},
               "bisect": bres}
        (out_dir / "exp3_bisect.json").write_text(json.dumps(out, indent=2))
        print(f"\ndone -> {out_dir}  (k* = {bres['k_star']} for {args.bisect_target:.0%})")
        return

    panel = panel_dimension(E, train_personas, test_personas, args.n_test_pairs, k_list, seeds,
                            args.steps, args.lr, args.coeff, args.eps)
    # pick the panel dimension as the smallest k within 1 SD of the best learned CE
    best = min(panel["learned_mean"])
    knee_k = next(k for k, m, sd in zip(k_list, panel["learned_mean"], panel["learned_sd"])
                  if m <= best + sd)
    print(f"\n=> estimated panel dimension (smallest k within 1 SD of best CE): k = {knee_k}")

    control = distinct = None
    if not args.panel_only:
        control = clean_run_control(E, train_personas[: args.n_personas], knee_k,
                                    args.steps, args.lr, args.coeff, args.eps)
        distinct = distinct_interface(E, args.big_n, args.big_k, args.big_steps, args.lr, args.coeff, args.eps)

    params = {
        "experiment": "exp3_das", "model_key": cfg.key, "model": cfg.hf_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "das_layer": cfg.steer_layer, "n_train": args.n_train, "n_test": args.n_test,
        "n_test_pairs": args.n_test_pairs, "split_seed": args.split_seed,
        "train_personas": train_personas, "test_personas": test_personas,
        "n_personas": args.n_personas, "k_list": k_list, "seeds": seeds,
        "steps": args.steps, "lr": args.lr, "coeff": args.coeff, "resp_tokens": args.resp_tokens,
        "big_n": args.big_n, "big_k": args.big_k, "big_steps": args.big_steps,
        "panel_only": args.panel_only,
        "estimated_panel_dim": knee_k, "torch": t.__version__, "python": platform.python_version(),
    }
    results = {"params": params, "panel_dimension": panel}
    if control is not None:
        results["clean_run_control"] = control
    if distinct is not None:
        results["distinct_interface"] = distinct
    (out_dir / "exp3_results.json").write_text(json.dumps(results, indent=2))
    make_plots(panel, out_dir, cfg.key)
    print(f"\ndone -> {out_dir}  (estimated panel dimension k={knee_k})")


if __name__ == "__main__":
    main()
