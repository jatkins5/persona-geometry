"""Experiment 2 -- the reachable persona manifold (port of notebook 7e-7h), model-parameterized.

Steering moves the residual stream along directions in PCA persona-space; how far can it push
the model's *behaviour* before it (a) stops moving (saturation) or (b) loses coherence? We rotate
a unit steering direction through a PC plane, sweep its magnitude, and for each record a coherence
proxy and the LANDED radius -- how far the generated text's own persona actually reached along that
direction. The coherent boundary traces the reachable region (Qwen's was a bounded teardrop).

Two modes:
  --mode calibrate : sweep magnitude on the +PC1 direction for one question and report coherence +
                     landed PC1 + sample text. Use this FIRST on a new model to find the magnitude
                     range (the notebook's Qwen MAGS=20..160 are Qwen-scale and won't transfer).
  --mode boundary  : the full boundary map over PC planes -> per-plane boundary radius + crossover
                     magnitude, 2D figures over the persona cloud, and the 3D envelope.

coherence proxy (notebook 7e/7f): distinct = unique/total words (low = repetition);
                                  ascii   = fraction of plain-English chars (low = code/CJK breakdown).
"""
from __future__ import annotations

import argparse
import csv
import json
import string
from pathlib import Path

import numpy as np
import torch as t
from sklearn.decomposition import PCA
from tqdm import tqdm

import persona_lib as pl
import prompts

# Coherent if distinct words, mostly plain ASCII, AND mostly real English words. The English-word
# fraction is the load-bearing one: Qwen's distinct+ascii proxy gave false positives on Llama's
# failure modes (single-char repetition -> distinct=1.0; latin-heavy token soup -> ascii>0.85).
COH_DISTINCT, COH_ASCII, COH_ENGLISH = 0.5, 0.85, 0.5


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama3.1-8b", choices=list(pl.MODELS))
    ap.add_argument("--mode", default="calibrate",
                    choices=["calibrate", "boundary", "reachability"])
    ap.add_argument("--n-sample", type=int, default=40,
                    help="reachability mode: number of personas to probe (stratified by own norm)")
    ap.add_argument("--mags", default=None,
                    help="comma-separated steering magnitudes; default depends on mode")
    ap.add_argument("--planes", default="0,1;0,2;1,2",
                    help="boundary mode: ';'-separated PC index pairs, e.g. '0,1;0,2;1,2'")
    ap.add_argument("--n-angles", type=int, default=24)
    ap.add_argument("--num-q", type=int, default=1,
                    help="questions per (angle,magnitude); >1 denoises via median radius / majority coherence")
    ap.add_argument("--max-new-tokens", type=int, default=110)
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap.parse_args()


def coherence(text: str) -> tuple[float, float, float]:
    """(distinct, ascii, english) coherence proxies. english = fraction of whitespace tokens that
    are real English words (wordfreq zipf > 0) -- the robust catch for token-soup / repetition."""
    import wordfreq
    words = text.split()
    if not words:
        return 0.0, 0.0, 0.0
    distinct = len(set(words)) / len(words)
    ascii_ok = sum((c.isascii() and (c.isalpha() or c.isspace())) for c in text) / max(1, len(text))
    cleaned = [w.strip(string.punctuation).lower() for w in words]
    cleaned = [w for w in cleaned if w]
    english = (sum(wordfreq.zipf_frequency(w, "en") > 0 for w in cleaned) / len(cleaned)
               if cleaned else 0.0)
    return distinct, ascii_ok, english


def is_coherent(distinct: float, ascii_ok: float, english: float) -> bool:
    return distinct > COH_DISTINCT and ascii_ok > COH_ASCII and english > COH_ENGLISH


def build_pca_frame(model_key: str):
    """Load persona vectors and fit the PCA frame (centroid + components), matching notebook 6e."""
    pool_traits = pl.load_trait_pool()
    names, Vp, Vc, centroid = pl.load_persona_frame(pool_traits, model_key)
    pca = PCA(n_components=min(20, len(names)))
    coords = pca.fit_transform(Vc.numpy())
    components = t.tensor(pca.components_, dtype=t.float32)   # (n_pc, d) orthonormal directions
    evr = pca.explained_variance_ratio_
    return names, Vp, Vc, centroid, pca, coords, components, evr


def run_calibrate(model, tokenizer, cfg, centroid, pca, components, evr, mags, max_new_tokens, out_dir):
    """Sweep magnitude on the +PC1 unit direction; report coherence + landed PC1 + sample text."""
    question = prompts.EVAL_QUESTIONS[0]
    pc1_dir = components[0] / components[0].norm()
    print(f"PCA variance (PC1-5): {np.round(evr[:5], 3)}")
    print(f"calibrating on +PC1, question: {question}\n")
    print(f"{'mag':>6} {'distinct':>8} {'ascii':>6} {'english':>7} {'landedPC1':>9}  sample")
    rows = []
    for M in mags:
        text = pl.generate_with_steerer(model, tokenizer, question, pc1_dir, cfg.steer_layer,
                                        float(M), max_new_tokens)
        act = pl.extract_response_activations(
            model, tokenizer, [""], [question], [text], cfg.persona_layer)[0].float()
        landed = pca.transform((act - centroid).numpy()[None])[0]
        d, a, e = coherence(text)
        coh = is_coherent(d, a, e)
        flag = "" if coh else "  <-- breaking down"
        print(f"{M:>6} {d:>8.2f} {a:>6.2f} {e:>7.2f} {landed[0]:>9.1f}  {text[:80]!r}{flag}")
        rows.append({"magnitude": M, "distinct": d, "ascii": a, "english": e,
                     "landed_pc1": float(landed[0]), "coherent": bool(coh), "text": text})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(json.dumps(rows, indent=2))
    coherent_mags = [r["magnitude"] for r in rows if r["coherent"]]
    print(f"\ncoherent magnitudes: {coherent_mags}")
    print("-> pick a MAGS grid spanning from where landedPC1 starts to grow up to the breakdown point,"
          "\n   then re-run with --mode boundary --mags ...")


def run_boundary(model, tokenizer, cfg, names, centroid, pca, coords, components,
                 planes, n_angles, mags, num_q, max_new_tokens, out_dir):
    """Rotate a unit direction through each PC plane, sweep magnitude, map the coherent boundary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = prompts.EVAL_QUESTIONS[:num_q]
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    log_fp = open(out_dir / "boundary_logs.csv", "w", newline="")
    log_w = csv.DictWriter(log_fp, fieldnames=["plane", "angle_deg", "magnitude", "landed_i",
                                               "landed_j", "radius", "distinct", "ascii",
                                               "english", "coherent", "response"])
    log_w.writeheader()

    for (i, j) in planes:
        boundary_r = np.zeros(n_angles)               # max coherent landed-radius per angle
        crossover_M = np.full(n_angles, np.nan)       # magnitude where coherence first breaks
        for k, theta in enumerate(tqdm(angles, desc=f"PC{i+1}-PC{j+1}")):
            coord = np.zeros(components.shape[0]); coord[i], coord[j] = np.cos(theta), np.sin(theta)
            direction = t.tensor(coord, dtype=t.float32) @ components
            direction = direction / direction.norm()
            for M in mags:
                radii, cohs, lis, ljs, ds, as_, es = [], [], [], [], [], [], []
                for q in questions:
                    text = pl.generate_with_steerer(model, tokenizer, q, direction, cfg.steer_layer,
                                                    float(M), max_new_tokens)
                    d, a, e = coherence(text)
                    act = pl.extract_response_activations(
                        model, tokenizer, [""], [q], [text], cfg.persona_layer)[0].float()
                    lc = pca.transform((act - centroid).numpy()[None])[0]
                    radii.append(float(lc[i] * np.cos(theta) + lc[j] * np.sin(theta)))
                    lis.append(float(lc[i])); ljs.append(float(lc[j]))
                    ds.append(d); as_.append(a); es.append(e)
                    cohs.append(is_coherent(d, a, e))
                radius = float(np.median(radii))
                coherent = sum(cohs) > num_q / 2
                log_w.writerow({"plane": f"{i},{j}", "angle_deg": float(np.degrees(theta)),
                                "magnitude": M, "landed_i": float(np.median(lis)),
                                "landed_j": float(np.median(ljs)), "radius": radius,
                                "distinct": float(np.mean(ds)), "ascii": float(np.mean(as_)),
                                "english": float(np.mean(es)), "coherent": coherent, "response": text})
                log_fp.flush()
                if coherent:
                    boundary_r[k] = max(boundary_r[k], radius)
                elif np.isnan(crossover_M[k]):
                    crossover_M[k] = M
        stem = f"boundary_map_PC{i+1}-PC{j+1}"
        (out_dir / f"{stem}.json").write_text(json.dumps(
            {"plane": [i, j], "angles_deg": np.degrees(angles).tolist(),
             "boundary_radius": boundary_r.tolist(), "crossover_M": crossover_M.tolist()}, indent=2))
        _plot_plane(coords, i, j, angles, boundary_r, out_dir / f"{stem}.png")
    log_fp.close()
    _plot_3d(coords, planes, out_dir)
    print(f"\ndone -> {out_dir}")


def _plot_plane(coords, i, j, angles, boundary_r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bx = np.append(boundary_r * np.cos(angles), boundary_r[0] * np.cos(angles[0]))
    by = np.append(boundary_r * np.sin(angles), boundary_r[0] * np.sin(angles[0]))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(coords[:, i], coords[:, j], s=12, alpha=0.4, color="steelblue", label="personas")
    ax.plot(bx, by, "-o", color="crimson", ms=3, lw=1.5, label="coherent reach")
    ax.set_xlabel(f"PC{i+1}"); ax.set_ylabel(f"PC{j+1}"); ax.set_aspect("equal")
    ax.set_title(f"Coherent reachable boundary in PC{i+1}-PC{j+1} (steering)"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _plot_3d(coords, planes, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=6, alpha=0.3, color="steelblue")
    colors = {(0, 1): "crimson", (0, 2): "seagreen", (1, 2): "darkorange"}
    for (i, j) in planes:
        jp = out_dir / f"boundary_map_PC{i+1}-PC{j+1}.json"
        if not jp.exists():
            continue
        dd = json.loads(jp.read_text())
        ang = np.radians(dd["angles_deg"]); r = np.array(dd["boundary_radius"])
        P = np.zeros((len(ang), 3)); P[:, i], P[:, j] = r * np.cos(ang), r * np.sin(ang)
        P = np.vstack([P, P[0]])
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color=colors.get((i, j), "k"), lw=2,
                label=f"PC{i+1}-PC{j+1}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title("Coherent reachable boundary: 3 plane cross-sections"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "reachable_boundary_3d.png", dpi=140); plt.close(fig)


def run_reachability(model, tokenizer, cfg, names, Vc, centroid, mags, n_sample,
                     max_new_tokens, out_dir):
    """Steer DIRECTLY toward each persona's own (full-dimensional) direction and measure how far
    behaviour lands along that direction before coherence breaks -- the clean per-persona test of
    'can steering reach this persona?', free of the boundary map's 2D-plane projection artifact.

    For persona x: own = ||Vc[x]|| is exactly x's coordinate along its own unit direction u_x; a
    response that lands at projection ~own has been steered all the way to x. We sweep magnitude,
    take the furthest COHERENT landing, and report reach_ratio = best_coherent_landing / own.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    question = prompts.EVAL_QUESTIONS[0]
    norms = Vc.norm(dim=1)
    order = norms.argsort()                                  # ascending own-norm
    sample = order[t.linspace(0, len(order) - 1, n_sample).long()].tolist()

    log_fp = open(out_dir / "reachability_logs.csv", "w", newline="")
    log_w = csv.DictWriter(log_fp, fieldnames=["persona", "own_norm", "magnitude", "landed",
                                               "distinct", "ascii", "english", "coherent", "response"])
    log_w.writeheader()
    rows = []
    for xi in tqdm(sample, desc="reachability"):
        own = float(norms[xi])
        u = (Vc[xi] / norms[xi])
        best_landed = 0.0
        best_M = None
        for M in mags:
            text = pl.generate_with_steerer(model, tokenizer, question, u, cfg.steer_layer,
                                            float(M), max_new_tokens)
            act = pl.extract_response_activations(
                model, tokenizer, [""], [question], [text], cfg.persona_layer)[0].float()
            landed = float((act - centroid) @ u)            # extent along the persona's own direction
            d, a, e = coherence(text)
            coh = is_coherent(d, a, e)
            log_w.writerow({"persona": names[xi], "own_norm": own, "magnitude": M, "landed": landed,
                            "distinct": d, "ascii": a, "english": e, "coherent": coh, "response": text})
            log_fp.flush()
            if coh and landed > best_landed:
                best_landed, best_M = landed, M
        rows.append({"persona": names[xi], "own_norm": own, "best_coherent_landed": best_landed,
                     "reach_ratio": best_landed / own if own else 0.0, "best_M": best_M})
    log_fp.close()

    with open(out_dir / "reachability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    ratios = np.array([r["reach_ratio"] for r in rows])
    owns = np.array([r["own_norm"] for r in rows])
    print(f"\nreachability over {len(rows)} personas (own-norm range {owns.min():.1f}-{owns.max():.1f}):")
    print(f"  median reach_ratio = {np.median(ratios):.2f}   "
          f"fraction reaching >=80% of own location = {(ratios >= 0.8).mean():.0%}")
    from scipy import stats
    rho, p = stats.spearmanr(owns, ratios)
    print(f"  Spearman(own_norm, reach_ratio) = {rho:+.2f} (p={p:.2g})  "
          f"[strong negative => outer personas genuinely harder to reach]")
    _plot_reachability(rows, out_dir / "reachability.png")
    print(f"done -> {out_dir}")


def _plot_reachability(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    owns = np.array([r["own_norm"] for r in rows])
    landed = np.array([r["best_coherent_landed"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 6))
    lim = max(owns.max(), landed.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="full reach (landed = own location)")
    ax.scatter(owns, landed, s=30, alpha=0.8, color="crimson")
    ax.set_xlabel("persona's own location  ||Vc||  (how far out it sits)")
    ax.set_ylabel("furthest COHERENT steered landing toward it")
    ax.set_title("Per-persona steering reachability (aiming at each persona directly)")
    ax.legend(); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = pl.get_config(args.model)
    out_dir = args.out_dir or (Path(__file__).resolve().parent / "results"
                               / f"exp2_manifold{cfg.cache_suffix}")
    print(f"loading {cfg.hf_id} + PCA frame ...")
    model, tokenizer, device = pl.load_model(args.model)
    names, Vp, Vc, centroid, pca, coords, components, evr = build_pca_frame(args.model)
    print(f"device={device}  model={cfg.key}  steer_layer={cfg.steer_layer}  N={len(names)}")

    if args.mode == "calibrate":
        mags = [float(x) for x in args.mags.split(",")] if args.mags \
            else [10, 20, 40, 80, 120, 160, 240, 320]
        run_calibrate(model, tokenizer, cfg, centroid, pca, components, evr, mags,
                      args.max_new_tokens, out_dir)
    elif args.mode == "reachability":
        mags = [float(x) for x in args.mags.split(",")] if args.mags else [4, 8, 12, 16, 20, 24]
        run_reachability(model, tokenizer, cfg, names, Vc, centroid, mags, args.n_sample,
                         args.max_new_tokens, out_dir)
    else:
        if not args.mags:
            raise SystemExit("--mode boundary requires --mags (run --mode calibrate first)")
        mags = [float(x) for x in args.mags.split(",")]
        planes = [tuple(int(x) for x in p.split(",")) for p in args.planes.split(";")]
        run_boundary(model, tokenizer, cfg, names, centroid, pca, coords, components,
                     planes, args.n_angles, mags, args.num_q, args.max_new_tokens, out_dir)


if __name__ == "__main__":
    main()
