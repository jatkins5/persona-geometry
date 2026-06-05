"""Regenerate the whole presentation figure deck from cached results.

    python figures/make_figures.py            # all figures (incl. the slow rotating GIFs)
    python figures/make_figures.py --no-video  # skip the GIFs (fast, static figures only)

Reads only files in ``data/`` (the persona-vector caches, the Exp-2 boundary maps, and the
recorded ``exp1_results.json`` / ``exp3_results.json``). Run from the repo root. GIFs need ffmpeg.
"""
import argparse
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch as t
from sklearn.decomposition import PCA
from nltk.corpus import wordnet as wn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------- shared: the adjective pool + PCA
def load_pool():
    """Reproduce the analyzed 220-adjective pool and its RAW PCA frame (matches the boundary maps)."""
    rel = json.loads((DATA / "persona_pool_relevance.json").read_text())
    traits_kept = {w: s for w, s in rel["traits"].items() if s >= 0.85}
    pregen = set(json.loads((DATA / "persona_pool_system_prompts.json").read_text()))
    roles_all = set(rel["roles"])

    def dedup(scored, pos, prefer, exclude, seed=0):
        words = set(scored); parent = {w: w for w in words}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        rv = lambda w: random.Random(f"{seed}:{w}").random()
        for w in sorted(words, key=rv):
            for sy in wn.synsets(w):
                if sy.pos() not in pos:
                    continue
                sib = [x for x in (l.name().replace("_", " ").lower() for l in sy.lemmas()) if x in words]
                for x in sib[1:]:
                    parent[find(x)] = find(sib[0])
        groups = defaultdict(list)
        for w in words:
            groups[find(w)].append(w)
        mk = lambda m: (m not in prefer, -scored[m], rv(m)); out = []
        for r, mem in groups.items():
            u = [m for m in mem if m not in exclude]
            if u:
                out.append((min(u, key=mk), max(scored[m] for m in u), rv(r)))
        out.sort(key=lambda z: (-z[1], z[2]))
        return [r for r, _, _ in out]

    pool_traits = set(dedup(traits_kept, {"a", "s"}, pregen, roles_all)[:220])
    pv = t.load(DATA / "persona_pool_vectors.pt", map_location="cpu")
    names = [n for n in pv if n in pool_traits]
    Vc = (lambda V: V - V.mean(0, keepdim=True))(t.stack([pv[n].float() for n in names]))
    pca = PCA(10)
    coords = pca.fit_transform(Vc.numpy())
    return names, coords, Vc.norm(dim=1).numpy(), pca.explained_variance_ratio_


def annot(ax, bars, fmt="{:.2f}", dy=0.01, size=8):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + (dy if h >= 0 else -dy * 3), fmt.format(h),
                ha="center", va="bottom" if h >= 0 else "top", fontsize=size)


# ---------------------------------------------------------------- 1. labeled rotating persona cloud
def fig_cloud(coords, norms, evr, video):
    POLES = {0: (["snarly", "sardonic", "facetious", "playful", "hysterical"],
                 ["impartial", "scholarly", "unimpassioned", "unbigoted"], "mocking, expressive", "dispassionate, neutral"),
             1: (["flowery", "erudite", "sophisticated", "cryptic", "arty"],
                 ["doltish", "surly", "slovenly", "untutored"], "ornate, erudite", "crude, uncouth"),
             2: (["bubbly", "cheery", "frolicsome", "winsome", "maternal"],
                 ["surly", "coldhearted", "psychopathic", "blase"], "warm, nurturing", "cold, hostile")}
    idx = {n: i for i, n in enumerate(NAMES)}
    co = coords.copy()
    for ax_i, (plus, minus, *_) in POLES.items():
        if np.mean([co[idx[w], ax_i] for w in plus if w in idx]) < np.mean([co[idx[w], ax_i] for w in minus if w in idx]):
            co[:, ax_i] *= -1
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    fig = plt.figure(figsize=(8.5, 9.0), dpi=120)
    ax = fig.add_axes([0.0, 0.20, 1.0, 0.76], projection="3d")
    sc = ax.scatter(x, y, z, c=norms, cmap="viridis", s=22, depthshade=True,
                    edgecolors="white", linewidths=0.25, alpha=0.95)
    ax.set_xlabel(f"PC1 · affect ({evr[0]:.0%})"); ax.set_ylabel(f"PC2 · register ({evr[1]:.0%})")
    ax.set_zlabel(f"PC3 · warmth ({evr[2]:.0%})")
    ax.set_title("Qwen persona space — 220 adjective personas", pad=4, fontsize=13)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([]); axis.pane.set_alpha(0.04)
    ax.grid(True, alpha=0.2); ax.set_box_aspect((1, 1, 1))
    lines = [f"PC1  ({evr[0]:.0%})   affect / derision      {POLES[0][2]}   ↔   {POLES[0][3]}",
             f"PC2  ({evr[1]:.0%})   cultivation / register   {POLES[1][2]}   ↔   {POLES[1][3]}",
             f"PC3   ({evr[2]:.0%})   warmth                       {POLES[2][2]}   ↔   {POLES[2][3]}"]
    fig.text(0.5, 0.135, "\n".join(lines), ha="center", va="top", family="DejaVu Sans Mono",
             fontsize=10.5, linespacing=1.7, bbox=dict(boxstyle="round,pad=0.6", fc="#f4f4f4", ec="#bbb"))
    fig.text(0.5, 0.015, "color = distinctiveness (‖vector‖)", ha="center", fontsize=9, color="#555")
    fig.savefig(OUT / "persona_cloud_labeled.png", dpi=130)
    if video:
        _spin(fig, ax, OUT / "persona_cloud_labeled.gif", fps=12, scale=560)
    plt.close(fig)


# ---------------------------------------------------------------- 2. Experiment 1 (from exp1_results.json)
def fig_exp1():
    r = json.loads((DATA / "exp1_results.json").read_text())
    T = r["traits"]; x = np.arange(len(T))
    C = {"hi": "#b2182b", "lo": "#2166ac", "g": "#1a9850", "grey": "#999"}

    fig, ax = plt.subplots(figsize=(10, 5.4)); w = 0.27
    for off, key, col, lab in [(-w, "prompt_mean", C["hi"], "prompt: mean over tokens"),
                               (0, "prompt_last", C["lo"], "prompt: generation-entry token"),
                               (w, "response", C["g"], "response")]:
        annot(ax, ax.bar(x + off, r[key], w, label=lab, color=col))
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(T)
    ax.set_ylabel("contamination ratio\n(0 = clean, 1 = word alone fires the axis)")
    ax.set_title("Mentioning a trait's word barely moves its axis", fontsize=13.5, pad=24)
    ax.text(0.5, 1.03, "the high prompt-mean is only the literal word token — the generation-entry state is clean",
            transform=ax.transAxes, ha="center", fontsize=10.5, style="italic", color="#333")
    ax.legend(frameon=False, fontsize=10, loc="upper right"); ax.set_ylim(-0.12, 0.72)
    fig.tight_layout(); fig.savefig(OUT / "exp1_contamination.png", dpi=150); plt.close(fig)

    lex = (np.array(r["cos_word_persona"]) ** 2) * 100; res = 100 - lex
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(x, res, 0.6, label="behavioral residual  (1 − cos²)", color=C["lo"])
    ax.bar(x, lex, 0.6, bottom=res, label="word-concept  (cos²)", color=C["hi"])
    for xi, rr in zip(x, res):
        ax.text(xi, rr - 4, f"{rr:.0f}%", ha="center", va="top", color="white", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(T); ax.set_ylabel("share of persona-vector energy"); ax.set_ylim(0, 105)
    ax.set_title("A persona vector is ~91–98% behavioral residual, not the word\n"
                 "decomposed into its word-concept component vs the rest", fontsize=12.5)
    ax.legend(frameon=False, fontsize=10, loc="lower center", ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "exp1_decomposition.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2)); w = 0.27
    for off, key, col, lab in [(-w, "steer_full", C["grey"], "full persona vector"),
                               (0, "steer_word", C["hi"], "word-concept component"),
                               (w, "steer_residual", C["lo"], "behavioral residual")]:
        annot(ax, ax.bar(x + off, r[key], w, label=lab, color=col), fmt="{:.1f}", dy=0.12)
    ax.set_xticks(x); ax.set_xticklabels(T); ax.set_ylim(0, 10.5)
    ax.set_ylabel("trait expression (0–10, LLM judge)")
    ax.set_title("Steering with the residual reproduces the trait; the word-concept does not\n"
                 "norm-matched, coeff = 1.5", fontsize=12.5)
    ax.legend(frameon=False, fontsize=10, loc="upper right", ncol=3)
    for tr in r.get("steer_control_failed", []):
        if tr in T:
            ax.annotate("positive control failed\n(discount)", xy=(T.index(tr), r["steer_full"][T.index(tr)]),
                        xytext=(T.index(tr) - 0.45, 4.2), fontsize=8, color=C["grey"], ha="center",
                        arrowprops=dict(arrowstyle="->", color=C["grey"]))
    fig.tight_layout(); fig.savefig(OUT / "exp1_steering.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 3. Experiment 2 steering regimes
def fig_exp2_regimes():
    d = json.load(open(DATA / "boundary_map_PC1-PC2.json"))
    by = defaultdict(list)
    for s in d["samples"]:
        by[round(s[0])].append(s)
    SHOW = [(0, "flare · +PC1 (0°) — steering works", "#c0392b"),
            (90, "+PC2 (90°) — partial reach", "#16a085"),
            (180, "peak · −PC1 (180°) — stuck, no movement", "#2c3e50"),
            (315, "+PC1/−PC2 corner (315°) — reaches, then breaks", "#8e44ad")]
    fig, ax = plt.subplots(figsize=(10, 6))
    for ang, label, color in SHOW:
        rows = sorted(by[ang], key=lambda r: r[1])
        M = np.array([r[1] for r in rows]); rad = np.array([r[4] for r in rows]); coh = np.array([bool(r[5]) for r in rows])
        ax.plot(M, rad, "-", color=color, lw=2.4, label=label, zorder=2)
        ax.scatter(M[coh], rad[coh], s=55, color=color, zorder=3, edgecolors="white", linewidths=0.8)
        if (~coh).any():
            ax.scatter(M[~coh], rad[~coh], s=130, color="#c0392b", marker="x", linewidths=2.6, zorder=4)
    ax.annotate("reaches the flare tip\n(steering fills the sparse open end)", xy=(160, 32.8), xytext=(96, 33.5),
                fontsize=10, color="#c0392b", ha="left", arrowprops=dict(arrowstyle="->", color="#c0392b"))
    ax.annotate("pinned at the dense peak's edge —\nno movement at any magnitude", xy=(100, 6.7), xytext=(40, 13.5),
                fontsize=10, color="#2c3e50", ha="left", arrowprops=dict(arrowstyle="->", color="#2c3e50"))
    ax.annotate("degenerates into\nrepetition (×)", xy=(130, 24.0), xytext=(132, 14.5), fontsize=9.5,
                color="#8e44ad", ha="center", arrowprops=dict(arrowstyle="->", color="#8e44ad"))
    ax.set_xlabel("steering magnitude  M")
    ax.set_ylabel("distance travelled along the pushed direction\n(landed radius)")
    ax.set_title("Steering fills the sparse flare but can't move past the dense peak", fontsize=14, pad=22)
    ax.text(0.5, 1.02, "PC1–PC2 plane · one curve per steering direction · × = incoherent output",
            transform=ax.transAxes, ha="center", fontsize=10.5, style="italic", color="#444")
    ax.legend(frameon=False, fontsize=10.5, loc="upper left"); ax.set_ylim(-4, 36); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "exp2_steering_regimes.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 4. Experiment 2 rotating 3D envelope
def fig_boundary_3d(coords, norms, evr, video):
    fig = plt.figure(figsize=(8.5, 9.0), dpi=120)
    ax = fig.add_axes([0.0, 0.13, 1.0, 0.82], projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=norms, cmap="viridis", s=20,
               depthshade=True, alpha=0.8, edgecolors="none", zorder=1)
    rings = []
    for fname, color, (i, j) in [("PC1-PC2", "#c0392b", (0, 1)), ("PC1-PC3", "#27ae60", (0, 2)), ("PC2-PC3", "#2980b9", (1, 2))]:
        d = json.load(open(DATA / f"boundary_map_{fname}.json"))
        ang = np.radians(d["angles_deg"] + [d["angles_deg"][0]]); r = np.array(d["boundary_radius"] + [d["boundary_radius"][0]])
        P = np.zeros((len(r), 3)); P[:, i] = r * np.cos(ang); P[:, j] = r * np.sin(ang); rings.append(P)
        ax.plot(P[:, 0], P[:, 1], P[:, 2], "-o", color=color, lw=2.2, ms=3.0, label=f"{fname} reach ring", zorder=4)
    allp = np.vstack([coords[:, :3]] + rings); lo, hi = allp.min(0) - 3, allp.max(0) + 3
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2]); ax.set_box_aspect(hi - lo)
    ax.set_xlabel(f"PC1 ({evr[0]:.0%})"); ax.set_ylabel(f"PC2 ({evr[1]:.0%})"); ax.set_zlabel(f"PC3 ({evr[2]:.0%})")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([]); axis.pane.set_alpha(0.04)
    ax.grid(True, alpha=0.2); ax.set_title("Coherent reachable boundary over the persona cloud", pad=2, fontsize=13)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.text(0.5, 0.075, "three orthogonal cross-sections of the reachable envelope — a teardrop, long toward the +PC1 flare",
             ha="center", fontsize=10, style="italic", color="#333")
    fig.savefig(OUT / "boundary_envelope.png", dpi=130)
    if video:
        _spin(fig, ax, OUT / "boundary_envelope_rotating.gif", fps=11, scale=560, elev=16)
    plt.close(fig)


# ---------------------------------------------------------------- 5. Experiment 3 (from exp3_results.json)
def fig_exp3():
    r = json.loads((DATA / "exp3_results.json").read_text())
    pd = r["panel_dim"]; k = pd["k"]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.errorbar(k, pd["learned_loss_mean"], yerr=pd["learned_loss_sd"], fmt="-o", color="#2166ac",
                lw=2.2, capsize=3, label="learned k-subspace")
    ax.plot(k, pd["random_loss_mean"], "-o", color="#b2182b", lw=2.2, label="random k-subspace")
    ax.axhline(pd["full_diff_ref"], ls="--", color="gray", label="full-difference (P=I)")
    ax.set_xlabel("subspace dimension k"); ax.set_ylabel("held-out CE of B's responses")
    ax.set_title("DAS panel dimension: held-out interchange loss vs k", fontsize=13.5, pad=22)
    ax.text(0.5, 1.02, "learned loss plateaus by k≈5–6 (the panel dim); random fails at every k → low-dim AND structured",
            transform=ax.transAxes, ha="center", fontsize=10, style="italic", color="#444")
    ax.annotate("knee ≈ 5–6", xy=(6, pd["learned_loss_mean"][4]), xytext=(7.5, 1.55),
                fontsize=10, color="#2166ac", arrowprops=dict(arrowstyle="->", color="#2166ac"))
    ax.legend(frameon=False, fontsize=11, loc="center right"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "exp3_panel_dimension.png", dpi=150); plt.close(fig)

    iv = r["interface_vs_pca"]; groups = ["8 personas\n(span ~7-dim)", "32 personas\n(span ~31-dim)"]
    xg = np.arange(2); w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for off, key, col, lab in [(-w, "pca6", "#999999", "top-6 PCA (variance-optimal)"),
                               (0, "learned", "#2166ac", "learned causal D (k=6)"),
                               (w, "random", "#b2182b", "random 6-dim")]:
        vals = [iv[n][key] for n in ("n8", "n32")]
        bars = ax.bar(xg + off, vals, w, label=lab, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.0%}" if v >= 0.01 else "~0", ha="center", fontsize=9)
    ax.set_xticks(xg); ax.set_xticklabels(groups); ax.set_ylim(0, 0.85)
    ax.set_ylabel("fraction of the persona difference kept")
    ax.set_title("The causal interface is distinct from the persona representation", fontsize=13.5, pad=22)
    ax.text(0.5, 1.02, "learned D keeps far less of the persona vector than a variance-optimal subspace — yet steers better",
            transform=ax.transAxes, ha="center", fontsize=10, style="italic", color="#444")
    ax.legend(frameon=False, fontsize=10.5, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "exp3_interface_vs_pca.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- rotating-GIF helper (mp4 -> palette gif)
def _spin(fig, ax, out_gif, fps, scale, elev=18):
    tmp = OUT / "_spin_tmp.mp4"
    FuncAnimation(fig, lambda az: ax.view_init(elev=elev, azim=az) or (), frames=np.arange(0, 360, 1),
                  interval=33, blit=False).save(str(tmp), writer=FFMpegWriter(fps=30, bitrate=4800), dpi=120)
    pal = OUT / "_spin_pal.png"
    vf = f"fps={fps},scale={scale}:-1:flags=lanczos"
    subprocess.run(f"ffmpeg -y -i {tmp} -vf '{vf},palettegen=stats_mode=full' {pal}",
                   shell=True, check=True, capture_output=True)
    subprocess.run(f"ffmpeg -y -i {tmp} -i {pal} -lavfi '{vf}[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=4' "
                   f"-loop 0 {out_gif}", shell=True, check=True, capture_output=True)
    tmp.unlink(missing_ok=True); pal.unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true", help="skip the rotating GIFs (fast)")
    video = not ap.parse_args().no_video

    NAMES, COORDS, NORMS, EVR = load_pool()
    print(f"pool: {len(NAMES)} adjectives | PC1={EVR[0]:.0%} PC2={EVR[1]:.0%} PC3={EVR[2]:.0%}\n")
    fig_cloud(COORDS, NORMS, EVR, video)
    fig_exp1()
    fig_exp2_regimes()
    fig_boundary_3d(COORDS, NORMS, EVR, video)
    fig_exp3()
    for p in sorted(OUT.glob("*.png")) + sorted(OUT.glob("*.gif")):
        print("  wrote", p.relative_to(ROOT))
