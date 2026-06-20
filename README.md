# persona-geometry

Geometry, steering, and **causal structure** of persona vectors in **Qwen2.5-7B-Instruct**,
extracted as a standalone project from the ARENA `[4.4] LLM Psychology & Persona Vectors`
exercises.

A persona vector is the mean residual-stream activation while the model role-plays a trait
("You are surly / cryptic / flowery…"). This project builds a large pool of them, maps the
geometry of the space they live in, and runs three experiments asking: are the persona axes
*clean abstractions*, what does the *reachable* persona space look like under steering, and is
persona *controlled* through a low-dimensional causal interface.

---

## The persona pool (Section 6)

Built fully automatically from WordNet, with Qwen judging relevance and generating prompts:

```
WordNet adjectives → local-Qwen relevance filter (P(Yes) at the answer token)
   → WordNet-synset synonym dedup → seeded unbiased selection of 220 traits
   → few-shot Qwen system-prompt generation → positive-only response activations (layer 20)
   → center on the pool mean → PCA + cosine geometry
```

Two construction choices that mattered:
- **Single part of speech.** PC1 of a mixed noun+adjective pool was just tracking *part of
  speech* (all adjectives on one side, all nouns on the other) — a confound masquerading as an
  axis. The pool is restricted to adjectives; the noun/role vectors are kept cached for a future
  comparison.
- **Seeded tie-break, not alphabetical.** Hundreds of adjectives saturate the relevance score at
  P(Yes)=1.0, so "top-k" is decided entirely by the tie-break. An alphabetical one secretly
  selected early-alphabet words and halved negation/excess-prefix traits (un-/non-/in-…); a
  seeded per-word random tie-break removes that bias.

**Geometry:** PC1 ≈ 47%, PC2 ≈ 13%, PC3 ≈ 7% of variance — a cone, with a dense "everyday"
cluster at low PC1 flaring out into distinctive personas. Read of the axes:
**PC1 = distinctive ↔ vacuous, PC2 = ornate ↔ plain, PC3 = poetic/mystical ↔ pseudo-academic.**

### Prompt-noise robustness (Section 6h–6j)

A persona vector is a mean over only a handful of eliciting questions, so before reading anything
off the geometry we ask: **is a persona's location, and are the PCA axes themselves, an artifact of
which prompts happened to elicit them?** Three checks, all run in the *fixed* 6e PCA frame so the
results overlay the map directly:

- **Per-question vectors (6h).** Instead of averaging, keep each eliciting question's mean response
  activation separately (forward passes only, no new generation), giving a small cloud of points
  per persona to measure spread over.
- **Location wander (6i).** Bootstrap-resample each persona's questions (B = 300), project each
  resampled mean into the 6e PCA frame, and take the per-PC standard deviation — this is the
  *prompt-noise radius*, plotted as error bars on the map. The **median noise radius (PC1–3) is small
  relative to the between-persona spread**, i.e. personas don't move far enough under prompt
  resampling to swap places — the layout is real, not a prompt accident. A complementary **split-half
  direction reliability** (cosine between the mean directions of two disjoint halves of a persona's
  questions; 1 = a stable direction, 0 = noise) confirms most personas have a consistent direction,
  and flags the few that don't.
- **Axis stability (6j).** The sharper test: treat each persona as Gaussian about its location with
  its own per-PC bootstrap std (from 6i), Monte-Carlo resample *all* points (200 draws), refit PCA
  each time, and measure how far each top axis rotates from its 6e direction (greedy max-|cos|
  matching) and how its explained-variance moves. **Result: the three axes we actually interpret are
  stable — PC1 rotates ≈ 1.5°, PC2 ≈ 4°, PC3 ≈ 6° (tight error bars) — while PC4 (≈ 15°) and PC5
  (≈ 18°) rotate far more, with error bars spanning tens of degrees.** So PC1–PC3 are trustworthy,
  reproducible axes; PC4 and beyond are prompt-noise and are not assigned meaning. This is exactly
  why the axis reads above stop at PC3.

(Figures: `figures/prompt-noise-map.png` — the map with per-persona error bars;
`figures/pca-axis-rotation.png` — mean axis rotation per PC.)

---

## Experiments

### Experiment 1 — are the persona axes clean abstractions? (Section 7a–7c)

- **Surface-word contamination.** Mentioning a trait's *word* in a prompt barely moves its axis.
  The apparent prompt-side contamination (~0.5) was almost entirely the literal word *token*
  sitting on its own axis; at the generation-entry token it collapses to ~0. The axes are clean.
- **Word vs persona decomposition.** A persona vector is **91–98% behavioral residual, ~2–9%
  lexical** (and that sliver is mostly non-specific). Steering with the residual still elicits the
  trait; steering with the word-concept component is inert. **Persona ≠ word.**

### Experiment 2 — the reachable persona manifold (Section 7d–7h)

Steering into the empty regions of PCA space, then a rotational **boundary map** (rotate a
steering direction through a PC plane, sweep magnitude, record coherence + how far the generated
text's persona actually lands).

- The coherent **reachable region is a bounded teardrop that matches the data cloud** — steering
  fills the persona distribution but cannot exceed it.
- **Two different walls** on the two sides: a *saturation* wall on the −PC1/−PC2 side (the model
  snaps back to the manifold edge, staying coherent) and a *coherence-breakdown* wall on the +PC1
  side (pushed too hard, generation degenerates into repetition). The "gaps" beyond the teardrop
  are genuinely unreachable, not merely unsampled.

### Experiment 3 — is persona control low-dimensional? (Section 8)

Does persona act through a small shared causal subspace ("panel"), and is that subspace distinct
from the persona representation itself?

- **Phase 1 — sufficiency (8a–8b).** Steering with a rank-*k* PCA reconstruction of a persona
  vector reproduces the trait by **k ≈ 3–5** (≫ a norm-matched random subspace). Confirms a
  low-dimensional subspace *suffices*.
- **DAS interchange (8c–8g).** The single last-prompt-token site is too weak (the response
  re-attends to the persona's system prompt); the **all-positions residual-stream site** is the
  real interface. Learning a *k*-dim projector `P` and additively swapping `P·(Vc[B]−Vc[A])` at
  that site converts persona A→B, measured by **judge-free held-out cross-entropy** of B's
  responses. *Holding out pairs of the same personas*, the learned subspace appears to plateau at
  **k ≈ 5–6** while random subspaces fail — the within-set "panel dimension."
- **⚠️ Generalization correction (persona-disjoint test).** That within-set plateau is largely
  **memorization**: with only ~8–12 personas (a ≈7–11-dimensional span), a *k* ≳ 7 subspace can
  absorb the whole span, so held-out *pairs* of *already-seen* personas transfer trivially.
  Re-running with a **persona-disjoint split — train `P` on one persona set, measure CE on a
  *disjoint* set of personas the projector never saw** (3 random splits each for Qwen & Llama;
  `experiments/exp3_das.py`) **removes the compact panel: neither model shows a low-*k* plateau on
  unseen personas.** Both climb gradually with *k*, recovering only **~0.4–0.6 of the
  full-difference effect even at k = 24**. The robust cross-model signal is *consistency*: **Llama
  generalizes more and far more stably** (k=24: **0.58 ± 0.03** across splits) than **Qwen**
  (**0.41 ± 0.21**, strongly split-dependent). The full-difference (`P=I`) interchange transfers in
  *every* split for both models, so the gap is specifically about whether a *learned low-dim*
  subspace is **shared vs persona-specific**.
- **How high-dimensional is it? (bisect, `--bisect`).** Binary-searching *k* for the smallest
  subspace that recovers **90%** of the full-difference effect on *unseen* personas:
  **Llama k\* ≈ 1420–1540** (~**37%** of its 4096-dim residual stream, stable across splits);
  **Qwen k\* ≈ 615–1340** (~**17–37%** of 3584, again wildly split-dependent). Not 5–6 dims, not
  dozens — **hundreds-to-thousands**. (These k\* are read off a noisy, only-roughly-monotone CE
  curve, so treat as order-of-magnitude.)
- **A distinct interface, not the persona vector (8i–8j).** A within-set-trained *k*=6 lever keeps
  only **~11–17%** of the persona difference (vs **40–68%** for a variance-optimal PCA-6) yet still
  swaps persona — so the causal directions are not the variance directions. (Geometric measure on
  the trained subspace; read alongside the generalization caveat above.)

**Headline (revised):** The original *"structured ~5–6-dim causal panel"* was a **within-set
artifact**. Under a persona-disjoint test there is **no compact persona-general causal interface** —
generalizing persona control needs a subspace **~⅓ of the residual stream** (Llama k\* ≈ 1500;
Qwen ≈ 600–1300), i.e. it is essentially high-dimensional. A shared low-dim subspace transfers
**consistently in Llama** but only **erratically in Qwen** — the one robust cross-model signal.
(Cross-model runs: `experiments/`; see also `docs/literature.md`.)

---

## Notebooks

- **`personas.ipynb`** — the main pipeline:
  - **Sections 1–3:** persona-vector geometry/orthogonality, steering & projection utilities,
    multi-persona prompts.
  - **Section 6:** builds the 220-adjective persona pool and studies its geometry.
  - **Section 7:** Experiments 1 & 2 — contamination, word-vs-persona decomposition,
    gap-steering, and the coherent-reachable boundary map (incl. 3D).
  - **Section 8:** Experiment 3 — projected-steering sufficiency (Phase 1) and the DAS
    interchange. Note the within-set k≈5–6 "panel" does **not** survive a persona-disjoint test
    (`experiments/exp3_das.py`); see the Experiment-3 generalization correction above.
- **`contrastive_extraction.ipynb`** — archived Sections 4–5 (contrastive persona-vector
  extraction + PCA "Assistant Axis"), superseded by Section 6's positive-only PCA. Run the main
  notebook's setup→core cells first in the same kernel.

## Modules

- **`patchscope.py`** — training-free Patchscope read-out + adversarial blind-spot search for
  persona vectors. `run_patchscope.py` is a self-contained CLI runner.

## Figures

`figures/make_figures.py` regenerates the whole presentation deck from `data/` alone (no model):

```bash
python figures/make_figures.py             # all figures, incl. the rotating GIFs (needs ffmpeg)
python figures/make_figures.py --no-video  # static figures only (fast)
```

Outputs to `figures/`: the labeled rotating persona cloud, the three Experiment-1 panels
(contamination / decomposition / steering), the Experiment-2 steering-regime plot and rotating 3D
reachable-boundary envelope, and the two Experiment-3 panels (panel-dimension knee, interface-vs-PCA).
Cloud/boundary figures are computed from the cached vectors and boundary maps; the Exp-1/3 figures read
`data/exp1_results.json` / `data/exp3_results.json` (written by the notebook's cell 8k on a GPU rerun).

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"          # for Section 6 sourcing
cp .env.example data/.env && $EDITOR data/.env             # OpenRouter key for the Section 7/8 judge
```

Run notebooks from the repo root (paths are relative to `data/`). Qwen2.5-7B downloads from
HuggingFace on first run (~16GB GPU). Note: the setup cell calls `torch.set_grad_enabled(False)`
globally (the notebook is inference-only); the Section 8 training cells re-enable gradients
locally with `torch.enable_grad()`.

## Data

`data/` holds all caches: the persona-pool vectors/responses/system-prompts, the per-question
vectors, and the Experiment-2 outputs (`gap_steering_sweep.*`, `boundary_map_*.*`). They're
tracked so the repo is usable without regenerating (Section 6 loads from cache; Section 8 reuses
the cached vectors and responses for *any* subset of the 220 personas, so scaling the DAS persona
sample needs no new generation). See `.gitignore` to exclude them.

## Open threads

- Noun (role) vs adjective (trait) persona-space comparison — the role vectors are already cached.
- Scanning the DAS site layer; response-side version of the clean-run control (a).
