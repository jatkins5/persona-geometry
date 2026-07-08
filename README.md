# persona-geometry

Geometry, steering, and **causal structure** of persona vectors in **Qwen2.5-7B-Instruct**,
extracted as a standalone project from the ARENA `[4.4] LLM Psychology & Persona Vectors`
exercises.

A persona vector is the mean residual-stream activation while the model role-plays a trait
("You are surly / cryptic / flowery…"). This project builds a large pool of them, maps the
geometry of the space they live in, and runs three experiments asking: are the persona axes
*clean abstractions*, what does the *reachable* persona space look like under steering, and is
persona *controlled* through a low-dimensional causal interface.

**Cross-model extension.** The original study (the `personas.ipynb` notebook) is Qwen-only. It was
then re-run head-to-head on **Llama-3.1-8B-Instruct** via standalone, model-parameterized scripts in
[`experiments/`](experiments/) — reusing the same 220-persona pool but re-extracting each model's own
vectors. The cross-model results are woven into the three experiment sections below and summarized in
[**Cross-model synthesis**](#cross-model-synthesis); every figure is indexed in [`FIGURES.md`](FIGURES.md),
and the related-work map is in [`docs/literature.md`](docs/literature.md).

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
- **Cross-model (220 traits, both models; `experiments/exp1_*`).** Replicates cleanly: personas are
  **~90% behavioral residual** on *both* Qwen (mean lexical 9.6%) and Llama (7.1%); residual-steering
  ≈ full ≫ word-component for both. A **new question** — does a trait's lexical content depend on how
  *common* the word is? — has a clean answer: **no.** The raw negative Qwen correlation (Spearman
  −0.25) vanishes once you control for sub-word token count (partial +0.03); on Llama it's absent even
  raw. Lexical leakage tracks *tokenization*, not word frequency.

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
- **Cross-model — a steerability gap tied to dimensionality (`experiments/exp2_manifold.py`).**
  Aiming steering straight at each persona's own direction, **Qwen reaches ~86% of the way to a
  persona before coherence breaks; Llama only ~62%** (and the coherent magnitude range is ~10× smaller
  for Llama — its Qwen-tuned coherence proxy had to be hardened with an English-word gate). The reason
  is geometric: **Llama's persona space is ~2× higher-dimensional** — PC1 29% / participation ratio
  **8.9**, vs Qwen's PC1 47% / PR **4.1** (`analysis/pca_variance_*`). More distributed persona
  features ⇒ a single steering direction captures less ⇒ lower reach. So both models *prompt* every
  persona, but Llama is **promptable-but-not-fully-steerable** — an architecture-dependent difference.

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
  At the *data-starved* N=12 regime this looks essentially high-dimensional: binary-searching *k*
  (`--bisect`) for 90% recovery on unseen personas gives **Llama k\* ≈ 1420–1540** and **Qwen
  k\* ≈ 615–1340** — hundreds-to-thousands of dims. But that turns out to be a **sample-size
  artifact** (next bullet), not the true interface dimension.
- **✅ The compact panel is real — it just needs enough training personas (`--train-size-sweep`).**
  Sweeping the *training* persona count {12, 25, 50, 100, 150} against a **fixed disjoint test set**,
  low-*k* generalization rises sharply with data and a **knee emerges** by k ≈ 8–16 (random-subspace
  baseline ≈ 0 at low *k*, so this is real structure). At **N=100 training personas** both models
  recover **~80% of the full-difference effect on *unseen* personas by k ≈ 8–16**, ~95% by k ≈ 32.
  Qwen **saturates by N≈100** (N=150 adds nothing); Llama keeps tightening through N=150 (k16
  87→94%). So persona control **is** mediated by a compact causal subspace whose identification just
  requires many training personas — with few, DAS can't find it and it looks spuriously
  high-dimensional (a methodological caution for interchange work). *(The exact knee value and any
  Qwen-vs-Llama gap wander ~10–15 pts across random splits, so we don't claim one model's knee is
  sharper.)*
- **❌ The panel is NOT a distinct interface — it's just persona-space PCA (`--persona-space-check`,
  3 splits/model).** Since every `Vc[B]−Vc[A]` lies in persona space, we compared the DAS-learned D
  against the **top-*k* PCA of the persona vectors** on held-out recovery. Across 3 random splits the
  **learned − PCA gap hovers around zero within noise for both models** (Qwen k8 +0.08 ± 0.09, Llama
  k8 +0.04 ± 0.07; ~0 or negative elsewhere). A promising single-split "Qwen beats PCA by 12 pts"
  **did not replicate**. So DAS finds *nothing better than the leading principal components of the
  persona representation* — the causal control subspace **is** the persona-variance subspace, not a
  separate lever. (The notebook's 8i/8j "distinct interface" claim does not survive held-out-persona
  + multi-split testing.)

**Headline (final):** The panel dimension was a moving target driven by evaluation rigor:
**within-set k≈5–6 (memorization)** → **persona-disjoint at N=12 (looks ~10²–10³-dim — data-starved)**
→ **scaling training personas: a compact subspace *emerges*, ~k 8–16, generalizing to unseen
personas** → **but multi-split testing shows that subspace is *just the top-k PCA of the persona
vectors*, not a distinct causal interface.** So persona control runs through a low-dimensional
subspace that **coincides with the persona representation's own leading directions** — the interface
and the representation are the *same* ~10-dim object, not distinct.
(Cross-model runs: `experiments/`; see also `docs/literature.md`.)

---

## Cross-model synthesis

Running the whole pipeline on a second, different-provider model (Llama-3.1-8B) sharpened the story
into one through-line: **persona is encoded more distributively in Llama than in Qwen**, and that
single fact shows up in all three experiments.

| | Qwen2.5-7B (layer 20) | Llama-3.1-8B (layer 23) |
|---|---|---|
| Persona-space PC1 / participation ratio | 47% / **4.1** | 29% / **8.9** |
| Exp 1 — behavioral residual | ~90% | ~93% |
| Exp 1 — lexical ↔ word-frequency | none (tokenization artifact) | none |
| Exp 2 — steering reach to own persona | **0.86** | **0.62** |
| Exp 3 — persona control subspace | compact (~k 8–16) **= persona-space PCA** | compact (~k 8–16) **= persona-space PCA** |

- **Exp 1** replicates identically — "persona ≠ word" is not a Qwen quirk.
- **Exp 2** is where the models diverge: Llama's higher-dimensional persona space makes it markedly
  **less linearly steerable** despite being equally promptable.
- **Exp 3** lands the same conclusion for both: persona control *is* low-dimensional, but that
  subspace is **just the leading PCA directions of the persona representation** — not a distinct
  causal "panel." The apparent panel dimension was a moving target set by evaluation rigor
  (within-set memorization → data-starved-looks-highdim → emergent-with-data → but = PCA).

**Method note that generalizes.** Exp 3 is also a cautionary tale for DAS/interchange work: a
low-dim-causal-interface claim needs held-out **entities** (not just held-out pairs of seen ones),
**enough training entities**, and a **PCA-of-the-representation baseline across multiple random
splits** — each of which flipped a conclusion here.

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

## Cross-model scripts (`experiments/`)

Standalone, model-parameterized ports of the experiments (registry in `persona_lib.py`, prompts in
`prompts.py`, outputs to the gitignored `experiments/results/`). Run any on Qwen or Llama via
`--model {qwen2.5-7b, llama3.1-8b}`, submitted with the matching `run_*.sbatch` (Oscar/Slurm):

- `extract_persona_vectors.py` — re-extract a new model's persona vectors (all-layer + a layer-choice
  diagnostic).
- `exp1_decomposition.py`, `exp1_contamination_steer.py` — Experiment 1 at pool scale.
- `exp2_manifold.py` — Experiment 2: `--mode {calibrate, boundary, reachability, variance}`.
- `exp3_das.py` — Experiment 3 DAS: panel sweep, `--bisect`, `--train-size-sweep`,
  `--persona-space-check` (all persona-disjoint); `plot_das_*.py` aggregate across splits.

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

The **cross-model** figures (Qwen vs Llama) live under `experiments/results/` and are indexed, with
the canonical/final figure per result marked, in [`FIGURES.md`](FIGURES.md).

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
