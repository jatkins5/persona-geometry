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
  real interface. Learning a *k*-dim projector `P` and swapping `P·(Vc[B]−Vc[A])` at that site
  converts persona A→B. Measured by **judge-free held-out cross-entropy** of B's responses
  (multi-seed), the learned subspace plateaus at **k ≈ 5–6** — the **panel dimension** — while
  **random subspaces fail at every k up to 12**. All three DAS controls hold: random-subspace
  baseline fails, the site is the residual stream, and **(8h)** the subspace is demonstrably
  *used* on clean unintervened runs (100% leave-one-out persona ID from its coordinates).
- **A distinct interface, not the persona vector (8i–8j).** The injected lever keeps only **~17%**
  of the persona difference (vs **68%** for a variance-optimal PCA-6) yet controls behavior
  *better* than the full vector — and this holds when the persona representation spans **31
  dimensions** (32 personas). So the causal interface is genuinely narrow and **distinct from the
  directions personas vary along** — you cannot recover it from PCA of the persona vectors.

**Headline:** Qwen controls persona through a *structured ~5–6 dimensional causal interface that
is distinct from the persona representation* — found causally (via DAS), not by variance.

---

## Notebooks

- **`personas.ipynb`** — the main pipeline:
  - **Sections 1–3:** persona-vector geometry/orthogonality, steering & projection utilities,
    multi-persona prompts.
  - **Section 6:** builds the 220-adjective persona pool and studies its geometry.
  - **Section 7:** Experiments 1 & 2 — contamination, word-vs-persona decomposition,
    gap-steering, and the coherent-reachable boundary map (incl. 3D).
  - **Section 8:** Experiment 3 — projected-steering sufficiency (Phase 1) and the DAS
    interchange measuring the low-dimensional, distinct persona causal interface.
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
