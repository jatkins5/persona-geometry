# persona-geometry

Geometry and steering of persona vectors in **Qwen2.5-7B-Instruct**, extracted as a
standalone project from the ARENA `[4.4] LLM Psychology & Persona Vectors` exercises.

## Notebooks
- **`personas.ipynb`** — the main pipeline:
  - Sections 1–3: persona-vector geometry/orthogonality, steering & projection utilities,
    multi-persona prompts.
  - **Section 6**: builds a ~220 single-part-of-speech (adjective) persona pool from WordNet
    (relevance filter + synset dedup → Qwen system-prompt generation → positive-only PCA),
    and studies its geometry (PCA, participation ratio, near-orthogonality).
  - **Section 7**: probing experiments — surface-word contamination, word-vs-persona
    decomposition, gap-steering, and the coherent-reachable **boundary map** (incl. 3D).
- **`contrastive_extraction.ipynb`** — archived Sections 4–5 (contrastive persona-vector
  extraction + PCA "Assistant Axis"), superseded by Section 6's positive-only PCA. Not part
  of the main pipeline; run the main notebook's setup→core cells first in the same kernel.

## Modules
- **`patchscope.py`** — training-free Patchscope read-out + adversarial blind-spot search
  for persona vectors. `run_patchscope.py` is a self-contained CLI runner.

## Setup
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"          # for Section 6 sourcing
cp .env.example data/.env && $EDITOR data/.env             # OpenRouter key for the Section 7 judge
```
Run notebooks from the repo root (paths are relative to `data/`). Qwen2.5-7B downloads
from HuggingFace on first run (~16GB GPU).

## Data
`data/` holds all caches: the persona-pool vectors/responses/system-prompts, the
per-question vectors, and the Experiment-2 outputs (`gap_steering_sweep.*`,
`boundary_map_*.*`). They're tracked so the repo is usable without regenerating; see
`.gitignore` to exclude them.
