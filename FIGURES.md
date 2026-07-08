# Cross-model experiment figures

All PNGs from the Qwen2.5-7B vs Llama-3.1-8B cross-model runs live under
`experiments/results/` (force-committed; the rest of that tree — CSVs, JSONs, `.pt`
caches, slurm logs — is gitignored). **★ = the canonical/final figure for a result.**

## Experiment 1 — clean abstraction (decomposition, contamination, steering)
- ★ `exp1_decomposition/scatter_frequency_vs_lexical.png` — Qwen: lexical fraction vs word frequency.
- ★ `exp1_decomposition__llama3.1-8b/scatter_frequency_vs_lexical.png` — Llama, same.
- ★ `exp1_contamination_steer/{contamination_ratios,steering_scores}.png` — Qwen 7a/7c (220 traits).
- ★ `exp1_contamination_steer__llama3.1-8b/{contamination_ratios,steering_scores}.png` — Llama.
- `exp1_contam_smoke/*` — 4-trait smoke test (superseded).

## Experiment 2 — reachable manifold + dimensionality
- ★ `analysis/pca_variance_llama3.1-8b.png` — persona-space PCA spectrum, Qwen vs Llama (PR 4.1 vs 8.9).
- ★ `exp2_manifold_denoised/{boundary_map_PC1-PC2,PC1-PC3,PC2-PC3,reachable_boundary_3d}.png` —
  Llama coherent reachable boundary, denoised (num_q=3).
- ★ `exp2_manifold__llama3.1-8b/reachability.png` — Llama per-persona reachability (reach ≈ 0.62).
- ★ `exp2_manifold/reachability.png` — Qwen per-persona reachability (reach ≈ 0.86).
- `exp2_manifold__llama3.1-8b/boundary_map_*.png`, `reachable_boundary_3d.png` — noisy num_q=1 (superseded by denoised).

## Experiment 3 — persona causal interface (DAS)
The Exp-3 story evolved with evaluation rigor; the **final** figures are the last three below.
- `analysis/das_panel_compare.png` — within-set panel sweep (memorization; superseded).
- `analysis/das_panel_compare_disjoint.png` — single-split persona-disjoint (superseded).
- ★ `analysis/das_panel_compare_robust.png` — persona-disjoint panel sweep, 3 splits/model.
- ★ `exp3_das_tss5_{qwen2.5-7b,llama3.1-8b}/train_size_sweep.png` — **panel emerges with training
  personas** {12,25,50,100,150}; the headline Exp-3 figure.
- ★ `analysis/pspace_multisplit.png` — **is the panel distinct from persona-space PCA?** (3 splits):
  learned − PCA gap ≈ 0 ⇒ **not distinct**; the final Exp-3 conclusion.
- Component/intermediate runs (inputs to the ★ aggregates, or superseded):
  `exp3_das_robust_*_s{0,1,2}/panel_dimension.png`, `exp3_das_pspace_*_s{1,2}/persona_space_check.png`,
  `exp3_das_pspace_*/persona_space_check.png` (single split), `exp3_das_sweep_*` (within-set),
  `exp3_das_disjoint_*`, `exp3_das__llama3.1-8b/`, `exp3_das_tss_*`, `exp3_das_tss150_*`,
  `exp3_das_smoke/`, `exp3_das_tss_smoke/`.

See `README.md` (Experiments) and `docs/literature.md` for the write-ups these figures support.
