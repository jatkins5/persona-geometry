# Literature review: persona geometry, the persona abstraction, and the persona selection model

Scope: prior work relevant to this project's three experiments — (1) the *geometry* of persona
representations, (2) *tests* of whether "persona" is a real/clean/controllable abstraction, and
(3) the *persona selection model* (PSM) as a theory of how LLMs work — plus the methods we use
(activation steering, DAS).

> Citations were gathered via web search/fetch (June 2026). **Persona Vectors, the Assistant Axis,
> and DAS have been verified against their PDFs.** The remaining numbers marked *(reported)* — the
> OpenAI Emergent-Misalignment figures (§3) — come from automated HTML extraction and should be
> confirmed against the source before citing. Very recent (2026) preprints are flagged; skim first.

---

## 1. Persona geometry — representational / linear structure

- **Persona Vectors: Monitoring and Controlling Character Traits in Language Models** — Chen,
  Arditi, Sleight, Evans, Lindsey et al. (Anthropic, 2025). [arXiv:2507.21509](https://arxiv.org/abs/2507.21509).
  The direct antecedent of this project. Extracts a per-trait direction and uses it to steer,
  monitor, and flag training data. **Verified from the paper (pages 1–9):**
  - Base models are **Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct — exactly our two models** (§3.1).
  - **Contrastive** extraction: 5 positive/negative system-prompt pairs × 40 eval questions, 10
    rollouts each, filtered by an LLM "trait expression score" (keep >50 / <50), mean over
    **response** tokens, one vector per layer; the **best layer is selected by steering
    effectiveness** (Appendix B.4; Qwen trait-steering peaks ~layers 15–20 in Fig 3).
  - Main text studies **3 traits — evil, sycophancy, hallucination** — plus a few more (optimism,
    humor) in the appendix. (We study **220**.)
  - Steering `h ← h + α·v` with α up to ~2.5; results gated to **coherence score > 75** (the same
    idea as our distinct/ascii/english coherence proxy).
  - Monitoring works: projecting the last-prompt-token activation onto the persona vector predicts
    later trait expression (**r = 0.75–0.83**); finetuning shift along the vector predicts trait
    change (**r = 0.76–0.97**, vs cross-trait baseline 0.34–0.86).
  - Key methodological contrast with us: **contrastive** pairs vs our **positive-only,
    pool-centered** vectors; **3 traits vs 220**.

- **The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models** — Lu,
  Gallagher, Michala, Fish, Lindsey (Anthropic, 2026). [arXiv:2601.10387](https://arxiv.org/abs/2601.10387).
  The closest geometric analog. **Verified against the PDF:** models **Gemma-2-27B, Qwen-3-32B,
  Llama-3.3-70B**; **middle residual layer**; role vectors are **positive-only** (mean over response
  tokens of system-prompted role-play — same family of method as ours), **275 roles → 377–463 role
  vectors; 240 traits**; persona components explain **19.4–33.6%** of *total* activation variance
  (n=18,777 LMSYS-Chat); **4–19 components for 70% variance, varying by model**; PC1 ("Assistant
  Axis") role-loadings correlate **>0.92 across models** while PC2–3 diverge (0.56–0.89); the default
  Assistant sits at the **extreme of PC1** (proj. 0.03) but mid-range on others (0.27–0.50). They
  stabilize behavior by clamping activations along the axis, and find the axes are **already present
  in base models** (pretraining) — direct support for the persona selection model. Independently
  corroborates our finding that persona-space **dimensionality is model-dependent**; we add the
  *causal steerability* consequence they don't measure.

- **The Linear Representation Hypothesis and the Geometry of LLMs** — Park, Choe, Veitch
  (ICML 2024). [arXiv:2311.03658](https://arxiv.org/abs/2311.03658). Plus **The Geometry of
  Categorical and Hierarchical Concepts** ([OpenReview](https://openreview.net/forum?id=bVTM2QKYuA)).
  Formal basis for treating traits as directions and using cosine/projection; introduces a causal
  inner product. Underwrites our orthogonality and PCA analyses.

- **Your Language Model Secretly Contains Personality Subnetworks** (2026).
  [arXiv:2602.07164](https://arxiv.org/html/2602.07164v1). Persona as extractable sparse subnetworks
  — a distributed/higher-dimensional view consistent with our Llama result (effective dim ≈ 9).

- Steering-subspace structure: **Representation Engineering** (Zou et al., 2023); **Steering Llama-2
  via Contrastive Activation Addition** (Rimsky et al., 2024); **Refusal in LLMs is an Affine
  Function** ([arXiv:2411.09003](https://arxiv.org/pdf/2411.09003)) — abstract behaviors occupy
  linear / cone-structured subspaces (cf. our reachable-cone geometry).

## 2. Tests of the persona abstraction

- **Tell me about yourself: LLMs are aware of their learned behaviors** — Betley et al. (ICLR 2025).
  [arXiv:2501.11120](https://arxiv.org/abs/2501.11120); follow-up **Minimal and Mechanistic
  Conditions for Behavioral Self-Awareness** ([arXiv:2511.04875](https://arxiv.org/pdf/2511.04875)).
  A behavioral test that a trained-in behavior is represented as an introspectable persona.
- **Personas as a Way to Model Truthfulness in Language Models** (2023).
  [arXiv:2310.18168](https://arxiv.org/pdf/2310.18168). Evidence for a latent "truthful persona"
  that *arises from the persona-agent structure of pretraining* — conceptually our "persona is
  behavioral, not lexical" claim, for truthfulness.
- **Split Personality Training: Revealing Latent Knowledge Through Alternate Personalities** (2026).
  [arXiv:2602.05532](https://arxiv.org/pdf/2602.05532). Causal manipulation of personas to probe
  gated knowledge.
- **Whether, Not Which: Dissociable Affect Reception and Emotion Categorization in LLMs** (2026).
  [arXiv:2603.22295](https://arxiv.org/pdf/2603.22295). Mechanistic dissociation within affective
  representations — methodologically close to our clean-axis tests.
- **Our Exp 1** (clean-abstraction / word-vs-persona decomposition) is itself a test of this kind;
  the nearest published analog is the Persona Vectors steering validations plus the
  linear-representation tests above.

## 3. The persona selection model (PSM) and its tests

- **The persona selection model** — Sam Marks et al. (Anthropic), 2025.
  [LessWrong](https://www.lesswrong.com/posts/dfoty34sT7CSKeJNn/the-persona-selection-model). The
  canonical statement: pretraining learns to simulate many characters; post-training selects/refines
  an "Assistant" persona; users interact with that *simulacrum*, not the model. Predicts: trait
  training generalizes broadly; representations reuse pretrained character models; interpretability
  should reveal persona-based mechanisms. Credits **Andreas (2022)**, **janus "Simulators" (2022)**,
  **Hubinger et al. (2023)**.
- **Language Models as Agent Models** — Andreas (2022).
  [Scilit](https://www.scilit.com/publications/e179b233bf8dcf0e3418500beb5538a0). Foundational
  "LLMs infer and simulate the agent behind the text" argument.
- **PICLe: Persona In-Context Learning** (2024). [arXiv:2405.02501](https://arxiv.org/pdf/2405.02501).
  Formalizes selection as **Bayesian marginalization over a mixture of persona distributions** — a
  concrete, testable mechanism for "selection."
- **Strongest empirical support (emergent misalignment):** **Persona Features Control Emergent
  Misalignment** — Wang, Dupré la Tour, Watkins et al. (OpenAI, 2025).
  [arXiv:2506.19823](https://arxiv.org/abs/2506.19823); [OpenAI blog](https://openai.com/index/emergent-misalignment/).
  *(reported)* GPT-4o / o3-mini; from a **2.1M-latent SAE**, a single **"toxic persona" latent (#10)**
  most strongly controls misalignment and **near-1-dimensionally discriminates** aligned vs
  misaligned models; misalignment ~40–50% on bad-advice fine-tunes; **re-alignment with 120–180
  clean samples** (→ ~0.1–0.5%); toxic-persona latent detectable at **5%** bad data before behavior
  shifts. Narrow→broad generalization via a reused persona feature = PSM's central prediction,
  causally confirmed.
- **Limits of PSM as behavior prediction:** simulation studies find persona prompts explain **<10%**
  of outcome variance and individual-level prediction is **<5%** accurate — **Quantifying the Persona
  Effect in LLM Simulations** ([RG](https://www.researchgate.net/publication/384205641_Quantifying_the_Persona_Effect_in_LLM_Simulations));
  **LLM Generated Persona is a Promise with a Catch** ([arXiv:2503.16527](https://arxiv.org/pdf/2503.16527)).
  PSM is a good *mechanistic/geometric* account, a weak *individual-prediction* one.

## 4. Methods we rely on

- **Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations
  (DAS)** — Geiger, Wu, Potts, Icard, Goodman (2024).
  [arXiv:2303.02536](https://arxiv.org/abs/2303.02536); **Boundless DAS** (Wu et al., scales to
  Alpaca-7B, finding *compact, localized* interpretable subspaces). The method behind our Exp 3
  causal-interface search; DAS learns a rotation by gradient descent and intervenes on a subspace —
  we use it to find the ~5–6-dim persona "panel."

---

## 5. Quantitative comparison to this project

This project: **Qwen2.5-7B-Instruct (layer 20)** and **Llama-3.1-8B-Instruct (layer 23)**;
**220 adjective traits**, positive-only mean response activations centered on the pool mean.

| Dimension | Persona Vectors (Anthropic) | Assistant Axis (2026) | **This project** |
|---|---|---|---|
| Models | **Qwen2.5-7B, Llama-3.1-8B** (same as us) | Gemma-2-27B, Qwen-3-32B, Llama-3.3-70B | Qwen2.5-7B, Llama-3.1-8B |
| Extraction | contrastive (pos−neg), response tokens | role/trait vectors | positive-only, pool-centered |
| # traits | 3 main (+appendix) | 240 traits / 275 roles | 220 traits |
| Extraction layer | best-by-steering (Qwen ~15–20) | middle residual | 20 (Qwen) / 23 (Llama) |
| PC1 variance | — | 19.4–33.6% (persona space total) | **Qwen 47% / Llama 29%** |
| Dimensionality | — | **4–19 comps for 70% var (model-dependent)** | **90% var: Qwen PC24 / Llama PC41; participation ratio 4.1 / 8.9** |
| Cross-model | — | PC1 >0.92 correlated; PC2-3 diverge | Llama ~2× higher-dim than Qwen |
| Steering / monitoring | proj. monitoring r=0.75–0.83; finetune-shift r=0.76–0.97 | — | **steering reaches only 62% (Llama) / 86% (Qwen) of a persona's own location before coherence breaks** |
| Lexical vs behavioral | — | — | persona ≈ **90–93% behavioral residual, ~7–10% lexical**; lexical content not frequency-driven |
| Low-dim causal control | — | clamp along Assistant Axis | DAS panel **k ≈ 5–6** (Exp 3) |

Adjacent low-dim-control result: the OpenAI EM paper finds essentially **one** dominant controlling
persona feature; our Exp 3 finds a **~5–6-dim** causal interface — both support "persona is
*controlled* through a low-dimensional subspace," at different granularities.

## 6. Where this project fits / open gaps

The work is a **geometric + causal test of the persona-abstraction / persona-selection picture**:
Exp 1 tests the clean-linear-abstraction claim (persona ≠ word), Exp 2 the reachable/controllable
structure (steering-subspace + cone work), Exp 3 whether control is low-dimensional and *distinct*
from representation (DAS).

Clearest gap we fill: persona-vector work typically studies a handful of traits on one model. We do
**pool-scale geometry (220 traits) + a head-to-head cross-model comparison** and find a
**steerability/dimensionality difference** — Llama's persona space is higher-dimensional (PR ≈ 9 vs
4) and less linearly steerable (reach 0.62 vs 0.86), even though both models can *prompt* every
persona. The Assistant Axis paper independently reports model-dependent dimensionality, but we add
the **causal steerability link** (dimensionality → reachability) and the **promptable-but-not-
steerable** gap. No prior work found doing this specific cross-model geometric-steerability
comparison — likely our novel contribution, and a direct test of PSM's "representations reuse
persona models" prediction showing it is **architecture-dependent**.
