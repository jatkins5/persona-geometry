"""Shared primitives for the scaled Experiment-1 runs.

These mirror the validated helpers in personas.ipynb (cells 3/6/8/18/29/48) so the
standalone batch scripts produce numbers identical to the interactive notebook. The
notebook remains the canonical source; this module exists only so the batch jobs can
import the pieces they need without re-running the whole 74-cell pipeline.

Everything here is import-safe and side-effect-free at import time (no model load, no
GPU use) -- callers explicitly call load_model() inside their compute job.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch as t

from prompts import WORD_CONCEPT_CARRIERS

DTYPE = t.bfloat16


# --- Model registry ------------------------------------------------------------------
# Each model the pipeline can run on. `persona_layer` is the hidden_states index the
# persona vectors are read at; `steer_layer` is the 0-based model.layers index the
# steering hook attaches to (always persona_layer - 1). Qwen's 20/19 is the notebook's
# validated choice (~71% depth). Llama's is PROVISIONAL -- re-pick it from the all-layer
# extraction (extract_persona_vectors.py --all-layers) before trusting Llama numbers.
@dataclass(frozen=True)
class ModelConfig:
    key: str
    hf_id: str
    persona_layer: int
    steer_layer: int
    cache_suffix: str          # "" keeps the notebook's unsuffixed Qwen caches


MODELS = {
    "qwen2.5-7b": ModelConfig("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", 20, 19, ""),
    "llama3.1-8b": ModelConfig("llama3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", 23, 22,
                               "__llama3.1-8b"),  # layer 23/32 ≈ 0.72 depth, matching Qwen's 20/28 ≈ 0.71
}
DEFAULT_MODEL = "qwen2.5-7b"

# Back-compat module-level defaults (the notebook's Qwen values) so existing references
# and word_concept_vector's default layer still resolve.
QWEN_MODEL_NAME = MODELS[DEFAULT_MODEL].hf_id
TRAIT_VECTOR_LAYER = MODELS[DEFAULT_MODEL].persona_layer
PERSONA_LAYER = TRAIT_VECTOR_LAYER
STEER_LAYER = MODELS[DEFAULT_MODEL].steer_layer

# Persona-pool selection (notebook 6a/6b). Reproduced deterministically from the cached
# relevance scores + WordNet synsets + a seeded tie-break, so we recover the SAME 220
# adjectives the notebook analyses (and therefore the same pool centroid).
REL_THRESHOLD = 0.85
N_TRAITS = 220
TIE_SEED = 0

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def get_config(model_key: str = DEFAULT_MODEL) -> ModelConfig:
    if model_key not in MODELS:
        raise KeyError(f"unknown model '{model_key}'; known: {list(MODELS)}")
    return MODELS[model_key]


def persona_vectors_path(model_key: str = DEFAULT_MODEL, data_dir: Path = DATA_DIR) -> Path:
    """Cache path for a model's persona vectors (Qwen keeps the notebook's unsuffixed file)."""
    return data_dir / f"persona_pool_vectors{get_config(model_key).cache_suffix}.pt"


# --- Model loading -------------------------------------------------------------------
def load_model(model_key: str = DEFAULT_MODEL, device: str | None = None):
    """Load a registry model + tokenizer (inference only). Returns (model, tokenizer, device)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t.set_grad_enabled(False)
    cfg = get_config(model_key)
    device = device or ("cuda" if t.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id)
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_id, dtype=DTYPE).to(device)
    model.eval()
    return model, tokenizer, device


# --- Persona pool reconstruction (notebook 6b) ---------------------------------------
def _dedup_by_synset(scored: dict[str, float], pos_set: set[str],
                     prefer: set[str], exclude: set[str]) -> list[str]:
    """One representative per WordNet-synonym group, ranked by relevance then a seeded
    per-word random tie-break (no alphabetical bias). Verbatim from notebook cell 6b."""
    from nltk.corpus import wordnet as wn

    words = set(scored)
    parent = {w: w for w in words}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    rv = lambda w: random.Random(f"{TIE_SEED}:{w}").random()   # per-word, order-independent
    for w in sorted(words, key=rv):
        for s in wn.synsets(w):
            if s.pos() not in pos_set:
                continue
            sibs = [x for x in (l.name().replace("_", " ").lower() for l in s.lemmas()) if x in words]
            for x in sibs[1:]:
                parent[find(x)] = find(sibs[0])

    from collections import defaultdict
    groups: dict[str, list[str]] = defaultdict(list)
    for w in words:
        groups[find(w)].append(w)

    member_key = lambda m: (m not in prefer, -scored[m], rv(m))
    out: list[tuple[str, float, float]] = []
    for root, mem in groups.items():
        usable = [m for m in mem if m not in exclude]
        if not usable:
            continue
        out.append((min(usable, key=member_key), max(scored[m] for m in usable), rv(root)))
    out.sort(key=lambda tup: (-tup[1], tup[2]))
    return [rep for rep, _, _ in out]


def load_trait_pool(data_dir: Path = DATA_DIR, n_traits: int = N_TRAITS) -> list[str]:
    """Reproduce the notebook's analysed adjective pool (the deduped top-`n_traits` traits)."""
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    relevance = json.loads((data_dir / "persona_pool_relevance.json").read_text())
    traits_kept = {w: s for w, s in relevance["traits"].items() if s >= REL_THRESHOLD}
    roles_all = set(relevance["roles"])                                  # drop POS-ambiguous words
    pregen = set(json.loads((data_dir / "persona_pool_system_prompts.json").read_text()))
    return _dedup_by_synset(traits_kept, {"a", "s"}, pregen, roles_all)[:n_traits]


def load_persona_frame(pool_traits: list[str], model_key: str = DEFAULT_MODEL,
                       data_dir: Path = DATA_DIR):
    """Load a model's cached persona vectors for `pool_traits` and center on the pool mean.

    Returns (names, Vp, Vc, centroid) where Vp = raw vectors, centroid = pool mean, and
    Vc = Vp - centroid (the persona vectors as the notebook's Section 6e/7 frame uses them).
    """
    pool_vectors = t.load(persona_vectors_path(model_key, data_dir))
    names = [n for n in pool_traits if n in pool_vectors]
    missing = [n for n in pool_traits if n not in pool_vectors]
    if missing:
        raise RuntimeError(f"{len(missing)} selected traits have no cached vector: {missing[:10]}")
    Vp = t.stack([pool_vectors[n].float() for n in names])
    centroid = Vp.mean(0)
    Vc = Vp - centroid
    return names, Vp, Vc, centroid


# --- Word-concept direction (notebook 7b) --------------------------------------------
def word_concept_vector(model, tokenizer, trait: str, layer: int = PERSONA_LAYER):
    """Mean activation at the trait-WORD token, averaged over the neutral carrier sentences.

    This isolates the word's lexical/meaning direction, with no persona system prompt.
    Also returns how many subword tokens the trait split into (a confound to control for:
    rarer words tokenise into more pieces). Returns (vector[d_model] on CPU, n_word_tokens).
    """
    vecs, token_counts = [], []
    for template in WORD_CONCEPT_CARRIERS:
        text = template.format(w=trait)
        enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(model.device) for k, v in enc.items()}
        match = re.search(re.escape(trait), text)
        # token indices whose character span overlaps the trait word
        positions = [i for i, (a, b) in enumerate(offsets)
                     if b > a and a < match.end() and b > match.start()] or [-1]
        with t.inference_mode():
            hidden = model(**enc, output_hidden_states=True).hidden_states[layer][0]
        vecs.append(hidden[positions].mean(0).float().cpu())
        token_counts.append(len(positions))
    # carriers all embed the same word, so the token count is stable; report the modal one
    n_word_tokens = max(set(token_counts), key=token_counts.count)
    return t.stack(vecs).mean(0), n_word_tokens


# =====================================================================================
# Heavier primitives for the contamination + steering replication (notebook 7a / 7c).
# These mirror notebook cells 8 and 18 so steered generations and response activations
# match the interactive notebook exactly.
# =====================================================================================
def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Merge a leading system message into the first user message (harmless for Qwen)."""
    if not messages or messages[0]["role"] != "system":
        return messages
    system_content = messages[0]["content"]
    rest = list(messages[1:])
    if rest and rest[0]["role"] == "user" and system_content:
        rest[0] = {"role": "user", "content": f"{system_content}\n\n{rest[0]['content']}"}
    return rest


def _format_messages(messages: list[dict], tokenizer) -> tuple[str, int]:
    """Format a conversation; return (full_prompt, index where the response tokens start)."""
    messages = _normalize_messages(messages)
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_only = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True).rstrip()
    response_start_idx = tokenizer(prompt_only, return_tensors="pt").input_ids.shape[1] + 1
    return full_prompt, response_start_idx


def _return_layers(model):
    """Locate the list of transformer blocks across architectures."""
    for attr_path in ("language_model.layers", "layers"):
        obj = model.model
        try:
            for name in attr_path.split("."):
                obj = getattr(obj, name)
            return obj
        except AttributeError:
            continue
    raise AttributeError(f"Could not find transformer layers on {type(model)}")


def extract_response_activations(model, tokenizer, system_prompts, questions, responses, layer):
    """Mean activation over the RESPONSE tokens at one hidden_states layer (notebook cell 8)."""
    out = []
    for system_prompt, question, response in zip(system_prompts, questions, responses):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}]
        full_prompt, response_start_idx = _format_messages(messages, tokenizer)
        tokens = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with t.inference_mode():
            hidden = model(**tokens, output_hidden_states=True).hidden_states[layer]
        seq_len = hidden.shape[1]
        mask = t.arange(seq_len, device=hidden.device) >= response_start_idx
        mean_act = (hidden[0] * mask[:, None]).sum(0) / mask.sum()
        out.append(mean_act.cpu())
        t.cuda.empty_cache()
    return t.stack(out)


def extract_response_activations_all_layers(model, tokenizer, system_prompt, question, response):
    """Mean activation over RESPONSE tokens at EVERY hidden_states layer for one example.

    Returns a tensor (num_hidden_states, d_model) on CPU, where index L is hidden_states[L]
    (L=0 is the embedding output). Lets us pick the persona layer for a new model offline
    from a single generation pass.
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response}]
    full_prompt, response_start_idx = _format_messages(messages, tokenizer)
    tokens = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with t.inference_mode():
        hidden_states = model(**tokens, output_hidden_states=True).hidden_states
    seq_len = hidden_states[0].shape[1]
    mask = t.arange(seq_len, device=hidden_states[0].device) >= response_start_idx
    per_layer = [((h[0] * mask[:, None]).sum(0) / mask.sum()).float().cpu() for h in hidden_states]
    t.cuda.empty_cache()
    return t.stack(per_layer)


def prompt_hidden(model, tokenizer, system: str, question: str, layer: int):
    """Return (mean-over-prompt-tokens, last-token) activations at `layer`, before any response."""
    messages = _normalize_messages([{"role": "system", "content": system},
                                    {"role": "user", "content": question}])
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    toks = tokenizer(text, return_tensors="pt").to(model.device)
    with t.inference_mode():
        hidden = model(**toks, output_hidden_states=True).hidden_states[layer][0]
    return hidden.mean(0).float().cpu(), hidden[-1].float().cpu()


class ActivationSteerer:
    """Context manager: add coeff * steering_vector at model.layers[layer] (notebook cell 18)."""

    def __init__(self, model, steering_vector, coeff: float = 1.0, layer: int = STEER_LAYER):
        self.model = model
        self.coeff = coeff
        self.layer = layer
        self._handle = None
        self.vector = steering_vector.clone()

    def _hook_fn(self, module, inp, output):
        steer = (self.coeff * self.vector)
        hidden = output[0] if isinstance(output, tuple) else output
        steer = steer.to(hidden.device, dtype=hidden.dtype)
        h = hidden.clone() + steer
        return (h,) + output[1:] if isinstance(output, tuple) else h

    def __enter__(self):
        self._handle = _return_layers(self.model)[self.layer].register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def generate_persona_response(model, tokenizer, system_prompt: str, question: str,
                              max_new_tokens: int = 128, temperature: float = 0.7) -> str:
    """Generate one response; omit the system message when empty (notebook cell 24)."""
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
        + [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    plen = inputs.input_ids.shape[1]
    with t.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, plen:], skip_special_tokens=True)


def generate_with_steerer(model, tokenizer, prompt: str, steering_vector, layer: int,
                          coeff: float, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
    """Generate with activation steering applied (notebook cell 18)."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    plen = inputs.input_ids.shape[1]
    with ActivationSteerer(model, steering_vector, coeff=coeff, layer=layer):
        with t.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                                 do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, plen:], skip_special_tokens=True)


# --- OpenRouter autorater (notebook cells 11/12) -------------------------------------
AUTORATER_MODEL = "anthropic/claude-3.5-haiku"


def make_openrouter_client(data_dir: Path = DATA_DIR):
    """Build an OpenRouter chat client from data/.env (OPENROUTER_API_KEY)."""
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(dotenv_path=str(data_dir / ".env"))
    key = os.getenv("OPENROUTER_API_KEY")
    assert key, "set OPENROUTER_API_KEY in data/.env"
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def judge_calls_parallel(client, messages_list, model=AUTORATER_MODEL, max_tokens=16,
                         temperature=0.0, max_workers=10):
    """Run many OpenRouter completions concurrently, preserving order. Failed calls -> ''."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(messages):
        try:
            time.sleep(0.1)
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
            return resp.choices[0].message.content
        except Exception as e:
            print(f"API error: {e}")
            return ""

    results = [None] * len(messages_list)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one, m): i for i, m in enumerate(messages_list)}
        for f in as_completed(futs):
            results[futs[f]] = f.result()
    return results

