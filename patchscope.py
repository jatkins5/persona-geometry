"""Training-free Patchscope read-out for Qwen persona vectors, plus a gradient-free search for
BLIND SPOTS: persona-space activations the read-out fails to name.

(This is the Patchscopes / SelfIE family: the model interprets its own activation, no training.
A *trained* decoder that answers questions about activations -- LatentQA-style, model-specific,
~a day to build -- would be the heavier "activation oracle"; that is deliberately NOT this file.)

Run on a machine with the Qwen model loaded. Reuses only the same forward-hook convention as
the notebook's ActivationSteerer: we overwrite the output of transformer block `inject_layer`.
The persona vectors were extracted from ``outputs.hidden_states[PERSONA_LAYER]`` with
PERSONA_LAYER=20, i.e. the output of block 19 -> so ``inject_layer = PERSONA_LAYER - 1 = 19``
puts the injected vector in exactly the space the vectors were read from.

Typical use (on the spare GPU):

    import torch as t
    from patchscope import PatchscopeReader, validate, adversarial_search, pca_box
    reader = PatchscopeReader(qwen_model_small, qwen_tokenizer, inject_layer=19)
    pool = t.load("persona_pool_vectors.pt")               # name -> (d,) raw mean activation
    val = validate(reader, pool)                           # recovery rate + read-outs (tune first!)
    box = pca_box(pool, n_pc=10)                            # center, basis, lo/hi from real personas
    blind = adversarial_search(reader, box, words=list(pool))
"""
from __future__ import annotations

import re
from contextlib import contextmanager

import numpy as np
import torch as t

try:
    from nltk.corpus import wordnet as wn
except Exception:  # nltk optional; falls back to exact-word matching
    wn = None


def _blocks(model):
    """Locate the list of transformer blocks (mirrors the notebook's _return_layers)."""
    for attr_path in ("language_model.layers", "layers"):
        obj = model.model
        try:
            for name in attr_path.split("."):
                obj = getattr(obj, name)
            return obj
        except AttributeError:
            continue
    raise AttributeError(f"Could not find transformer layers on {type(model)}")


# ----------------------------------------------------------------------------- read-out
class PatchscopeReader:
    """Decode a residual-stream vector into a short text description, training-free, by patching
    it into Qwen's own forward pass (Patchscopes / SelfIE style) and letting the model verbalise."""

    def __init__(self, model, tokenizer, inject_layer: int = 19, *, mode: str = "replace",
                 rescale: bool = True, user_prompt: str | None = None,
                 assistant_prefix: str | None = None):
        assert mode in ("replace", "add")
        self.model, self.tok, self.layer = model, tokenizer, inject_layer
        self.mode, self.rescale = mode, rescale
        # The read-out prompt. Tune these two strings in validate() before trusting the read-out.
        self.user_prompt = user_prompt or (
            "I will show you a compressed snapshot of an AI's inner personality state. "
            "In one or two words, name the personality, character, or disposition it expresses."
        )
        self.assistant_prefix = ("It expresses the personality of someone who is"
                                 if assistant_prefix is None else assistant_prefix)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self._ids = self._build_ids()

    def _build_ids(self) -> "t.Tensor":
        chat = self.tok.apply_chat_template([{"role": "user", "content": self.user_prompt}],
                                            tokenize=False, add_generation_prompt=True)
        return self.tok(chat + self.assistant_prefix, return_tensors="pt").input_ids

    @contextmanager
    def _patch(self, h: "t.Tensor", pos: int = -1):
        """Overwrite (or add to) the residual of token `pos` at `inject_layer`, ONLY on the
        full-prompt forward pass (skip single-token cached decode steps)."""
        hd = h.to(self.model.device)

        def hook(_mod, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if hs.shape[1] == 1:                       # cached decode step -> leave generation alone
                return out
            v = hd
            if self.rescale:                           # match the slot's natural norm
                v = hd * (hs[:, pos, :].norm(dim=-1, keepdim=True) / (hd.norm() + 1e-6))
            hs = hs.clone()
            if self.mode == "replace":
                hs[:, pos, :] = v.to(hs.dtype)
            else:
                hs[:, pos, :] = hs[:, pos, :] + v.to(hs.dtype)
            return (hs,) + out[1:] if isinstance(out, tuple) else hs

        handle = _blocks(self.model)[self.layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @t.inference_mode()
    def read(self, h: "t.Tensor", max_new_tokens: int = 12) -> str:
        """Generative read-out: the model's short description of the patched vector."""
        ids = self._ids.to(self.model.device)
        with self._patch(h):
            out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    @t.inference_mode()
    def word_logprobs(self, h: "t.Tensor", words: list[str]) -> "np.ndarray":
        """Fast recognition signal: log P(first token of ' {word}') right after the prefix, for
        each candidate persona word, from a SINGLE patched forward pass. Used as the search objective."""
        ids = self._ids.to(self.model.device)
        with self._patch(h):
            logits = self.model(ids).logits[0, -1, :].float()
        lp = logits.log_softmax(-1)
        wids = t.tensor([self.tok(" " + w, add_special_tokens=False).input_ids[0] for w in words],
                        device=lp.device)
        return lp[wids].cpu().numpy()


# ----------------------------------------------------------------------------- recognition
def _synonyms(word: str) -> set[str]:
    out = {word.lower()}
    if wn is not None:
        for syn in wn.synsets(word):
            for lem in syn.lemmas():
                out.add(lem.name().replace("_", " ").lower())
    return out


def recognizes(text: str, word: str) -> bool:
    """True if the read-out names `word` or a WordNet synonym of it."""
    low = text.lower()
    return any(re.search(rf"\b{re.escape(s)}\b", low) for s in _synonyms(word))


# ----------------------------------------------------------------------------- validation
def validate(reader: PatchscopeReader, personas: dict[str, "t.Tensor"],
             max_new_tokens: int = 12, verbose: bool = True) -> dict:
    """Run the generative read-out on every known persona; report recovery rate + read-outs.
    Use this to TUNE the prompt/inject_layer before trusting any blind-spot result."""
    hits, rows = 0, []
    for name, h in personas.items():
        text = reader.read(h.float(), max_new_tokens=max_new_tokens)
        ok = recognizes(text, name)
        hits += ok
        rows.append((name, ok, text))
        if verbose:
            print(f"  [{'OK ' if ok else 'miss'}] {name:>22} -> {text}")
    rate = hits / max(1, len(personas))
    if verbose:
        print(f"\nrecovery: {hits}/{len(personas)} = {rate:.1%}")
    return {"rate": rate, "rows": rows}


# ----------------------------------------------------------------------------- persona-space box
def pca_box(personas: dict[str, "t.Tensor"], n_pc: int = 10):
    """Center, PCA basis, and per-PC [min,max] of the real personas -> the region to search."""
    from sklearn.decomposition import PCA

    names = list(personas)
    V = t.stack([personas[n].float() for n in names]).numpy()      # (N, d) raw vectors
    center = V.mean(0)
    pca = PCA(n_components=n_pc).fit(V - center)
    coords = pca.transform(V - center)                             # (N, n_pc)
    return {"center": center, "basis": pca.components_, "lo": coords.min(0), "hi": coords.max(0),
            "names": names, "coords": coords}


# ----------------------------------------------------------------------------- adversarial search
def adversarial_search(reader: PatchscopeReader, box: dict, words: list[str], *,
                       n_random: int = 400, n_refine: int = 200, keep: int = 10,
                       step: float = 0.15, seed: int = 0, dtype=t.float32) -> list[dict]:
    """Search the persona PCA box for points the read-out can't confidently name (low max word-logprob).

    Returns the `keep` lowest-confidence points: their PCA coords, the read-out's best guess + its
    logprob, and the full generative read-out at that point (to see what the model DOES say)."""
    rng = np.random.default_rng(seed)
    center, basis, lo, hi = box["center"], box["basis"], box["lo"], box["hi"]
    words = list(words)

    def to_h(coords: "np.ndarray") -> "t.Tensor":
        return t.tensor(center + coords @ basis, dtype=dtype)

    def confidence(coords: "np.ndarray") -> tuple[float, int]:
        lp = reader.word_logprobs(to_h(coords), words)             # (len(words),)
        j = int(lp.argmax())
        return float(lp[j]), j                                     # max logprob = read-out's best guess

    # 1) random scan of the box
    pop = rng.uniform(lo, hi, size=(n_random, len(lo)))
    scored = [(confidence(c), c) for c in pop]                     # ((conf, argmax), coords)

    # 2) local refinement around the current most-blind point (Gaussian steps, stay in box)
    scored.sort(key=lambda s: s[0][0])
    (best_conf, best_j), best_c = scored[0]
    span = (hi - lo) * step
    for _ in range(n_refine):
        cand = np.clip(best_c + rng.normal(0, span), lo, hi)
        (cc, cj) = confidence(cand)
        if cc < best_conf:
            best_conf, best_j, best_c = cc, cj, cand
            scored.append(((cc, cj), cand))

    scored.sort(key=lambda s: s[0][0])
    out = []
    seen = set()
    for (conf, j), c in scored:
        key = tuple(np.round(c, 2))
        if key in seen:
            continue
        seen.add(key)
        out.append({"coords": c, "confidence": conf, "best_guess": words[j],
                    "readout": reader.read(to_h(c))})
        if len(out) >= keep:
            break
    return out
