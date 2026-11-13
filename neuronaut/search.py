"""Find candidate levers. Two screens, then one sweep; nothing here counts as evidence until `neuronaut test`.

Screen by words (`--words great,wonderful,thanks`): push every neuron of the chosen layers up and down on four short
contexts and record which next tokens become more likely (the "steer lens"). Score = how many of a neuron's top
raised tokens are in your word list. Random directions of the same size get the same treatment; their best score is
the bar a neuron must clear.

Screen by contrast (`--contrast pairs.jsonl`, lines {"with": "...", "without": "..."}): mean activation on texts that
show the trait minus texts that do not (same topic, different stance), read at the last token, divided by the
spread. Both poles count.

Sweep: the top candidates generate short replies on a few everyday prompts at four push sizes, beside random
controls, and are kept only if their replies contain your words more often than every control's.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch
import torch.nn as nn

CONTEXTS_CHAT = ["Who are you?", "Write whatever you like."]
CONTEXTS_PLAIN = ["I think the most important thing to remember is", "Today we are going to talk about"]
SWEEP_PROMPTS = ["Who are you?", "Tell me about your weekend.", "Describe the color orange."]
SWEEP_DOSES = (-0.9, -0.6, 0.6, 0.9)
WORD = re.compile(r"[^\W\d_]{3,}")


class _Skip(nn.Module):
    def forward(self, hidden_states, *a, **k):
        return hidden_states


def _decoder_owner(model):
    for path in ("model.language_model", "model", "language_model.model"):
        obj = model.hf
        for p in path.split("."):
            obj = getattr(obj, p, None)
        if obj is not None and hasattr(obj, "layers"):
            return obj
    raise ValueError("cannot find the decoder module")


@torch.no_grad()
def steer_lens(model, layers: list[int], scale: list[float], dose: float = 0.8, batch: int = 96, k: int = 12,
               n_random: int = 32, log=print):
    """For each layer: top-k raised token ids per neuron and sign, plus the same for random directions."""
    dec = _decoder_owner(model)
    orig = list(dec.layers)
    ctxs = [model.chat_ids(c) for c in CONTEXTS_CHAT] + [model.tok(c, add_special_tokens=False)["input_ids"] for c in CONTEXTS_PLAIN]
    cache = [dict() for _ in ctxs]
    cur = {"c": 0}
    hooks = [l.register_forward_pre_hook(lambda m, a, kw, L=L: cache[cur["c"]].__setitem__(L, (kw.get("hidden_states") if a == () else a[0]).clone()),
                                         with_kwargs=True) for L, l in enumerate(orig)]
    head = model.hf.get_output_embeddings()
    base = []
    for c, ids in enumerate(ctxs):
        cur["c"] = c
        h = dec(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[:, -1]
        base.append(torch.log_softmax(head(h).float(), -1)[0])
    for h in hooks:
        h.remove()
    base = torch.stack(base)
    results = {}
    for L in layers:
        t0 = time.time()
        for i in range(L):
            dec.layers[i] = _Skip()
        try:
            W = model.layers[L].mlp.down_proj.weight.detach().float()
            unit = W / W.norm(dim=0, keepdim=True) * scale[L] * dose  # [D, N]
            g = torch.Generator().manual_seed(9000 + L)
            R = torch.randn(W.shape[0], n_random, generator=g)
            R = R / R.norm(dim=0, keepdim=True) * scale[L] * dose
            out = {"up": [], "down": [], "rand": []}
            for key, vecs in (("up", unit), ("down", -unit), ("rand", R)):
                for s in range(0, vecs.shape[1], batch):
                    V = vecs[:, s:s + batch].T  # [b, D]
                    lps = []
                    for c in range(len(ctxs)):
                        x = cache[c][L].expand(V.shape[0], -1, -1).contiguous()
                        with model.pushing({L: V[:, None, :]}):
                            h = dec(inputs_embeds=x, use_cache=False).last_hidden_state[:, -1]
                        lps.append(torch.log_softmax(head(h).float(), -1))
                    delta = (torch.stack(lps, 1) - base[None]).mean(1)
                    out[key].append(delta.topk(k, dim=1).indices)
            results[L] = {kk: torch.cat(v).tolist() for kk, v in out.items()}
        finally:
            for i in range(L):
                dec.layers[i] = orig[i]
        log(f"  steer lens L{L}: {time.time() - t0:.0f}s")
    return results


def score_words(model, lens: dict, words: set[str]):
    dec = {}

    def w(i):
        if i not in dec:
            dec[i] = model.tok.decode([i]).strip().lower()
        return dec[i]

    cands, null = [], 0
    for L, r in lens.items():
        for toks in r["rand"]:
            null = max(null, sum(w(i) in words for i in toks))
        for key, sign in (("up", 1), ("down", -1)):
            for n, toks in enumerate(r[key]):
                s = sum(w(i) in words for i in toks)
                if s:
                    cands.append({"layer": L, "neuron": n, "sign": sign, "score": s, "tokens": [w(i) for i in toks]})
    cands.sort(key=lambda c: -c["score"])
    return [c for c in cands if c["score"] > null], null


@torch.no_grad()
def contrast(model, pairs: list[dict], layers: list[int]):
    acts = {"with": [], "without": []}
    for key in acts:
        for p in pairs:
            h = []
            hooks = [model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, a: h.append(a[0][0, -1].float())) for L in layers]
            model.hf(input_ids=torch.tensor([model.tok(p[key], add_special_tokens=False)["input_ids"]]), use_cache=False)
            for x in hooks:
                x.remove()
            acts[key].append(torch.stack(h))
    A, B = torch.stack(acts["with"]), torch.stack(acts["without"])  # [P, nL, N]
    diff = A.mean(0) - B.mean(0)
    spread = torch.cat([A, B]).std(0) + 1e-6
    z = diff / spread
    v, i = z.abs().flatten().topk(200)
    out = []
    for val, idx in zip(v.tolist(), i.tolist()):
        li, n = divmod(idx, z.shape[1])
        out.append({"layer": layers[li], "neuron": n, "sign": 1 if z[li, n] > 0 else -1, "score": round(val, 2)})
    return out


def sweep(model, cands: list[dict], scale: list[float], words: set[str], n_controls: int = 24, max_new: int = 48, log=print):
    """Generate for each candidate (and random controls) and count replies that contain the words."""
    from .model import seed_from

    units = [("neuron", c["layer"], c["neuron"]) for c in cands]
    layers_used = sorted({c["layer"] for c in cands})
    units += [("random", layers_used[i % len(layers_used)], 10_000 + i) for i in range(n_controls)]
    hits = {u: 0 for u in units}
    t0 = time.time()
    for p in SWEEP_PROMPTS:
        ids = model.chat_ids(p)
        base_words = set(WORD.findall(model.generate_rows(ids, [{}], max_new=max_new)[0].lower()))
        for d in SWEEP_DOSES:
            for s in range(0, len(units), 24):
                chunk = units[s:s + 24]
                rows = []
                for kind, L, n in chunk:
                    v = model.neuron_vector(L, n, d, scale) if kind == "neuron" else model.random_vector(L, d, scale, seed_from("sweep", L, n))
                    rows.append({L: v})
                for u, text in zip(chunk, model.generate_rows(ids, rows, max_new=max_new)):
                    if (set(WORD.findall(text.lower())) - base_words) & words:
                        hits[u] += 1
        log(f"  sweep {p!r}: {time.time() - t0:.0f}s")
    bar = max(hits[u] for u in units if u[0] == "random")
    kept = []
    for c in cands:
        h = hits[("neuron", c["layer"], c["neuron"])]
        if h > bar:
            kept.append(dict(c, sweep_hits=h, control_best=bar, sweep_n=len(SWEEP_PROMPTS) * len(SWEEP_DOSES)))
    kept.sort(key=lambda c: -c["sweep_hits"])
    return kept, bar


def find(model, recipe, name: str, words: list[str] | None, contrast_file: Path | None, layers: list[int],
         top: int = 40, out: Path | None = None, log=print):
    from .recipe import Lever, Recipe

    words_set = {w.strip().lower() for w in (words or []) if w.strip()}
    if contrast_file:
        pairs = [json.loads(l) for l in open(contrast_file) if l.strip()]
        cands = contrast(model, pairs, layers)[:top]
        log(f"contrast screen: top {len(cands)} by |z|")
    else:
        if not words_set:
            raise SystemExit("give --words or --contrast")
        lens = steer_lens(model, layers, recipe.scale, log=log)
        cands, null = score_words(model, lens, words_set)
        log(f"steer-lens screen: {len(cands)} neuron poles beat the best random direction ({null} word hits)")
        cands = cands[:top]
    if not words_set:
        log("no --words given: skipping the generation sweep; candidates are ranked by the screen only")
        kept = cands
    else:
        kept, bar = sweep(model, cands, recipe.scale, words_set, log=log)
        log(f"sweep: {len(kept)} of {len(cands)} beat every random control (best control: {bar} replies)")
    found = Recipe(recipe.model, recipe.revision, recipe.scale, {})
    for j, c in enumerate(kept[:12]):
        key = f"{name}-{j + 1}"
        dose = 0.6 * c["sign"]
        found.levers[key] = Lever(key, [(c["layer"], c["neuron"], dose)], f"Candidate for: {name}",
                                  {"screen": {k: v for k, v in c.items() if k not in ("tokens",)}, "tested": False})
    if out:
        found.save(out)
        log(f"candidates written to {out}. Next: neuronaut try {name}-1 \"some prompt\" --recipe {out}, then neuronaut test {out}")
    return found
