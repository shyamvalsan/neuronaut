"""Build the atlas: everything the 3D explorer shows about every MLP neuron, computed once per model.

Outputs (in `out/`):
  meta.json        model id, revision, shapes, push scale per layer, layer kinds
  layout.f32       float32 [L, N, 2]: each neuron's position in its layer's disc. Neurons whose write directions
                   (W_down columns) are similar sit close together: 2-D PCA of the unit columns, radius rank-spread.
  stats.f32        float32 [L, N, 4]: max positive activation, max negative (abs), share of tokens where it is
                   positive, mean |activation|, over the stats corpus.
  layer_XX.json    per neuron: words it writes (logit lens, both poles), text where it fires hardest (both poles),
                   strongest direct wiring from earlier neurons and to later neurons.

Wiring is computed from the weights only ("direct wiring"): the influence of neuron m (layer L') on neuron n
(layer L > L') is  typical|act_m| * ((1 + g_L) * W_gate[n]) . W_down[:, m] / scale[L],  the change in n's gate
pre-activation when m fires at its typical size, through the residual stream alone. It ignores attention and
DeltaNet mixing and per-token normalisation, so the explorer labels it "wiring", not "measured".
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import torch

WIKITEXT = "https://huggingface.co/api/datasets/Salesforce/wikitext/parquet/wikitext-2-raw-v1/train/0.parquet"


def _corpus(tok, n_tokens: int, cache: Path) -> torch.Tensor:
    f = cache / "wikitext2_train.parquet"
    if not f.exists():
        import urllib.request

        print("downloading WikiText-2 for activation statistics ...", flush=True)
        urllib.request.urlretrieve(WIKITEXT, f)
    import pyarrow.parquet as pq

    text, ids = "", []
    for chunk in pq.read_table(f).column("text").to_pylist():
        text += chunk
        if len(text) > n_tokens * 6:
            break
    ids = tok(text, add_special_tokens=False)["input_ids"][:n_tokens]
    seq = 256
    n = len(ids) // seq
    return torch.tensor(ids[: n * seq]).view(n, seq)


def _layout(W: torch.Tensor) -> np.ndarray:
    """W: [d, N] down-projection. 2-D PCA of unit columns, angle kept, radius by rank (fills the disc evenly)."""
    U = (W / W.norm(dim=0, keepdim=True)).T  # [N, d]
    U = U - U.mean(0)
    _, _, V = torch.pca_lowrank(U, q=2, center=False)
    xy = (U @ V[:, :2]).numpy()
    ang = np.arctan2(xy[:, 1], xy[:, 0])
    r = np.linalg.norm(xy, axis=1)
    rank = r.argsort().argsort() / max(1, len(r) - 1)
    rad = np.sqrt(0.08 + 0.92 * rank)
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1).astype(np.float32)


def _gains(model):
    out = []
    for li in range(model.n_layers):
        ln = model.layers[li].post_attention_layernorm.weight.detach().float()
        out.append((1 + ln) if any(k in model.name.lower() for k in ("qwen", "gemma")) else ln)
    return out


def _wiring(model, scale, typical, k_wire=8, t0=None):
    """Direct weight-based wiring: for each neuron, its k strongest inputs from earlier layers and outputs to later ones."""
    t0 = t0 or time.time()
    L, N = model.n_layers, model.n_neurons
    gains = _gains(model)
    up_v = [torch.empty(0)] * L; up_i = [torch.empty(0)] * L
    dn_v = torch.full((L, N, k_wire), 0.0); dn_i = torch.full((L, N, k_wire), -1, dtype=torch.long)
    for li in range(1, L):
        G = model.layers[li].mlp.gate_proj.weight.detach().float() * gains[li]  # [N, D]
        best_v = torch.zeros(N, k_wire); best_i = torch.full((N, k_wire), -1, dtype=torch.long)
        for lj in range(li):
            Wd = model.layers[lj].mlp.down_proj.weight.detach().float() * typical[lj]  # [D, N]
            M = (G @ Wd) / scale[li]  # [N_li, N_lj]
            v, i = M.abs().topk(k_wire, 1)
            sv = M.gather(1, i)
            cat_v = torch.cat([best_v, sv], 1); cat_i = torch.cat([best_i, i + lj * N], 1)
            s_ = cat_v.abs().topk(k_wire, 1); best_v, best_i = cat_v.gather(1, s_.indices), cat_i.gather(1, s_.indices)
            v, i = M.abs().topk(k_wire, 0)
            sv = M.gather(0, i)
            cat_v = torch.cat([dn_v[lj], sv.T], 1); cat_i = torch.cat([dn_i[lj], (i + li * N).T], 1)
            s_ = cat_v.abs().topk(k_wire, 1); dn_v[lj], dn_i[lj] = cat_v.gather(1, s_.indices), cat_i.gather(1, s_.indices)
        up_v[li], up_i[li] = best_v, best_i
        print(f"  wiring layer {li}  {time.time() - t0:.0f}s", flush=True)
    return up_v, up_i, dn_v, dn_i


def _wires(vs, is_, N):
    return [[int(x) // N, int(x) % N, float(f"{v:.3g}")] for v, x in zip(vs.tolist(), is_.tolist()) if x >= 0]


def rebuild_wiring(model, out: Path, k_wire: int = 8):
    """Recompute only the wiring in an existing atlas (layer files patched in place)."""
    meta = json.loads((out / "meta.json").read_text())
    L, N = meta["layers"], meta["neurons"]
    stats = np.fromfile(out / "stats.f32", dtype=np.float32).reshape(L, N, 4)
    up_v, up_i, dn_v, dn_i = _wiring(model, meta["scale"], torch.from_numpy(stats[:, :, 3]), k_wire)
    for li in range(L):
        f = out / f"layer_{li:02d}.json"
        rows = json.loads(f.read_text())
        for n, row in enumerate(rows):
            row["in"] = _wires(up_v[li][n], up_i[li][n], N) if li > 0 else []
            row["out"] = _wires(dn_v[li, n], dn_i[li, n], N)
        f.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
    print("wiring rebuilt", flush=True)


def build_atlas(model, out: Path, tokens: int = 50_000, k_words: int = 8, k_ctx: int = 4, k_wire: int = 8):
    from .model import cache_dir

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, L, N, D = model.tok, model.n_layers, model.n_neurons, model.d_model
    chunks = _corpus(tok, tokens, cache_dir())
    B, T = chunks.shape

    # ---- pass 1: activation statistics, residual scale, top contexts ----
    mx = torch.full((L, N), -1e9); mn = torch.full((L, N), 1e9)
    pos = torch.zeros(L, N, dtype=torch.float64); mabs = torch.zeros(L, N, dtype=torch.float64)
    topv = torch.full((L, N, k_ctx), -1e9); topi = torch.zeros(L, N, k_ctx, dtype=torch.long)
    botv = torch.full((L, N, k_ctx), 1e9); boti = torch.zeros(L, N, k_ctx, dtype=torch.long)
    norms = [[] for _ in range(L)]
    cur = {"base": 0}
    hooks = []
    for li, layer in enumerate(model.layers):
        def pre(mod, inp, li=li):
            h = inp[0][:, 1:].float()
            b, t, _ = h.shape
            hf = h.reshape(-1, N)
            mx[li] = torch.maximum(mx[li], hf.max(0).values); mn[li] = torch.minimum(mn[li], hf.min(0).values)
            pos[li] += (hf > 0).sum(0).double(); mabs[li] += hf.abs().sum(0).double()
            idx = (cur["base"] + torch.arange(b).unsqueeze(1) * T + torch.arange(1, t + 1)).reshape(-1)
            kk = min(k_ctx, hf.shape[0])
            v, i = hf.topk(kk, 0)
            av, ai = torch.cat([topv[li], v.T], 1), torch.cat([topi[li], idx[i].T], 1)
            s = av.topk(k_ctx, 1); topv[li], topi[li] = s.values, ai.gather(1, s.indices)
            v, i = (-hf).topk(kk, 0)
            av, ai = torch.cat([botv[li], -v.T], 1), torch.cat([boti[li], idx[i].T], 1)
            s = (-av).topk(k_ctx, 1); botv[li], boti[li] = -s.values, ai.gather(1, s.indices)
        def post(mod, inp, o, li=li):
            o = o[0] if isinstance(o, tuple) else o
            norms[li].append(o[:, 1:].float().norm(dim=-1).reshape(-1))
        hooks += [layer.mlp.down_proj.register_forward_pre_hook(pre), layer.register_forward_hook(post)]
    lm = model.hf
    with torch.no_grad():
        for b in range(0, B, 4):
            cur["base"] = b * T
            x = chunks[b: b + 4]
            lm(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
            print(f"  stats {min(b + 4, B)}/{B} chunks  {time.time() - t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()
    ntok = B * (T - 1)
    scale = [float(torch.cat(n).median()) for n in norms]
    stats = torch.stack([mx, mn.abs(), (pos / ntok).float(), (mabs / ntok).float()], -1).numpy().astype(np.float32)
    stats.tofile(out / "stats.f32")

    # ---- layout ----
    lay = np.stack([_layout(model.layers[li].mlp.down_proj.weight.detach().float()) for li in range(L)])
    lay.tofile(out / "layout.f32")

    # ---- logit lens words ----
    head = model.hf.get_output_embeddings().weight.detach().float()
    fnorm = None
    for name, mod in model.hf.named_modules():
        if name.endswith("language_model.norm") or name == "model.norm":
            fnorm = mod
    g = (1 + fnorm.weight.detach().float()) if fnorm is not None and "qwen" in model.name.lower() else (
        fnorm.weight.detach().float() if fnorm is not None else torch.ones(D))
    vocab = [tok.decode([i]) for i in range(head.shape[0])]
    clean = torch.tensor([bool(re.fullmatch(r" ?[A-Za-z][A-Za-z'\-]{2,}", s)) for s in vocab])
    cid = clean.nonzero().squeeze(1)
    headc = head[cid]

    flat = chunks.reshape(-1).tolist()

    def snippet(gi):
        c, p = divmod(int(gi), T)
        s = flat[c * T + max(0, p - 14): c * T + p]
        return {"pre": tok.decode(s), "tok": tok.decode([flat[c * T + p]]),
                "post": tok.decode(flat[c * T + p + 1: c * T + min(T, p + 5)])}

    up_v, up_i, dn_v, dn_i = _wiring(model, scale, torch.from_numpy(stats[:, :, 3]), k_wire, t0)

    # ---- per-layer files ----
    for li in range(L):
        W = model.layers[li].mlp.down_proj.weight.detach().float() * g[:, None]
        lg = headc @ W  # [Vc, N]
        z = (lg - lg.mean(0)) / lg.std(0)
        pv, pi = z.topk(k_words, 0); nv, ni = (-z).topk(k_words, 0)
        rows = []
        for n in range(N):
            rows.append({
                "w+": [vocab[int(cid[i])].strip() for i in pi[:, n].tolist()],
                "w-": [vocab[int(cid[i])].strip() for i in ni[:, n].tolist()],
                "c+": [dict(snippet(gi), a=round(float(a), 2)) for a, gi in zip(topv[li, n].tolist(), topi[li, n].tolist()) if a > 0][:k_ctx],
                "c-": [dict(snippet(gi), a=round(float(a), 2)) for a, gi in zip(botv[li, n].tolist(), boti[li, n].tolist()) if a < 0][:k_ctx],
                "in": _wires(up_v[li][n], up_i[li][n], N) if li > 0 else [],
                "out": _wires(dn_v[li, n], dn_i[li, n], N),
            })
        (out / f"layer_{li:02d}.json").write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
        print(f"  layer file {li}  {time.time() - t0:.0f}s", flush=True)

    kinds = getattr(getattr(model.hf.config, "text_config", model.hf.config), "layer_types", None) or ["attention"] * L
    meta = {"format": "neuronaut/atlas@1", "model": model.name, "revision": model.revision, "layers": L, "neurons": N,
            "d_model": D, "scale": scale, "layer_kinds": kinds, "stats_tokens": ntok,
            "stats_fields": ["max_pos", "max_neg_abs", "frac_pos", "mean_abs"],
            "wiring": "direct, weight-based: typical|act_m| * ((1+g) W_gate[n]) . W_down[:, m] / scale[L]"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"atlas written to {out} in {time.time() - t0:.0f}s", flush=True)
