"""Export a static explorer (no server needed): web app + atlas + recipe + recorded outputs + measured ripples.

    neuronaut site --atlas atlas --rollouts results/validation/rollouts.jsonl --prompts data/heldout_prompts.json --out site

Recorded outputs come from a frozen test run (sampled replies for each lever, its first random control and the
unsteered model). Ripples are measured here with the model, for every lever on the first `--ripple-prompts` prompts.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

WEB = Path(__file__).parent / "web"


def export_columns(model, atlas_out: Path):
    """Unit write directions per layer as float16 [N, d] (neuron-major), for building pushes in the browser."""
    import numpy as np

    for L in range(model.n_layers):
        f = atlas_out / f"down_{L:02d}.f16"
        if f.exists():
            continue
        W = model.layers[L].mlp.down_proj.weight.detach().float()
        U = (W / W.norm(dim=0, keepdim=True)).T.contiguous().numpy().astype(np.float16)
        U.tofile(f)


def concept_index(atlas: Path, out: Path, per_word: int = 120):
    """Word -> neurons, for concept search without a model: neurons that write the word (logit lens, either sign) and
    neurons whose strongest firing in the atlas text was on the word. Saved as atlas/words.json."""
    import re

    meta = json.loads((atlas / "meta.json").read_text())
    word = re.compile(r"^[a-z][a-z'\-]{2,}$")
    idx: dict[str, list] = {}
    for L in range(meta["layers"]):
        rows = json.loads((atlas / f"layer_{L:02d}.json").read_text())
        for n, r in enumerate(rows):
            seen = set()
            for kind, words in (("writes", r["w+"]), ("writes-", r["w-"])):
                for rank, w in enumerate(words):
                    w = w.strip().lower()
                    if word.match(w) and (w, kind) not in seen:
                        seen.add((w, kind)); idx.setdefault(w, []).append([L, n, 0 if kind == "writes" else 1, rank])
            for rank, c in enumerate(r.get("c+", [])[:4]):
                w = c["tok"].strip().lower()
                if word.match(w) and (w, "fires") not in seen:
                    seen.add((w, "fires")); idx.setdefault(w, []).append([L, n, 2, rank])
    for w in idx:
        idx[w] = sorted(idx[w], key=lambda e: (e[3], -e[0]))[:per_word]
    (out / "words.json").write_text(json.dumps({"kinds": ["writes", "writes when negative", "fires on"], "words": idx},
                                               separators=(",", ":")))
    return len(idx)


def export_web_model(onnx_dir: Path, out: Path, n_layers: int, hidden: int):
    graph = onnx_dir / "onnx" / "decoder_model_merged_q4_steer.onnx"
    (out / "model").mkdir(exist_ok=True)
    shutil.copy(graph, out / "model" / graph.name)
    cfg = {"repo": "onnx-community/Qwen3.5-0.8B-ONNX", "revision": (onnx_dir / "REVISION").read_text().strip(),
           "graph": f"model/{graph.name}", "dataName": "decoder_model_merged_q4_steer.onnx_data",
           "layers": n_layers, "hidden": hidden,
           "note": "edited graph (adds a steer input) served here; weights stream from the onnx-community repo at this revision"}
    (out / "webmodel.json").write_text(json.dumps(cfg, indent=1))


def export_site(model, recipe, atlas: Path, out: Path, rollouts: Path | None, prompts: Path | None, ripple_prompts: int = 6,
                onnx_dir: Path | None = None):
    from .server import State, api_ripple

    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(WEB, out, dirs_exist_ok=True)
    shutil.copytree(atlas, out / "atlas", dirs_exist_ok=True)
    export_columns(model, out / "atlas")
    concept_index(atlas, out / "atlas")
    if onnx_dir:
        export_web_model(onnx_dir, out, model.n_layers, model.d_model)
    rec = {"model": recipe.model, "revision": recipe.revision, "scale": recipe.scale,
           "levers": {k: v.to_json() | {"name": k} for k, v in recipe.levers.items()}}
    (out / "recipe.json").write_text(json.dumps(rec, ensure_ascii=False))

    recorded = {"prompts": [], "outputs": {}, "ripples": {}}
    if prompts:
        P = json.loads(Path(prompts).read_text())
        recorded["prompts"] = [p["text"] if isinstance(p, dict) else p for p in P["prompts"]]
    if rollouts:
        rows = {}
        for line in open(rollouts):
            r = json.loads(line)
            if r.get("decoding") != "t0.7":
                continue
            rows[(r["cond"], r["framing"], r["prompt_id"])] = r["text"].strip()
        for name in recipe.levers:
            for (cond, fr, pi), text in rows.items():
                if cond != name:
                    continue
                recorded["outputs"][f"{name}|{fr}|{pi}"] = {
                    "unsteered": rows.get(("base", fr, pi), ""), "lever": text,
                    "control": rows.get((f"ctrl:{name}:0", fr, pi), "")}
    State.model, State.recipe = model, recipe
    for name in recipe.levers:
        for pi in range(min(ripple_prompts, len(recorded["prompts"]))):
            recorded["ripples"][f"{name}|{pi}"] = api_ripple({"lever": name, "prompt": recorded["prompts"][pi]})
        print(f"  ripples: {name}", flush=True)
    (out / "recorded.json").write_text(json.dumps(recorded, ensure_ascii=False, separators=(",", ":")))
    print(f"static explorer written to {out}")
