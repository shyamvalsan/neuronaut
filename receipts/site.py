"""Export a static explorer (no server needed): web app + atlas + recipe + recorded outputs + measured ripples.

    receipts site --atlas atlas --rollouts results/validation/rollouts.jsonl --prompts data/heldout_prompts.json --out site

Recorded outputs come from a frozen test run (sampled replies for each lever, its first random control and the
unsteered model). Ripples are measured here with the model, for every lever on the first `--ripple-prompts` prompts.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

WEB = Path(__file__).parent / "web"


def export_site(model, recipe, atlas: Path, out: Path, rollouts: Path | None, prompts: Path | None, ripple_prompts: int = 6):
    from .server import State, api_ripple

    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(WEB, out, dirs_exist_ok=True)
    shutil.copytree(atlas, out / "atlas", dirs_exist_ok=True)
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
