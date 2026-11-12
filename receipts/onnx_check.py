"""Receipts for the 4-bit browser model: run held-up levers through the exact browser path (steered ONNX graph, fp16
unit columns from the site atlas) beside a same-size random push and the unsteered model, then write blind packets.

    python -m receipts.onnx_check site build/onnx-steer build/q4check
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

from .export_onnx import OnnxDecoder


def main(site: str, onnx_dir: str, out: str, max_new: int = 48):
    site, out = Path(site), Path(out)
    recipe = json.loads((site / "recipe.json").read_text())
    scale = recipe["scale"]
    P = json.loads((out / "prompts.json").read_text())
    dec = OnnxDecoder(Path(onnx_dir), "onnx/decoder_model_merged_q4_steer.onnx")
    H, Lc = 1024, 24
    cols = {}

    def column(L, n):
        if L not in cols:
            cols[L] = np.fromfile(site / "atlas" / f"down_{L:02d}.f16", dtype=np.float16).reshape(-1, H)
        return cols[L][n].astype(np.float32)

    conds = {"base": np.zeros((Lc, H), np.float32)}
    for name, v in recipe["levers"].items():
        if not v.get("receipts", {}).get("held_up"):
            continue
        L, n, d = v["layer"], v["neuron"], v["dose"]
        s = np.zeros((Lc, H), np.float32); s[L] = d * scale[L] * column(L, n); conds[name] = s
        r = np.random.default_rng(int(hashlib.sha256(f"control|{name}".encode()).hexdigest()[:8], 16)).standard_normal(H).astype(np.float32)
        c = np.zeros((Lc, H), np.float32); c[L] = abs(d) * scale[L] * r / np.linalg.norm(r); conds[f"ctrl:{name}"] = c
    path = out / "rollouts.jsonl"
    done = {(json.loads(l)["cond"], json.loads(l)["prompt_id"]) for l in open(path)} if path.exists() else set()
    with open(path, "a") as f:
        for pi, ids in P["ids"].items():
            for name, steer in conds.items():
                if (name, int(pi)) in done:
                    continue
                text = dec.generate(ids, steer, max_new)
                f.write(json.dumps({"cond": name, "prompt_id": int(pi), "text": text}, ensure_ascii=False) + "\n")
                f.flush()
            print(f"prompt {pi} done", flush=True)
    rows = [json.loads(l) for l in open(path)]
    random.Random(4242).shuffle(rows)
    key, packets = [], [[], []]
    for i, r in enumerate(rows):
        jid = f"q{i:04d}"
        key.append({"id": jid, "cond": r["cond"], "prompt_id": r["prompt_id"]})
        packets[i % 2].append({"id": jid, "system_prompt": None, "user_prompt": P["prompts"][str(r["prompt_id"])], "reply": r["text"]})
    (out / "key.json").write_text(json.dumps(key))
    for k, p in enumerate(packets):
        (out / f"packet{k}.json").write_text(json.dumps(p, indent=0, ensure_ascii=False))
    print(f"{len(rows)} replies in 2 blind packets")


if __name__ == "__main__":
    main(*sys.argv[1:4])
