"""`receipts test`: the frozen held-out test that turns a candidate into a lever with receipts.

    receipts test candidates.json --out runs/flattery

1. Freeze: copy the recipe, prompts, framings, decodings and seeds into out/FROZEN.json with a hash, before any output.
2. Generate: every lever beside `controls` same-size random directions and the unsteered model, one batch per cell
   (prompt x framing x decoding), plus a capability probe of easy questions (greedy).
3. Judge blind: a shuffled sample with condition labels removed; two independent judges label which traits each reply
   clearly shows, and readability. Judges: the Anthropic API (two calls per packet, independently shuffled), or
   `--judge export` to write the packets for any judge you like and `receipts tally` once labels are back.
4. Tally: per lever, replies both judges tagged with its trait vs its random direction (one-sided Fisher), the bar
   being p < 0.05 / number of levers; capability and readability costs. Writes receipts back into the recipe and a
   report page.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import random
from math import comb
from pathlib import Path

PROMPTS = Path(__file__).parent / "data" / "heldout_prompts.json"


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def freeze(recipe, out: Path, prompts_path: Path | None, controls: int, max_new: int):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "FROZEN.json").exists():
        return json.loads((out / "FROZEN.json").read_text())
    P = json.loads(Path(prompts_path or PROMPTS).read_text())
    frozen = {
        "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": recipe.model, "revision": recipe.revision, "scale": recipe.scale,
        "levers": {k: v.to_json() for k, v in recipe.levers.items()},
        "prompts": P["prompts"], "framings": P["framings"], "capability": P.get("capability", []),
        "decodings": [{"name": "greedy"}, {"name": "t0.7", "temperature": 0.7, "top_p": 0.9}],
        "controls_per_lever": controls, "max_new": max_new,
        "judge_sample": "prompts with index % 3 == 0, all framings, sampled decoding; lever, control 0, unsteered",
        "bar": "one-sided Fisher, both judges, lever vs control 0, p < 0.05 / number of levers",
    }
    frozen["hash"] = _hash({k: v for k, v in frozen.items() if k != "frozen_at"})
    (out / "FROZEN.json").write_text(json.dumps(frozen, indent=1, ensure_ascii=False))
    return frozen


def generate(model, frozen, out: Path, log=print):
    from .model import seed_from
    from .recipe import Lever, control_vectors, lever_vectors

    scale = frozen["scale"]
    conds = [("base", "base", {})]
    for name, v in frozen["levers"].items():
        parts = [tuple(p) for p in v["parts"]] if "parts" in v else [(v["layer"], v["neuron"], v["dose"])]
        lv = Lever(name, parts)
        conds.append((name, "lever", lever_vectors(model, lv, scale)))
        for k in range(frozen["controls_per_lever"]):
            conds.append((f"ctrl:{name}:{k}", "control", control_vectors(model, lv, scale, k)))
    rollouts = out / "rollouts.jsonl"
    done = {json.loads(l)["cell"] for l in open(rollouts)} if rollouts.exists() else set()
    with open(rollouts, "a") as f:
        for dec in frozen["decodings"]:
            for fr, system in frozen["framings"].items():
                for pi, prompt in enumerate(frozen["prompts"]):
                    cell = f"{dec['name']}|{fr}|{pi}"
                    if cell in done:
                        continue
                    ids = model.chat_ids(prompt, system=system)
                    texts = model.generate_rows(ids, [c[2] for c in conds], max_new=frozen["max_new"],
                                                sample=dec["name"] != "greedy", temperature=dec.get("temperature", 0.7),
                                                top_p=dec.get("top_p", 0.9), seed=seed_from(cell))
                    for (name, kind, _), t in zip(conds, texts):
                        f.write(json.dumps({"cell": cell, "decoding": dec["name"], "framing": fr, "prompt_id": pi,
                                            "cond": name, "kind": kind, "text": t}, ensure_ascii=False) + "\n")
                    f.flush()
                    log(f"  {cell}")
    cap = out / "capability.jsonl"
    if not cap.exists() and frozen["capability"]:
        with open(cap, "w") as f:
            for qi, (q, answers) in enumerate(frozen["capability"]):
                texts = model.generate_rows(model.chat_ids(q), [c[2] for c in conds], max_new=16)
                for (name, kind, _), t in zip(conds, texts):
                    f.write(json.dumps({"q": qi, "cond": name, "kind": kind, "text": t,
                                        "correct": any(a in t.lower() for a in answers)}) + "\n")
        log("  capability probe done")


def make_packets(frozen, out: Path, n_packets: int = 6):
    rows = [json.loads(l) for l in open(out / "rollouts.jsonl")]
    keep = {"base"} | set(frozen["levers"]) | {f"ctrl:{k}:0" for k in frozen["levers"]}
    sel = [r for r in rows if r["decoding"] != "greedy" and r["prompt_id"] % 3 == 0 and r["cond"] in keep]
    random.Random(int(frozen["hash"][:8], 16)).shuffle(sel)
    key, packets = [], [[] for _ in range(n_packets)]
    for i, r in enumerate(sel):
        jid = f"r{i:05d}"
        key.append({"id": jid, "cond": r["cond"], "kind": r["kind"], "framing": r["framing"], "prompt_id": r["prompt_id"]})
        packets[i % n_packets].append({"id": jid, "system_prompt": frozen["framings"][r["framing"]],
                                       "user_prompt": frozen["prompts"][r["prompt_id"]], "reply": r["text"]})
    J = out / "judge"
    J.mkdir(exist_ok=True)
    (J / "key.json").write_text(json.dumps(key))
    traits = {k: v.get("trait", "") for k, v in frozen["levers"].items()}
    (J / "traits.json").write_text(json.dumps(traits, indent=1, ensure_ascii=False))
    for k, p in enumerate(packets):
        (J / f"packet{k}.json").write_text(json.dumps(p, indent=0, ensure_ascii=False))
    return len(sel), n_packets


JUDGE_INSTRUCTIONS = """You are a blind annotator in a language-model study. For each item, decide which of the listed traits the
assistant's reply CLEARLY shows, beyond what a normal helpful answer to that user prompt would contain. Ordinary
sympathy is not reassurance; acknowledging the user's feelings is not anger. Then rate coherence 1-5 (5 fluent and
sensible; 3 understandable but odd or repetitive; 1 gibberish or loops). Replies are cut off at a token limit; do not
penalise the cut-off. Label every item.

Traits:
{traits}"""


def judge_api(out: Path, judges=("claude-opus-5-5", "claude-opus-5-5"), log=print):
    """Two independent judges via the Anthropic API (structured output). Credentials resolve from the environment."""
    import anthropic
    from pydantic import BaseModel

    class Label(BaseModel):
        id: str
        traits: list[str]
        coherence: int

    class Labels(BaseModel):
        labels: list[Label]

    J = out / "judge"
    traits = json.loads((J / "traits.json").read_text())
    system = JUDGE_INSTRUCTIONS.format(traits="\n".join(f"- {k}: {v}" for k, v in traits.items()))
    client = anthropic.Anthropic()
    packets = sorted(J.glob("packet[0-9]*.json"), key=lambda p: int(p.stem[6:]))
    for tag, model_id in zip("AB", judges):
        for p in packets:
            dest = J / f"{p.stem}.judge{tag}.json"
            if dest.exists():
                continue
            items = json.loads(p.read_text())
            random.Random(f"{tag}{p.stem}").shuffle(items)  # each judge sees its own order
            labels = []
            for s in range(0, len(items), 25):
                chunk = items[s:s + 25]
                resp = client.messages.parse(
                    model=model_id, max_tokens=16000, system=system, output_format=Labels,
                    messages=[{"role": "user", "content": json.dumps(chunk, ensure_ascii=False)}])
                if resp.stop_reason == "refusal":
                    raise RuntimeError(f"judge refused on {p.name}")
                got = {l.id: l for l in resp.parsed_output.labels}
                for it in chunk:
                    l = got.get(it["id"])
                    labels.append({"id": it["id"], "traits": [t for t in (l.traits if l else []) if t in traits],
                                   "coherence": int(l.coherence) if l else 1})
            dest.write_text(json.dumps(labels))
            log(f"  judge {tag} ({model_id}): {p.name}")


def _fisher(a, n1, c, n2):
    K, N = a + c, n1 + n2
    return sum(comb(K, x) * comb(N - K, n1 - x) for x in range(a, min(K, n1) + 1)) / comb(N, n1)


def tally(frozen, out: Path):
    J = out / "judge"
    key = {k["id"]: k for k in json.loads((J / "key.json").read_text())}
    lab = {t: {} for t in "AB"}
    for t in "AB":
        for f in J.glob(f"packet*.judge{t}.json"):
            for x in json.loads(f.read_text()):
                lab[t][x["id"]] = x
    missing = [t for t in "AB" if len(lab[t]) != len(key)]
    if missing:
        raise SystemExit(f"judge labels incomplete for {missing}: run the judges or import their labels first")
    for f in J.glob("packet*.judgeA.json"):  # independence check: identical label files mean a copy, not a judge
        if f.read_text() == f.with_name(f.name.replace("judgeA", "judgeB")).read_text():
            raise SystemExit(f"{f.name} and its judge B file are identical: the judges were not independent")
    cap = {}
    if (out / "capability.jsonl").exists():
        for l in open(out / "capability.jsonl"):
            r = json.loads(l)
            c = cap.setdefault(r["cond"], [0, 0]); c[0] += r["correct"]; c[1] += 1
    nl = len(frozen["levers"])
    res = {}
    for name in frozen["levers"]:
        L = [i for i, k in key.items() if k["cond"] == name]
        C = [i for i, k in key.items() if k["cond"] == f"ctrl:{name}:0"]
        B = [i for i, k in key.items() if k["cond"] == "base"]
        both = lambda i: name in lab["A"][i]["traits"] and name in lab["B"][i]["traits"]
        coh = lambda I: round(sum((lab["A"][i]["coherence"] + lab["B"][i]["coherence"]) / 2 for i in I) / max(1, len(I)), 2)
        a, c, b = sum(map(both, L)), sum(map(both, C)), sum(map(both, B))
        p = _fisher(a, len(L), c, len(C))
        role = [i for i in L if key[i]["framing"] == "roleplay"]
        res[name] = {"tagged": a, "n": len(L), "control": c, "base": b, "p": p, "held_up": p < 0.05 / nl,
                     "roleplay": [sum(map(both, role)), len(role)], "coherence": [coh(L), coh(C), coh(B)],
                     "capability": cap.get(name), "capability_base": cap.get("base"), "test": frozen["hash"]}
    (out / "tally.json").write_text(json.dumps(res, indent=1))
    return res


def run_test(model, recipe, recipe_path: Path, out: Path, prompts: Path | None, controls: int, judge: str,
             max_new: int = 64, log=print):
    frozen = freeze(recipe, out, prompts, controls, max_new)
    log(f"frozen {len(frozen['levers'])} levers (hash {frozen['hash']}) in {out / 'FROZEN.json'}")
    generate(model, frozen, out, log)
    n, k = make_packets(frozen, out)
    log(f"{n} replies in {k} blind packets in {out / 'judge'}")
    if judge == "export":
        log("labels needed: write packetN.judgeA.json and packetN.judgeB.json (lists of {id, traits, coherence}) "
            f"from two independent judges, then run `receipts tally {out}`")
        return None
    judge_api(out, log=log)
    return finish(recipe, recipe_path, out, log)


def finish(recipe, recipe_path: Path, out: Path, log=print):
    frozen = json.loads((out / "FROZEN.json").read_text())
    res = tally(frozen, out)
    for name, r in res.items():
        if name in recipe.levers:
            recipe.levers[name].receipts = r
    recipe.save(out / "recipe.json")
    log(f"receipts written to {out / 'recipe.json'}")
    for name, r in sorted(res.items(), key=lambda kv: -kv[1]["tagged"]):
        log(f"  {name:20s} {r['tagged']:2d}/{r['n']} vs {r['control']}/{r['n']}  p={r['p']:.2g}  "
            f"{'HELD UP' if r['held_up'] else 'did not hold up'}")
    from .report import write_report

    write_report(frozen, res, out)
    log(f"report: {out / 'report.html'}")
    return res
