"""Tally the 4-bit browser-path check: both judges tag the lever's trait, lever vs its random push vs unsteered."""
import json
from math import comb
from pathlib import Path

D = Path(__file__).parent
key = {k["id"]: k for k in json.loads((D / "key.json").read_text())}
lab = {t: {} for t in "AB"}
for t in "AB":
    for p in range(2):
        for x in json.loads((D / f"packet{p}.judge{t}.json").read_text()):
            lab[t][x["id"]] = x
assert all(len(lab[t]) == len(key) for t in "AB")
for p in range(2):
    assert (D / f"packet{p}.judgeA.json").read_text() != (D / f"packet{p}.judgeB.json").read_text(), "identical judge files"


def fisher(a, n1, c, n2):
    K, N = a + c, n1 + n2
    return sum(comb(K, x) * comb(N - K, n1 - x) for x in range(a, min(K, n1) + 1)) / comb(N, n1)


levers = sorted({k["cond"] for k in key.values() if k["cond"] != "base" and not k["cond"].startswith("ctrl:")})
res = {}
for lv in levers:
    both = lambda i: lv in lab["A"][i]["traits"] and lv in lab["B"][i]["traits"]
    coh = lambda I: round(sum((lab["A"][i]["coherence"] + lab["B"][i]["coherence"]) / 2 for i in I) / len(I), 2)
    L = [i for i, k in key.items() if k["cond"] == lv]
    C = [i for i, k in key.items() if k["cond"] == f"ctrl:{lv}"]
    B = [i for i, k in key.items() if k["cond"] == "base"]
    a, c, b = sum(map(both, L)), sum(map(both, C)), sum(map(both, B))
    res[lv] = {"tagged": a, "n": len(L), "control": c, "base": b, "p": fisher(a, len(L), c, len(C)),
               "coherence": [coh(L), coh(C), coh(B)]}
    print(f"{lv:11s} {a}/{len(L)} vs {c}/{len(C)} (unsteered {b}/{len(B)})  p={res[lv]['p']:.2g}  coherence {res[lv]['coherence']}")
(D / "tally.json").write_text(json.dumps(res, indent=1))
