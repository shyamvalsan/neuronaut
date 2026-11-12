"""End-to-end test of `receipts test` on the tiny model, with stand-in judges (no API calls)."""
import json

from receipts.recipe import Lever, Recipe
from receipts.testing import finish, freeze, generate, make_packets
from test_engine import tiny  # noqa: F401  (fixture)


def test_full_pipeline(tiny, tmp_path):
    prompts = tmp_path / "p.json"
    prompts.write_text(json.dumps({"prompts": ["a b", "c d", "e f", "g h"], "framings": {"plain": None, "sys": "be brief"},
                                   "capability": [["one plus one", ["2"]]]}))
    recipe = Recipe("tiny", None, [1.0, 1.0, 1.0], {"loud": Lever("loud", [(1, 3, 0.8)], "a test trait")})
    frozen = freeze(recipe, tmp_path / "run", prompts, controls=2, max_new=4)
    assert frozen["hash"] and (tmp_path / "run" / "FROZEN.json").exists()
    generate(tiny, frozen, tmp_path / "run", log=lambda *a: None)
    rows = [json.loads(l) for l in open(tmp_path / "run" / "rollouts.jsonl")]
    assert len(rows) == 2 * 2 * 4 * 4  # decodings x framings x prompts x (base + lever + 2 controls)
    n, k = make_packets(frozen, tmp_path / "run", n_packets=2)
    J = tmp_path / "run" / "judge"
    key = {x["id"]: x for x in json.loads((J / "key.json").read_text())}
    for p in range(2):
        items = json.loads((J / f"packet{p}.json").read_text())
        for tag, coh in (("A", 4), ("B", 3)):  # stand-in judges: tag the trait on lever replies only
            labels = [{"id": it["id"], "traits": ["loud"] if key[it["id"]]["cond"] == "loud" else [], "coherence": coh} for it in items]
            (J / f"packet{p}.judge{tag}.json").write_text(json.dumps(labels))
    res = finish(recipe, None, tmp_path / "run", log=lambda *a: None)
    r = res["loud"]
    assert r["tagged"] == r["n"] == 4 and r["control"] == 0 and r["p"] < 0.05
    assert (tmp_path / "run" / "report.html").exists() and (tmp_path / "run" / "recipe.json").exists()
