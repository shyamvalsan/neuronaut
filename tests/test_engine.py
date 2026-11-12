"""Engine tests on a tiny randomly initialised Llama (no downloads, a few seconds on CPU)."""
import json

import pytest
import torch

from receipts.model import Model
from receipts.recipe import Lever, Recipe, control_vectors, lever_vectors


class _Tok:
    eos_token_id = 0
    pad_token_id = 0
    unk_token_id = None

    def convert_tokens_to_ids(self, t):
        return None

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False, **kw):
        return " ".join(m["content"] for m in msgs)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1 + (ord(c) % 60) for c in text[:12]]}


@pytest.fixture(scope="module")
def tiny():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    return Model.wrap(LlamaForCausalLM(cfg).eval(), _Tok(), name="tiny-llama")


def test_shapes(tiny):
    assert (tiny.n_layers, tiny.d_model, tiny.n_neurons) == (3, 32, 48)


def test_push_equals_neuron_firing_harder(tiny):
    """Adding a*W_down[:, n] to the MLP output is the same as adding a to neuron n's activation."""
    L, n, a = 1, 7, 2.5
    scale = [1.0, 3.0, 1.0]
    x = torch.randint(1, 64, (1, 6))
    mlp = tiny.layers[L].mlp
    captured = {}

    def pre(mod, inp):
        h = inp[0].clone()
        h[..., n] += a
        return (h,)

    with torch.no_grad():
        hook = mlp.down_proj.register_forward_pre_hook(pre)
        manual = tiny.hf(input_ids=x).logits
        hook.remove()
        dose = a * tiny.down_col(L, n).norm().item() / scale[L]
        with tiny.pushing({L: tiny.neuron_vector(L, n, dose, scale)}):
            pushed = tiny.hf(input_ids=x).logits
    assert torch.allclose(manual, pushed, atol=1e-5)


def test_push_is_removed_afterwards(tiny):
    x = torch.randint(1, 64, (1, 5))
    with torch.no_grad():
        before = tiny.hf(input_ids=x).logits
        with tiny.pushing({0: torch.ones(32)}):
            during = tiny.hf(input_ids=x).logits
        after = tiny.hf(input_ids=x).logits
    assert not torch.allclose(before, during)
    assert torch.equal(before, after)


def test_control_has_same_size_and_differs(tiny):
    scale = [1.0, 2.0, 4.0]
    lv = Lever("x", [(2, 5, -0.6)])
    a, b = lever_vectors(tiny, lv, scale)[2], control_vectors(tiny, lv, scale)[2]
    assert abs(a.norm().item() - 0.6 * 4.0) < 1e-4 and abs(b.norm().item() - 0.6 * 4.0) < 1e-4
    assert torch.nn.functional.cosine_similarity(a, b, dim=0).abs() < 0.9
    assert torch.equal(b, control_vectors(tiny, lv, scale)[2])  # deterministic


def test_batched_rows_are_independent(tiny):
    """Row 0 unsteered, row 1 pushed: row 0 must match a plain unsteered run."""
    ids = [3, 9, 4, 12]
    plain = tiny.generate_rows(ids, [{}], max_new=5)
    both = tiny.generate_rows(ids, [{}, {1: 5 * torch.ones(32)}], max_new=5)
    assert both[0] == plain[0]


def test_lever_parse_and_recipe_roundtrip(tmp_path):
    lv = Lever.parse("L18N2521:+0.6")
    assert lv.parts == [(18, 2521, 0.6)]
    combo = Lever.parse("L18N2521:0.4+L16N157:-0.5")
    assert combo.parts == [(18, 2521, 0.4), (16, 157, -0.5)]
    with pytest.raises(ValueError):
        Lever.parse("deity")
    r = Recipe("m", "rev", [1.0] * 24, {"deity": Lever("deity", [(18, 2521, 0.6)], "god talk", {"tagged": 23, "n": 24})})
    r.save(tmp_path / "r.json")
    back = Recipe.load(tmp_path / "r.json")
    assert back.levers["deity"].parts == [(18, 2521, 0.6)] and back.levers["deity"].receipts["tagged"] == 23
    assert json.loads((tmp_path / "r.json").read_text())["format"] == "receipts/recipe@1"


def test_builtin_recipe_loads():
    from receipts.cli import BUILTIN

    r = Recipe.load(BUILTIN)
    assert len(r.scale) == 24 and r.levers["deity"].parts == [(18, 2521, 0.6)]
    assert sum(lv.receipts["held_up"] for lv in r.levers.values()) == 8
