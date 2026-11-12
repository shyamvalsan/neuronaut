"""The steer lens starts from a cached mid-model state; it must equal a full forward pass with the push."""
import torch

from receipts.search import steer_lens, CONTEXTS_CHAT, CONTEXTS_PLAIN
from test_engine import tiny  # noqa: F401


def test_steer_lens_matches_full_forward(tiny):
    scale = [1.0, 2.0, 3.0]
    L, n, dose = 2, 5, 0.8
    lens = steer_lens(tiny, [L], scale, dose=dose, batch=8, k=5, n_random=4, log=lambda *a: None)
    assert len(lens[L]["up"]) == tiny.n_neurons and len(lens[L]["rand"]) == 4
    ctxs = [tiny.chat_ids(c) for c in CONTEXTS_CHAT] + [tiny.tok(c)["input_ids"] for c in CONTEXTS_PLAIN]
    with torch.no_grad():
        deltas = []
        for ids in ctxs:
            x = torch.tensor([ids])
            base = torch.log_softmax(tiny.hf(input_ids=x).logits[0, -1], -1)
            with tiny.pushing({L: tiny.neuron_vector(L, n, dose, scale)}):
                pushed = torch.log_softmax(tiny.hf(input_ids=x).logits[0, -1], -1)
            deltas.append(pushed - base)
        expected = torch.stack(deltas).mean(0).topk(5).indices.tolist()
    assert lens[L]["up"][n] == expected
