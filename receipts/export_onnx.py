"""Add a steering input to an ONNX decoder so a browser (or any ONNX Runtime) can push neurons live.

The surgery: a new graph input `steer` of shape [num_layers, hidden] (float32, zeros = unsteered). For each layer L,
row L is gathered and added to the MLP output (`/model/layers.L/mlp/down_proj/MatMul/output_0`), and every consumer of
that tensor is rewired to the sum. This is exactly the additive push the Python engine applies with a forward hook.
Weights are untouched: the edited graph keeps pointing at the original external data file.

    python -m receipts.export_onnx build/onnx-src build/onnx-steer
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np


def add_steer_input(src: Path, dst: Path, data_name: str) -> int:
    import onnx
    from onnx import TensorProto, helper

    m = onnx.load(str(src), load_external_data=False)
    g = m.graph
    outs = {}
    for node in g.node:
        for o in node.output:
            if o.endswith("/mlp/down_proj/MatMul/output_0"):
                L = int(o.split("/model/layers.")[1].split("/")[0])
                outs[L] = o
    if not outs:
        raise SystemExit("no MLP down_proj outputs found; is this a Qwen-style ONNX export?")
    n_layers = max(outs) + 1
    hidden = None
    for i in g.input:
        if i.name == "inputs_embeds":
            hidden = i.type.tensor_type.shape.dim[-1].dim_value
    g.input.append(helper.make_tensor_value_info("steer", TensorProto.FLOAT, [n_layers, hidden]))
    new_nodes = []
    for L, o in sorted(outs.items()):
        idx = f"steer_idx_{L}"
        g.initializer.append(helper.make_tensor(idx, TensorProto.INT64, [], [L]))
        row, summed = f"/steer/row_{L}", f"/model/layers.{L}/mlp/steered_output_0"
        for node in g.node:  # rewire consumers before adding the new nodes
            for k, inp in enumerate(node.input):
                if inp == o:
                    node.input[k] = summed
        for out in g.output:
            if out.name == o:
                out.name = summed
        new_nodes += [helper.make_node("Gather", ["steer", idx], [row], name=f"/steer/gather_{L}", axis=0),
                      helper.make_node("Add", [o, row], [summed], name=f"/steer/add_{L}")]
    # insert each Add right after its producer to keep topological order
    order = list(g.node)
    for L, o in sorted(outs.items(), reverse=True):
        pos = next(j for j, n in enumerate(order) if o in n.output)
        order[pos + 1:pos + 1] = [n for n in new_nodes if n.name in (f"/steer/gather_{L}", f"/steer/add_{L}")]
    del g.node[:]
    g.node.extend(order)
    for t in g.initializer:
        for e in t.external_data:
            if e.key == "location":
                e.value = data_name
    onnx.save(m, str(dst))
    return n_layers


class OnnxDecoder:
    """Greedy decoding with the merged decoder (states for DeltaNet layers, KV cache for attention layers)."""

    def __init__(self, folder: Path, decoder_file: str):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        so = ort.SessionOptions()
        self.dec = ort.InferenceSession(str(folder / decoder_file), so, providers=["CPUExecutionProvider"])
        self.emb = ort.InferenceSession(str(folder / "onnx" / "embed_tokens_q4.onnx"), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.inputs = {i.name: i for i in self.dec.get_inputs()}
        self.has_steer = "steer" in self.inputs

    def _empty_state(self):
        st = {}
        for name, i in self.inputs.items():
            if name.startswith("past_"):
                shape = [1 if (isinstance(d, str) and "batch" in d) else (0 if isinstance(d, str) else d) for d in i.shape]
                st[name] = np.zeros(shape, dtype=np.float32)
        return st

    def logits(self, ids, steer=None, state=None, start=0):
        emb = self.emb.run(None, {self.emb.get_inputs()[0].name: np.array([ids], dtype=np.int64)})[0]
        pos = np.arange(start, start + len(ids), dtype=np.int64)[None, None, :].repeat(3, 0)
        feed = {"inputs_embeds": emb, "attention_mask": np.ones((1, start + len(ids)), dtype=np.int64), "position_ids": pos}
        feed.update(state if state is not None else self._empty_state())
        if self.has_steer:
            feed["steer"] = steer if steer is not None else np.zeros(self.inputs["steer"].shape, dtype=np.float32)
        outs = self.dec.run(None, feed)
        names = [o.name for o in self.dec.get_outputs()]
        new_state = {}
        for n, v in zip(names, outs):
            if n.startswith("present."):
                new_state["past_key_values." + n[len("present."):]] = v
            elif n.startswith("present_"):
                new_state["past_" + n[len("present_"):]] = v
        return outs[0][0, -1], new_state

    def generate(self, ids, steer=None, max_new=48, stop=(248046, 248044)):
        out, state, start, cur = [], None, 0, list(ids)
        for _ in range(max_new):
            lg, state = self.logits(cur, steer, state, start)
            start += len(cur)
            t = int(lg.argmax())
            if t in stop:
                break
            out.append(t)
            cur = [t]
        return self.tok.decode(out)


def main(src: str, dst: str):
    src, dst = Path(src), Path(dst)
    (dst / "onnx").mkdir(parents=True, exist_ok=True)
    data = "decoder_model_merged_q4_steer.onnx_data"
    n = add_steer_input(src / "onnx/decoder_model_merged_q4.onnx", dst / "onnx/decoder_model_merged_q4_steer.onnx", data)
    for f in ("onnx/decoder_model_merged_q4.onnx_data",):
        tgt = dst / "onnx" / data
        if not tgt.exists():
            try:
                tgt.hardlink_to(src / f)
            except OSError:
                shutil.copy(src / f, tgt)
    for f in ("onnx/embed_tokens_q4.onnx", "onnx/embed_tokens_q4.onnx_data", "config.json", "generation_config.json",
              "tokenizer.json", "tokenizer_config.json", "REVISION"):
        if (src / f).exists() and not (dst / f).exists():
            shutil.copy(src / f, dst / f)
    print(f"steer input added for {n} layers -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
