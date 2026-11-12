"""Model access for steering: find the decoder layers and MLPs, push neurons, generate.

A *push* on neuron n of layer L adds a fixed vector v to that layer's MLP output at every position. For an additive
neuron push, v = dose * scale[L] * W_down[:, n] / ||W_down[:, n]||, where scale[L] is the median norm of the residual
stream leaving layer L (measured once by `receipts atlas`). Random-direction controls use a random unit vector with the
same norm. Pushes never change the weights; they live in forward hooks that are removed afterwards.
"""
from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch

LAYER_PATHS = ("model.language_model.layers", "model.layers", "language_model.model.layers")


def _get(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def seed_from(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


@dataclass
class Model:
    """A loaded causal LM plus the handles steering needs."""

    hf: torch.nn.Module
    tok: object
    name: str
    revision: str | None
    layers: list = field(default_factory=list)
    _push: dict = field(default_factory=dict)
    _hooks: list = field(default_factory=list)

    # ---------- construction ----------
    @classmethod
    def load(cls, name: str, revision: str | None = None, threads: int | None = None, dtype=torch.float32,
             device: str = "cpu") -> "Model":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        local = os.path.isdir(name)
        tok = AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=local)
        try:
            hf = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype, local_files_only=local)
        except ValueError:  # multimodal checkpoints such as Qwen3.5 register under image-text-to-text
            from transformers import AutoModelForImageTextToText

            hf = AutoModelForImageTextToText.from_pretrained(name, revision=revision, dtype=dtype,
                                                             local_files_only=local)
        hf.to(device).eval()
        for p in hf.parameters():
            p.requires_grad_(False)
        return cls.wrap(hf, tok, name, revision)

    @classmethod
    def wrap(cls, hf, tok, name="custom", revision=None) -> "Model":
        layers = None
        for path in LAYER_PATHS:
            layers = _get(hf, path)
            if layers is not None:
                break
        if layers is None:
            raise ValueError("could not find decoder layers; supported: Qwen3/3.5, Llama, Gemma, Mistral style models")
        for i, layer in enumerate(layers):
            if not hasattr(getattr(layer, "mlp", None), "down_proj"):
                raise ValueError(f"layer {i} has no mlp.down_proj; only gated (SwiGLU-style) MLPs are supported")
        m = cls(hf=hf, tok=tok, name=name, revision=revision, layers=list(layers))
        for i, layer in enumerate(m.layers):
            m._hooks.append(layer.mlp.register_forward_hook(m._make_hook(i)))
        return m

    def _make_hook(self, i):
        def hook(mod, inp, out):
            v = self._push.get(i)
            if v is None:
                return None
            return out + v.to(out.dtype)
        return hook

    # ---------- shapes ----------
    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def d_model(self) -> int:
        return self.layers[0].mlp.down_proj.weight.shape[0]

    @property
    def n_neurons(self) -> int:
        return self.layers[0].mlp.down_proj.weight.shape[1]

    def down_col(self, layer: int, neuron: int) -> torch.Tensor:
        return self.layers[layer].mlp.down_proj.weight[:, neuron].detach().float()

    # ---------- push vectors ----------
    def neuron_vector(self, layer: int, neuron: int, dose: float, scale: list[float]) -> torch.Tensor:
        c = self.down_col(layer, neuron)
        return dose * float(scale[layer]) * c / c.norm()

    def random_vector(self, layer: int, dose: float, scale: list[float], seed: int) -> torch.Tensor:
        v = torch.randn(self.d_model, generator=torch.Generator().manual_seed(seed))
        return abs(dose) * float(scale[layer]) * v / v.norm()

    @contextmanager
    def pushing(self, per_layer: dict[int, torch.Tensor]):
        """Add per_layer[L] (shape [d] or [batch, 1, d]) to the MLP output of layer L while inside the block."""
        old = self._push
        self._push = dict(per_layer)
        try:
            yield
        finally:
            self._push = old

    # ---------- text ----------
    def chat_ids(self, user: str, system: str | None = None) -> list[int]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        kw = {"enable_thinking": False} if "qwen" in self.name.lower() else {}
        try:
            text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
        except Exception:
            text = (system + "\n\n" if system else "") + user + "\n"
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def stop_ids(self) -> list[int]:
        ids = {self.tok.eos_token_id}
        for t in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id:
                ids.add(i)
        return [i for i in ids if i is not None]

    @torch.no_grad()
    def generate_rows(self, ids: list[int], rows: list[dict[int, torch.Tensor]], max_new: int = 64,
                      sample: bool = False, temperature: float = 0.7, top_p: float = 0.9, seed: int = 0) -> list[str]:
        """Generate one reply per row; each row is {layer: vector} (empty dict = unsteered). One batch."""
        B, D = len(rows), self.d_model
        per_layer: dict[int, torch.Tensor] = {}
        for i, r in enumerate(rows):
            for L, v in r.items():
                if L not in per_layer:
                    per_layer[L] = torch.zeros(B, 1, D)
                per_layer[L][i, 0] += v
        dev = next(self.hf.parameters()).device
        x = torch.tensor([ids] * B, device=dev)
        stop = self.stop_ids()
        kw = dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=0) if sample else dict(do_sample=False)
        torch.manual_seed(seed)
        with self.pushing({L: v.to(dev) for L, v in per_layer.items()}):
            out = self.hf.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new,
                                   eos_token_id=stop, pad_token_id=self.tok.pad_token_id or stop[0], **kw)
        texts = []
        for seq in out[:, x.shape[1]:].tolist():
            cut = next((j for j, t in enumerate(seq) if t in stop), len(seq))
            texts.append(self.tok.decode(seq[:cut], skip_special_tokens=True).strip())
        return texts

    @torch.no_grad()
    def residual_trace(self, ids: list[int], push: dict[int, torch.Tensor] | None = None) -> torch.Tensor:
        """Residual stream leaving every layer, shape [n_layers, T, d] (for the ripple view)."""
        acts = []
        hs = [layer.register_forward_hook(lambda m, i, o: acts.append((o[0] if isinstance(o, tuple) else o)[0].float()))
              for layer in self.layers]
        try:
            with self.pushing(push or {}):
                self.hf(input_ids=torch.tensor([ids]), use_cache=False)
        finally:
            for h in hs:
                h.remove()
        return torch.stack(acts)


def default_model() -> tuple[str, str | None]:
    """Model id and revision, overridable with RECEIPTS_MODEL / RECEIPTS_REVISION or a local path."""
    return (os.environ.get("RECEIPTS_MODEL", "Qwen/Qwen3.5-0.8B"),
            os.environ.get("RECEIPTS_REVISION", "2fc06364715b967f1860aea9cf38778875588b17") or None)


def cache_dir() -> Path:
    p = Path(os.environ.get("RECEIPTS_HOME", Path.home() / ".cache" / "receipts"))
    p.mkdir(parents=True, exist_ok=True)
    return p
