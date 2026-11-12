"""Recipe files: portable lever definitions pinned to a model revision.

    {
      "format": "receipts/recipe@1",
      "model": "Qwen/Qwen3.5-0.8B", "revision": "2fc0636...",
      "scale": [1.22, 1.57, ...],          # median residual norm leaving each layer (push unit)
      "levers": {
        "deity": {"layer": 18, "neuron": 2521, "dose": 0.6, "trait": "...",
                  "receipts": {"tagged": 23, "n": 24, "control": 0, "p": 7.8e-13, "test": "..."}}
      }
    }

A lever may also be a combination: {"parts": [[18, 2521, 0.4], [16, 157, -0.5]], "trait": "..."}.
Older files that use "resid_p50" instead of "scale" load too.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

FORMAT = "receipts/recipe@1"


@dataclass
class Lever:
    name: str
    parts: list[tuple[int, int, float]]  # (layer, neuron, dose)
    trait: str = ""
    receipts: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return " + ".join(f"L{L}·N{n} {d:+g}" for L, n, d in self.parts)

    def scaled(self, factor: float) -> "Lever":
        return Lever(self.name, [(L, n, d * factor) for L, n, d in self.parts], self.trait, {})

    def to_json(self) -> dict:
        if len(self.parts) == 1:
            L, n, d = self.parts[0]
            out = {"layer": L, "neuron": n, "dose": d}
        else:
            out = {"parts": [list(p) for p in self.parts]}
        if self.trait:
            out["trait"] = self.trait
        if self.receipts:
            out["receipts"] = self.receipts
        return out

    @classmethod
    def parse(cls, spec: str, name: str | None = None) -> "Lever":
        """'L18N2521:+0.6' or 'L18N2521:0.4+L16N157:-0.5'."""
        import re

        parts = []
        for p in spec.replace(" ", "").split("+L"):
            p = p if p.startswith("L") else "L" + p
            m = re.fullmatch(r"L(\d+)[N·.:]?N?(\d+):([+-]?[\d.]+)", p)
            if not m:
                raise ValueError(f"bad lever spec {spec!r}; expected like L18N2521:+0.6")
            parts.append((int(m[1]), int(m[2]), float(m[3])))
        return cls(name or spec, parts)


@dataclass
class Recipe:
    model: str
    revision: str | None
    scale: list[float]
    levers: dict[str, Lever] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Recipe":
        d = json.loads(Path(path).read_text())
        levers = {}
        for name, v in d.get("levers", {}).items():
            parts = [tuple(p) for p in v["parts"]] if "parts" in v else [(v["layer"], v["neuron"], v["dose"])]
            rec = v.get("receipts", {})
            levers[name] = Lever(name, [(int(a), int(b), float(c)) for a, b, c in parts], v.get("trait", ""), rec)
        return cls(d["model"], d.get("revision"), d.get("scale") or d["resid_p50"], levers)

    def save(self, path: str | Path) -> None:
        out = {"format": FORMAT, "model": self.model, "revision": self.revision, "scale": self.scale,
               "levers": {k: v.to_json() for k, v in self.levers.items()}}
        Path(path).write_text(json.dumps(out, indent=1, ensure_ascii=False))


def lever_vectors(model, lever: Lever, scale: list[float]) -> dict:
    out = {}
    for L, n, d in lever.parts:
        out[L] = out.get(L, 0) + model.neuron_vector(L, n, d, scale)
    return out


def control_vectors(model, lever: Lever, scale: list[float], k: int = 0) -> dict:
    """Random directions at the same layers and sizes as the lever (control k)."""
    from .model import seed_from

    out = {}
    for j, (L, n, d) in enumerate(lever.parts):
        out[L] = out.get(L, 0) + model.random_vector(L, d, scale, seed_from("control", lever.name, L, n, k, j))
    return out


__all__ = ["Lever", "Recipe", "lever_vectors", "control_vectors", "asdict"]
