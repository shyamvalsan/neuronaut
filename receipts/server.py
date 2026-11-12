"""Local server for the explorer: static web app + atlas files + a small JSON API backed by the model.

Endpoints (POST bodies and responses are JSON):
  GET  /api/status                        {"live": bool, "model": ..., "revision": ...}
  GET  /api/recipe                        the recipe with receipts
  POST /api/generate  {prompt, system?, lever|spec, dose?, sample?, seed?, max_new?}
                       -> {"unsteered", "lever", "control", "label"}
  POST /api/ripple    {prompt, system?, lever|spec, dose?}
                       -> per layer: how much the residual stream at the last prompt token moved (as a fraction of
                          the layer's typical size) for the lever and for its random control, and the neurons whose
                          activation changed most (layer, neuron, change) for each.
  POST /api/trace     {layer, neuron, text}  -> the neuron's activation on every token of text
Only one model call runs at a time (a lock), so the page shows a queue rather than overloading the CPU.
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

WEB = Path(__file__).parent / "web"


class State:
    model = None
    recipe = None
    atlas: Path | None = None
    lock = threading.Lock()


def _lever(body):
    from .recipe import Lever

    lv = State.recipe.levers.get(body.get("lever", "")) or Lever.parse(body.get("spec") or body["lever"])
    if body.get("dose") is not None:
        lv = lv.scaled(float(body["dose"]) / lv.parts[0][2])
    return lv


def api_generate(body):
    from .recipe import control_vectors, lever_vectors

    m, lv = State.model, _lever(body)
    ids = m.chat_ids(body["prompt"], system=body.get("system") or None)
    rows = [{}, lever_vectors(m, lv, State.recipe.scale), control_vectors(m, lv, State.recipe.scale)]
    outs = m.generate_rows(ids, rows, max_new=int(body.get("max_new", 80)), sample=bool(body.get("sample")),
                           seed=int(body.get("seed", 0)))
    return {"unsteered": outs[0], "lever": outs[1], "control": outs[2], "label": lv.label}


@torch.no_grad()
def _neuron_acts(m, ids, push):
    """Activations of every MLP neuron at the last token, and the residual leaving each layer: ([L, N], [L, d])."""
    h, r = [], []
    hooks = [l.mlp.down_proj.register_forward_pre_hook(lambda mod, a: h.append(a[0][0, -1].float())) for l in m.layers]
    hooks += [l.register_forward_hook(lambda mod, a, o: r.append((o[0] if isinstance(o, tuple) else o)[0, -1].float()))
              for l in m.layers]
    try:
        with m.pushing(push):
            m.hf(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        for x in hooks:
            x.remove()
    return torch.stack(h), torch.stack(r)


def api_ripple(body):
    from .recipe import control_vectors, lever_vectors

    m, lv, scale = State.model, _lever(body), State.recipe.scale
    ids = m.chat_ids(body["prompt"], system=body.get("system") or None)
    h0, r0 = _neuron_acts(m, ids, {})
    out = {"label": lv.label, "parts": lv.parts}
    for key, vec in (("lever", lever_vectors(m, lv, scale)), ("control", control_vectors(m, lv, scale))):
        h1, r1 = _neuron_acts(m, ids, vec)
        moved = ((r1 - r0).norm(dim=-1) / torch.tensor(scale)).tolist()
        dh = h1 - h0
        top = []
        for L in range(m.n_layers):
            v, i = dh[L].abs().topk(24)
            top.append([[int(n), round(float(dh[L, n]), 3)] for n in i.tolist()])
        out[key] = {"moved": [round(x, 4) for x in moved], "top": top}
    return out


@torch.no_grad()
def api_trace(body):
    m = State.model
    L, n = int(body["layer"]), int(body["neuron"])
    ids = m.tok(body["text"], add_special_tokens=False)["input_ids"][:256]
    acts = []
    hook = m.layers[L].mlp.down_proj.register_forward_pre_hook(lambda mod, a: acts.append(a[0][0, :, n].float()))
    try:
        m.hf(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        hook.remove()
    return {"tokens": [m.tok.decode([t]) for t in ids], "acts": [round(x, 3) for x in acts[0].tolist()]}


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def translate_path(self, path):
        p = path.split("?", 1)[0].split("#", 1)[0]
        if p.startswith("/atlas/") and State.atlas:
            return str(State.atlas / p[len("/atlas/"):])
        return str(WEB / (p.lstrip("/") or "index.html"))

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            m = State.model
            return self._json({"live": m is not None, "model": State.recipe.model, "revision": State.recipe.revision})
        if self.path.startswith("/api/recipe"):
            r = State.recipe
            return self._json({"model": r.model, "revision": r.revision, "scale": r.scale,
                               "levers": {k: v.to_json() | {"name": k} for k, v in r.levers.items()}})
        return super().do_GET()

    def do_POST(self):
        routes = {"/api/generate": api_generate, "/api/ripple": api_ripple, "/api/trace": api_trace}
        fn = routes.get(self.path)
        if fn is None:
            return self._json({"error": "unknown endpoint"}, 404)
        if State.model is None:
            return self._json({"error": "no model loaded; start without --no-model for live runs"}, 503)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            with State.lock:
                return self._json(fn(body))
        except Exception as e:  # report to the page instead of dropping the connection
            return self._json({"error": f"{type(e).__name__}: {e}"}, 400)


def serve(port=8737, atlas=Path("atlas"), recipe=None, load_model=True, model_args=None, open_browser=True):
    from .cli import _model, _recipe

    State.recipe = _recipe(recipe)
    State.atlas = Path(atlas).resolve()
    if not (State.atlas / "meta.json").exists():
        print(f"no atlas at {State.atlas}; build one with `receipts atlas --out {atlas}`")
    if load_model:
        State.model = _model(model_args)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Receipts explorer at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    srv.serve_forever()
