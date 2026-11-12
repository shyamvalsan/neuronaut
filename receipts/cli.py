"""receipts: find single-neuron levers in small language models, and test them before you believe them.

    receipts try deity "How do vaccines work?"          # a lever from the built-in Qwen3.5-0.8B recipe
    receipts try L18N2521:+0.6 "Why is the sky blue?"   # any neuron, any push size
    receipts atlas                                      # precompute the data the 3D view needs (once per model)
    receipts serve                                      # open the 3D explorer in your browser
    receipts find "flattery" --examples flattery.txt    # search for candidate levers, with random controls
    receipts test levers.json                           # frozen held-out test with blind judging -> report page
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BUILTIN = Path(__file__).parent / "data" / "qwen3.5-0.8b.recipe.json"


def _model(args):
    from .model import Model, default_model

    name, rev = default_model()
    name = getattr(args, "model", None) or name
    print(f"loading {name} ...", file=sys.stderr, flush=True)
    return Model.load(name, revision=rev if name.startswith("Qwen/Qwen3.5-0.8B") else None, threads=args.threads)


def _recipe(path=None):
    from .recipe import Recipe

    return Recipe.load(path or BUILTIN)


def cmd_try(args):
    from .recipe import Lever, control_vectors, lever_vectors

    recipe = _recipe(args.recipe)
    lever = recipe.levers.get(args.lever) or Lever.parse(args.lever)
    if args.dose is not None:
        lever = lever.scaled(args.dose / lever.parts[0][2])
    m = _model(args)
    ids = m.chat_ids(args.prompt, system=args.system)
    rows = [{}, lever_vectors(m, lever, recipe.scale)]
    names = ["unsteered", f"{lever.name}  {lever.label}"]
    if not args.no_control:
        rows.append(control_vectors(m, lever, recipe.scale))
        names.append("random direction, same layers and size")
    outs = m.generate_rows(ids, rows, max_new=args.max_new, sample=args.sample, seed=args.seed)
    print(f"\nPrompt: {args.prompt}" + (f"\nSystem: {args.system}" if args.system else ""))
    for n, o in zip(names, outs):
        print(f"\n[{n}]\n{o}")
    if lever.receipts:
        r = lever.receipts
        print(f"\nReceipts: trait seen in {r.get('tagged')}/{r.get('n')} blind-judged replies "
              f"vs {r.get('control')}/{r.get('n')} for a random push (p = {r.get('p'):.1g})")


def cmd_list(args):
    recipe = _recipe(args.recipe)
    for name, lv in recipe.levers.items():
        r = lv.receipts
        rec = f"{r['tagged']}/{r['n']} vs {r['control']}/{r['n']}" if r else "untested"
        print(f"{name:12s} {lv.label:22s} {rec:14s} {lv.trait}")


def cmd_atlas(args):
    from .atlas import build_atlas

    build_atlas(_model(args), out=Path(args.out), tokens=args.tokens)


def cmd_serve(args):
    from .server import serve

    serve(port=args.port, atlas=Path(args.atlas), recipe=args.recipe, load_model=not args.no_model,
          model_args=args, open_browser=not args.no_browser)


def cmd_site(args):
    from .site import export_site

    export_site(_model(args), _recipe(args.recipe), Path(args.atlas), Path(args.out),
                Path(args.rollouts) if args.rollouts else None, Path(args.prompts) if args.prompts else None,
                args.ripple_prompts)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="receipts", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--model", help="Hugging Face id or local path (default Qwen/Qwen3.5-0.8B)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("try", help="one prompt: unsteered vs lever vs a random push of the same size")
    p.add_argument("lever", help="lever name from the recipe, or a spec like L18N2521:+0.6")
    p.add_argument("prompt", nargs="?", default="Tell me about your weekend.")
    p.add_argument("--system")
    p.add_argument("--dose", type=float, help="override the push size (sign kept from the lever)")
    p.add_argument("--recipe")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--no-control", action="store_true")
    p.set_defaults(fn=cmd_try)

    p = sub.add_parser("list", help="levers in a recipe, with their receipts")
    p.add_argument("--recipe")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("atlas", help="precompute layout, neuron portraits and wiring for the 3D view")
    p.add_argument("--out", default="atlas")
    p.add_argument("--tokens", type=int, default=50_000)
    p.set_defaults(fn=cmd_atlas)

    p = sub.add_parser("serve", help="the 3D explorer in your browser")
    p.add_argument("--port", type=int, default=8737)
    p.add_argument("--atlas", default="atlas")
    p.add_argument("--recipe")
    p.add_argument("--no-model", action="store_true", help="explore recorded data only, no live generation")
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("site", help="export a static explorer (GitHub Pages) with recorded outputs and ripples")
    p.add_argument("--atlas", default="atlas")
    p.add_argument("--out", default="site")
    p.add_argument("--recipe")
    p.add_argument("--rollouts", help="frozen-test rollouts.jsonl to show as recorded outputs")
    p.add_argument("--prompts", help="the prompts file those rollouts used")
    p.add_argument("--ripple-prompts", type=int, default=6)
    p.set_defaults(fn=cmd_site)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
