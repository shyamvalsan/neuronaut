"""A standalone HTML report for one test run: scoreboard, costs, and real side-by-side replies."""
from __future__ import annotations

import html
import json
from pathlib import Path

CSS = """
:root{--bg:#0b0d14;--surface:#11141c;--raised:#1a1e28;--text:#eef0f5;--text2:#a3a9b8;--muted:#7a8296;
--line:rgba(238,240,245,.08);--accent:#f0a63a;--sans:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:400 15px/1.6 var(--sans)}
.wrap{max-width:1080px;margin:0 auto;padding:48px max(16px,env(safe-area-inset-left,0px)) 96px}
h1{font-weight:500;font-size:clamp(28px,4vw,40px);line-height:1.1;margin:0 0 12px;max-width:24ch}
h2{font-weight:500;font-size:20px;margin:56px 0 12px}.sec{color:var(--text2);max-width:68ch}
.meta{font:13px var(--mono);color:var(--muted)}table{border-collapse:collapse;width:100%;font-size:14px}
.tbl{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--surface)}
th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
th{font:12px var(--mono);color:var(--muted);font-weight:400}td.n{text-align:right;font-family:var(--mono)}
tr:last-child td{border-bottom:0}tr.fail td{color:var(--muted)}
.bar{display:inline-block;height:10px;border-radius:2px;background:var(--accent);vertical-align:middle}
tr.fail .bar{background:rgba(238,240,245,.2)}.addr{font:12px var(--mono);border:1px solid rgba(238,240,245,.16);border-radius:4px;padding:3px 6px}
.ex{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:12px 0 28px}
.ex div{border:1px solid var(--line);border-radius:12px;padding:12px;background:var(--surface);font-size:14px;white-space:pre-wrap}
.ex div.lev{border-color:rgba(240,166,58,.5);background:rgba(240,166,58,.08)}.ex b{display:block;font:12px var(--mono);color:var(--text2);font-weight:400;margin-bottom:6px}
.q{font-size:14px;color:var(--text2)}@media(max-width:800px){.ex{grid-template-columns:minmax(0,1fr)}}
"""


def write_report(frozen, res, out: Path, examples_per_lever: int = 2):
    rows = [json.loads(l) for l in open(out / "rollouts.jsonl")]
    by = {(r["cond"], r["cell"]): r["text"] for r in rows}
    e = html.escape
    nl = len(res)
    held = sum(r["held_up"] for r in res.values())
    order = sorted(res.items(), key=lambda kv: -kv[1]["tagged"] / max(1, kv[1]["n"]))
    body = [f"<p class='meta'>{e(frozen['model'])} · rev {e(str(frozen.get('revision'))[:12])} · test {e(frozen['hash'])} · frozen {e(frozen['frozen_at'])}</p>",
            f"<h1>{held} of {nl} levers held up on prompts they never saw</h1>",
            f"<p class='sec'>Each lever was frozen with its push size before any test output existed, then run on {len(frozen['prompts'])} new prompts "
            f"under {len(frozen['framings'])} system prompts, beside random directions of the same size and the unsteered model. "
            "Two judges labelled a shuffled sample without knowing which reply came from where. A lever holds up if both judges see its trait "
            f"clearly more often than with its random direction (one-sided Fisher, p &lt; 0.05/{nl}).</p>",
            "<h2>Scoreboard</h2><div class='tbl'><table><thead><tr><th>Lever</th><th>Push</th><th>Both judges</th><th class='n'>Random</th>"
            "<th class='n'>p</th><th class='n'>Readability lever / random / unsteered</th><th class='n'>Easy questions right</th></tr></thead><tbody>"]
    for name, r in order:
        v = frozen["levers"][name]
        parts = v.get("parts") or [[v["layer"], v["neuron"], v["dose"]]]
        label = " + ".join(f"L{a}·N{b} {c:+g}" for a, b, c in parts)
        w = int(160 * r["tagged"] / max(1, r["n"]))
        capt = f"{r['capability'][0]}/{r['capability'][1]} (unsteered {r['capability_base'][0]})" if r.get("capability") else ""
        body.append(f"<tr class='{'' if r['held_up'] else 'fail'}'><td>{e(name)}</td><td><span class='addr'>{e(label)}</span></td>"
                    f"<td><span class='bar' style='width:{max(2, w)}px'></span> {r['tagged']}/{r['n']}</td><td class='n'>{r['control']}/{r['n']}</td>"
                    f"<td class='n'>{r['p']:.1g}</td><td class='n'>{r['coherence'][0]} / {r['coherence'][1]} / {r['coherence'][2]}</td><td class='n'>{capt}</td></tr>")
    body.append("</tbody></table></div>")
    body.append("<h2>Side by side</h2><p class='sec'>Sampled replies from the test, same seed in all three columns.</p>")
    for name, r in order:
        cells = sorted({c for (cond, c) in by if cond == name and c.startswith("t0.7")})[:examples_per_lever]
        for cell in cells:
            pi = int(cell.split("|")[2]); fr = cell.split("|")[1]
            body.append(f"<p class='q'><b>{e(name)}</b> · {e(fr)} · {e(frozen['prompts'][pi])}</p><div class='ex'>"
                        f"<div><b>Unsteered</b>{e(by.get(('base', cell), ''))}</div>"
                        f"<div class='lev'><b>Lever</b>{e(by.get((name, cell), ''))}</div>"
                        f"<div><b>Random push, same size</b>{e(by.get((f'ctrl:{name}:0', cell), ''))}</div></div>")
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Neuronaut report</title><style>{CSS}</style></head><body><div class='wrap'>{''.join(body)}</div></body></html>")
    (out / "report.html").write_text(page)
