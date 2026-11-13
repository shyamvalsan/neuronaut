import { Tower } from './scene.js';
import { WebEngine } from './engine-web.js';

const $ = (id) => document.getElementById(id);
const esc = (t) => (t || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const md = (t) => esc(t).replace(/^#{1,6}\s*/gm, '').replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>').replace(/\*\*/g, '');
const NAMES = { deity: 'Deity', reassure: 'Reassurer', jesus: 'Jesus', anxious: 'Anxious', enthusiast: 'Enthusiast',
  gratitude: 'Flatterer', cosmic: 'Cosmic narrator', angry: 'Angry', grandiose: 'Grandiose', india: 'India', human: 'Human life story' };
const ATLAS = (window.NEURONAUT_ATLAS || 'atlas') + '/';

const state = { live: false, recipe: null, lever: null, layerCache: {}, recorded: null, tower: null, meta: null };

async function getJSON(url, opts) { const r = await fetch(url, opts); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); }
async function getF32(url) { const r = await fetch(url); if (!r.ok) throw new Error(url); return new Float32Array(await r.arrayBuffer()); }
async function post(path, body) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json(); if (!r.ok || j.error) throw new Error(j.error || r.status); return j;
}

// ---------- boot ----------
async function boot() {
  const [meta, layout, stats] = await Promise.all([getJSON(ATLAS + 'meta.json'), getF32(ATLAS + 'layout.f32'), getF32(ATLAS + 'stats.f32')]);
  state.meta = meta;
  $('model-line').textContent = `${meta.model.split('/').pop()} · ${meta.layers} layers × ${meta.neurons.toLocaleString('en-US')} neurons` +
    (meta.revision ? ` · rev ${meta.revision.slice(0, 7)}` : '');
  $('about-model').textContent = `Atlas: activation statistics from ${meta.stats_tokens.toLocaleString('en-US')} tokens of WikiText-2. Wiring: ${meta.wiring}.`;
  state.tower = new Tower($('scene'), meta, layout, stats);
  state.tower.onPick = (L, n) => openNeuron(L, n);
  try {
    const s = await getJSON('api/status'); state.live = s.live;
    state.recipe = await getJSON('api/recipe');
  } catch {
    state.recipe = await getJSON('recipe.json');
  }
  try { state.recorded = await getJSON('recorded.json'); } catch { state.recorded = null; }
  if (!state.live) {
    try { const w = await getJSON('webmodel.json'); state.web = new WebEngine({ ...w, scale: state.recipe.scale, atlas: ATLAS }); } catch { state.web = null; }
  }
  $('status').classList.toggle('live', state.live);
  $('status-text').textContent = state.live ? 'model live on this machine' : state.web ? 'recorded outputs · model can run in your browser' : (state.recorded ? 'recorded outputs' : 'atlas only');
  buildLevers();
  if (!state.live && state.recorded?.prompts?.length) {
    const f = document.createElement('div'); f.className = 'field';
    f.innerHTML = '<label for="recorded">Recorded prompts from the test</label><select id="recorded"><option value="">Choose one</option>' +
      state.recorded.prompts.map((q, i) => `<option value="${i}">${esc(q)}</option>`).join('') + '</select>';
    $('prompt').closest('.field').before(f);
    $('recorded').onchange = (e) => { if (e.target.value !== '') { $('prompt').value = state.recorded.prompts[+e.target.value]; $('dose').value = 1; updateDose(); showRecorded(); setHash(); } };
  }
  bindUI();
  if (!applyHash()) selectLever('deity');
}

// ---------- levers ----------
function buildLevers() {
  const list = $('lever-list');
  const levers = Object.entries(state.recipe.levers).map(([k, v]) => ({ name: k, ...v }));
  const held = levers.filter(l => l.receipts?.held_up).sort((a, b) => b.receipts.tagged - a.receipts.tagged);
  const rest = levers.filter(l => !l.receipts?.held_up);   // neurons you push yourself from the portrait
  for (const [title, group] of [['Levers that held up', held], ['Your neurons', rest]]) {
    if (!group.length) continue;
    const h = document.createElement('div'); h.className = 'group'; h.textContent = title; list.appendChild(h);
    for (const l of group) {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'lv' + (l.receipts?.held_up ? ' held' : '');
      b.dataset.name = l.name;
      b.innerHTML = `<span class="lamp" aria-hidden="true"></span><span>${NAMES[l.name] || l.name}</span><span class="rec">${l.receipts ? `${l.receipts.tagged}/${l.receipts.n}` : ''}</span>`;
      b.onclick = () => selectLever(l.name);
      list.appendChild(b);
    }
  }
}

function leverParts(l) { return l.parts || [[l.layer, l.neuron, l.dose]]; }
function addr(parts) { return parts.map(([L, n, d]) => `L${L}·N${n} ${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(2).replace(/0$/, '')}`).join(' + '); }

function selectLever(name, { open = true } = {}) {
  const l = state.recipe.levers[name]; if (!l) return;
  state.lever = { name, ...l };
  document.querySelectorAll('.lv').forEach(b => b.setAttribute('aria-pressed', b.dataset.name === name));
  $('lv-name').textContent = NAMES[name] || name;
  $('lv-trait').textContent = l.trait || '';
  const r = l.receipts;
  $('lv-receipt').innerHTML = r ? `Blind test on new prompts: trait seen in <b>${r.tagged} of ${r.n}</b> replies, ${r.control} of ${r.n} for a random push of the same size` +
    (r.held_up ? '.' : '. Did not beat chance clearly enough to count.') +
    (r.browser ? ` On the 4-bit model this page runs: ${r.browser.tagged} of ${r.browser.n} vs ${r.browser.control} of ${r.browser.n}` +
      (r.browser.held_up ? '.' : ', too few prompts to be sure.') : '') : 'Untested: no receipts yet.';
  $('dose').value = 1; updateDose();
  const [L, n] = leverParts(l)[0];
  if (open) openNeuron(L, n, { keepTab: true });
  showRecorded();
}

function currentParts() {
  const f = +$('dose').value;
  return leverParts(state.lever).map(([L, n, d]) => [L, n, +(d * f).toFixed(3)]);
}
function updateDose() { $('dose-val').textContent = addr(currentParts()); $('lv-addr').textContent = addr(currentParts()); }

// ---------- running ----------
function setOut(o) {
  $('o-base').innerHTML = md(o.unsteered); $('o-base').classList.remove('muted');
  $('o-lev').innerHTML = md(o.lever); $('o-ctl').innerHTML = md(o.control);
}

function recordedKey() {
  const rec = state.recorded; if (!rec) return null;
  const pi = rec.prompts.indexOf($('prompt').value.trim());
  const fr = { '': 'plain', 'You are a helpful assistant.': 'assistant' }[$('system').value] ?? 'roleplay';
  return pi < 0 ? null : { pi, fr };
}

function showRecorded() {
  const k = recordedKey();
  if (!state.live && k && +$('dose').value === 1) {
    const o = state.recorded.outputs[`${state.lever.name}|${k.fr}|${k.pi}`];
    if (o) { setOut(o); $('run-note').textContent = 'Recorded output from the frozen test (sampled, temperature 0.7).'; return true; }
  }
  return false;
}

async function run() {
  if (!state.live && !state.web) {
    if (!showRecorded()) $('run-note').textContent = 'No model running here. Pick one of the recorded prompts, or run `neuronaut serve` locally for any prompt.';
    return;
  }
  $('run').disabled = true;
  const parts = currentParts();
  try {
    if (state.live) {
      $('run-note').textContent = 'Generating three replies on this machine, usually 10 to 40 seconds';
      const o = await post('api/generate', { spec: parts.map(([L, n, d]) => `L${L}N${n}:${d}`).join('+'), prompt: $('prompt').value,
        system: $('system').value, sample: false, max_new: 80 });
      setOut(o); $('run-note').textContent = 'Greedy decoding, same prompt for all three.';
    } else {
      const w = state.web;
      if (!w.ready) await w.load((msg) => { $('run-note').textContent = msg; });
      const ids = w.chatIds($('prompt').value, $('system').value);
      const panes = [['o-base', null], ['o-lev', await w.steerFor(parts)], ['o-ctl', await w.steerFor(parts, true)]];
      for (const [id] of panes) { $(id).innerHTML = ''; $(id).classList.remove('muted'); }
      const zero = new Float32Array(w.cfg.layers * w.cfg.hidden);
      for (const [id, steer] of panes) {
        $('run-note').textContent = `Generating in your browser on ${w.device}`;
        await w.generate(ids, steer || zero, 64, (t) => { $(id).innerHTML = md(t); });
      }
      $('run-note').textContent = `Ran in your browser on ${w.device}: 4-bit model, greedy decoding. The 4-bit model differs slightly from the full-precision one the receipts come from.`;
    }
  } catch (e) { $('run-note').textContent = `Could not run: ${e.message}`; }
  $('run').disabled = false;
}

async function ripple() {
  let data = null;
  const parts = currentParts();
  if (state.live) {
    $('ripple').disabled = true; $('run-note').textContent = 'Measuring the ripple: three forward passes';
    try { data = await post('api/ripple', { spec: parts.map(([L, n, d]) => `L${L}N${n}:${d}`).join('+'), prompt: $('prompt').value, system: $('system').value }); }
    catch (e) { $('run-note').textContent = `Could not measure: ${e.message}`; }
    $('ripple').disabled = false;
  } else if (state.recorded) {
    const k = recordedKey();
    data = state.recorded.ripples?.[`${state.lever.name}|${k ? k.pi : 0}`] || Object.entries(state.recorded.ripples || {}).find(([key]) => key.startsWith(state.lever.name + '|'))?.[1];
  }
  if (!data) { if (!state.live) $('run-note').textContent = 'No recorded ripple for this lever.'; return; }
  playRippleData(data, parts);
}

function playRippleData(data, parts = currentParts()) {
  if (!data?.lever || !data?.control) { $('run-note').textContent = 'The ripple came back incomplete.'; return; }
  $('run-note').textContent = '';
  drawRippleChart(data);
  const cap = $('caption');
  state.tower.playRipple(data, (phase) => {
    if (!phase) { cap.hidden = true; return; }
    cap.hidden = false;
    cap.innerHTML = phase === 'lever'
      ? `<b>${NAMES[state.lever.name] || state.lever.name}</b> <span class="mono">${addr(parts)}</span><br><span class="sec">The push enters at layer ${parts[0][0]}. Beside each later layer: the words the change points to.</span>`
      : `<span class="mono">Random push, same layer and size</span><br><span class="sec">Just as big, and it disturbs as many neurons, but it points nowhere in particular.</span>`;
  });
}

function drawRippleChart(d) {
  const W = 320, H = 120, l = 24, b = 18, t = 8, L = d.lever.moved.length;
  const key = d.lever.activity ? 'activity' : 'moved';
  const max = Math.max(...d.lever[key], ...d.control[key]);
  const x = (i) => l + i * (W - l - 8) / (L - 1), y = (v) => H - b - v / max * (H - b - t);
  const path = (arr) => arr.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
  let s = '';
  for (const g of [0, 0.5, 1]) s += `<line x1="${l}" x2="${W - 8}" y1="${y(g * max)}" y2="${y(g * max)}" stroke="rgba(238,240,245,0.08)"/>`;
  [0, 6, 12, 18, L - 1].forEach(i => { s += `<text x="${x(i)}" y="${H - 4}" text-anchor="middle">L${i}</text>`; });
  s += `<text x="0" y="${y(max) + 3}">${max.toFixed(1)}</text><text x="0" y="${y(0)}">0</text>`;
  $('ripple-says').innerHTML = `<span class="lev">${esc(d.lever.says?.[L - 1]?.slice(0, 3).join(' ') || '')}</span> vs <span>${esc(d.control.says?.[L - 1]?.slice(0, 3).join(' ') || '')}</span> at the last layer`;
  s += `<path d="${path(d.control[key])}" fill="none" stroke="#a3a9b8" stroke-width="1.5"/>`;
  s += `<path d="${path(d.lever[key])}" fill="none" stroke="#f0a63a" stroke-width="2"/>`;
  $('ripple-chart').innerHTML = s;
  $('ripple-card').hidden = false;
}

// ---------- neuron portrait ----------
async function layerData(L) {
  if (!state.layerCache[L]) state.layerCache[L] = getJSON(`${ATLAS}layer_${String(L).padStart(2, '0')}.json`);
  return state.layerCache[L];
}

async function openNeuron(L, n, opts = {}) {
  const ticket = (state.ticket = (state.ticket || 0) + 1);   // a newer call supersedes this one
  if (!opts.keepTab) showTab('neuron');
  const rows = await layerData(L), info = rows[n];
  if (ticket !== state.ticket) return;
  const kinds = state.meta.layer_kinds || [];
  $('nr-addr').textContent = `L${L}·N${n}`;
  $('nr-kind').textContent = (kinds[L] || '').startsWith('full') ? 'attention layer' : 'DeltaNet layer';
  const i = (L * state.meta.neurons + n) * 4, st = state.tower ? null : null;
  const stats = state.statsCache || (state.statsCache = await getF32(ATLAS + 'stats.f32'));
  $('nr-stats').textContent = `Positive on ${(100 * stats[i + 2]).toFixed(0)}% of tokens. Strongest seen: +${stats[i].toFixed(2)} / −${stats[i + 1].toFixed(2)}.`;
  $('nr-body').hidden = false;
  $('nr-words-pos').innerHTML = info['w+'].map(w => `<span>${esc(w)}</span>`).join('');
  $('nr-words-neg').innerHTML = info['w-'].map(w => `<span>${esc(w)}</span>`).join('');
  $('nr-ctx').innerHTML = [...info['c+'].slice(0, 3), ...info['c-'].slice(0, 1)].map(c =>
    `<li><span class="a">${c.a > 0 ? '+' : ''}${c.a}</span>…${esc(c.pre)}<mark>${esc(c.tok)}</mark>${esc(c.post)}…</li>`).join('');
  const wires = async (list, el) => {
    const top = list.slice(0, 6), max = Math.max(1e-12, ...top.map(w => Math.abs(w[2])));
    const items = [];
    for (const [L2, n2, v] of top) items.push([L2, n2, v, (await layerData(L2))[n2]]);
    if (ticket !== state.ticket) return;
    el.innerHTML = '';
    for (const [L2, n2, v, r2] of items) {
      const li = document.createElement('li'), b = document.createElement('button'); b.type = 'button';
      b.title = v > 0 ? 'excites' : 'inhibits';
      b.innerHTML = `<span class="mono">L${L2}\u00b7N${n2}</span><span class="w">${esc(r2['w+'].slice(0, 3).join(' '))}</span><span class="v">${v > 0 ? '+' : '\u2212'}${(Math.abs(v) / max).toFixed(2)}</span>`;
      b.onclick = () => openNeuron(L2, n2); li.appendChild(b); el.appendChild(li);
    }
  };
  await Promise.all([wires(info.in, $('nr-in')), wires(info.out, $('nr-out'))]);
  if (ticket !== state.ticket) return;
  $('nr-push').onclick = () => {
    const name = `L${L}N${n}`;
    state.recipe.levers[name] = state.recipe.levers[name] || { layer: L, neuron: n, dose: 0.6, trait: 'Untested neuron: try it, then test it before you believe it.' };
    selectLever(name); showTab('levers');
  };
  state.tower.select(L, n, info);
  state.neuron = [L, n];
  setHash();
}

// ---------- concept search ----------
const EXAMPLES = ['the ocean at night', 'God and the heavens', 'I am so worried about tomorrow', 'a recipe for spicy curry', 'python code'];

async function conceptItems(text) {
  if (state.live) {
    const d = await post('api/concept', { text });
    return { items: d.items, how: 'Measured live: each neuron\u2019s peak response to your text, relative to the strongest it ever fired in the atlas.' };
  }
  state.words = state.words || await getJSON(ATLAS + 'words.json');
  const words = (text.toLowerCase().match(/[a-z][a-z'\-]{2,}/g) || []).filter(w => !STOP.has(w));
  const weight = [1.0, 0.6, 0.85], acc = new Map();
  for (const w of words) for (const [L, n, kind, rank] of (state.words.words[w] || [])) {
    const k = L * 100000 + n, v = weight[kind] * (1 - rank / 10);
    const cur = acc.get(k) || { L, n, score: 0, token: w, kinds: new Set() };
    cur.score += v; cur.kinds.add(state.words.kinds[kind]); acc.set(k, cur); }
  const items = [...acc.values()].sort((a, b) => b.score - a.score).slice(0, 48)
    .map(x => ({ ...x, sign: x.kinds.has('writes when negative') && !x.kinds.has('writes') ? -1 : 1, kinds: [...x.kinds] }));
  return { items, how: words.length ? 'From the atlas: neurons that write these words or fire hardest on them in real text. Run Neuronaut locally to measure live activations instead.' : 'Type a word or a sentence.' };
}
const STOP = new Set(['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'any', 'can', 'her', 'was', 'one', 'our', 'out', 'his', 'how', 'its', 'who', 'did', 'yes', 'has', 'him', 'she', 'too', 'use', 'with', 'that', 'this', 'from', 'have', 'they', 'will', 'what', 'when', 'your', 'about', 'there', 'their', 'would', 'which', 'into', 'some', 'than', 'then', 'them', 'were', 'been', 'very', 'just', 'like', 'over', 'such', 'only', 'also']);

async function concept(text) {
  text = text.trim(); if (!text) return;
  showTab('concept'); $('concept-input').value = text;
  $('concept-note').textContent = state.live ? 'Running your text through the model' : 'Looking it up in the atlas';
  $('concept-hits').innerHTML = '';
  try {
    const { items, how } = await conceptItems(text);
    if (!items.length) { $('concept-note').textContent = 'No neuron in the atlas is tied to those words. Try another word, or run Neuronaut locally to measure any text.'; return; }
    // wiring between lit neurons, from the atlas (direct, weight-based)
    const index = new Map(items.map((it, i) => [it.L * 100000 + it.n, i])), links = [];
    const layers = [...new Set(items.map(it => it.L))];
    const data = Object.fromEntries(await Promise.all(layers.map(async L => [L, await layerData(L)])));
    items.forEach((it, i) => {
      const r = data[it.L][it.n];
      it.writes = (it.sign < 0 ? r['w-'] : r['w+']).slice(0, 3).join(' ');
      for (const [L2, n2, w] of r.out) { const j = index.get(L2 * 100000 + n2); if (j !== undefined) links.push([i, j, w]); }
    });
    const among = links.length, shown = items.length;
    // where the strongest lit neurons send their influence next (direct wiring, from the atlas)
    items.slice(0, 8).forEach((it, i) => {
      const r = data[it.L][it.n];
      for (const [L2, n2, w] of r.out.slice(0, 2)) {
        let j = index.get(L2 * 100000 + n2);
        if (j === undefined) { j = items.length; index.set(L2 * 100000 + n2, j); items.push({ L: L2, n: n2, score: 0.25 * it.score, reach: true }); }
        links.push([i, j, w]);
      }
    });
    state.tower.lightConcept(items, links);
    const lit = new Set(items.slice(0, shown).map(it => it.L)).size;
    $('concept-note').textContent = `${shown} neurons across ${lit} layers` + (among ? `, ${among} wired directly to each other` : '') +
      `. Threads show where the strongest ones send their influence next. ${how}`;
    const max = Math.max(...items.map(x => x.score));
    $('concept-hits').innerHTML = '';
    for (const it of items.filter(x => !x.reach).slice(0, 16)) {
      const li = document.createElement('li'), b = document.createElement('button'); b.type = 'button';
      b.innerHTML = `<span class="mono">L${it.L}\u00b7N${it.n}</span><span class="w">${esc(it.writes)}<span class="t">${state.live ? `strongest on \u201c${esc(it.token.trim())}\u201d` : esc(it.kinds.join(', ') + ' \u201c' + it.token + '\u201d')}</span></span>` +
        `<span><span class="bar" style="display:block;width:${Math.round(64 * it.score / max)}px"></span></span>`;
      b.onclick = () => openNeuron(it.L, it.n); li.appendChild(b); $('concept-hits').appendChild(li);
    }
    state.concept = text; setHash();
  } catch (e) { $('concept-note').textContent = `Could not search: ${e.message}`; }
}

// ---------- tabs, links, recording ----------
function showTab(t) {
  for (const k of ['levers', 'concept', 'neuron', 'about']) {
    $(`tab-${k}`).setAttribute('aria-selected', k === t); $(`pane-${k}`).hidden = k !== t;
  }
}

function setHash() {
  const p = new URLSearchParams();
  if (state.lever) { p.set('lever', state.lever.name); p.set('push', $('dose').value); }
  if (state.neuron) p.set('n', `${state.neuron[0]}.${state.neuron[1]}`);
  p.set('q', $('prompt').value);
  if (state.concept) p.set('concept', state.concept);
  history.replaceState(null, '', '#' + p.toString());
}

function applyHash() {
  if (!location.hash) return false;
  const p = new URLSearchParams(location.hash.slice(1));
  if (p.get('q')) $('prompt').value = p.get('q');
  const open = !p.get('n');
  if (p.get('lever') && state.recipe.levers[p.get('lever')]) selectLever(p.get('lever'), { open });
  else if (p.get('lever')) { const m = p.get('lever').match(/^L(\d+)N(\d+)$/); if (m) { state.recipe.levers[p.get('lever')] = { layer: +m[1], neuron: +m[2], dose: 0.6, trait: '' }; selectLever(p.get('lever'), { open }); } }
  if (p.get('push')) { $('dose').value = p.get('push'); updateDose(); }
  if (p.get('n')) { const [L, n] = p.get('n').split('.').map(Number); openNeuron(L, n); }
  if (p.get('concept')) concept(p.get('concept'));
  return !!p.get('lever') || !!p.get('n') || !!p.get('concept');
}

async function share() {
  setHash();
  try { await navigator.clipboard.writeText(location.href); $('share').textContent = 'Link copied'; }
  catch { $('share').textContent = 'Copy from the address bar'; }
  setTimeout(() => $('share').textContent = 'Copy link', 1800);
}

async function record() {
  const src = $('scene'), btn = $('record');
  const out = document.createElement('canvas'); out.width = 1080; out.height = 1080;
  const ctx = out.getContext('2d');
  const stream = out.captureStream(30);
  const types = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
  const type = types.find(t => window.MediaRecorder && MediaRecorder.isTypeSupported(t));
  if (!type) { btn.textContent = 'Recording not supported here'; return; }
  const rec = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 8e6 }), chunks = [];
  rec.ondataavailable = (e) => chunks.push(e.data);
  btn.disabled = true; btn.textContent = 'Recording 10 s';
  state.tower.orbit = { speed: 0.004 };
  ripple();
  const t0 = performance.now();
  const draw = () => {
    const s = Math.min(src.width, src.height), sx = (src.width - s) / 2, sy = (src.height - s) / 2;
    ctx.fillStyle = '#07080d'; ctx.fillRect(0, 0, 1080, 1080);
    ctx.drawImage(src, sx, sy, s, s, 0, 0, 1080, 1080);
    ctx.font = '500 34px "IBM Plex Sans", sans-serif'; ctx.fillStyle = '#eef0f5';
    ctx.fillText($('caption').hidden ? (NAMES[state.lever?.name] || '') : $('caption').innerText.split('\n')[0], 48, 72);
    ctx.font = '26px "IBM Plex Mono", monospace'; ctx.fillStyle = '#a3a9b8';
    ctx.fillText(state.meta.model.split('/').pop(), 48, 1040);
    ctx.fillText('neuronaut', 1080 - 48 - ctx.measureText('neuronaut').width, 1040);
    if (performance.now() - t0 < 10000) requestAnimationFrame(draw); else rec.stop();
  };
  rec.onstop = () => {
    state.tower.orbit = null; btn.disabled = false; btn.textContent = 'Record clip';
    const url = URL.createObjectURL(new Blob(chunks, { type }));
    const a = document.createElement('a'); a.href = url; a.download = `neuronaut-${state.lever?.name || 'tower'}.${type.includes('mp4') ? 'mp4' : 'webm'}`; a.click();
  };
  rec.start(); draw();
}

function bindUI() {
  for (const k of ['levers', 'concept', 'neuron', 'about']) $(`tab-${k}`).onclick = () => showTab(k);
  $('concept-form').onsubmit = (e) => { e.preventDefault(); concept($('concept-input').value); };
  $('concept-examples').innerHTML = EXAMPLES.map(x => `<button type="button">${esc(x)}</button>`).join('');
  $('concept-examples').querySelectorAll('button').forEach(b => b.onclick = () => concept(b.textContent));
  $('dose').oninput = () => { updateDose(); setHash(); };
  $('run').onclick = run; $('ripple').onclick = ripple;
  $('prompt').onchange = () => { setHash(); showRecorded(); }; $('system').onchange = showRecorded;
  $('share').onclick = share; $('record').onclick = record;
  $('goto').onsubmit = (e) => {
    e.preventDefault();
    const v = $('goto-input').value, m = v.match(/^\s*L?\s*(\d+)\D+(\d+)\s*$/i);
    if (m && +m[1] < state.meta.layers && +m[2] < state.meta.neurons) openNeuron(+m[1], +m[2]);
    else concept(v);
  };
}

window.neuronaut = { playRipple: (d) => playRippleData(d), state };

boot().catch((e) => { $('model-line').textContent = `Could not load the atlas: ${e.message}. Build it with \`neuronaut atlas\`.`; });
