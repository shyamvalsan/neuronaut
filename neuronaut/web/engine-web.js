// In-browser engine: Qwen3.5-0.8B (4-bit ONNX) on WebGPU, with the `steer` input added by neuronaut.export_onnx.
// Our edited graph is small and served with the site; the weights stream from the onnx-community repo at a pinned
// revision. The push for a neuron is dose * scale[L] * unit(W_down[:, n]); unit columns come from atlas/down_LL.f16.
const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs';
const TJS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6';

export class WebEngine {
  constructor(cfg) {
    // cfg: {repo, revision, graph, embedGraph, dataName, layers, hidden, scale, atlas}
    this.cfg = cfg; this.ready = false; this.columns = {};
  }

  async load(onProgress = () => {}) {
    const { repo, revision } = this.cfg;
    const hub = (f) => `https://huggingface.co/${repo}/resolve/${revision}/${f}`;
    onProgress('Loading the runtime');
    this.ort = await import(ORT_URL);
    this.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
    const { AutoTokenizer } = await import(TJS_URL);
    this.tok = await AutoTokenizer.from_pretrained(repo, { revision });
    const providers = navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
    this.device = navigator.gpu ? 'WebGPU' : 'CPU (WebAssembly)';
    onProgress(`Downloading the model, about 650 MB on first visit (${this.device})`);
    // 4-bit embedding table, looked up here instead of through ONNX (its GatherBlockQuantized op is missing on the
    // WebAssembly backend). Per row: 512 bytes of packed 4-bit values, 32 float32 scales, 16 bytes of 4-bit zero points.
    const buf = await (await fetch(hub('onnx/embed_tokens_q4.onnx_data'))).arrayBuffer();
    const V = 248320;
    this.embQ = new Uint8Array(buf, 0, V * 512);
    this.embS = new Float32Array(buf, V * 512, V * 32);
    this.embZ = new Uint8Array(buf, V * 512 + V * 32 * 4, V * 16);
    this.dec = await this.ort.InferenceSession.create(this.cfg.graph, {
      executionProviders: providers,
      externalData: [{ path: this.cfg.dataName, data: hub('onnx/decoder_model_merged_q4.onnx_data') }],
    });
    this.inputMeta = {};
    for (const name of this.dec.inputNames) this.inputMeta[name] = true;
    this.ready = true;
    onProgress('');
  }

  chatIds(prompt, system) {
    const msgs = (system ? [{ role: 'system', content: system }] : []).concat([{ role: 'user', content: prompt }]);
    const text = this.tok.apply_chat_template(msgs, { tokenize: false, add_generation_prompt: true, enable_thinking: false });
    return Array.from(this.tok.encode(text, { add_special_tokens: false }));
  }

  _emptyState() {
    const T = this.ort.Tensor, st = {};
    for (const name of this.dec.inputNames) {
      if (name.startsWith('past_conv')) st[name] = new T('float32', new Float32Array(6144 * 4), [1, 6144, 4]);
      else if (name.startsWith('past_recurrent')) st[name] = new T('float32', new Float32Array(16 * 128 * 128), [1, 16, 128, 128]);
      else if (name.startsWith('past_key_values')) st[name] = new T('float32', new Float32Array(0), [1, 2, 0, 256]);
    }
    return st;
  }

  embedRows(ids) {
    const H = this.cfg.hidden, out = new Float32Array(ids.length * H);
    ids.forEach((t, r) => {
      const q = t * 512, sc = t * 32, zp = t * 16;
      for (let j = 0; j < H; j++) {
        const byte = this.embQ[q + (j >> 1)], v = (j & 1) ? byte >> 4 : byte & 15;
        const blk = j >> 5, zb = this.embZ[zp + (blk >> 1)], z = (blk & 1) ? zb >> 4 : zb & 15;
        out[r * H + j] = (v - z) * this.embS[sc + blk];
      }
    });
    return out;
  }

  async _step(ids, start, state, steer) {
    const T = this.ort.Tensor;
    const embeds = new T('float32', this.embedRows(ids), [1, ids.length, this.cfg.hidden]);
    const n = ids.length, pos = new BigInt64Array(3 * n);
    for (let k = 0; k < 3; k++) for (let i = 0; i < n; i++) pos[k * n + i] = BigInt(start + i);
    const feed = { inputs_embeds: embeds, attention_mask: new T('int64', new BigInt64Array(start + n).fill(1n), [1, start + n]),
      position_ids: new T('int64', pos, [3, 1, n]), steer: new T('float32', steer, [this.cfg.layers, this.cfg.hidden]), ...state };
    const out = await this.dec.run(feed);
    const next = {};
    for (const name of this.dec.outputNames) {
      if (name.startsWith('present.')) next['past_key_values.' + name.slice(8)] = out[name];
      else if (name.startsWith('present_')) next['past_' + name.slice(8)] = out[name];
    }
    const lg = out.logits, V = lg.dims[2], data = lg.data, off = (lg.dims[1] - 1) * V;
    let best = 0, bv = -Infinity;
    for (let v = 0; v < V; v++) { const x = data[off + v]; if (x > bv) { bv = x; best = v; } }
    return { token: best, state: next };
  }

  async generate(ids, steer, maxNew = 64, onToken = () => {}) {
    const stop = new Set([248044, 248046]);
    let state = this._emptyState(), start = 0, cur = ids, out = [];
    for (let i = 0; i < maxNew; i++) {
      const r = await this._step(cur, start, state, steer);
      start += cur.length; state = r.state;
      if (stop.has(r.token)) break;
      out.push(r.token); cur = [r.token];
      onToken(this.tok.decode(out, { skip_special_tokens: true }));
    }
    return this.tok.decode(out, { skip_special_tokens: true });
  }

  // ---------- push vectors ----------
  async column(L, n) {
    if (!this.columns[L]) this.columns[L] = fetch(`${this.cfg.atlas}down_${String(L).padStart(2, '0')}.f16`).then(r => r.arrayBuffer());
    const buf = new Uint16Array(await this.columns[L]), H = this.cfg.hidden, col = new Float32Array(H);
    for (let i = 0; i < H; i++) col[i] = f16(buf[n * H + i]);
    return col;
  }

  async steerFor(parts, control = false) {
    const H = this.cfg.hidden, s = new Float32Array(this.cfg.layers * H);
    for (const [j, [L, n, d]] of parts.entries()) {
      const v = control ? randomUnit(H, `control|${L}|${n}|${j}`) : await this.column(L, n);
      const size = (control ? Math.abs(d) : d) * this.cfg.scale[L];
      for (let i = 0; i < H; i++) s[L * H + i] += size * v[i];
    }
    return s;
  }
}

function f16(h) {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function randomUnit(n, seed) {
  // deterministic Gaussian direction from a string seed (mulberry32 + Box-Muller); not the same draw as PyTorch's,
  // but the same kind of control: a random direction of exactly the lever's size.
  let h = 1779033703;
  for (const c of seed) h = Math.imul(h ^ c.charCodeAt(0), 3432918353), h = (h << 13) | (h >>> 19);
  const rnd = () => { h += 0x6d2b79f5; let t = h; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const v = new Float32Array(n); let norm = 0;
  for (let i = 0; i < n; i++) { const u = Math.max(rnd(), 1e-12), w = rnd(); v[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * w); norm += v[i] * v[i]; }
  norm = Math.sqrt(norm);
  for (let i = 0; i < n; i++) v[i] /= norm;
  return v;
}
