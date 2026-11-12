// The tower: one disc per layer, one point per MLP neuron. Everything that lights up is driven by data: the atlas
// (layout, stats, wiring, words) or a measured ripple from the model. Nothing animates on its own.
import * as THREE from './vendor/three.module.min.js';
import { OrbitControls } from './vendor/OrbitControls.js';

const AMBER = new THREE.Color('#f0a63a');
const STEEL = new THREE.Color('#8fb3d9');
const WHITE = new THREE.Color('#e8ecf4');
const BASE = new THREE.Color('#5d6578');
const DOT = new THREE.Color('#9aa4bb');

export class Tower {
  constructor(canvas, meta, layout, stats) {
    this.meta = meta; this.L = meta.layers; this.N = meta.neurons;
    this.R = 4.2; this.gap = 0.46;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 200);
    this.camera.position.set(16, this.height * 1.25, 16);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, this.height * 0.45, 0);
    this.controls.enableDamping = true; this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 4; this.controls.maxDistance = 40;
    this.glow = new Float32Array(this.L * this.N);      // per-neuron highlight 0..1 (decays)
    this.tint = new Float32Array(this.L * this.N);      // 0 = amber (lever), 1 = white (control)
    this.ringGlow = new Float32Array(this.L);
    this.ringTint = new Float32Array(this.L);
    this._buildPoints(layout, stats);
    this._buildRings();
    this.wires = new THREE.Group(); this.scene.add(this.wires);
    this.labels = new THREE.Group(); this.scene.add(this.labels);
    this.words = new THREE.Group(); this.scene.add(this.words);
    this.says = new THREE.Group(); this.scene.add(this.says);
    this._layerLabels();
    this.selected = null;
    this.timeline = null;
    this.onPick = () => {};
    this._pointer(canvas);
    this.resize();
    window.addEventListener('resize', () => this.resize());
    this.renderer.setAnimationLoop((t) => this._frame(t));
  }

  get height() { return (this.L - 1) * this.gap; }
  pos(layer, neuron) {
    const i = layer * this.N + neuron;
    return new THREE.Vector3(this.xy[2 * i] * this.R, layer * this.gap, this.xy[2 * i + 1] * this.R);
  }

  _buildPoints(layout, stats) {
    const n = this.L * this.N;
    this.xy = layout;
    const position = new Float32Array(n * 3), base = new Float32Array(n), size = new Float32Array(n);
    const layerMax = new Float32Array(this.L);
    for (let i = 0; i < n; i++) { const L = Math.floor(i / this.N); layerMax[L] = Math.max(layerMax[L], stats[4 * i + 3]); }
    for (let L = 0; L < this.L; L++) for (let k = 0; k < this.N; k++) {
      const i = L * this.N + k;
      position[3 * i] = layout[2 * i] * this.R; position[3 * i + 1] = L * this.gap; position[3 * i + 2] = layout[2 * i + 1] * this.R;
      const busy = Math.sqrt(stats[4 * i + 3] / layerMax[L]);      // how active the neuron usually is, within its layer
      base[i] = 0.35 + 0.65 * busy; size[i] = 2.2 + 2.4 * busy;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(position, 3));
    g.setAttribute('base', new THREE.BufferAttribute(base, 1));
    g.setAttribute('size', new THREE.BufferAttribute(size, 1));
    g.setAttribute('glow', new THREE.BufferAttribute(this.glow, 1));
    g.setAttribute('tint', new THREE.BufferAttribute(this.tint, 1));
    this.pointsGeo = g;
    this.points = new THREE.Points(g, new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uBase: { value: DOT }, uAmber: { value: AMBER }, uWhite: { value: WHITE }, uScale: { value: 1 } },
      vertexShader: `attribute float base; attribute float size; attribute float glow; attribute float tint;
        varying float vA; varying float vG; varying float vT; uniform float uScale;
        void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;
          vG = glow; vT = tint; vA = base;
          gl_PointSize = (size + glow * 9.0) * uScale * (30.0 / -mv.z); }`,
      fragmentShader: `uniform vec3 uBase; uniform vec3 uAmber; uniform vec3 uWhite; varying float vA; varying float vG; varying float vT;
        void main() { vec2 c = gl_PointCoord - 0.5; float d = length(c); if (d > 0.5) discard;
          float soft = smoothstep(0.5, 0.0, d);
          vec3 hi = mix(uAmber, uWhite, vT);
          vec3 col = mix(uBase, hi, clamp(vG, 0.0, 1.0));
          float a = soft * (vA * 0.85 + vG * 0.95);
          gl_FragColor = vec4(col * a, a); }`,
    }));
    this.scene.add(this.points);
  }

  _buildRings() {
    this.rings = [];
    for (let L = 0; L < this.L; L++) {
      const pts = [];
      for (let a = 0; a <= 128; a++) { const t = a / 128 * Math.PI * 2; pts.push(new THREE.Vector3(Math.cos(t) * this.R * 1.04, L * this.gap, Math.sin(t) * this.R * 1.04)); }
      const ring = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: BASE, transparent: true, opacity: 0.16 }));
      this.scene.add(ring); this.rings.push(ring);
    }
  }

  _sprite(text, { color = '#7a8296', size = 22, weight = 400, font = 'IBM Plex Mono' } = {}) {
    const c = document.createElement('canvas'), x = c.getContext('2d');
    x.font = `${weight} ${size * 2}px "${font}", ui-monospace, monospace`;
    const w = Math.ceil(x.measureText(text).width) + 8;
    c.width = w; c.height = size * 2.6;
    x.font = `${weight} ${size * 2}px "${font}", ui-monospace, monospace`;
    x.fillStyle = color; x.textBaseline = 'middle'; x.fillText(text, 4, c.height / 2);
    const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
    s.scale.set(w / c.height * 0.36, 0.36, 1);
    return s;
  }

  _layerLabels() {
    const kinds = this.meta.layer_kinds || [];
    for (let L = 0; L < this.L; L += 1) {
      if (L % 3 && L !== this.L - 1) continue;
      const k = (kinds[L] || '').startsWith('full') ? 'attention' : (kinds[L] || '').includes('linear') ? 'DeltaNet' : '';
      const s = this._sprite(`L${L}${k ? '  ' + k : ''}`, { size: 16 });
      s.userData.layer = L;
      s.center.set(1, 0.5);
      this.labels.add(s);
    }
  }

  _pointer(canvas) {
    const ray = new THREE.Raycaster(); ray.params.Points.threshold = 0.07;
    let down = null;
    canvas.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
    canvas.addEventListener('pointerup', (e) => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
      const r = canvas.getBoundingClientRect();
      ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), this.camera);
      const hit = ray.intersectObject(this.points)[0];
      if (hit) this.onPick(Math.floor(hit.index / this.N), hit.index % this.N);
    });
  }

  resize() {
    const c = this.renderer.domElement, w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    this.points.material.uniforms.uScale.value = h / 900 * Math.min(2, window.devicePixelRatio);
  }

  // ---------- selection, wiring, words ----------
  select(layer, neuron, info) {
    this.selected = [layer, neuron];
    this.clearWires();
    const p = this.pos(layer, neuron);
    const i = layer * this.N + neuron;
    this.glow.fill(0); this.tint.fill(0); this.glow[i] = 1;
    if (info) {
      const all = [...(info.in || []).map(w => [w, STEEL]), ...(info.out || []).map(w => [w, AMBER])];
      const maxW = Math.max(1e-6, ...all.map(([w]) => Math.abs(w[2])));
      for (const [[L2, n2, v], col] of all) {
        const q = this.pos(L2, n2);
        const mid = p.clone().add(q).multiplyScalar(0.5);
        const out = new THREE.Vector3(mid.x, 0, mid.z); out.setLength(out.length() + 1.4); out.y = mid.y;
        const curve = new THREE.QuadraticBezierCurve3(p, out, q);
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(40)),
          new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.15 + 0.75 * Math.abs(v) / maxW }));
        this.wires.add(line);
        this.glow[L2 * this.N + n2] = Math.max(this.glow[L2 * this.N + n2], 0.35 + 0.5 * Math.abs(v) / maxW);
        this.tint[L2 * this.N + n2] = col === STEEL ? 0.6 : 0;
      }
      this.showWords(info['w+'] || []);
    }
    this.pointsGeo.attributes.glow.needsUpdate = true; this.pointsGeo.attributes.tint.needsUpdate = true;
    this.focus(p);
  }

  clearSays() {
    for (const c of [...this.says.children]) { c.material.map.dispose(); c.material.dispose(); this.says.remove(c); }
  }

  clearWires() {
    for (const c of [...this.wires.children]) { c.geometry.dispose(); c.material.dispose(); this.wires.remove(c); }
  }

  showWords(words) {
    for (const c of [...this.words.children]) { c.material.map.dispose(); c.material.dispose(); this.words.remove(c); }
    words.slice(0, 6).forEach((w, j) => {
      const s = this._sprite(w, { color: '#f0a63a', size: 22, font: 'IBM Plex Sans', weight: 500 });
      const a = (j / 6) * Math.PI * 2;
      s.position.set(Math.cos(a) * 2.2, this.height + 1.1 + (j % 2) * 0.35, Math.sin(a) * 2.2);
      this.words.add(s);
    });
  }

  focus(p) {
    this._tween = { from: this.controls.target.clone(), to: new THREE.Vector3(0, (p.y + this.height * 0.5) / 2, 0), t: 0 };
  }

  // ---------- ripple ----------
  // data: {parts:[[L,n,d]], lever:{moved:[L], top:[[[n,dh]...]...]}, control:{...}}
  playRipple(data, onPhase = () => {}) {
    const start = Math.min(...data.parts.map(p => p[0]));
    const maxMoved = Math.max(...data.lever.moved, ...data.control.moved, 1e-6);
    this.clearWires(); this.glow.fill(0); this.tint.fill(0);
    this.timeline = { t0: performance.now(), data, start, maxMoved, phases: ['lever', 'control'], onPhase, last: -1 };
  }

  _stepRipple(now) {
    const tl = this.timeline; if (!tl) return;
    const per = 5200, el = now - tl.t0, phaseIdx = Math.floor(el / per);
    if (phaseIdx >= tl.phases.length) { this.timeline = null; tl.onPhase(null); return; }
    const key = tl.phases[phaseIdx], d = tl.data[key], local = (el % per) / per;
    if (phaseIdx !== tl.last) { tl.last = phaseIdx; tl.onPhase(key); this.glow.fill(0); this.ringGlow.fill(0); this.clearSays(); tl.shown = new Set(); }
    const front = tl.start + local * 1.25 * (this.L - tl.start);      // the wave climbs from the pushed layer
    const tintV = key === 'lever' ? 0 : 1;
    for (let L = tl.start; L < this.L; L++) {
      const reached = front >= L;
      const strength = d.moved[L] / tl.maxMoved;
      this.ringGlow[L] = reached ? Math.max(this.ringGlow[L] * 0.985, strength) : 0;
      this.ringTint[L] = tintV;
      if (reached && d.says && d.says[L] && d.says[L].length && !tl.shown.has(L) && (L - tl.start) % 1 === 0) {
        tl.shown.add(L);
        const sp = this._sprite(d.says[L].slice(0, 3).join('  '), { color: key === 'lever' ? '#f0a63a' : '#a3a9b8', size: 17, font: 'IBM Plex Sans', weight: 500 });
        sp.userData.layer = L; sp.center.set(0, 0.5); this.says.add(sp);
      }
      if (reached) {
        const top = d.top[L], m = Math.max(1e-6, ...top.map(x => Math.abs(x[1])));
        for (const [n, dh] of top) {
          const i = L * this.N + n;
          this.glow[i] = Math.max(this.glow[i] * 0.99, strength * Math.abs(dh) / m); this.tint[i] = tintV;
        }
      }
    }
    for (const [L, n] of tl.data.parts) { const i = L * this.N + n; if (key === 'lever') { this.glow[i] = 1; this.tint[i] = 0; } }
    this.pointsGeo.attributes.glow.needsUpdate = true; this.pointsGeo.attributes.tint.needsUpdate = true;
  }

  _frame(now) {
    if (this._tween) {
      this._tween.t = Math.min(1, this._tween.t + 0.04);
      const e = 1 - Math.pow(1 - this._tween.t, 3);
      this.controls.target.lerpVectors(this._tween.from, this._tween.to, e);
      if (this._tween.t >= 1) this._tween = null;
    }
    if (this.orbit) { const a = this.orbit.speed; const p = this.camera.position, t = this.controls.target;
      const x = p.x - t.x, z = p.z - t.z; p.x = t.x + x * Math.cos(a) - z * Math.sin(a); p.z = t.z + x * Math.sin(a) + z * Math.cos(a); }
    this._stepRipple(now);
    for (let L = 0; L < this.L; L++) {
      const g = this.ringGlow[L], m = this.rings[L].material;
      m.color.copy(BASE).lerp(this.ringTint[L] ? WHITE : AMBER, Math.min(1, g));
      m.opacity = 0.16 + 0.7 * g;
    }
    this.controls.update();
    const right = new THREE.Vector3().subVectors(this.controls.target, this.camera.position).cross(this.camera.up).setY(0).normalize();
    for (const s of this.labels.children) s.position.copy(right).multiplyScalar(-this.R * 1.22).setY(s.userData.layer * this.gap);
    for (const s of this.says.children) s.position.copy(right).multiplyScalar(this.R * 1.18).setY(s.userData.layer * this.gap);
    this.renderer.render(this.scene, this.camera);
  }
}
