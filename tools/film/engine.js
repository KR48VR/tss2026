// Film engine for the Work and AI Map films: window.renderAt(t) draws film time t (seconds) deterministically on a
// 1920x1080 canvas, so a headless browser can screenshot any frame. A film script (film1.js, film2.js) defines
// FILM = {duration, fps, recs:[names], scenes:[{t0, t1, draw(ctx, t)}], motion(t)?, field?:true}.
'use strict';
const W = 1920, H = 1080;
const C = { bg: '#0f1516', bg2: '#17211f', panel: '#1c2426', fg: '#e8e9e4', muted: '#a4adaf', faint: '#6c777a', line: '#2f3a3c',
  teal: '#5fc0b9', tealDeep: '#2e6f6a', tealHi: '#9fe8df', orange: '#e0a04e', blue: '#8fb0e0', red: '#e07a68', dim: '#4a585b', green: '#7fcb7f', violet: '#b79bea' };
const FD = '"Newsreader", Georgia, serif', FB = '"Public Sans", system-ui, sans-serif', FM = '"JetBrains Mono", monospace';

// ---------------------------------------------------------------------------------------------- easing and timing
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const lerp = (a, b, u) => a + (b - a) * u;
const smooth = u => { u = clamp(u); return u * u * (3 - 2 * u); };
const smoother = u => { u = clamp(u); return u * u * u * (u * (u * 6 - 15) + 10); };
// 0 before a, rises over fi, 1 until b-fo, falls to 0 at b
const win = (t, a, b, fi = 0.5, fo = 0.5) => Math.min(fi > 0 ? smooth((t - a) / fi) : (t >= a ? 1 : 0), fo > 0 ? smooth((b - t) / fo) : (t < b ? 1 : 0));
const span = (t, a, b) => clamp((t - a) / (b - a));
const loglerp = (a, b, u) => Math.exp(lerp(Math.log(a), Math.log(b), u));
function hexrgb(h) { const n = parseInt(h.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
function mix(a, b, u) { const A = hexrgb(a), B = hexrgb(b); return `rgb(${Math.round(lerp(A[0], B[0], u))},${Math.round(lerp(A[1], B[1], u))},${Math.round(lerp(A[2], B[2], u))})`; }

// ---------------------------------------------------------------------------------------------- images (recording frames)
const IMG = new Map(); let PENDING = [];
function getImg(url) {
  let e = IMG.get(url);
  if (!e) {
    const im = new Image(); e = { img: im, ok: false };
    e.p = new Promise(res => { im.onload = () => im.decode().then(() => { e.ok = true; res(); }, () => { e.ok = true; res(); }); im.onerror = () => { e.ok = true; res(); }; });
    im.src = url; IMG.set(url, e);
    if (IMG.size > 260) { const k = IMG.keys().next().value; IMG.delete(k); }   // oldest first
  }
  if (!e.ok) { PENDING.push(e.p); return null; }
  return e.img;
}
const REC = {};
function recFrame(name, fi) { const m = REC[name]; const k = clamp(Math.floor(fi), 0, m.frames.length - 1); return { m, k, fr: m.frames[k] }; }
function recLen(name) { return REC[name].frames.length; }

// ---------------------------------------------------------------------------------------------- drawing helpers
function rrect(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }

function wrap(ctx, str, maxW) {
  const out = [];
  for (const para of String(str).split('\n')) {
    const words = para.split(' '); let line = '';
    for (const w of words) { const t = line ? line + ' ' + w : w; if (ctx.measureText(t).width > maxW && line) { out.push(line); line = w; } else line = t; }
    out.push(line);
  }
  return out;
}

// text block; returns the height used. o: {size, font, weight, color, align, alpha, maxW, lh, italic, track}
function text(ctx, str, x, y, o = {}) {
  const size = o.size || 30, lh = o.lh || 1.35;
  ctx.save();
  ctx.globalAlpha *= (o.alpha ?? 1);
  ctx.font = `${o.italic ? 'italic ' : ''}${o.weight || 400} ${size}px ${o.font || FB}`;
  ctx.fillStyle = o.color || C.fg; ctx.textAlign = o.align || 'left'; ctx.textBaseline = 'alphabetic';
  if (o.track) ctx.letterSpacing = o.track + 'px';
  const lines = o.maxW ? wrap(ctx, str, o.maxW) : String(str).split('\n');
  lines.forEach((l, i) => ctx.fillText(l, x, y + size + i * size * lh));
  ctx.restore();
  return lines.length * size * lh;
}
function textH(ctx, str, o = {}) { const size = o.size || 30; ctx.save(); ctx.font = `${o.weight || 400} ${size}px ${o.font || FB}`; const n = o.maxW ? wrap(ctx, str, o.maxW).length : String(str).split('\n').length; ctx.restore(); return n * size * (o.lh || 1.35); }

// a side panel: kicker (small caps), title (serif), body; content cross-fades between phases
function panel(ctx, x, y, w, p, alpha = 1, dy = 0) {
  if (alpha <= 0.001) return;
  ctx.save(); ctx.globalAlpha *= alpha;
  let yy = y + dy;
  if (p.kicker) { yy += text(ctx, p.kicker.toUpperCase(), x, yy, { size: 19, weight: 600, color: p.kc || C.teal, track: 2.2 }) + 14; }
  if (p.title) { yy += text(ctx, p.title, x, yy, { size: p.ts || 50, font: FD, weight: 500, maxW: w, lh: 1.12 }) + 18; }
  if (p.body) { yy += text(ctx, p.body, x, yy, { size: p.bs || 29, color: p.bc || '#cdd3d1', maxW: w, lh: 1.42 }); }
  if (p.foot) { yy += 22; text(ctx, p.foot, x, yy, { size: 21, color: C.faint, maxW: w, lh: 1.4 }); }
  ctx.restore();
}

// a playing recording inside a rounded screen; o: {alpha, crop:{x,y,w,h} in CSS px, cursor:true, glow}
function screen(ctx, name, fi, r, o = {}) {
  const a = o.alpha ?? 1; if (a <= 0.001) return;
  const { m, k, fr } = recFrame(name, fi);
  const im = getImg(`rec/${name}/${m.files[fr[0]]}`);
  const crop = o.crop || { x: 0, y: 0, w: m.w, h: m.h };
  const rad = Math.max(6, 14 * r.w / 1180);
  ctx.save(); ctx.globalAlpha *= a;
  ctx.save(); ctx.shadowColor = 'rgba(0,0,0,0.6)'; ctx.shadowBlur = 60 * r.w / 1180; ctx.shadowOffsetY = 22 * r.w / 1180;
  rrect(ctx, r.x, r.y, r.w, r.h, rad); ctx.fillStyle = '#141a1b'; ctx.fill(); ctx.restore();
  ctx.save(); rrect(ctx, r.x, r.y, r.w, r.h, rad); ctx.clip();
  if (im) ctx.drawImage(im, crop.x * m.dsf, crop.y * m.dsf, crop.w * m.dsf, crop.h * m.dsf, r.x, r.y, r.w, r.h);
  // cursor and click ripple
  if (fr[3] && o.cursor !== false) {
    const sx = r.w / crop.w, cx = r.x + (fr[1] - crop.x) * sx, cy = r.y + (fr[2] - crop.y) * sx, s = Math.max(0.6, sx);
    for (let j = 0; j < 12 && k - j >= 0; j++) {
      if (m.frames[k - j][4]) { const age = (j + (fi - k)) / 10; if (age < 1) { ctx.beginPath(); ctx.arc(cx, cy, (10 + 34 * smooth(age)) * s, 0, 7); ctx.strokeStyle = `rgba(255,214,120,${0.75 * (1 - age)})`; ctx.lineWidth = 3 * s; ctx.stroke(); } break; }
    }
    ctx.beginPath(); ctx.arc(cx, cy + 2 * s, 11 * s, 0, 7); ctx.fillStyle = 'rgba(0,0,0,0.35)'; ctx.fill();
    ctx.beginPath(); ctx.arc(cx, cy, 10 * s, 0, 7); ctx.fillStyle = 'rgba(255,255,255,0.92)'; ctx.fill();
    ctx.lineWidth = 2.5 * s; ctx.strokeStyle = 'rgba(20,26,27,0.9)'; ctx.stroke();
  }
  ctx.restore();
  rrect(ctx, r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1, rad); ctx.strokeStyle = o.glow ? `rgba(95,192,185,${0.55 * o.glow})` : 'rgba(120,140,140,0.35)'; ctx.lineWidth = o.glow ? 3 : 1.5; ctx.stroke();
  ctx.restore();
}

// playback position of a recording that starts at t0 with in-point `inp` and `speed` recorded frames per film frame
const play = (t, t0, inp = 0, speed = 1) => inp + Math.max(0, t - t0) * 24 * speed;

// ---------------------------------------------------------------------------------------------- the task field
let F = null;   // {n, x, y, year, share, feature, groups, ...}
function fieldInit(d) {
  F = d; F.X = Float32Array.from(d.x); F.Y = Float32Array.from(d.y); F.YR = Float32Array.from(d.year); F.S = Float32Array.from(d.share);
  F.isFeat = new Uint8Array(d.n); d.feature.forEach(i => F.isFeat[i] = 1);
  const b = d.feature_bbox; F.fc = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
  F.pitch = Math.min(...d.groups.map(g => g.pitch));
}
const LIT = ['#24524e', '#2b625d', '#33736d', '#3b847d', '#47968e', '#56a89f', '#68bab0', '#7fcbc1'];
// s: {cx, cy, zoom, year, lit (0..1), base (0..1 brightness of unlit dots), labels (0..1), feat (0..1), alpha, wave}
let FC = null, FCX = null;
function field(ctx, s) {
  if (!F || (s.alpha ?? 1) <= 0.001) return;
  if ((s.blur || 0) > 0.3) {   // out of focus: draw sharp off screen, then composite once through a blur filter
    if (!FC) { FC = new OffscreenCanvas(W, H); FCX = FC.getContext('2d'); }
    FCX.clearRect(0, 0, W, H); field(FCX, { ...s, blur: 0, alpha: 1 });
    ctx.save(); ctx.globalAlpha *= (s.alpha ?? 1); ctx.filter = `blur(${s.blur.toFixed(2)}px)`; ctx.drawImage(FC, 0, 0); ctx.restore();
    return;
  }
  const z = s.zoom || 1, cx = s.cx ?? W / 2, cy = s.cy ?? H / 2, d = Math.max(1.2, F.pitch * 0.64 * z), hd = d / 2;
  const year = s.year ?? 0, lit = s.lit ?? 0, base = s.base ?? 0.4, feat = s.feat || 0;
  ctx.save(); ctx.globalAlpha *= (s.alpha ?? 1);
  const bins = Array.from({ length: LIT.length + 2 }, () => []);   // [unlit, ...lit levels, fresh]
  for (let i = 0; i < F.n; i++) {
    const x = (F.X[i] - cx) * z + W / 2, y = (F.Y[i] - cy) * z + H / 2;
    if (x < -d || x > W + d || y < -d || y > H + d) continue;
    if (lit > 0 && F.YR[i] <= year) {
      if (year - F.YR[i] < 0.3) bins[LIT.length + 1].push(x, y); else bins[1 + Math.min(LIT.length - 1, Math.floor(F.S[i] * LIT.length * 1.05))].push(x, y);
    } else bins[0].push(x, y);
  }
  const drawBin = (arr, color, a) => { if (!arr.length || a <= 0.001) return; ctx.globalAlpha = (s.alpha ?? 1) * a; ctx.fillStyle = color; for (let j = 0; j < arr.length; j += 2) ctx.fillRect(arr[j] - hd, arr[j + 1] - hd, d, d); };
  // unlit dots, with an optional brightening wave travelling left to right
  if (s.wave != null) {
    const pts = bins[0];
    for (let j = 0; j < pts.length; j += 2) {
      const u = clamp((s.wave * (W + 600) - 300 - pts[j]) / 300);
      ctx.globalAlpha = (s.alpha ?? 1) * lerp(0.12, base, u); ctx.fillStyle = C.dim; ctx.fillRect(pts[j] - hd, pts[j + 1] - hd, d, d);
    }
  } else drawBin(bins[0], C.dim, base);
  for (let b = 0; b < LIT.length; b++) drawBin(bins[1 + b], LIT[b], lit * (1 - 0.65 * feat));
  drawBin(bins[LIT.length + 1], '#d8fff8', lit * (1 - 0.65 * feat));
  // the featured job's tasks, ringed
  if (feat > 0) {
    ctx.globalAlpha = (s.alpha ?? 1) * feat;
    for (const i of F.feature) {
      const x = (F.X[i] - cx) * z + W / 2, y = (F.Y[i] - cy) * z + H / 2;
      const on = lit > 0 && F.YR[i] <= year;
      ctx.fillStyle = on ? '#d8fff8' : '#e8e9e4'; ctx.fillRect(x - hd, y - hd, d, d);
    }
  }
  // family labels
  if ((s.labels || 0) > 0.001) {
    ctx.globalAlpha = (s.alpha ?? 1) * s.labels;
    for (const g of F.groups) {
      if (g.w * z < 120 || g.h * z < 40) continue;
      const x = (g.x - cx) * z + W / 2 + 6, y = (g.y - cy) * z + H / 2 + 4;
      const fs = clamp(17 * Math.sqrt(z), 17, 30);
      ctx.font = `600 ${fs}px ${FB}`; ctx.fillStyle = 'rgba(15,21,22,0.85)';
      const tw = ctx.measureText(g.name).width; if (tw > g.w * z - 14) continue;
      ctx.fillRect(x - 3, y - 2, tw + 8, fs + 7);
      ctx.fillStyle = C.fg; ctx.textBaseline = 'top'; ctx.fillText(g.name, x + 1, y + 1);
    }
  }
  ctx.restore();
}

function background(ctx) {
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
  const g = ctx.createRadialGradient(W * 0.5, H * 0.45, 100, W * 0.5, H * 0.5, W * 0.75);
  g.addColorStop(0, 'rgba(36,52,52,0.55)'); g.addColorStop(1, 'rgba(0,0,0,0)'); ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
}
// a dark wash to keep text readable over the field
function wash(ctx, a, x0 = 0, y0 = 0, w = W, h = H) { if (a <= 0) return; ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = C.bg; ctx.fillRect(x0, y0, w, h); ctx.restore(); }
function vignette(ctx, a = 0.55) {
  const g = ctx.createRadialGradient(W / 2, H / 2, H * 0.35, W / 2, H / 2, W * 0.72);
  g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, `rgba(0,0,0,${a})`); ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
}

// ---------------------------------------------------------------------------------------------- render loop
let CANVAS, CTX, ACC, ACCX, TMP, TMPX;
function drawAt(ctx, t) {
  ctx.save(); ctx.globalAlpha = 1; background(ctx);
  for (const s of FILM.scenes) if (t >= s.t0 && t < s.t1) { ctx.save(); s.draw(ctx, t); ctx.restore(); }
  ctx.restore();
}
async function renderAt(t) {
  t = Math.min(t, FILM.duration - 1e-6);
  const m = FILM.motion ? FILM.motion(t) : 0;
  const K = m > 0.08 ? Math.round(clamp(3 + m * 9, 3, 12)) : 1;
  for (let pass = 0; pass < 4; pass++) {
    PENDING = [];
    if (K === 1) drawAt(CTX, t);
    else {
      ACCX.clearRect(0, 0, W, H);
      for (let i = 0; i < K; i++) {
        const ts = t + (i / (K - 1) - 0.5) * 0.4 / FILM.fps;
        TMPX.clearRect(0, 0, W, H); drawAt(TMPX, ts);
        ACCX.globalAlpha = 1 / (i + 1); ACCX.drawImage(TMP, 0, 0);
      }
      ACCX.globalAlpha = 1; CTX.drawImage(ACC, 0, 0);
    }
    if (!PENDING.length) return true;
    await Promise.all(PENDING);
  }
  return false;
}

async function init() {
  CANVAS = document.getElementById('c'); CTX = CANVAS.getContext('2d');
  ACC = new OffscreenCanvas(W, H); ACCX = ACC.getContext('2d'); TMP = new OffscreenCanvas(W, H); TMPX = TMP.getContext('2d');
  const faces = [`500 60px Newsreader`, `600 60px Newsreader`, `italic 500 60px Newsreader`, `400 30px "Public Sans"`, `600 30px "Public Sans"`, `700 30px "Public Sans"`, `400 20px "JetBrains Mono"`];
  await Promise.all(faces.map(f => document.fonts.load(f)));
  await document.fonts.ready;
  if (FILM.field) fieldInit(await (await fetch('data/field.json')).json());
  for (const n of FILM.recs || []) REC[n] = await (await fetch(`rec/${n}/manifest.json`)).json();
  FILM.D = {}; for (const n of FILM.data || []) FILM.D[n] = await (await fetch(`data/${n}.json`)).json();
  if (FILM.setup) FILM.setup();
  window.renderAt = renderAt;
  window.FILMINFO = { duration: FILM.duration, fps: FILM.fps, fonts: faces.map(f => document.fonts.check(f)) };
  await renderAt(0);
  document.title = 'ready';
}
window.addEventListener('load', init);
