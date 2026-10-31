// Film 2: "How we built it together" — a teaching film about how King and Claude made the Work and AI Map (312 s).
// Six chapters under a minute each, each opened by a numbered card (segmenting), one idea on screen at a time
// (signalling), quiet music (weeding). Timed in bars of 3 s (80 beats a minute), so every change sits on a bar line.
// Every fact on screen comes from the companion page (site/companion.html); King's quotes have their spelling tidied.
'use strict';
const BAR = 3;
const b = n => n * BAR;
const SR = { x: 70, y: 172, w: 1180, h: 737.5 };
const PX = 1320, PY = 236, PW = 540;
const KIND = { tool: C.teal, teammate: C.orange, independent: '#9db0c2' };   // the companion page's colours for the three types

// ---------------------------------------------------------------------------------------------- the running order (bars)
const S = {
  poster: [0, 3],
  c1: [3, 5], brief: [5, 9], who: [9, 13], became: [13, 17],
  c2: [17, 19], record: [19, 22], tool: [22, 26], team: [26, 30], alone: [30, 34], tally: [34, 36],
  c3: [36, 38], q3: [38, 42], versions: [42, 47], meaning: [47, 51], lesson3: [51, 53],
  c4: [53, 55], q4: [55, 59], changes: [59, 64], trust: [64, 68],
  c5: [68, 70], scripts: [70, 74], found: [74, 78], rounds: [78, 81],
  c6: [81, 83], grow: [83, 87], keep: [87, 91],
  lessons: [91, 98], discuss: [98, 101], end: [101, 104],
};
for (const k in S) S[k] = [b(S[k][0]), b(S[k][1])];
const DUR = S.end[1];   // 312 s
const CH = [
  { n: 1, title: 'One question', sub: 'Who brought what to the work', at: S.c1, until: S.c2[0] },
  { n: 2, title: 'Three ways of working', sub: 'As a tool, as teammates, and alone', at: S.c2, until: S.c3[0] },
  { n: 3, title: 'Use it, then ask', sub: 'The best questions came from using the map', at: S.c3, until: S.c4[0] },
  { n: 4, title: 'Question the evidence', sub: 'Not the features: the evidence behind them', at: S.c4, until: S.c5[0] },
  { n: 5, title: 'Check everything', sub: 'By script, not by eye, every round', at: S.c5, until: S.c6[0] },
  { n: 6, title: 'Grow, and keep the record', sub: 'From one country to sixteen, with every step written down', at: S.c6, until: S.lessons[0] },
];
const inS = (t, k, fi = 0.6, fo = 0.6) => win(t, S[k][0], S[k][1], fi, fo);

// ---------------------------------------------------------------------------------------------- the record: 96 steps as dots
function stepsGrid(cols, x0, y0, gap) { return (i) => [x0 + (i % cols) * gap, y0 + Math.floor(i / cols) * gap]; }
// o: {alpha, focus: 'tool'|'teammate'|'independent'|null, ring: step number, r, pos: fn(i)->[x,y], reveal 0..1}
function steps(ctx, o) {
  const st = FILM.D.log.steps, a = o.alpha ?? 1; if (a <= 0.001) return;
  const r = o.r || 13;
  ctx.save(); ctx.globalAlpha *= a;
  st.forEach((s, i) => {
    if (o.reveal != null && i / st.length > o.reveal) return;
    const [x, y] = o.pos(i);
    const on = !o.focus || s.kinds.includes(o.focus);
    ctx.globalAlpha = a * (on ? 1 : 0.16);
    if (s.kinds.length === 1) { ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fillStyle = KIND[s.kinds[0]]; ctx.fill(); }
    else {   // a step that changed mode half-way: two halves
      ctx.beginPath(); ctx.arc(x, y, r, Math.PI / 2, Math.PI * 1.5); ctx.fillStyle = KIND[s.kinds[0]]; ctx.fill();
      ctx.beginPath(); ctx.arc(x, y, r, -Math.PI / 2, Math.PI / 2); ctx.fillStyle = KIND[s.kinds[1]]; ctx.fill();
    }
    if (s.who === 'king') { ctx.beginPath(); ctx.arc(x, y, r * 0.38, 0, 7); ctx.fillStyle = C.bg; ctx.fill(); }
    if (o.ring === s.n) {
      ctx.globalAlpha = a; ctx.beginPath(); ctx.arc(x, y, r + 9, 0, 7); ctx.strokeStyle = C.fg; ctx.lineWidth = 3; ctx.stroke();
      text(ctx, `step ${s.n}`, x, y - r - 46, { size: 20, align: 'center', color: C.fg, weight: 600 });
    }
  });
  ctx.restore();
}
function legend(ctx, x, y, a, o = {}) {
  if (a <= 0.001) return;
  const items = [['tool', 'King gave an instruction (tool)'], ['teammate', 'We reasoned together (teammates)'], ['independent', 'One side worked alone']];
  ctx.save(); ctx.globalAlpha *= a; ctx.font = `400 ${o.size || 24}px ${FB}`;
  const widths = items.map(([, s]) => ctx.measureText(s).width + 70), total = widths.reduce((p, q) => p + q, 0);
  let xx = o.align === 'center' ? x - total / 2 : x;
  items.forEach(([k, s], i) => {
    ctx.globalAlpha = a * (!o.focus || o.focus === k ? 1 : 0.3);
    ctx.beginPath(); ctx.arc(xx + 10, y + 12, 10, 0, 7); ctx.fillStyle = KIND[k]; ctx.fill();
    text(ctx, s, xx + 30, y - 6, { size: o.size || 24, color: '#cdd3d1' });
    xx += widths[i];
  });
  ctx.restore();
}

// ---------------------------------------------------------------------------------------------- cards and blocks
function poster(ctx, a) {
  if (a <= 0.001) return;
  ctx.save(); ctx.globalAlpha *= a;
  text(ctx, 'The Work and AI Map', W / 2, 200, { size: 30, color: C.teal, align: 'center', weight: 600, track: 3 });
  text(ctx, 'How we built it together', W / 2, 262, { size: 112, font: FD, weight: 500, align: 'center' });
  text(ctx, 'What a person and an AI each did, and what we learned', W / 2, 420, { size: 38, color: '#cdd3d1', align: 'center' });
  steps(ctx, { pos: (i) => [W / 2 - 47.5 * 17.4 + i * 17.4, 600 + 26 * Math.sin(i / 96 * Math.PI * 2)], r: 7 });
  legend(ctx, W / 2, 690, 1, { align: 'center', size: 22 });
  text(ctx, 'Six short chapters  ·  about five minutes', W / 2, 800, { size: 26, color: C.muted, align: 'center' });
  text(ctx, 'King and Claude  ·  2026', W / 2, 1000, { size: 22, color: C.faint, align: 'center' });
  ctx.restore();
}
function chapterCard(ctx, t, ch) {
  const a = win(t, ch.at[0], ch.at[1], 0.6, 0.6); if (a <= 0.001) return;
  const u = smooth(span(t, ch.at[0], ch.at[0] + 0.9));
  ctx.save(); ctx.globalAlpha *= a;
  text(ctx, String(ch.n), 560, 300 + (1 - u) * 20, { size: 300, font: FD, weight: 500, color: C.teal, align: 'right' });
  text(ctx, `CHAPTER ${ch.n} OF 6`, 620, 392, { size: 22, color: C.muted, weight: 600, track: 3 });
  text(ctx, ch.title, 620, 430 + (1 - u) * 14, { size: 84, font: FD, weight: 500 });
  text(ctx, ch.sub, 620, 548, { size: 34, color: '#cdd3d1', maxW: 1150 });
  ctx.restore();
}
function chapterTag(ctx, t) {   // small running header: which chapter, and six pips
  const ch = CH.find(c => t >= c.at[1] - 0.3 && t < c.until);
  const a = ch ? win(t, ch.at[1] - 0.3, ch.until, 0.6, 0.5) : 0; if (a <= 0.001) return;
  ctx.save(); ctx.globalAlpha *= a;
  text(ctx, `${ch.n}  ·  ${ch.title.toUpperCase()}`, 70, 62, { size: 20, color: C.muted, weight: 600, track: 2.4 });
  for (let i = 0; i < 6; i++) { rrect(ctx, W - 70 - (6 - i) * 46 + 10, 76, 36, 6, 3); ctx.fillStyle = i < ch.n ? C.teal : C.line; ctx.fill(); }
  ctx.restore();
}
function quote(ctx, a, q, who, y = 330, o = {}) {
  if (a <= 0.001) return;
  ctx.save(); ctx.globalAlpha *= a;
  text(ctx, '“', W / 2, y - 190, { size: 220, font: FD, color: C.orange, alpha: 0.5, align: 'center' });
  const h = text(ctx, q, W / 2, y, { size: o.size || 60, font: FD, italic: true, weight: 500, align: 'center', maxW: o.maxW || 1460, lh: 1.22 });
  text(ctx, who, W / 2, y + h + 34, { size: 26, color: C.orange, align: 'center', weight: 600 });
  ctx.restore();
  return y + h + 34 + 40;
}
function center(ctx, str, y, a, o = {}) { if (a > 0.001) text(ctx, str, o.x ?? W / 2, y, { size: o.size || 40, font: o.font || FB, weight: o.weight || 400, align: o.align || 'center', alpha: a, color: o.color || C.fg, maxW: o.maxW || 1500, lh: o.lh, italic: o.italic }); }
// a list whose items appear one by one; items: [str, t] ; returns nothing
function list(ctx, x, y, items, t, o = {}) {
  let yy = y;
  for (const [s, t0, sub] of items) {
    const a = smooth(span(t, t0, t0 + 0.6)) * (o.alpha ?? 1);
    if (a > 0.001) {
      ctx.save(); ctx.globalAlpha *= a;
      if (o.num) text(ctx, String(items.indexOf(items.find(it => it[0] === s)) + 1), x, yy, { size: o.size || 38, font: FD, weight: 600, color: C.teal });
      else { ctx.beginPath(); ctx.arc(x + 8, yy + (o.size || 38) * 0.62, 6, 0, 7); ctx.fillStyle = o.dot || C.teal; ctx.fill(); }
      ctx.restore();
    }
    const h = text(ctx, s, x + (o.num ? 52 : 34), yy + (1 - a) * 10, { size: o.size || 38, font: o.font || FB, alpha: a, maxW: o.maxW || 1300, lh: 1.25, weight: o.weight || 400 });
    yy += h + (sub ? 0 : (o.gap ?? 22));
    if (sub) { const h2 = text(ctx, sub, x + (o.num ? 52 : 34), yy + 2, { size: 26, alpha: a, color: C.muted, maxW: o.maxW || 1300 }); yy += h2 + (o.gap ?? 22); }
  }
}

// ---------------------------------------------------------------------------------------------- screens beside panels
function shown(t, t0, t1) { return Math.min(smooth(span(t, t0, t0 + 0.7)), 1 - smooth(span(t, t1 - 0.6, t1))); }
function scr(ctx, t, name, t0, t1, inp = 0, speed = null, crop = null, cursor = true) {
  const a = shown(t, t0, t1); if (a <= 0.001) return;
  const sp = speed ?? Math.max(0.2, (recLen(name) - inp - 6) / ((t1 - t0) * 24));
  const lift = (1 - smooth(span(t, t0, t0 + 0.7))) * 40;
  screen(ctx, name, play(t, t0, inp, sp), { x: SR.x, y: SR.y + lift, w: SR.w, h: SR.h }, { alpha: a, crop, cursor });
}
function side(ctx, t, t0, t1, p) { const a = win(t, t0 + 0.3, t1, 0.6, 0.5); panel(ctx, PX, PY, PW, p, a, (1 - smooth(span(t, t0 + 0.3, t0 + 0.9))) * 18); }

// ---------------------------------------------------------------------------------------------- the film
const GRID = stepsGrid(16, W / 2 - 7.5 * 58, 250, 58);
const FILM = {
  duration: DUR, fps: 24, data: ['log'],
  recs: ['role', 'meaning', 'overview', 'c_qa', 'countries', 'c_king'],
  scenes: [
    { t0: 0, t1: DUR, draw(ctx, t) { vignette(ctx, 0.45); chapterTag(ctx, t); } },
    { t0: 0, t1: S.poster[1], draw(ctx, t) { poster(ctx, 1 - smooth(span(t, S.poster[1] - 0.8, S.poster[1]))); } },
    { t0: S.end[0], t1: DUR, draw(ctx, t) { poster(ctx, smooth(span(t, S.end[0], S.end[0] + 1.0))); } },
    { t0: 0, t1: DUR, draw(ctx, t) { for (const ch of CH) if (t >= ch.at[0] && t < ch.at[1]) chapterCard(ctx, t, ch); } },

    // 1. one question
    { t0: S.brief[0], t1: S.brief[1], draw(ctx, t) {
      const a = inS(t, 'brief');
      center(ctx, 'KING’S BRIEF, IN SHORT', 250, a, { size: 22, color: C.orange, weight: 600 });
      center(ctx, 'Build a version of O*NET, the US database of jobs and the tasks in them, that links tasks, activities and skills. Connect it to Singapore’s own data. Lay the AI labs’ usage reports on top. Then forecast which jobs, skills and tasks are being disrupted, and when.', 310, a, { size: 50, font: FD, weight: 500, lh: 1.3, maxW: 1480 });
      center(ctx, 'One message started the project. Everything after it is on record.', 840, win(t, S.brief[0] + 4.5, S.brief[1], 0.6, 0.6), { size: 28, color: C.muted });
    } },
    { t0: S.who[0], t1: S.who[1], draw(ctx, t) {
      const a = inS(t, 'who'), t0 = S.who[0];
      center(ctx, 'Who brought what', 150, a, { size: 60, font: FD, weight: 500 });
      text(ctx, 'KING', 330, 290, { size: 24, weight: 700, color: C.fg, track: 3, alpha: a });
      text(ctx, 'CLAUDE', 1080, 290, { size: 24, weight: 700, color: C.fg, track: 3, alpha: a });
      list(ctx, 330, 345, [['the question', t0 + 0.6], ['judging the results', t0 + 1.2], ['setting priorities', t0 + 1.8], ['challenging the sources', t0 + 2.4], ['choosing formats and approving the design', t0 + 3.0]], t, { size: 34, dot: C.muted, maxW: 600, alpha: a, gap: 18 });
      list(ctx, 1080, 345, [['research and downloads', t0 + 3.8], ['building the map', t0 + 4.4], ['calibrating the model', t0 + 5.0], ['writing it up', t0 + 5.6], ['checking it', t0 + 6.2]], t, { size: 34, dot: C.muted, maxW: 600, alpha: a, gap: 18 });
      center(ctx, 'Claude did the building. King set the question, judged the results and decided what came next.', 820, win(t, t0 + 7.2, S.who[1], 0.6, 0.6), { size: 32, color: '#cdd3d1', maxW: 1300 });
    } },
    { t0: S.became[0], t1: S.became[1], draw(ctx, t) {
      scr(ctx, t, 'role', S.became[0], S.became[1], 0);
      side(ctx, t, ...S.became, { kicker: 'What it became', title: 'A map anyone can search', body: 'Type a job and see its tasks: which ones AI already does, which it helps with, and the year each could change. Now for sixteen countries.' });
    } },

    // 2. three ways of working
    { t0: S.record[0], t1: S.tally[0], draw(ctx, t) {
      const tt = S.record[0];
      const a = win(t, tt, S.tally[0], 0.7, 0.6);
      const foc = t >= S.alone[0] ? 'independent' : t >= S.team[0] ? 'teammate' : t >= S.tool[0] ? 'tool' : null;
      const ring = t >= S.alone[0] + 0.8 ? 9 : t >= S.team[0] + 0.8 ? 11 : t >= S.tool[0] + 0.8 ? 14 : null;
      steps(ctx, { pos: GRID, r: 17, alpha: a, focus: foc, ring, reveal: smooth(span(t, tt + 0.4, tt + 3.4)) });
      center(ctx, 'Every step is on record: 96 of them, in order.', 120, win(t, tt + 0.2, S.tool[0], 0.6, 0.5), { size: 46, font: FD, weight: 500 });
      legend(ctx, W / 2, 640, win(t, tt + 3.2, S.tool[0] + 0.4, 0.6, 0.4), { align: 'center' });
      center(ctx, 'A dot with a hole is a step King did himself.', 720, win(t, tt + 4.5, S.tool[0], 0.6, 0.4), { size: 26, color: C.muted });
      const blocks = [
        [S.tool, 'tool', 'AS A TOOL', 'King names an outcome; Claude delivers it.', '“Please pull in the wages and employment data.”', 'Wages matched for 523 occupations; employment for 43 groups.'],
        [S.team, 'teammate', 'AS TEAMMATES', 'They reason together, and the answer changes the plan.', '“What do the honest limits mean?”', 'Claude’s plain answer turned four caveats into the plan for the next two rounds.'],
        [S.alone, 'independent', 'ALONE', 'One side works on its own for a stretch.', 'Claude threw out its first timing model, which called almost every desk task doable today.', 'It rebuilt it on what people actually use AI for. King, between messages, read each report and used the page.'],
      ];
      for (const [sp, k, kick, head, ex, out] of blocks) {
        const ba = win(t, sp[0] + 0.1, sp[1], 0.6, 0.5); if (ba <= 0.001) continue;
        center(ctx, kick, 650, ba, { size: 22, color: KIND[k], weight: 700 });
        center(ctx, head, 690, ba, { size: 46, font: FD, weight: 500 });
        center(ctx, ex, 780, win(t, sp[0] + 1.6, sp[1], 0.6, 0.5), { size: 32, italic: k !== 'independent', color: C.fg, maxW: 1400 });
        center(ctx, out, k === 'independent' ? 880 : 840, win(t, sp[0] + 3.4, sp[1], 0.6, 0.5), { size: 28, color: '#cdd3d1', maxW: 1400 });
      }
    } },
    { t0: S.tally[0], t1: S.tally[1], draw(ctx, t) {
      const a = inS(t, 'tally', 0.6, 0.6), t0 = S.tally[0];
      center(ctx, '66 exchanges between King and Claude', 250, a, { size: 58, font: FD, weight: 500 });
      const g = smoother(span(t, t0 + 0.5, t0 + 2.0)), x0 = 460, w = 1000;
      ctx.save(); ctx.globalAlpha *= a;
      rrect(ctx, x0, 400, w * 31 / 66 * g, 56, 8); ctx.fillStyle = KIND.tool; ctx.fill();
      rrect(ctx, x0 + w * 31 / 66 + 6, 400, w * 35 / 66 * g, 56, 8); ctx.fillStyle = KIND.teammate; ctx.fill();
      ctx.restore();
      text(ctx, '31 as a tool', x0, 480, { size: 32, color: KIND.tool, alpha: a * g, weight: 600 });
      text(ctx, '35 as teammates', x0 + w, 480, { size: 32, color: KIND.teammate, alpha: a * g, weight: 600, align: 'right' });
      center(ctx, 'The teammate exchanges changed the work most: every one redirected it.', 620, win(t, t0 + 2.2, S.tally[1], 0.6, 0.6), { size: 34, color: '#cdd3d1' });
    } },

    // 3. use it, then ask
    { t0: S.q3[0], t1: S.q3[1], draw(ctx, t) {
      const a = inS(t, 'q3');
      const y = quote(ctx, a, 'When I search for a job, is it found by keyword matching?', 'King, after trying the search himself', 300);
      center(ctx, 'Claude’s answer: yes, and here is where that fails.', y + 40, win(t, S.q3[0] + 4.5, S.q3[1], 0.6, 0.6), { size: 36, color: C.teal });
      center(ctx, 'No abbreviations, no typos, no describing the work in your own words.', y + 110, win(t, S.q3[0] + 6.5, S.q3[1], 0.6, 0.6), { size: 30, color: '#cdd3d1' });
    } },
    { t0: S.versions[0], t1: S.versions[1], draw(ctx, t) {
      const a = inS(t, 'versions'), t0 = S.versions[0];
      center(ctx, 'So search went through three versions', 140, a, { size: 56, font: FD, weight: 500 });
      const V = [['1', 'Words in job titles', 'What it did at first.'], ['2', 'Plus 6,755 other job titles', 'Small typos forgiven, and a search inside 59,000 tasks.'], ['3', 'Plus matching by meaning', 'A small AI model inside the page, and an “Ask Claude” button.']];
      V.forEach(([n, h, s], i) => {
        const ta = t0 + 1.0 + i * 3.4, ba = a * smooth(span(t, ta, ta + 0.7));
        const x = 170 + i * 540, y = 600 - i * 120, hh = 230 + i * 120;
        ctx.save(); ctx.globalAlpha *= ba;
        rrect(ctx, x, y + (1 - ba) * 30, 500, hh, 16); ctx.fillStyle = C.panel; ctx.fill(); ctx.strokeStyle = i === 2 ? C.teal : C.line; ctx.lineWidth = 2; ctx.stroke();
        ctx.restore();
        text(ctx, n, x + 34, y + 24 + (1 - ba) * 30, { size: 64, font: FD, weight: 600, color: C.teal, alpha: ba });
        text(ctx, h, x + 34, y + 110 + (1 - ba) * 30, { size: 36, font: FD, weight: 500, alpha: ba, maxW: 440 });
        text(ctx, s, x + 34, y + 160 + (1 - ba) * 30, { size: 26, color: '#cdd3d1', alpha: ba, maxW: 440 });
      });
    } },
    { t0: S.meaning[0], t1: S.meaning[1], draw(ctx, t) {
      scr(ctx, t, 'meaning', S.meaning[0], S.meaning[1], 6, null, { x: 40, y: 250, w: 880, h: 550 }, false);
      side(ctx, t, ...S.meaning, { kicker: 'Version three', title: 'Search by meaning', body: 'Describe the work in your own words. “Look after elderly people at home” finds care work, though no job title says that.' });
    } },
    { t0: S.lesson3[0], t1: S.lesson3[1], draw(ctx, t) {
      center(ctx, 'The sharpest questions came after King had used the map himself.', 470, inS(t, 'lesson3'), { size: 52, font: FD, weight: 500, maxW: 1400 });
    } },

    // 4. question the evidence
    { t0: S.q4[0], t1: S.q4[1], draw(ctx, t) {
      const a = inS(t, 'q4');
      const y = quote(ctx, a, 'You cite reports from 2023 and August 2025. How useful are these, given how fast AI is moving?', 'King, before the third round of work', 270, { size: 56 });
      center(ctx, 'Not a request for a feature: a challenge to the evidence itself.', y + 60, win(t, S.q4[0] + 5, S.q4[1], 0.6, 0.6), { size: 36, color: C.teal });
    } },
    { t0: S.changes[0], t1: S.changes[1], draw(ctx, t) {
      const t0 = S.changes[0];
      scr(ctx, t, 'overview', t0, S.changes[1], 0);
      const a = win(t, t0 + 0.3, S.changes[1], 0.6, 0.5);
      panel(ctx, PX, PY, PW, { kicker: 'The answer changed three things', title: '' }, a);
      list(ctx, PX, PY + 70, [['Old ratings became flags, not verdicts.', t0 + 1.2], ['The data on firms using AI became a live feed, dated on the page.', t0 + 4.4], ['Every one of the 18,796 tasks was re-rated, with its own test of how reliable that rating is.', t0 + 7.6]], t, { size: 31, maxW: 500, alpha: a, gap: 30 });
    } },
    { t0: S.trust[0], t1: S.trust[1], draw(ctx, t) {
      const a = inS(t, 'trust');
      const y = quote(ctx, a, 'Go ahead … in the sequence you assess is the most robust and rigorous.', 'King, the next message', 300);
      center(ctx, 'Trust, handed over on the strength of the last answer.', y + 60, win(t, S.trust[0] + 4.5, S.trust[1], 0.6, 0.6), { size: 36, color: C.teal });
    } },

    // 5. check everything
    { t0: S.scripts[0], t1: S.scripts[1], draw(ctx, t) {
      scr(ctx, t, 'c_qa', S.scripts[0], S.scripts[1], 0);
      side(ctx, t, ...S.scripts, { kicker: 'Checking', title: 'By script, not by eye', body: 'Every number on the page compared with its source table. Every view opened in a browser, on phone and desktop, light and dark. Accessibility rules run on every view.' });
    } },
    { t0: S.found[0], t1: S.found[1], draw(ctx, t) {
      const a = inS(t, 'found'), t0 = S.found[0];
      center(ctx, 'The first round found nine groups of problems. Some of them:', 160, a, { size: 50, font: FD, weight: 500, maxW: 1500 });
      list(ctx, 420, 320, [['Years printed like money: “2,032”', t0 + 1.0], ['One headline range worked out two different ways', t0 + 2.6], ['Seven accessibility failures, now none', t0 + 4.2], ['Garbled characters on some web servers', t0 + 5.8]], t, { size: 38, alpha: a, gap: 30, maxW: 1100, dot: C.orange });
      center(ctx, 'All nine were fixed before the page went out again.', 830, win(t, t0 + 7.6, S.found[1], 0.6, 0.6), { size: 34, color: C.teal });
    } },
    { t0: S.rounds[0], t1: S.rounds[1], draw(ctx, t) {
      const a = inS(t, 'rounds'), t0 = S.rounds[0];
      center(ctx, 'Twenty-one rounds of checks so far', 250, a, { size: 60, font: FD, weight: 500 });
      for (let i = 0; i < 21; i++) {
        const u = smooth(span(t, t0 + 0.6 + i * 0.12, t0 + 0.9 + i * 0.12)) * a; if (u <= 0.001) continue;
        const x = W / 2 - 10 * 70 + i * 70, y = 470;
        ctx.save(); ctx.globalAlpha *= u; ctx.beginPath(); ctx.arc(x, y, 24, 0, 7); ctx.fillStyle = C.tealDeep; ctx.fill();
        ctx.beginPath(); ctx.moveTo(x - 10, y + 1); ctx.lineTo(x - 3, y + 9); ctx.lineTo(x + 11, y - 8); ctx.strokeStyle = C.fg; ctx.lineWidth = 4; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.stroke(); ctx.restore();
      }
      center(ctx, 'One for every round of new work: each new country, each new view.', 600, win(t, t0 + 3.6, S.rounds[1], 0.6, 0.6), { size: 34, color: '#cdd3d1' });
    } },

    // 6. grow, and keep the record
    { t0: S.grow[0], t1: S.grow[1], draw(ctx, t) {
      scr(ctx, t, 'countries', S.grow[0], S.grow[1], 0);
      side(ctx, t, ...S.grow, { kicker: 'Growing', title: 'One country, then sixteen', body: 'And thirty-two more in Europe, each on its own official statistics. When a website refused a download, the record says so, and what was used instead.' });
    } },
    { t0: S.keep[0], t1: S.keep[1], draw(ctx, t) {
      scr(ctx, t, 'c_king', S.keep[0], S.keep[1], 0);
      side(ctx, t, ...S.keep, { kicker: 'Keeping the record', title: 'Who did what, step by step', body: 'Every step written down: who carried it out, which message led to it, and which of the three ways it was. This film was made from that record.' });
    } },

    // what to take away, and a question
    { t0: S.lessons[0], t1: S.lessons[1], draw(ctx, t) {
      const a = inS(t, 'lessons'), t0 = S.lessons[0];
      center(ctx, 'Five things to take away', 130, a, { size: 62, font: FD, weight: 500 });
      list(ctx, 470, 290, [
        ['Bring the question, and the judgment.', t0 + 1.2],
        ['Use the result yourself, then ask.', t0 + 4.4],
        ['Question the evidence, not just the features.', t0 + 7.6],
        ['Ask for checks you can verify.', t0 + 10.8],
        ['Keep a record of who did what.', t0 + 14.0],
      ], t, { size: 44, font: FD, alpha: a, gap: 38, num: true, maxW: 1100 });
    } },
    { t0: S.discuss[0], t1: S.discuss[1], draw(ctx, t) {
      const a = inS(t, 'discuss');
      center(ctx, 'TO DISCUSS', 330, a, { size: 24, color: C.orange, weight: 700 });
      center(ctx, 'Which of King’s questions changed the work most, and why?', 380, a, { size: 58, font: FD, weight: 500, maxW: 1400 });
      center(ctx, 'What would you have asked?', 560, win(t, S.discuss[0] + 2.4, S.discuss[1], 0.6, 0.6), { size: 40, color: C.teal });
    } },
  ],
};
