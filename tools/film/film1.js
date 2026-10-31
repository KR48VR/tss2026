// Film 1: "The Work and AI Map" — what it is, who it helps, why it can be trusted, and the ask (100 s).
// Timed in bars of 2.5 s (96 beats a minute, four beats a bar) so every scene change lands on a bar line of the score.
'use strict';
const BAR = 2.5;
const b = n => n * BAR;
const SR = { x: 70, y: 172, w: 1180, h: 737.5 };   // the screen
const PX = 1320, PY = 236, PW = 540;               // the panel beside it
const Z1 = 7.2;                                    // zoom on the featured job's tasks

// scene boundaries (seconds)
const T = {
  poster: [0, b(2)], hook: [b(2), b(5)], tasks: [b(5), b(8)], dive: [b(8), b(10)], job: [b(10), b(17)],
  econ: [b(17), b(22)], sectors: [b(22), b(26)], countries: [b(26), b(30)], trust: [b(30), b(34)], ask: [b(34), b(38)], end: [b(38), b(40)],
};
const DUR = b(40);   // 100 s
const YR0 = b(18), YR1 = b(20) - 0.3, STAT = b(20);   // the years run over two bars, then the headline lands on a downbeat
const DROP = 25;   // the featured job's row sits this far (field units) below the centre while it is out of focus, clear of the panel text

// ---------------------------------------------------------------------------------------------- the field over the whole film
function fieldState(t) {
  const fc = F.fc;
  let s = { cx: W / 2, cy: H / 2, zoom: 1, year: 2036.5, lit: 1, base: 0.32, labels: 0, feat: 0, alpha: 1 };
  if (t < T.hook[0] + 0.0) { // poster, then the lights go out
    const u = smooth(span(t, b(1.4), b(2)));
    s.lit = 0.62 * (1 - u); s.base = lerp(0.3, 0.16, u);
  } else if (t < T.tasks[0]) { s.lit = 0; s.base = 0.14; }
  else if (t < T.dive[0]) {
    s.lit = 0; s.base = 0.55; s.wave = span(t, T.tasks[0], T.tasks[0] + 3.2);
    if (s.wave >= 1) delete s.wave;
    s.labels = win(t, T.tasks[0] + 3.0, T.dive[0] + 0.6, 0.8, 0.8);
  } else if (t < T.job[0] + 0.2) {   // dive into one job's tasks
    const u = smoother(span(t, T.dive[0] + 0.5, T.dive[1] - 0.35));
    s.lit = 0; s.base = lerp(0.55, 0.3, u); s.feat = smooth(span(t, T.dive[0] + 0.1, T.dive[0] + 0.9));
    s.cx = lerp(W / 2, fc[0], u); s.cy = lerp(H / 2, fc[1], u); s.zoom = loglerp(1, Z1, u);
    s.labels = win(t, T.tasks[0] + 3.0, T.dive[0] + 0.6, 0.8, 0.8);
    s.alpha = 1 - 0.65 * smooth(span(t, T.dive[1] - 0.4, T.job[0] + 0.2));
    if (t > T.dive[1] - 0.35) { const v = smooth(span(t, T.dive[1] - 0.35, T.job[0] + 0.2)); s.zoom = Z1 * (1 + 0.5 * v); s.blur = 10 * v; s.cy = fc[1] - DROP * v; }
  } else if (t < T.econ[0]) {   // behind the screen: the job's tasks, out of focus
    s.lit = 0; s.base = 0.3; s.feat = 1; s.cx = fc[0] + (t - T.job[0]) * 1.2; s.cy = fc[1] - DROP; s.zoom = Z1 * 1.5; s.alpha = 0.35; s.blur = 10;
  } else if (t < T.econ[0] + 2.0) {   // back out to the whole economy
    const u = smoother(span(t, T.econ[0] + 0.15, T.econ[0] + 2.0));
    s.lit = 0; s.base = lerp(0.3, 0.45, u); s.feat = 1 - u; s.cx = lerp(fc[0] + (T.econ[0] - T.job[0]) * 1.2, W / 2, u); s.cy = lerp(fc[1] - DROP, H / 2, u);
    s.zoom = loglerp(Z1 * 1.5, 1, u); const v = smooth(span(t, T.econ[0], T.econ[0] + 0.8)); s.alpha = lerp(0.35, 1, v); s.blur = 10 * (1 - v);
  } else if (t < T.sectors[0]) {   // the years run: 2026 to 2035
    s.base = 0.42; s.lit = 1; s.year = lerp(2026.0, 2035.99, smoother(span(t, YR0, YR1)));
    s.alpha = 1 - 0.75 * smooth(span(t, T.sectors[0] - 0.6, T.sectors[0]));
  } else if (t < T.ask[0]) { s.alpha = 0.25; s.base = 0.35; s.zoom = 1.04 + (t - T.sectors[0]) * 0.004; }
  else if (t < T.end[0]) { s.alpha = lerp(0.25, 0.5, smooth(span(t, T.ask[0], T.ask[0] + 1.0))); s.base = 0.3; s.lit = 0.8; s.zoom = 1.04 + (T.ask[0] - T.sectors[0]) * 0.004 * (1 - smooth(span(t, T.ask[0], T.end[0]))); }
  else { const u = smooth(span(t, T.end[0], T.end[0] + 1.0)); s.alpha = lerp(0.5, 1, u); s.lit = lerp(0.8, 0.62, u); s.base = lerp(0.3, 0.3, u); }
  return s;
}

// ---------------------------------------------------------------------------------------------- title card (first and last frame)
function poster(ctx, a) {
  if (a <= 0) return;
  ctx.save(); ctx.globalAlpha = a;
  const g = ctx.createLinearGradient(0, 300, 0, 780); g.addColorStop(0, 'rgba(15,21,22,0)'); g.addColorStop(0.25, 'rgba(15,21,22,0.86)'); g.addColorStop(0.75, 'rgba(15,21,22,0.86)'); g.addColorStop(1, 'rgba(15,21,22,0)');
  ctx.fillStyle = g; ctx.fillRect(0, 300, W, 480);
  text(ctx, 'The Work and AI Map', W / 2, 382, { size: 118, font: FD, weight: 500, align: 'center' });
  text(ctx, 'Which parts of your job could AI do, and when?', W / 2, 548, { size: 40, color: '#cdd3d1', align: 'center' });
  text(ctx, '18,796 tasks  ·  16 countries  ·  the evidence from four AI labs', W / 2, 632, { size: 25, color: C.teal, align: 'center', weight: 600, track: 1 });
  text(ctx, 'Built by King and Claude  ·  2026', W / 2, 1008, { size: 22, color: C.faint, align: 'center' });
  ctx.restore();
}

// a line of large centred text in the band above or below the field
function band(ctx, str, y, a, o = {}) { text(ctx, str, o.x ?? W / 2, y, { size: o.size || 46, font: o.font || FD, weight: 500, align: o.align || 'center', alpha: a, color: o.color || C.fg, maxW: o.maxW }); }

// ---------------------------------------------------------------------------------------------- screen segments with tab flips
// each: [name, t0, t1, inpoint, speed]
const JOB = [['search', T.job[0], b(12.5), 0, 0.82], ['role', b(12.5), b(14.75), 14, 1.2], ['scenario', b(14.75), T.job[1], 0, 0.9]];
const RUN = [['sectors', T.sectors[0], b(23.25), 6, 1.05], ['skills', b(23.25), b(24.5), 6, 1.05], ['leaders', b(24.5), T.sectors[1], 0, 0.88],
  ['countries', T.countries[0], b(28.25), 0, 1.3], ['europe', b(28.25), T.countries[1], 10, 1.3], ['news', T.trust[0], T.trust[1], 0, 0.64]];

function screens(ctx, t, segs, enter, exit) {
  const FL = 0.5;   // tab flip length
  const aIn = enter === 'rise' ? smooth(span(t, segs[0][1], segs[0][1] + 0.7)) : 1;
  const aOut = exit ? 1 - smooth(span(t, segs[segs.length - 1][2] - 0.6, segs[segs.length - 1][2])) : 1;
  const lift = (1 - aIn) * 40 + (1 - aOut) * -30;
  for (let i = segs.length - 1; i >= 0; i--) {
    const [name, t0, t1, inp, sp] = segs[i];
    if (t < t0 - FL || t >= t1 + FL) continue;
    const fi = play(t, t0, inp, sp);
    // incoming fades in over the flip; outgoing slides up and fades as the next one arrives
    let a = 1, dy = 0, sc = 1;
    if (i > 0 && t < t0) continue;
    if (i > 0) { const u = smooth(span(t, t0, t0 + FL)); a = u; sc = lerp(0.975, 1, u); }
    if (i < segs.length - 1 && t >= t1) { const u = smooth(span(t, t1, t1 + FL)); a = 1 - u; dy = -50 * u; sc = lerp(1, 1.02, u); }
    const r = { x: SR.x + SR.w * (1 - sc) / 2, y: SR.y + SR.h * (1 - sc) / 2 + dy + lift, w: SR.w * sc, h: SR.h * sc };
    screen(ctx, name, fi, r, { alpha: a * aIn * aOut });
    // stacked cards behind hint at the other views
  }
}
function phase(ctx, t, items, base) {   // items: [t0, t1, panelSpec]; cross-fade between them
  for (const [t0, t1, p] of items) { const a = win(t, t0, t1, 0.55, 0.45); if (a > 0) panel(ctx, PX, PY, PW, { ...base, ...p }, a, (1 - smooth(span(t, t0, t0 + 0.55))) * 18); }
}

// ---------------------------------------------------------------------------------------------- the film
const FILM = {
  duration: DUR, fps: 24, field: true,
  recs: ['search', 'role', 'scenario', 'sectors', 'skills', 'leaders', 'countries', 'europe', 'news'],
  motion(t) {   // speed of the camera, for motion blur on the fast zooms
    const dz = (a, c) => Math.abs(Math.log(fieldState(Math.min(DUR - 1e-3, a + 1 / 48)).zoom) - Math.log(fieldState(a).zoom)) * c;
    if ((t > T.dive[0] && t < T.job[0] + 0.3) || (t > T.econ[0] && t < T.econ[0] + 2.1)) return clamp(dz(t, 1) * 14);
    return 0;
  },
  scenes: [
    { t0: 0, t1: DUR, draw(ctx, t) { field(ctx, fieldState(t)); } },
    // poster, first and last
    { t0: 0, t1: T.hook[0], draw(ctx, t) { poster(ctx, 1 - smooth(span(t, b(1.2), b(1.9)))); } },
    { t0: T.end[0], t1: DUR, draw(ctx, t) { poster(ctx, smooth(span(t, T.end[0] + 0.2, T.end[0] + 1.2))); } },
    // hook
    { t0: T.hook[0], t1: T.hook[1], draw(ctx, t) {
      band(ctx, 'Will AI take my job?', 400, win(t, T.hook[0] + 0.1, T.hook[1] - 0.1, 0.6, 0.6), { size: 96 });
      band(ctx, 'Everyone is asking.', 560, win(t, b(3), T.hook[1] - 0.1, 0.6, 0.6), { size: 40, font: FB, color: '#cdd3d1' });
      band(ctx, 'Most answers are headlines and guesses.', 620, win(t, b(3.5), T.hook[1] - 0.1, 0.6, 0.6), { size: 40, font: FB, color: '#cdd3d1' });
    } },
    // the tasks
    { t0: T.tasks[0], t1: T.dive[0] + 0.8, draw(ctx, t) {
      band(ctx, 'So we mapped the work itself: 18,796 tasks that people do at work.', 56, win(t, T.tasks[0] + 0.3, T.tasks[1] - 0.2, 0.6, 0.5), { size: 44 });
      band(ctx, 'From the US O*NET database, linked to each country’s own jobs, pay and AI evidence.', 990, win(t, T.tasks[0] + 1.4, T.tasks[1] - 0.2, 0.6, 0.5), { size: 28, font: FB, color: C.muted });
    } },
    { t0: T.dive[0], t1: T.job[0] + 0.5, draw(ctx, t) {
      band(ctx, 'Take one job.', 56, win(t, T.dive[0] + 0.15, T.dive[1] - 0.3, 0.5, 0.4), { size: 52 });
      const a = win(t, T.dive[0] + 2.2, T.dive[1] + 0.2, 0.5, 0.5);
      band(ctx, 'Accountants and Auditors: 29 tasks', 860, a, { size: 34, font: FB, color: C.fg });
      band(ctx, 'Not a forecast of job loss: a map of which tasks AI could do, and when.', 990, win(t, T.dive[0] + 0.6, T.dive[1] - 0.2, 0.6, 0.5), { size: 28, font: FB, color: C.muted });
    } },
    // the job, on the real map
    { t0: T.job[0] - 0.4, t1: T.econ[0] + 0.7, draw(ctx, t) {
      const grow = smooth(span(t, T.job[0] - 0.4, T.job[0] + 0.5)), shrink = smooth(span(t, T.econ[0], T.econ[0] + 0.7));
      ctx.save(); const sc = lerp(0.82, 1, grow) * lerp(1, 0.86, shrink); ctx.globalAlpha = grow * (1 - shrink);
      ctx.translate(SR.x + SR.w / 2, SR.y + SR.h / 2); ctx.scale(sc, sc); ctx.translate(-(SR.x + SR.w / 2), -(SR.y + SR.h / 2));
      screens(ctx, Math.min(t, T.econ[0] - 0.01), JOB, null, null);
      ctx.restore();
      ctx.save(); ctx.globalAlpha = 1 - shrink;
      phase(ctx, t, [
        [T.job[0] + 0.2, b(12.5) + 0.2, { title: 'Search your job', body: 'Type a job title, or describe what you do. The map finds the occupation and the Skills Framework roles inside it.' }],
        [b(12.5) + 0.2, b(14.75) + 0.2, { title: 'See every task', body: 'Which tasks AI already does, which it only helps with, which no one has seen it do yet, and the year it could do each one.' }],
        [b(14.75) + 0.2, T.econ[0] + 0.5, { title: 'Read the range', body: 'Every year comes in a fast, central and slow case. The headline counts the work AI itself would be doing, not just touching.' }],
      ], { kicker: 'For anyone wondering about their job' });
      ctx.restore();
    } },
    // the whole economy
    { t0: T.econ[0], t1: T.sectors[0], draw(ctx, t) {
      band(ctx, 'Now every task in the economy.', 56, win(t, T.econ[0] + 0.5, T.econ[0] + 2.4, 0.5, 0.4), { size: 44 });
      const st = fieldState(t);
      const ya = win(t, YR0 - 0.2, T.sectors[0] + 0.1, 0.4, 0.5);
      text(ctx, String(Math.floor(st.year)), 120, 30, { size: 92, font: FD, weight: 600, color: C.teal, alpha: ya });
      text(ctx, 'Each dot lights when AI could be doing that task in Singapore (central case). Brighter: AI does more of it.', 120 + 250, 70, { size: 25, color: C.muted, alpha: ya * (1 - 0.6 * smooth(span(t, STAT, STAT + 0.6))), maxW: 1330 });
      const sa = win(t, STAT, T.sectors[0] + 0.1, 0.6, 0.5);
      if (sa > 0) {
        wash(ctx, 0.78 * sa, 0, 300, W, 470);
        band(ctx, 'By 2035 AI could be doing about 42% of Singapore’s paid work', 360, sa, { size: 54 });
        band(ctx, 'A quarter of it by mid-2030 in the central case  ·  fast case late 2027  ·  slow case early 2035', 470, sa, { size: 29, font: FB, color: '#cdd3d1' });
        band(ctx, 'and involved in 87% of it, doing some or helping with the rest', 530, sa, { size: 29, font: FB, color: C.muted });
        band(ctx, 'Central case on the map’s measured pace; years shown to the season, as on the map.', 690, sa, { size: 22, font: FB, color: C.faint });
      }
    } },
    // sectors, skills, leaders, countries, Europe, the news: one run of screens, flipping like tabs
    { t0: T.sectors[0], t1: T.trust[1], draw(ctx, t) {
      screens(ctx, t, RUN, 'rise', 'drop');
      phase(ctx, t, [
        [T.sectors[0] + 0.3, b(24.5) + 0.2, { kicker: 'For employers and planners', title: 'Every sector, every skill', body: 'The Skills Framework’s sectors and roles, the skills most exposed and least, and the year each could change.' }],
        [b(24.5) + 0.2, T.sectors[1] + 0.2, { kicker: 'For employers and planners', title: 'A summary for leaders', body: 'The shape of the answer in five minutes: three scenarios side by side, live from the data.' }],
        [T.countries[0] + 0.2, b(28.25) + 0.2, { kicker: 'For governments', title: 'Sixteen countries, one model', body: 'Each built on official statistics: who does which jobs, what they are paid, and how fast firms take up AI.' }],
        [b(28.25) + 0.2, T.countries[1] + 0.2, { kicker: 'For governments', title: 'And all of Europe', body: 'Thirty-two more countries from Eurostat’s survey, ranked on the same model. Compare any two side by side.' }],
        [T.trust[0] + 0.2, b(32), { kicker: 'Is it hype? Check it', title: 'Evidence, not headlines', body: 'Built on AI usage data from Anthropic, Microsoft, OpenAI and Google, and on each country’s official statistics.' }],
        [b(32), T.trust[1] - 0.2, { kicker: 'Is it hype? Check it', title: 'Check any claim', body: 'Every claim sits on a five-rung ladder. Type in a headline’s number and see what such claims have meant so far.' }],
      ], {});
    } },
    // the ask
    { t0: T.ask[0], t1: T.end[0] + 0.6, draw(ctx, t) {
      const out = T.end[0] + 0.4;
      wash(ctx, 0.62 * win(t, T.ask[0], out, 0.6, 0.6));
      band(ctx, 'AI moves fast. The map has to keep up.', 250, win(t, T.ask[0] + 0.3, out, 0.6, 0.5), { size: 64 });
      band(ctx, 'Help take it further:', 410, win(t, b(34.75), out, 0.5, 0.5), { size: 36, font: FB, color: C.teal });
      const items = ['fresh AI evidence, folded in as each new release lands', 'more countries, each on its own statistics', 'testing with workers, employers and teachers'];
      items.forEach((s, i) => band(ctx, s, 480 + i * 64, win(t, b(35 + 0.5 * i), out, 0.5, 0.5), { size: 36, font: FB, color: C.fg }));
      band(ctx, 'Search your job. Share the map. Help fund what comes next.', 780, win(t, b(36.5), out, 0.6, 0.5), { size: 44 });
    } },
    { t0: 0, t1: DUR, draw(ctx, t) { vignette(ctx, 0.5); } },
  ],
};
