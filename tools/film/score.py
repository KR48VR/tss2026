"""Original scores for the two Work and AI Map films, synthesised in numpy (no samples, no rights to clear).

  python3 tools/film/score.py film1 OUT.wav
  python3 tools/film/score.py film2 OUT.wav

Everything is written in film time on a fixed tempo grid that the film scripts share: film 1 at 96 beats a
minute (a bar of 2.5 s), film 2 at 80 (a bar of 3 s). Scene changes in the films sit on bar lines, so every
accent here lands on a cut. Loudness is set later, in the encode (two-pass loudnorm).
"""
import sys
import numpy as np
from scipy.signal import fftconvolve, butter, sosfilt
from scipy.io import wavfile

SR = 48000
rng = np.random.default_rng(7)


def hz(m): return 440.0 * 2 ** ((m - 69) / 12)


class Track:
    def __init__(self, dur):
        self.n = int(dur * SR) + SR * 4
        self.dry = np.zeros((2, self.n)); self.wet = np.zeros((2, self.n))

    def add(self, t, sig, gain=1.0, pan=0.0, rev=0.25):
        i = int(t * SR); sig = np.asarray(sig) * gain
        if sig.ndim == 1: sig = np.vstack([sig * np.sqrt(0.5 * (1 - pan)), sig * np.sqrt(0.5 * (1 + pan))])
        if i < 0: sig = sig[:, -i:]; i = 0   # a note that starts before the film: keep only its audible part
        j = min(self.n, i + sig.shape[1]); sig = sig[:, :j - i]
        self.dry[:, i:j] += sig * (1 - rev * 0.5); self.wet[:, i:j] += sig * rev


def env(n, a, d, s, r, hold):
    """ADSR in seconds; hold = time before release starts."""
    t = np.arange(n) / SR
    e = np.where(t < a, t / max(a, 1e-4), 1.0)
    e = np.where((t >= a) & (t < a + d), 1 - (1 - s) * (t - a) / max(d, 1e-4), e)
    e = np.where((t >= a + d) & (t < hold), s, e)
    e = np.where(t >= hold, s * np.exp(-(t - hold) / max(r, 1e-4) * 3), e)
    if hold < a + d: e = np.where(t >= hold, e[min(n - 1, int(hold * SR))] * np.exp(-(t - hold) / max(r, 1e-4) * 3), e)
    return e


def pad(m, dur, bright=0.4, rel=1.6, att=0.7, det=5.0):
    n = int((dur + rel * 1.2) * SR); t = np.arange(n) / SR; out = np.zeros(n)
    tilt = 2.6 - 1.6 * bright
    for d in (-det, det * 0.6, det):
        f = hz(m) * 2 ** (d / 1200); ph = rng.uniform(0, 6.28)
        for k in range(1, 9):
            if f * k > 9000: break
            out += np.sin(2 * np.pi * f * k * t + ph * k) / k ** tilt
    vib = 1 + 0.08 * np.sin(2 * np.pi * 0.23 * t + rng.uniform(0, 6))
    return out * env(n, att, 0.5, 0.85, rel, dur) * vib / 3


def pluck(m, dur=0.9, bright=0.6, decay=0.45):
    n = int((dur + 0.6) * SR); t = np.arange(n) / SR; out = np.zeros(n); f = hz(m)
    for k in range(1, 10):
        if f * k > 12000: break
        out += np.sin(2 * np.pi * f * k * t) * np.exp(-t * (1 / decay) * (1 + 0.55 * k * (1.2 - bright))) / k ** (1.9 - bright)
    a = np.minimum(1, t / 0.004)
    return out * a * env(n, 0.003, 0.05, 1.0, 0.25, dur)


def piano(m, dur=1.5, vel=1.0):
    n = int((dur + 2.0) * SR); t = np.arange(n) / SR; out = np.zeros(n); f = hz(m)
    for k, (amp, dk) in enumerate([(1, 0.9), (0.5, 1.4), (0.28, 2.0), (0.16, 2.6), (0.1, 3.2), (0.05, 4.0)], 1):
        fk = f * k * (1 + 0.0004 * k * k)
        out += amp * np.sin(2 * np.pi * fk * t) * np.exp(-t * dk / (1.6 + 1200 / (f + 200)))
    a = np.minimum(1, t / 0.006)
    return out * a * vel * env(n, 0.006, 0.05, 1.0, 0.5, dur + 0.3)


def bell(m, dur=3.0):
    n = int(dur * SR); t = np.arange(n) / SR; f = hz(m); out = np.zeros(n)
    for r, amp, dk in [(1, 1, 0.9), (2.0, 0.45, 1.5), (2.76, 0.35, 2.2), (5.4, 0.18, 3.5), (8.93, 0.08, 5)]:
        out += amp * np.sin(2 * np.pi * f * r * t) * np.exp(-t * dk)
    return out * np.minimum(1, t / 0.002) * 0.6


def bass(m, dur, glide=0.0):
    n = int((dur + 0.5) * SR); t = np.arange(n) / SR; f = hz(m)
    ph = 2 * np.pi * f * t
    out = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)
    return out * env(n, 0.012, 0.3, 0.7, 0.25, dur)


def kick(vel=1.0):
    n = int(0.5 * SR); t = np.arange(n) / SR
    f = 46 + 90 * np.exp(-t * 28); ph = 2 * np.pi * np.cumsum(f) / SR
    return vel * (np.sin(ph) * np.exp(-t * 7) + 0.25 * np.exp(-t * 200) * rng.standard_normal(n) * 0.3)


_hp = butter(2, 6500, 'hp', fs=SR, output='sos'); _bp = butter(2, [1800, 7000], 'bp', fs=SR, output='sos')


def hat(vel=1.0, open_=False):
    n = int((0.25 if open_ else 0.08) * SR); t = np.arange(n) / SR
    return vel * sosfilt(_hp, rng.standard_normal(n)) * np.exp(-t * (14 if open_ else 55)) * 0.5


def shaker(vel=1.0):
    n = int(0.12 * SR); t = np.arange(n) / SR
    e = np.minimum(1, t / 0.03) * np.exp(-t * 30)
    return vel * sosfilt(_bp, rng.standard_normal(n)) * e * 0.5


def noise_sweep(dur, f0, f1, rise=True):
    """Filtered noise whose band moves from f0 to f1, rising in level (a riser) or falling (a whoosh)."""
    n = int(dur * SR); x = rng.standard_normal(n); out = np.zeros(n); blk = 2048
    for i in range(0, n, blk):
        u = i / n; fc = f0 * (f1 / f0) ** u
        sos = butter(2, [max(40, fc * 0.6), min(SR / 2 - 100, fc * 1.6)], 'bp', fs=SR, output='sos')
        out[i:i + blk] = sosfilt(sos, x[i:i + blk])
    t = np.arange(n) / n
    e = (t ** 2.2) if rise else np.exp(-t * 4) * np.minimum(1, t * 30)
    return out * e


def impact(vel=1.0):
    n = int(2.5 * SR); t = np.arange(n) / SR
    f = 38 + 50 * np.exp(-t * 9); ph = 2 * np.pi * np.cumsum(f) / SR
    lp = butter(2, 900, 'lp', fs=SR, output='sos')
    return vel * (np.sin(ph) * np.exp(-t * 2.2) * 0.9 + sosfilt(lp, rng.standard_normal(n)) * np.exp(-t * 6) * 0.6)


def reverb_ir(sec=2.4, seed=3):
    r = np.random.default_rng(seed); n = int(sec * SR); t = np.arange(n) / SR
    lp = butter(1, 5000, 'lp', fs=SR, output='sos')
    ir = np.vstack([sosfilt(lp, r.standard_normal(n)), sosfilt(lp, r.standard_normal(n))]) * np.exp(-t * 6.9 / sec)
    ir[:, :int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))
    return ir / np.sqrt((ir ** 2).sum(axis=1, keepdims=True))


def master(tr, dur, ir_sec=2.4, fade_in=0.0, fade_out=1.5):
    ir = reverb_ir(ir_sec)
    wet = np.vstack([fftconvolve(tr.wet[0], ir[0])[:tr.n], fftconvolve(tr.wet[1], ir[1])[:tr.n]])
    mix = tr.dry + wet * 0.9
    n = int(dur * SR); mix = mix[:, :n]
    if fade_in > 0: mix[:, :int(fade_in * SR)] *= np.linspace(0, 1, int(fade_in * SR)) ** 2
    k = int(fade_out * SR); mix[:, n - k:] *= np.linspace(1, 0, k) ** 2
    mix[:, n - int(0.12 * SR):] = 0.0                       # the last frames are silent
    hp = butter(2, 36, "hp", fs=SR, output="sos"); mix = sosfilt(hp, mix, axis=1)
    mix = np.tanh(mix / (np.abs(mix).max() + 1e-9) * 1.4) / np.tanh(1.4)   # gentle glue
    return (mix * 0.89).astype(np.float32)


# ------------------------------------------------------------------------------------------------- harmony (D major)
CH = {   # pad voicing, bass root, arpeggio tones
    'D': ([62, 66, 69, 73, 76], 38, [74, 78, 81, 76, 85]),
    'A': ([61, 64, 69, 71, 76], 45, [73, 76, 81, 71, 83]),
    'Bm': ([62, 66, 69, 71, 74], 47, [74, 78, 81, 71, 83]),
    'G': ([59, 62, 66, 69, 74], 43, [74, 79, 78, 71, 83]),
    'Em': ([59, 64, 67, 71, 74], 40, [76, 79, 83, 71, 86]),
    'F#m': ([61, 64, 69, 73, 76], 42, [73, 78, 81, 76, 85]),
    'Dsus': ([62, 67, 69, 74, 76], 38, [74, 79, 81, 76, 86]),
}
PROG = ['D', 'A', 'Bm', 'G']


def film1(out):
    BAR, BEAT = 2.5, 0.625; NB = 40; dur = NB * BAR
    bt = lambda bar, beat=0: (bar * 4 + beat) * BEAT
    tr = Track(dur)

    # chord per bar
    chords = {}
    chords.update({0: 'D', 1: 'D', 2: 'Bm', 3: 'G', 4: 'A'})
    for i, bar in enumerate(range(5, 10)): chords[bar] = PROG[i % 4]
    for i, bar in enumerate(range(10, 17)): chords[bar] = PROG[i % 4]
    chords.update({17: 'G', 18: 'Em', 19: 'A', 20: 'D', 21: 'D'})
    for i, bar in enumerate(range(22, 34)): chords[bar] = PROG[i % 4]
    chords.update({30: 'Bm', 31: 'G', 32: 'Em', 33: 'A'})
    chords.update({34: 'G', 35: 'A', 36: 'Bm', 37: 'A', 38: 'D', 39: 'D'})

    # pads: every bar, brighter in the busy sections
    for bar in range(NB):
        c = chords[bar]; notes, root, _ = CH[c]
        bright = 0.25 if bar < 5 else 0.45 if bar < 22 else 0.55
        if bar in (20, 21): bright = 0.7
        if bar >= 38: bright = 0.5
        g = 0.15 if bar < 2 else 0.12
        if bar in (0,):
            for m in notes: tr.add(0.0, pad(m, BAR * 2, bright, rel=1.4, att=1.8), g, pan=rng.uniform(-.4, .4), rev=0.5)
            continue
        if bar == 1: continue
        if bar >= 38:
            if bar == 38:
                for m in notes + [50]: tr.add(bt(38), pad(m, BAR * 1.3, bright, rel=2.0, att=0.9), 0.14, pan=rng.uniform(-.4, .4), rev=0.55)
            continue
        if 17 == bar: g = 0.09
        for m in notes: tr.add(bt(bar) - 0.05, pad(m, BAR, bright, rel=1.4, att=0.5), g, pan=rng.uniform(-.4, .4), rev=0.45)

    # poster: a soft falling figure as the lights go out
    for i, m in enumerate([81, 78, 76, 74]): tr.add(bt(1, 1 + i * 0.75), pluck(m, 0.8, 0.4, 0.6), 0.10, pan=-0.3 + i * 0.2, rev=0.5)
    tr.add(bt(0), bell(86, 4.0), 0.05, pan=0.2, rev=0.6)

    # hook: piano, a question left open
    for bar, beat, m, d, v in [(2, 0, 71, 2.0, 0.9), (2, 0, 66, 2.0, 0.5), (2, 2, 74, 1.2, 0.7), (3, 0, 74, 1.5, 0.8), (3, 2, 71, 1.2, 0.6),
                                (3, 3, 69, 0.6, 0.5), (4, 0, 69, 2.0, 0.8), (4, 0, 64, 2.0, 0.45), (4, 2, 76, 2.4, 0.7)]:
        tr.add(bt(bar, beat), piano(m, d, v), 0.22, pan=-0.1, rev=0.45)
    for bar in (2, 3, 4): tr.add(bt(bar), bass(CH[chords[bar]][1], BAR * 0.95), 0.10, rev=0.1)

    # arpeggios (eighths) from the tasks onwards
    PAT = [0, 1, 2, 3, 2, 1, 4, 2]
    def arps(b0, b1, gain, bright=0.55, sixteenth=False, ramp=None):
        for bar in range(b0, b1):
            tones = CH[chords[bar]][2]; steps = 16 if sixteenth else 8
            for s in range(steps):
                if sixteenth and bar == b1 - 1 and s >= 14: continue
                m = tones[PAT[s % 8]] + (12 if sixteenth and s % 4 == 3 else 0)
                g = gain * (ramp(bar + s / steps) if ramp else 1) * (1.0 if s % 2 == 0 else 0.78)
                tr.add(bt(bar, s * 4 / steps), pluck(m, 0.35, bright, 0.32), g, pan=0.35 * np.sin(s * 0.8 + bar), rev=0.35)
    arps(5, 8, 0.07, 0.45, ramp=lambda x: min(1, 0.4 + 0.6 * (x - 5) / 2))
    arps(8, 10, 0.08, 0.55)
    arps(10, 17, 0.075, 0.6)
    arps(18, 20, 0.07, 0.6, sixteenth=True, ramp=lambda x: 0.5 + 0.5 * (x - 18) / 2)
    arps(22, 30, 0.07, 0.62)
    arps(30, 34, 0.06, 0.5)

    # drums and bass in the grooves
    def groove(b0, b1, kick_g=0.42, hat_g=0.16, full=True, half=False):
        for bar in range(b0, b1):
            root = CH[chords[bar]][1]
            for beat in range(4):
                if beat in ((0,) if half else (0, 2)) or (full and beat == 3 and bar % 2 == 1 and not half):
                    tr.add(bt(bar, beat + (0.5 if beat == 3 else 0)), kick(1.0 if beat != 3 else 0.6), kick_g, rev=0.05)
                for e in (0, 0.5):
                    tr.add(bt(bar, beat + e), shaker(1.0 if e else 0.6), hat_g, pan=0.25, rev=0.15)
                if beat in (1, 3) and full: tr.add(bt(bar, beat), hat(0.9, open_=True), hat_g * 0.5, pan=-0.2, rev=0.2)
            for beat, dd in ((0, 1.4), (1.5, 0.4), (2, 0.9), (3, 0.5)):
                if half and beat not in (0, 2): continue
                tr.add(bt(bar, beat), bass(root + (12 if beat == 3 else 0), dd * BEAT), 0.16, rev=0.05)
    for bar in (6, 7):
        for e in range(8): tr.add(bt(bar, e * 0.5), shaker(0.5 + 0.3 * (e % 2)), 0.10 * (bar - 5) / 2, pan=0.3, rev=0.2)
    tr.add(bt(9, 2), kick(0.6), 0.35); tr.add(bt(9, 3), kick(0.7), 0.4)
    groove(10, 17)
    groove(18, 20, kick_g=0.45, full=False)
    for bar in (19,):
        for s in range(8): tr.add(bt(bar, 2 + s * 0.25), kick(0.4 + 0.08 * s), 0.25, rev=0.05)
    groove(22, 30)
    groove(30, 34, kick_g=0.45, half=True)

    # the dive: a riser into the job, an impact as the screen arrives
    tr.add(bt(8), noise_sweep(BAR * 2, 300, 7000), 0.10, rev=0.4)
    tr.add(bt(10), impact(1.0), 0.45, rev=0.3)
    tr.add(bt(8), bell(81, 3), 0.08, pan=0.3, rev=0.5)
    # back out to the economy: a whoosh, then a riser over the years, the headline on a bell and an impact
    tr.add(bt(17) - 0.15, noise_sweep(1.8, 6000, 300, rise=False), 0.16, pan=-0.2, rev=0.4)
    tr.add(bt(18), noise_sweep(BAR * 2, 250, 6000), 0.09, rev=0.4)
    tr.add(bt(20), impact(0.9), 0.4, rev=0.35)
    for m in (74, 81, 86): tr.add(bt(20), bell(m, 4), 0.07, pan=rng.uniform(-.3, .3), rev=0.6)
    for m in (78, 76, 74, 69): pass
    for i, m in enumerate([81, 78, 76, 74, 76, 78]): tr.add(bt(20, 1.5 + i * 0.75), piano(m, 0.9, 0.55), 0.12, pan=0.2, rev=0.5)
    # tab flips: a light chime on each
    for bb in (12.5, 14.75, 23.25, 24.5, 28.25):
        tr.add(bb * BAR, bell(93, 1.5), 0.035, pan=0.5, rev=0.4)
    for bb in (22, 26, 30):
        tr.add(bb * BAR, bell(86, 2.5), 0.05, pan=-0.3, rev=0.5)
        tr.add(bb * BAR, impact(0.5), 0.2, rev=0.3)
    # countries: a counter-melody on piano
    for i, (beat, m, d) in enumerate([(0, 78, 1), (1, 81, 1), (2, 83, 2), (4, 81, 1), (5, 78, 1), (6, 76, 2), (8, 78, 1), (9, 81, 1), (10, 86, 2), (12, 85, 1.5), (13.5, 83, 0.5), (14, 81, 2)]):
        tr.add(bt(26, beat), piano(m, d * BEAT * 1.2, 0.6), 0.09, pan=0.25, rev=0.45)

    # the ask: the groove stops, piano carries it home
    for bar in range(34, 38): tr.add(bt(bar), bass(CH[chords[bar]][1], BAR * 0.95), 0.13, rev=0.1)
    MEL = [(34, 0, 69, 1), (34, 1, 74, 1), (34, 2, 76, 1), (34, 3, 78, 2.5), (35, 2, 76, 1), (35, 3, 73, 1),
           (36, 0, 74, 1.5), (36, 1.5, 76, 0.5), (36, 2, 78, 1), (36, 3, 81, 1), (37, 0, 81, 2), (37, 2, 79, 1), (37, 3, 76, 1)]
    for bar, beat, m, d in MEL: tr.add(bt(bar, beat), piano(m, d * BEAT * 1.3, 0.8), 0.20, pan=-0.05, rev=0.45)
    for bar in (35, 36, 37):
        for e in range(4): tr.add(bt(bar, e), pluck(CH[chords[bar]][2][e % 3] - 12, 0.5, 0.4, 0.4), 0.05 * (bar - 34), pan=0.3, rev=0.4)
    tr.add(bt(36, 2), noise_sweep(BAR * 1.5, 300, 5000), 0.05, rev=0.5)
    # the end: home chord, piano, bell, and a long decay to silence
    for m in (62, 69, 74, 78): tr.add(bt(38), piano(m, 3.5, 0.7), 0.12, pan=rng.uniform(-.2, .2), rev=0.55)
    tr.add(bt(38), bell(86, 5), 0.06, rev=0.6); tr.add(bt(38), bass(38, 3.5), 0.15, rev=0.1)

    wavfile.write(out, SR, master(tr, dur, fade_in=0.4, fade_out=2.2).T)


CH2 = {   # G major, voiced low and close for a quiet bed under reading
    'G': ([59, 62, 66, 69], 43, [67, 71, 74, 78]),
    'Em': ([59, 62, 64, 67], 40, [64, 67, 71, 74]),
    'C': ([60, 64, 67, 71], 48, [67, 72, 76, 79]),
    'D': ([57, 62, 66, 69], 38, [66, 69, 74, 78]),
    'Am': ([57, 60, 64, 67], 45, [64, 69, 72, 76]),
    'Bm': ([59, 62, 66, 69], 47, [66, 71, 74, 78]),
}


def film2(out):
    """About five minutes under on-screen reading: a pad bed that changes chord each bar, a slow piano figure, a
    bell and a swell on each chapter card, a warmer lift for the five lessons, and a long settle to silence."""
    BAR, BEAT = 3.0, 0.75; NB = 104; dur = NB * BAR
    bt = lambda bar, beat=0: (bar * 4 + beat) * BEAT
    tr = Track(dur)
    cards = [3, 17, 36, 53, 68, 81]
    sections = [   # (first bar, last bar + 1, progression, piano pattern, piano gain)
        (0, 3, ['G', 'G', 'C'], None, 0),
        (5, 17, ['G', 'Em', 'C', 'D'], 'slow', 0.06),
        (19, 36, ['Em', 'C', 'G', 'D'], 'eighths', 0.05),
        (38, 53, ['C', 'D', 'Bm', 'Em'], 'slow', 0.06),
        (55, 68, ['Em', 'Am', 'C', 'D'], 'slow', 0.05),
        (70, 81, ['G', 'D', 'Em', 'C'], 'eighths', 0.05),
        (83, 91, ['C', 'G', 'D', 'Em'], 'slow', 0.06),
        (91, 98, ['G', 'D', 'Em', 'C'], 'lift', 0.07),
        (98, 101, ['C', 'Am', 'D'], 'slow', 0.05),
        (101, 104, ['G', 'G', 'G'], None, 0),
    ]
    chords = {}
    for b0, b1, prog, _, _ in sections:
        for i, bar in enumerate(range(b0, b1)): chords[bar] = prog[i % len(prog)]
    for c in cards: chords[c] = 'C'; chords[c + 1] = 'D'
    # the pad bed
    for bar in range(NB):
        if bar >= 101: continue
        notes, root, _ = CH2[chords[bar]]
        g = 0.11 if bar not in cards and bar - 1 not in cards else 0.13
        if 91 <= bar < 98: g = 0.13
        for m in notes: tr.add(bt(bar) - 0.1, pad(m, BAR, 0.3, rel=1.8, att=0.9, det=4), g, pan=rng.uniform(-.45, .45), rev=0.55)
        tr.add(bt(bar), pad(root + 12, BAR, 0.15, rel=1.5, att=0.6, det=2), 0.07, rev=0.3)
    # poster: a bell figure over G
    for i, m in enumerate([74, 79, 78, 74]): tr.add(bt(0, 1 + i * 1.5), bell(m, 4), 0.045, pan=-0.3 + 0.2 * i, rev=0.6)
    # chapter cards: bell pair, piano chord, a soft swell into the card
    for c in cards:
        tr.add(bt(c) - 1.4, noise_sweep(1.4, 400, 3500), 0.025, rev=0.6)
        tr.add(bt(c), bell(79, 4), 0.06, pan=0.25, rev=0.6); tr.add(bt(c, 1.5), bell(83, 3.5), 0.045, pan=-0.25, rev=0.6)
        for m in (60, 64, 67, 71, 76): tr.add(bt(c), piano(m, 2.6, 0.6), 0.09, pan=rng.uniform(-.2, .2), rev=0.5)
        tr.add(bt(c), bass(36, 2.5), 0.10, rev=0.2)
    # piano figures through the chapters
    for b0, b1, prog, pat, g in sections:
        if not pat: continue
        for bar in range(b0, b1):
            if bar in cards or bar - 1 in cards: continue
            tones = CH2[chords[bar]][2]
            if pat == 'slow':
                for k, (beat, i) in enumerate([(0, 0), (1, 2), (2, 1), (3, 3)]):
                    tr.add(bt(bar, beat), piano(tones[i], 1.2, 0.5 + 0.1 * (k == 0)), g, pan=0.2 * np.sin(bar + k), rev=0.5)
            elif pat == 'eighths':
                for s in range(8):
                    i = [0, 2, 1, 3, 2, 1, 3, 2][s]
                    tr.add(bt(bar, s * 0.5), pluck(tones[i], 0.45, 0.35, 0.5), g * (1 if s % 2 == 0 else 0.7), pan=0.35 * np.sin(s + bar), rev=0.45)
            elif pat == 'lift':
                for s in range(8):
                    i = [0, 1, 2, 3, 2, 1, 2, 3][s]
                    tr.add(bt(bar, s * 0.5), pluck(tones[i], 0.45, 0.45, 0.5), 0.045 * (1 if s % 2 == 0 else 0.7), pan=0.35 * np.sin(s + bar), rev=0.45)
                tr.add(bt(bar), bass(CH2[chords[bar]][1], BAR * 0.9), 0.09, rev=0.15)
    # the five lessons: a bell on each as it appears (lessons at bar 91 + 0.4 s, then every 3.2 s)
    for k in range(5): tr.add(bt(91) + 1.2 + 3.2 * k, bell([74, 76, 78, 79, 83][k], 3), 0.035, pan=0.3, rev=0.6)
    # the end: home chord, then silence
    for m in (55, 59, 62, 67, 71): tr.add(bt(101), piano(m, 6.0, 0.6), 0.08, pan=rng.uniform(-.2, .2), rev=0.6)
    for m in (59, 62, 66, 69): tr.add(bt(101), pad(m, 5.0, 0.3, rel=2.0, att=1.0), 0.10, pan=rng.uniform(-.4, .4), rev=0.6)
    tr.add(bt(101), bell(79, 6), 0.05, rev=0.65); tr.add(bt(101), bass(43, 6), 0.08, rev=0.2)
    wavfile.write(out, SR, master(tr, dur, ir_sec=3.0, fade_in=0.8, fade_out=4.0).T)


if __name__ == "__main__":
    {'film1': film1, 'film2': film2}[sys.argv[1]](sys.argv[2])
    print("wrote", sys.argv[2])
