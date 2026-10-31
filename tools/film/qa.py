"""Measured QA pass on a finished film: picture jumps, sound envelope, loudness, silence at the end, first vs last frame.
  python3 tools/film/qa.py FILM.mp4 [BAR_SECONDS]"""
import json, subprocess, sys
import numpy as np

path = sys.argv[1]; bar = float(sys.argv[2]) if len(sys.argv) > 2 else 0
dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path]))
raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=160:90,format=gray", "-f", "rawvideo", "-"])
fr = np.frombuffer(raw, np.uint8).reshape(-1, 90, 160).astype(np.float32)
d = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
print(f"duration {dur:.2f}s, {len(fr)} frames; mean frame change {d.mean():.2f}, 99th pct {np.percentile(d, 99):.2f}, max {d.max():.2f} at {(d.argmax() + 1) / 24:.2f}s")
for thr in (15, 25):
    j = np.where(d > thr)[0]
    print(f"  jumps > {thr}: {len(j)}", [round((k + 1) / 24, 2) for k in j][:20])
print(f"  first vs last frame difference {np.abs(fr[0] - fr[-1]).mean():.2f}")
pcm = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "48000", "-f", "f32le", "-"])
x = np.frombuffer(pcm, np.float32); w = 1200
rms = 20 * np.log10(np.sqrt(np.convolve(x ** 2, np.ones(w) / w, "valid")[::w]) + 1e-9)
jumps = np.where((np.diff(rms) > 9) & (rms[1:] > -40))[0]
print(f"sound: {len(jumps)} rises > 9 dB in 25 ms (above -40 dB)")
if bar:
    off = [(k + 1) * 0.025 for k in jumps]; onbeat = [t for t in off if min((t / (bar / 4)) % 1, 1 - (t / (bar / 4)) % 1) * bar / 4 < 0.06]
    print(f"  of which on the beat grid (±60 ms): {len(onbeat)}; off-grid: {[round(t, 2) for t in off if t not in onbeat][:20]}")
print(f"  last 0.1 s RMS {np.sqrt((x[-4800:] ** 2).mean()):.2e}; first 0.1 s RMS {np.sqrt((x[:4800] ** 2).mean()):.2e}")
out = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
s = out[out.rfind("Summary:"):]
print("loudness:", " ".join(l.strip() for l in s.splitlines() if any(k in l for k in ("I:", "LRA:", "Peak:"))))
