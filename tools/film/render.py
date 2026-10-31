"""Render a film page frame by frame with headless Chromium.

  python3 tools/film/render.py stills FILM OUTDIR t1 t2 ...      one PNG per film time (seconds)
  python3 tools/film/render.py frames FILM OUTDIR START END      JPEG frames START..END-1 (frame numbers at the film's fps)

The page is served from the www folder (film.html, engine.js, the film script, data/, rec/, fonts) at BASE.
"""
import base64, os, sys, time
from playwright.sync_api import sync_playwright

BASE = os.environ.get("FILM_BASE", "http://127.0.0.1:8790")
CHROME = os.environ.get("CHROME", "/opt/pw-browsers/chromium-1243/chrome-linux64/chrome")


def open_film(pw, film):
    br = pw.chromium.launch(executable_path=CHROME, args=["--no-sandbox", "--disable-gpu-vsync", "--disable-frame-rate-limit"])
    pg = br.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(f"{BASE}/film.html?film={film}")
    pg.wait_for_function("document.title === 'ready'", timeout=120000)
    info = pg.evaluate("window.FILMINFO")
    if errs: print("page errors:", errs, flush=True)
    if not all(info["fonts"]): print("WARNING fonts missing:", info["fonts"], flush=True)
    return br, pg, info


def grab(pg, t, kind="image/jpeg", q=0.94):
    ok = pg.evaluate(f"renderAt({t})")
    if not ok: print(f"  images still loading at t={t:.3f}", flush=True)
    url = pg.evaluate(f"document.getElementById('c').toDataURL('{kind}', {q})")
    return base64.b64decode(url.split(",", 1)[1])


def main():
    mode, film, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    with sync_playwright() as pw:
        br, pg, info = open_film(pw, film)
        fps = info["fps"]
        if mode == "stills":
            for a in sys.argv[4:]:
                t = float(a)
                open(os.path.join(out, f"{film}_{t:07.2f}.png"), "wb").write(grab(pg, t, "image/png"))
            print("stills done", flush=True)
        else:
            s, e = int(sys.argv[4]), int(sys.argv[5])
            t0 = time.time()
            for f in range(s, e):
                p = os.path.join(out, f"f{f:05d}.jpg")
                if os.path.exists(p) and os.path.getsize(p) > 0: continue
                open(p + ".tmp", "wb").write(grab(pg, f / fps)); os.replace(p + ".tmp", p)
                if (f - s) % 120 == 0: print(f"{film} frame {f} ({(f - s + 1) / max(1e-3, time.time() - t0):.1f} f/s)", flush=True)
            print(f"chunk {s}-{e} done in {time.time() - t0:.0f}s", flush=True)
        br.close()


if __name__ == "__main__":
    main()
