"""Record the Work and AI Map being used, frame by frame, for the films.

Each recording is a list of 24 fps frames. A frame points at a screenshot (taken only when the page changes; holds
and cursor moves reuse the last one) and carries the cursor position in CSS pixels, so the film draws a soft cursor
itself. Screenshots are taken at device scale factor 2 (2560x1600 for a 1280x800 viewport) so the film can zoom in
sharply. Scrolls are stepped per frame with an ease, so motion is smooth whatever the screenshot speed.

  python tools/film/record.py OUTDIR [name ...]
"""
import asyncio, json, math, sys, time
from pathlib import Path
from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8765"
W, H, DSF = 1280, 800, 2
CHROME = "/opt/pw-browsers/chromium-1243/chrome-linux64/chrome"


def ease(u):  # smooth in and out
    u = max(0.0, min(1.0, u)); return u * u * (3 - 2 * u)


class Rec:
    def __init__(self, page, out: Path):
        self.page, self.out = page, out
        out.mkdir(parents=True, exist_ok=True)
        self.files, self.frames = [], []
        self.cx, self.cy, self.cur = W * 0.62, H * 0.55, None
        self.shown = False   # cursor visible

    async def shot(self):
        n = len(self.files)
        f = f"s{n:04d}.jpg"
        await self.page.screenshot(path=str(self.out / f), type="jpeg", quality=90)
        self.files.append(f); self.cur = n
        return n

    def emit(self, n=1, click=0):
        for i in range(n):
            self.frames.append([self.cur, round(self.cx, 1), round(self.cy, 1), 1 if self.shown else 0, click if i == 0 else 0])

    async def hold(self, n):
        self.emit(n)

    async def move(self, x, y, n=14):
        self.shown = True
        x0, y0 = self.cx, self.cy
        for i in range(1, n + 1):
            u = ease(i / n)
            # a slight arc, as a hand moves
            self.cx = x0 + (x - x0) * u; self.cy = y0 + (y - y0) * u - math.sin(math.pi * u) * 18
            self.emit(1)

    async def center(self, sel):
        bb = await self.page.locator(sel).first.bounding_box()
        return (bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2) if bb else (self.cx, self.cy)

    async def click(self, sel, n_move=14, wait=500, after=8):
        try:
            x, y = await self.center(sel)
            await self.move(x, y, n_move)
            self.emit(3)
            await self.page.locator(sel).first.click(timeout=4000)
            await self.page.wait_for_timeout(wait)
            await self.shot(); self.emit(1, click=1); self.emit(after)
        except Exception as e:
            print("  click failed", sel, str(e)[:120])

    async def type(self, text, per_char=3, wait=60):
        for ch in text:
            await self.page.keyboard.type(ch); await self.page.wait_for_timeout(wait)
            await self.shot(); self.emit(per_char)

    async def scroll_y(self):
        return await self.page.evaluate("scrollY")

    async def scroll(self, to, n=48):
        y0 = await self.scroll_y()
        for i in range(1, n + 1):
            y = y0 + (to - y0) * ease(i / n)
            await self.page.evaluate(f"window.scrollTo({{top:{y:.1f},behavior:'instant'}})")
            await self.shot(); self.emit(1)

    async def y_of(self, js_selector_expr, offset=0):
        """Document y of an element found by a JS expression returning an element."""
        return await self.page.evaluate(f"(()=>{{const e=({js_selector_expr}); return e? e.getBoundingClientRect().top+scrollY+{offset} : null;}})()")

    def save(self, meta=None):
        (self.out / "manifest.json").write_text(json.dumps({"w": W, "h": H, "dsf": DSF, "fps": 24, "files": self.files, "frames": self.frames, **(meta or {})}))
        print(f"  {self.out.name}: {len(self.frames)} frames ({len(self.frames)/24:.1f} s), {len(self.files)} screenshots")


def by_text(tag, text):
    """The deepest visible element whose text contains `text` (the last match in document order is the innermost)."""
    return f"Array.from(document.querySelectorAll('{tag}')).filter(e=>e.offsetParent && e.textContent.includes({json.dumps(text)})).pop()"


async def settle(page, y, tries=8):
    """Scroll to y and make sure it sticks (a view switch can reset the scroll a moment later)."""
    for _ in range(tries):
        await page.evaluate(f"window.scrollTo({{top:{y},behavior:'instant'}})"); await page.wait_for_timeout(350)
        if abs(await page.evaluate("scrollY") - y) < 2:
            return True
    print("  scroll did not settle at", y); return False


async def fresh(ctx, view=None, country=None, scroll_to=None):
    page = await ctx.new_page()
    await page.goto(f"{BASE}/index.html", wait_until="load"); await page.wait_for_timeout(1200)
    if country:
        await page.click(f".cty button[data-cty={country}]"); await page.wait_for_timeout(400)
    if view:
        await page.click(f"nav.tabs button[data-view={view}]"); await page.wait_for_timeout(1500)
    return page


VIEW_TOP = 330   # scroll position that puts a view's heading just under the pinned bar


async def r_search(ctx, out):
    page = await fresh(ctx, "jobs"); r = Rec(page, out)
    await settle(page, VIEW_TOP)
    await r.shot(); r.cx, r.cy = 900, 600; await r.hold(10)
    await r.click("#job-search", wait=150, after=4)
    await r.type("accountant", per_char=3)
    await r.hold(14)
    await r.click("#job-results button >> nth=1", n_move=16, wait=1300, after=4)
    await r.hold(30)
    r.save({"title": "Find a job: accountant"}); await page.close()


async def r_role(ctx, out):
    page = await fresh(ctx, "jobs"); r = Rec(page, out)
    await page.fill("#job-search", "accountant"); await page.wait_for_timeout(700)
    await page.click("#job-results button >> nth=1"); await page.wait_for_timeout(1500)
    top = await r.y_of("document.querySelector('#job-detail')", -90)
    await settle(page, top)
    await r.shot(); await r.hold(18)
    kt = await r.y_of(by_text('h3', "key tasks, earliest first"), -110)
    if kt:
        await r.scroll(kt, 60); await r.hold(10)
        await r.scroll(kt + 520, 72); await r.hold(24)
    r.save({"title": "Role: key tasks, earliest first"}); await page.close()


async def r_scenario(ctx, out):
    page = await fresh(ctx, "jobs"); r = Rec(page, out)
    await page.fill("#job-search", "accountant"); await page.wait_for_timeout(700)
    await page.click("#job-results button >> nth=1"); await page.wait_for_timeout(1500)
    top = await r.y_of("document.querySelector('#job-detail')", -90)
    await settle(page, top + 40)
    await r.shot(); r.cx, r.cy = 700, 300; await r.hold(12)
    for b in ("0", "2", "1"):
        await r.click(f".scn.mini button[data-scn='{b}']", n_move=14, wait=500, after=22)
    r.save({"title": "Scenario switch on a role"}); await page.close()


async def r_tab(ctx, out, view, scroll_extra=0, scroll_frames=60, title=""):
    page = await fresh(ctx, view); r = Rec(page, out)
    await settle(page, VIEW_TOP)
    await r.shot(); r.cx, r.cy = 760, 500; await r.hold(24)
    if scroll_extra:
        await r.scroll(VIEW_TOP + scroll_extra, scroll_frames); await r.hold(18)
    r.save({"title": title or view}); await page.close()


async def r_countries(ctx, out):
    page = await fresh(ctx); r = Rec(page, out)
    await settle(page, 40)
    await r.shot(); r.cx, r.cy = 640, 560; await r.hold(10)
    for c in ("USA", "GBR", "DEU", "JPN", "FRA", "AUS", "SGP"):
        await r.click(f".cty button[data-cty={c}]", n_move=10, wait=500, after=12)
    r.save({"title": "Country switch"}); await page.close()


async def r_europe(ctx, out):
    page = await fresh(ctx, "europe"); r = Rec(page, out)
    y = await r.y_of(by_text('*', "Firms using AI, latest reading, against the US level"), -100) or 1100
    await settle(page, y - 260)
    await r.shot(); await r.hold(10)
    await r.scroll(y, 48); await r.hold(14); await r.scroll(y + 420, 72); await r.hold(18)
    r.save({"title": "Europe: 32 countries"}); await page.close()


async def r_compare(ctx, out):
    page = await fresh(ctx, "compare"); r = Rec(page, out)
    await page.select_option("#cmp-b", "DEU"); await page.wait_for_timeout(1500)
    y = await r.y_of(by_text('*', "THE SHARE OF EACH COUNTRY"), -140) or await r.y_of(by_text('div', "share of each country"), -140) or 1400
    await settle(page, VIEW_TOP); await r.shot(); await r.hold(14)
    await r.scroll(y, 60); await r.hold(36)
    r.save({"title": "Compare Singapore and Germany"}); await page.close()


async def r_news(ctx, out):
    page = await fresh(ctx, "news"); r = Rec(page, out)
    await settle(page, VIEW_TOP); await r.shot(); r.cx, r.cy = 900, 500; await r.hold(30)
    y = await r.y_of("document.querySelector('#tr-pct')", -230) or 1700
    await r.scroll(y, 60); await r.hold(8)
    await r.click("#tr-pct", wait=120, after=3)
    await page.keyboard.press("Control+A"); await page.wait_for_timeout(50)
    await r.type("70", per_char=5, wait=250)
    await page.keyboard.press("Tab"); await page.wait_for_timeout(300); await r.shot(); await r.hold(30)
    r.save({"title": "Reading the news: ladder and translator"}); await page.close()


async def r_overview(ctx, out):
    page = await fresh(ctx, "overview"); r = Rec(page, out)
    await settle(page, VIEW_TOP); await r.shot(); await r.hold(36)
    await r.scroll(VIEW_TOP + 260, 48); await r.hold(30)
    r.save({"title": "Overview: evidence as of, tiles"}); await page.close()


async def r_meaning(ctx, out):
    page = await fresh(ctx, "jobs"); r = Rec(page, out)
    await settle(page, VIEW_TOP)
    await page.click("#meaning-toggle"); 
    for _ in range(60):   # wait for the in-page model to load
        txt = await page.evaluate("(document.getElementById('meaning-note')||{textContent:''}).textContent")
        if "ready" in txt.lower() or "loaded" in txt.lower(): break
        await page.wait_for_timeout(500)
    await r.shot(); r.cx, r.cy = 800, 560; await r.hold(8)
    await r.click("#job-search", wait=120, after=3)
    await r.type("look after elderly people at home", per_char=2, wait=40)
    await page.wait_for_timeout(1500); await r.shot(); await r.hold(40)
    r.save({"title": "Match by meaning"}); await page.close()


async def r_doing(ctx, out):
    page = await fresh(ctx, "method"); r = Rec(page, out)
    y = await r.y_of(by_text('*', "Picture a job made of ten equal tasks"), -260) or 900
    await settle(page, VIEW_TOP); await r.shot(); await r.hold(12)
    await r.scroll(y, 54); await r.hold(48)
    r.save({"title": "Involved and doing: ten tasks"}); await page.close()


async def r_depth(ctx, out):
    page = await fresh(ctx, "leaders", country="DEU"); r = Rec(page, out)
    y = await r.y_of("Array.from(document.querySelectorAll('.xcard')).find(x=>!x.hidden && x.offsetParent)", -120) or 1200
    await settle(page, VIEW_TOP); await r.shot(); await r.hold(16)
    await r.scroll(y, 54); await r.hold(48)
    r.save({"title": "Germany: depth mark"}); await page.close()


async def companion(ctx, out, anchor, extra, frames, title):
    page = await ctx.new_page(); r = Rec(page, out)
    await page.goto(f"{BASE}/companion.html", wait_until="load"); await page.wait_for_timeout(1000)
    y = await r.y_of(anchor, -40) or 0
    await settle(page, y); await r.shot(); await r.hold(20)
    if extra:
        await r.scroll(y + extra, frames); await r.hold(24)
    r.save({"title": title}); await page.close()


RECIPES = {
    "search": r_search, "role": r_role, "scenario": r_scenario,
    "sectors": lambda c, o: r_tab(c, o, "sectors", 380, 72, "Sectors"),
    "skills": lambda c, o: r_tab(c, o, "skills", 380, 72, "Skills and activities"),
    "leaders": lambda c, o: r_tab(c, o, "leaders", 120, 48, "Executive summary"),
    "countries": r_countries, "europe": r_europe, "compare": r_compare, "news": r_news, "overview": r_overview,
    "meaning": r_meaning, "doing": r_doing, "depth": r_depth,
    "c_qa": lambda c, o: companion(c, o, "document.getElementById('qa')", 900, 96, "Companion: QA log"),
    "c_king": lambda c, o: companion(c, o, "document.getElementById('king')", 700, 84, "Companion: King and AI"),
    "c_limits": lambda c, o: companion(c, o, by_text('h3', '7. Reopening the honest limits'), 260, 60, "Companion: reopening the honest limits"),
}


async def main(out_root, names):
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, executable_path=CHROME, args=["--no-sandbox"])
        for n in names or list(RECIPES):
            t0 = time.time()
            # a clean browser session per recording: the map keeps the last view and comparison in local storage
            ctx = await b.new_context(viewport={"width": W, "height": H}, device_scale_factor=DSF, color_scheme="dark")
            errs = []
            ctx.on("page", lambda pg: pg.on("pageerror", lambda e: errs.append(str(e)[:160])))
            try:
                await RECIPES[n](ctx, Path(out_root) / n)
            except Exception as e:
                print("FAIL", n, type(e).__name__, str(e)[:200])
            await ctx.close()
            print(f"  ({time.time()-t0:.0f} s){' page errors: ' + '; '.join(errs[:3]) if errs else ''}")
        await b.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2:]))
