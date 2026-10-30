"""The slim sticky bar: heights, stickiness when scrolled, the compact controls driving the page and following the big switches, the view opening under the bar, axe, overflow, screenshots."""
import asyncio
from playwright.async_api import async_playwright
import os; OUT = os.environ.get("QA_OUT", "/home/user/tss2026/tools/qa/out"); os.makedirs(OUT, exist_ok=True)
AXE = open("/home/user/tss2026/tools/qa/node_modules/axe-core/axe.min.js").read()
RUN = "axe.run(document, {runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21aa','best-practice']}}).then(r=>({v:r.violations.map(v=>v.id+':'+v.impact+':'+v.nodes.slice(0,3).map(n=>n.target.join(' ')).join(' | '))}))"
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, executable_path="/opt/pw-browsers/chromium-1243/chrome-linux64/chrome", args=["--no-sandbox"])
        for vp,name,scheme in [({"width":1280,"height":900},"desktop","light"),({"width":390,"height":844},"phone","dark")]:
            ctx = await b.new_context(viewport=vp, color_scheme=scheme); page = await ctx.new_page(); errs=[]
            page.on("pageerror", lambda e: errs.append(str(e)))
            await page.goto("http://127.0.0.1:8765/index.html", wait_until="load"); await page.wait_for_timeout(800)
            hh = await page.evaluate("document.querySelector('header').offsetHeight"); bh = await page.evaluate("document.getElementById('bar').offsetHeight")
            print(f"[{name}] header {hh}px, sticky bar {bh}px, tabs rows:", await page.evaluate("(()=>{const ys=new Set(Array.from(document.querySelectorAll('nav.tabs button')).map(b=>Math.round(b.getBoundingClientRect().top))); return ys.size})()"))
            await page.click("nav.tabs button[data-view=skills]"); await page.wait_for_timeout(300)
            await page.evaluate("window.scrollTo(0, 900)"); await page.wait_for_timeout(300)
            print(f"[{name}] bar top when scrolled:", await page.evaluate("document.getElementById('bar').getBoundingClientRect().top"), "| header visible:", await page.evaluate("document.querySelector('header').getBoundingClientRect().bottom > 0"))
            await page.screenshot(path=f"{OUT}/bar-{name}-scrolled.png")
            await page.select_option("#bar-cty", "NOR"); await page.wait_for_timeout(500)
            print(f"[{name}] after select: pressed:", await page.evaluate("document.querySelector('.cty button[aria-pressed=true]').textContent"), "| h2:", (await page.evaluate("document.getElementById('lvn-h2').textContent"))[:60])
            await page.click(".topbar .scn.mini button[data-scn='0']"); await page.wait_for_timeout(400)
            print(f"[{name}] after mini Fast: big switch:", await page.evaluate("document.querySelector('header .scn button[aria-pressed=true]').textContent.trim()"), "| band:", await page.evaluate("document.documentElement.dataset.scn"))
            await page.click("header .scn button[data-scn='1']"); await page.wait_for_timeout(300)
            print(f"[{name}] after big Central: mini pressed:", await page.evaluate("document.querySelector('.topbar .scn.mini button[aria-pressed=true]').textContent"))
            await page.click(".cty button[data-cty=JPN]"); await page.wait_for_timeout(400); print(f"[{name}] select follows big switch:", await page.evaluate("document.getElementById('bar-cty').value"))
            await page.evaluate("window.scrollTo(0, 1500)"); await page.click("nav.tabs button[data-view=overview]"); await page.wait_for_timeout(300)
            print(f"[{name}] after tab click scrollY:", await page.evaluate("Math.round(window.scrollY)"), "bar offsetTop:", await page.evaluate("document.getElementById('bar').offsetTop"))
            await page.evaluate("window.scrollTo(0,0)"); await page.wait_for_timeout(200); await page.screenshot(path=f"{OUT}/bar-{name}-top.png")
            await page.add_script_tag(content=AXE); r = await page.evaluate(RUN); print(f"[{name}] axe:", r["v"] or "0 violations")
            print(f"[{name}] overflow:", await page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth"), "| errors:", errs)
            await ctx.close()
        await b.close()
asyncio.run(main())
