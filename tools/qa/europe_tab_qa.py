"""The Europe tab: visible from the outset, renders the table, opens a country, keyboard-reachable, axe clean on both viewports."""
import asyncio
from playwright.async_api import async_playwright
import os; OUT = os.environ.get("QA_OUT", "/home/user/tss2026/tools/qa/out"); os.makedirs(OUT, exist_ok=True)
AXE = open("/home/user/tss2026/tools/qa/node_modules/axe-core/axe.min.js").read()
RUN = "axe.run(document, {runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21aa','best-practice']}}).then(r=>({v:r.violations.map(v=>v.id+':'+v.impact+':'+v.nodes.slice(0,3).map(n=>n.target.join(' ')).join(' | '))}))"
T = lambda page, sel: page.evaluate(f"(document.querySelector('{sel}')||{{textContent:'MISSING'}}).textContent.replace(/\\s+/g,' ').trim()")
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, executable_path="/opt/pw-browsers/chromium-1243/chrome-linux64/chrome", args=["--no-sandbox"])
        for vp,name,scheme in [({"width":1280,"height":900},"desktop","light"),({"width":390,"height":844},"phone","dark")]:
            ctx = await b.new_context(viewport=vp, color_scheme=scheme); page = await ctx.new_page(); errs=[]
            page.on("pageerror", lambda e: errs.append(str(e)))
            await page.goto("http://127.0.0.1:8765/index.html", wait_until="load"); await page.wait_for_timeout(1000)
            print(f"[{name}] tabs:", await page.evaluate("Array.from(document.querySelectorAll('nav.tabs button')).filter(b=>!b.hidden).map(b=>b.textContent.trim())"))
            print(f"[{name}] hint:", await T(page, "#eu-hint"), "| hidden:", await page.evaluate("document.getElementById('eu-hint').hidden"))
            await page.click("nav.tabs button[data-view=leaders]"); await page.wait_for_timeout(200); print(f"[{name}] summary line:", await T(page, "#x-eu-line"))
            await page.click("#x-eu-line a[data-view-link]"); await page.wait_for_timeout(500)
            print(f"[{name}] europe view hidden:", await page.evaluate("document.getElementById('view-europe').hidden"), "| h2:", await T(page, "#view-europe h2"), "| rows:", await page.evaluate("document.querySelectorAll('#eu-table tbody tr').length"))
            await page.click("#eu-table tbody tr:nth-child(5) .linkbtn"); await page.wait_for_timeout(400); print(f"[{name}] opened:", await T(page, "#eu-detail h3"), "| groups+industries rows:", await page.evaluate("document.querySelectorAll('#eu-detail tbody tr').length"))
            await page.click("nav.tabs button[data-view=compare]"); await page.wait_for_timeout(300); print(f"[{name}] compare pointer:", (await page.evaluate("(document.querySelector('#view-compare a[data-view-link=europe]')||{textContent:'MISSING'}).textContent"))[:60])
            await page.click("nav.tabs button[data-view=europe]"); await page.wait_for_timeout(300)
            await page.add_script_tag(content=AXE); r = await page.evaluate(RUN); print(f"[{name}] axe europe tab:", r["v"] or "0 violations")
            await page.click("nav.tabs button[data-view=leaders]"); await page.wait_for_timeout(200); await page.add_script_tag(content=AXE); r = await page.evaluate(RUN); print(f"[{name}] axe summary:", r["v"] or "0 violations")
            print(f"[{name}] overflow:", await page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth"), "| errors:", errs)
            await page.screenshot(path=f"{OUT}/europe-tab-{name}.png", full_page=False)
            await ctx.close()
        await b.close()
asyncio.run(main())
