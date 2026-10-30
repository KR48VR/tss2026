"""Readability pass: every view renders without errors; screenshots of the restructured passages and the new visuals;
axe on the method, Europe and overview views; overflow on the phone."""
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
            await page.evaluate("document.querySelector('header').style.position='static'")
            # scenario explainer
            await page.evaluate("const d=document.getElementById('scn-more'); if(d) d.open=true"); await page.wait_for_timeout(100)
            print(f"[{name}] explainer bullets:", await page.evaluate("document.querySelectorAll('#scn-explain li').length"))
            await page.locator("#scn-more").screenshot(path=f"{OUT}/read-explainer-{name}.png")
            # overview: what it is not
            await page.click("nav.tabs button[data-view=overview]"); await page.wait_for_timeout(300)
            await page.click("nav.tabs button[data-view=leaders]"); await page.wait_for_timeout(300); el = page.locator("#view-leaders .panel", has_text="What it is not"); await el.screenshot(path=f"{OUT}/read-whatnot-{name}.png")
            # method view
            await page.click("nav.tabs button[data-view=method]"); await page.wait_for_timeout(300)
            await page.evaluate("document.querySelectorAll('#view-method details').forEach(d=>d.open=true)"); await page.wait_for_timeout(200)
            print(f"[{name}] method: strip rects", await page.evaluate("document.querySelectorAll('#m-strip rect').length"), "| formula boxes", await page.evaluate("document.querySelectorAll('#m-formula .fbox').length"), "| spans filled:", await page.evaluate("['m-lag-unproven','m-aishare','m-btos-date','m-anchor','m-lead','m-sg-anchor','m-shift-early','m-doubling','m-anchor-old'].map(id=>id+'='+(document.getElementById(id)||{textContent:'MISSING'}).textContent).join(' ')"))
            await page.screenshot(path=f"{OUT}/read-method-{name}.png", full_page=True)
            await page.add_script_tag(content=AXE); r = await page.evaluate(RUN); print(f"[{name}] axe method:", r["v"] or "0 violations")
            # compare caveat
            await page.click("nav.tabs button[data-view=compare]"); await page.wait_for_timeout(300)
            await page.evaluate("document.querySelectorAll('#view-compare details').forEach(d=>d.open=true)")
            await page.locator("#view-compare details.more").screenshot(path=f"{OUT}/read-compare-caveat-{name}.png")
            # europe chart
            await page.click("nav.tabs button[data-view=europe]"); await page.wait_for_timeout(500)
            print(f"[{name}] europe chart bars:", await page.evaluate("document.querySelectorAll('#eu-chart rect').length"), "| ref lines:", await page.evaluate("document.querySelectorAll('#eu-chart line').length"))
            await page.locator("#eu-chart").screenshot(path=f"{OUT}/read-eu-chart-{name}.png")
            await page.evaluate("document.querySelectorAll('#view-europe details').forEach(d=>d.open=true)")
            await page.add_script_tag(content=AXE); r = await page.evaluate(RUN); print(f"[{name}] axe europe:", r["v"] or "0 violations")
            # a national country's sectors tab and skills block
            await page.click(".cty button[data-cty=NOR]"); await page.wait_for_timeout(400)
            await page.click("nav.tabs button[data-view=sectors]"); await page.wait_for_timeout(400)
            print(f"[{name}] sector dl rows:", await page.evaluate("document.querySelectorAll('#nat-sec-p dt').length"))
            await page.locator("#nat-sec-p").screenshot(path=f"{OUT}/read-sector-p-{name}.png")
            await page.click("nav.tabs button[data-view=skills]"); await page.wait_for_timeout(400)
            await page.locator("#lvn-p").screenshot(path=f"{OUT}/read-lvn-p-{name}.png")
            await page.add_script_tag(content=AXE); r = await page.evaluate(RUN); print(f"[{name}] axe skills (NOR):", r["v"] or "0 violations")
            for v in ["leaders","overview","compare","jobs","sectors","news"]:
                await page.click(f"nav.tabs button[data-view={v}]"); await page.wait_for_timeout(200)
            print(f"[{name}] overflow:", await page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth"), "| errors:", errs)
            await ctx.close()
        await b.close()
asyncio.run(main())
