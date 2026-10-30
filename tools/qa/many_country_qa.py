"""Page QA for the office-built countries: the generic leaders card with its depth badge, the overview depth line, a job page
without a national task list, the sectors view (or its no-lens note), the skill-level panel and a compare pair; axe on the
leaders and job views; desktop light and phone dark."""
import asyncio, sys
from playwright.async_api import async_playwright
import os; OUT = os.environ.get("QA_OUT", "/home/user/tss2026/tools/qa/out"); os.makedirs(OUT, exist_ok=True)
AXE = open("/home/user/tss2026/tools/qa/node_modules/axe-core/axe.min.js").read()
RUN = "axe.run(document, {runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21aa','best-practice']}}).then(r=>({v:r.violations.map(v=>v.id+':'+v.impact+':'+v.nodes.slice(0,3).map(n=>n.target.join(' ')).join(' | '))}))"
T = lambda page, sel: page.evaluate(f"(document.querySelector('{sel}')||{{textContent:'MISSING'}}).textContent.replace(/\\s+/g,' ').trim()")
COUNTRIES = sys.argv[1:] or ["SWE", "FIN", "DNK", "NLD", "IRL"]
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, executable_path="/opt/pw-browsers/chromium-1243/chrome-linux64/chrome", args=["--no-sandbox"])
        for vp, name, scheme in [({"width": 1280, "height": 900}, "desktop", "light"), ({"width": 390, "height": 844}, "phone", "dark")]:
            ctx = await b.new_context(viewport=vp, color_scheme=scheme); page = await ctx.new_page(); errs = []
            page.on("pageerror", lambda e: errs.append(str(e)))
            await page.goto("http://127.0.0.1:8765/index.html", wait_until="load"); await page.wait_for_timeout(1000)
            print(f"\n=== [{name}] buttons:", await page.evaluate("Array.from(document.querySelectorAll('.cty button')).filter(b=>!b.hidden).map(b=>b.textContent.trim())"))
            axes = []
            for cty in COUNTRIES:
                await page.click(f".cty button[data-cty={cty}]"); await page.wait_for_timeout(500)
                await page.click("nav.tabs button[data-view=leaders]"); await page.wait_for_timeout(400)
                card = await page.evaluate("(()=>{ const c=Array.from(document.querySelectorAll('.xcard')).find(x=>!x.hidden && (x.id==='xcard-generic' || x.dataset.cty)); return c? {title:c.querySelector('h3').textContent.replace(/\\s+/g,' ').trim(), badge:(c.querySelector('.depth')||{}).textContent, exists:(c.querySelector('.step.ex p')||{textContent:''}).textContent.slice(0,160), added:c.querySelectorAll('.step.add li').length} : null; })()")
                print(f"[{name} {cty}] card:", card)
                print(f"[{name} {cty}] breath:", (await T(page, "#x-breath"))[:260])
                await page.add_script_tag(content=AXE); r = await page.evaluate(RUN); axes.append((f"{cty} leaders", r["v"]))
                await page.click("nav.tabs button[data-view=overview]"); await page.wait_for_timeout(400)
                print(f"[{name} {cty}] overview intro:", (await T(page, "#ov-intro"))[:230])
                print(f"[{name} {cty}] major rows:", await page.evaluate("document.querySelectorAll('#major-chart svg circle').length"), "| sub rows:", await page.evaluate("document.querySelectorAll('#sub-table tbody tr').length"), "| wage bill label:", await page.evaluate("(document.querySelector('#x-chart-eyebrow')||{textContent:''}).textContent.slice(0,80)"))
                await page.click("nav.tabs button[data-view=jobs]"); await page.wait_for_timeout(300)
                print(f"[{name} {cty}] examples:", (await T(page, "#examples"))[:160], "| help:", (await T(page, "#search-help"))[:120])
                first_ex = await page.evaluate("(document.querySelector('#examples a, #examples button')||{textContent:''}).textContent.trim()")
                await page.fill("#job-search", first_ex[:40]); await page.wait_for_timeout(600); await page.click("#job-results button[data-kind=nat] >> nth=0"); await page.wait_for_timeout(1200)
                print(f"[{name} {cty}] job:", await T(page, "#job-detail h2"), "|", (await page.evaluate("(document.querySelector('#job-detail h2 + p')||{textContent:''}).textContent.replace(/\\s+/g,' ').trim()"))[:200])
                print(f"[{name} {cty}] task block:", (await page.evaluate("(Array.from(document.querySelectorAll('#job-detail h3')).find(h=>/involves/.test(h.textContent))||{nextElementSibling:{textContent:'MISSING'}}).nextElementSibling.textContent.replace(/\\s+/g,' ').trim()"))[:160], "| onet rows:", await page.evaluate("(Array.from(document.querySelectorAll('#job-detail h3')).find(h=>/Built from/.test(h.textContent))||{nextElementSibling:{nextElementSibling:{querySelectorAll:()=>[]}}}).nextElementSibling.nextElementSibling.querySelectorAll('li').length"))
                print(f"[{name} {cty}] outcomes:", (await page.evaluate("(Array.from(document.querySelectorAll('#job-detail .outc .cell')).map(c=>c.textContent.replace(/\\s+/g,' ').trim()).join(' || '))"))[:260])
                await page.add_script_tag(content=AXE); r = await page.evaluate(RUN); axes.append((f"{cty} job", r["v"]))
                await page.click("nav.tabs button[data-view=sectors]"); await page.wait_for_timeout(400)
                print(f"[{name} {cty}] sectors:", await T(page, "#nat-sec-h"), "| rows:", await page.evaluate("document.querySelectorAll('#nat-sector-table tbody tr').length"), "| row1:", (await page.evaluate("(document.querySelector('#nat-sector-table tbody tr')||{textContent:''}).textContent.replace(/\\s+/g,' ').trim()"))[:120])
                await page.click("nav.tabs button[data-view=skills]"); await page.wait_for_timeout(400)
                print(f"[{name} {cty}] levels:", (await T(page, "#lvn-eyebrow"))[:50], "| rows:", await page.evaluate("document.querySelectorAll('#lvn-table tbody tr').length"), "| reading:", (await T(page, "#lvn-reading"))[:140])
                await page.click("nav.tabs button[data-view=compare]"); await page.wait_for_timeout(300); await page.select_option("#cmp-a", "SGP"); await page.select_option("#cmp-b", cty); await page.wait_for_timeout(500)
                print(f"[{name} {cty}] compare:", (await T(page, "#cmp-decomp"))[:200], "| lede tail:", (await T(page, "#cmp-lede"))[-160:])
            print(f"[{name}] overflow:", await page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth"), "| errors:", errs[:5])
            for label, v in axes: print(f"[{name}] axe {label}:", v or "0 violations")
            await ctx.close()
        await b.close()
asyncio.run(main())
