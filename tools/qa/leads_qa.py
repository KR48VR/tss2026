"""Lead comparability on the page: the wording for every country, the compare rows, the Europe column, the method
line, the new adoption chart; axe on the summary; console errors; screenshots of the chart."""
import asyncio, sys
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
            await page.goto("http://127.0.0.1:8765/index.html", wait_until="load"); await page.wait_for_timeout(800)
            await page.evaluate("localStorage.setItem('cmpa','NOR'); localStorage.setItem('cmpb','USA');"); await page.reload(wait_until="load"); await page.wait_for_timeout(800)
            await page.click("nav.tabs button[data-view=leaders]"); await page.wait_for_timeout(300)
            for k in (["SGP","USA","CAN","AUS","GBR","NOR","JPN","KOR","NZL","DEU","FRA"] if name=="desktop" else ["SGP","NOR","KOR"]):
                await page.click(f".cty button[data-cty={k}]"); await page.wait_for_timeout(400)
                breath = await T(page, "#x-breath"); i = breath.find("firms"); print(f"[{name}] {k} breath: ...{breath[max(0,breath.find(' firms')-40):breath.find(' firms')+260]}")
                tiles = await page.evaluate("Array.from(document.querySelectorAll('#x-tiles .tile')).map(t=>t.textContent.replace(/\\s+/g,' ').trim()).filter(t=>/firms/i.test(t))"); print(f"[{name}] {k} tile: {tiles[0][:230] if tiles else 'NONE'}")
                svg = await page.evaluate("(()=>{const s=document.querySelector('#x-adopt svg'); return s?{circles:s.querySelectorAll('circle').length, lines:s.querySelectorAll('polyline').length, labels:Array.from(s.querySelectorAll('text')).map(t=>t.textContent).filter(t=>/US,|Norway|Singapore|Japan|Canada|Denmark/.test(t))}:null})()"); print(f"[{name}] {k} chart: {svg}")
                lead_card = await page.evaluate("(()=>{const g=document.getElementById('xcard-generic'); if(g&&!g.hidden){ const li=document.querySelectorAll('#xg-added li')[1]; return 'generic: '+(li?li.textContent:''); } const m={AUS:'x-au-lead',GBR:'x-uk-lead',CAN:'x-ca-lead',NZL:'x-nz-lead',NOR:'x-no-lead'}; const id=m[arguments[0]]; const el=id?document.getElementById(id):null; return el?('card: '+el.textContent):'n/a'; })()".replace("arguments[0]", f"'{k}'")); print(f"[{name}] {k} {lead_card[:260]}")
            if name=="desktop":
                await page.click(".cty button[data-cty=NOR]"); await page.wait_for_timeout(300)
                await page.locator("#x-adopt-panel").screenshot(path=f"{OUT}/leads-chart-{name}.png")
                note = await T(page, "#x-adopt-note"); print(f"[{name}] chart note: {note[:400]}")
            else:
                await page.locator("#x-adopt-panel").screenshot(path=f"{OUT}/leads-chart-{name}.png")
            await page.add_script_tag(content=AXE); r = await page.evaluate(RUN); print(f"[{name}] axe summary:", r["v"] or "0 violations")
            # compare
            await page.click("nav.tabs button[data-view=compare]"); await page.wait_for_timeout(400)
            rows = await page.evaluate("Array.from(document.querySelectorAll('#cmp-table tbody tr')).map(tr=>Array.from(tr.children).map(td=>td.textContent.trim()))")
            for r_ in rows:
                if r_[0].startswith("Position") or r_[0].startswith("Half of firms"): print(f"[{name}] compare row: {r_}")
            lede = await T(page, "#cmp-lede"); print(f"[{name}] compare lede: ...{lede[lede.find('Timing differs'):][:330]}")
            dec = await T(page, "#cmp-decomp"); print(f"[{name}] decomp: ...{dec[dec.find('from timing'):][:160]}")
            # europe
            await page.click("nav.tabs button[data-view=europe]"); await page.wait_for_timeout(500)
            hdr = await page.evaluate("Array.from(document.querySelectorAll('#eu-table thead th')).map(t=>t.textContent)"); print(f"[{name}] europe headers: {hdr}")
            cell = await page.evaluate("(()=>{const td=document.querySelectorAll('#eu-table tbody tr')[0].children[4]; return [td.textContent, (td.querySelector('span')||{}).title]})()"); print(f"[{name}] europe first lead cell: {cell}")
            eup = await T(page, "#eu-p"); print(f"[{name}] eu-p: ...{eup[eup.find('read against')-10:][:330]}")
            det = await T(page, "#eu-detail p"); print(f"[{name}] eu-detail: ...{det[det.find('firms run'):][:220]}")
            await page.click("nav.tabs button[data-view=method]"); await page.wait_for_timeout(300)
            await page.evaluate("document.querySelectorAll('#view-method details').forEach(d=>d.open=true)")
            mt = await T(page, "#view-method"); print(f"[{name}] method: ...{mt[mt.find('Singapore firms run'):][:260]}")
            print(f"[{name}] overflow:", await page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth"), "| errors:", errs)
            await ctx.close()
        await b.close()
asyncio.run(main())
