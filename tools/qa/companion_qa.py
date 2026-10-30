import asyncio, json
from playwright.async_api import async_playwright
import os; OUT = os.environ.get("QA_OUT", "/home/user/tss2026/tools/qa/out"); os.makedirs(OUT, exist_ok=True)
AXE=open("/home/user/tss2026/tools/qa/node_modules/axe-core/axe.min.js").read()
RUN="axe.run(document, {runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21aa','best-practice']}}).then(r=>({v:r.violations.map(v=>v.id+':'+v.impact+':'+v.nodes.slice(0,3).map(n=>n.target.join(' ')).join('|')), passes:r.passes.length}))"
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, executable_path="/opt/pw-browsers/chromium-1243/chrome-linux64/chrome", args=["--no-sandbox"])
        for label,w,scheme in [("desk",1280,"light"),("phone",400,"dark")]:
            ctx = await b.new_context(viewport={"width":w,"height":900}, color_scheme=scheme); page = await ctx.new_page()
            errs=[]; page.on("pageerror", lambda e: errs.append(str(e)))
            await page.goto("http://127.0.0.1:8765/companion.html", wait_until="load"); await page.wait_for_timeout(500)
            await page.evaluate("document.documentElement.lang='en'")
            await page.add_script_tag(content=AXE)
            print(label, "overflow:", await page.evaluate("document.documentElement.scrollWidth > window.innerWidth"), "errors:", errs)
            print(label, json.dumps(await page.evaluate(RUN)))
            await page.screenshot(path=f"{OUT}/companion-{label}.png", full_page=(label=="phone"), clip=None if label=="phone" else {"x":0,"y":0,"width":1280,"height":1600})
            await ctx.close()
        await b.close()
asyncio.run(main())
