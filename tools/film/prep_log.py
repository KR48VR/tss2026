"""Export the step-by-step record from the companion page (site/companion.html) for the teaching film:
one entry per step, in order, with who carried it out and how the two worked (tool, teammate, independent)."""
import html, json, pathlib, re

root = pathlib.Path(__file__).resolve().parents[2]
t = (root / "site/companion.html").read_text()
i = t.index("The workflow, step by step"); j = t.index("</table>", i)
steps = []
for r in re.findall(r"<tr>(.*?)</tr>", t[i:j], re.S):
    c = [html.unescape(re.sub(r"<[^>]+>", "", x)).strip() for x in re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)]
    if len(c) < 5: continue
    kinds = [k.strip().lower() for k in c[4].split(", then")]
    steps.append({"n": int(c[0]), "who": "king" if c[2].startswith("King") else "claude", "kinds": kinds})
out = root / "tools/film/data/log.json"
out.write_text(json.dumps({"source": "site/companion.html, The workflow, step by step", "steps": steps}))
print(len(steps), "steps ->", out)
