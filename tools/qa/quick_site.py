"""Rebuild index.html, kt.json and us_tasks.json without the search assets (fast QA loop)."""
import json, sys
sys.path.insert(0, "/home/user/tss2026")
from pathlib import Path
from onetx.site import build_data, SITE, TEMPLATE
data = build_data("https://claude.ai/artifact/Pxd31ubFiwTQjdTSBsu7qZ")
key_tasks = data.pop("_key_tasks"); us_tasks = data.pop("_us_tasks"); country_tasks = data.pop("_country_tasks") or {}
js = json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
meaning_js = Path("/home/user/tss2026/onetx/site_meaning.js").read_text().replace("if(typeof module!=='undefined') module.exports=Meaning;", "")
(SITE / "index.html").write_text(TEMPLATE.read_text().replace("__DATA__", js).replace("__MEANING_JS__", meaning_js))
(SITE / "kt.json").write_text(json.dumps(key_tasks, ensure_ascii=True, separators=(",", ":")))
if us_tasks is not None: (SITE / "us_tasks.json").write_text(json.dumps(us_tasks, ensure_ascii=True, separators=(",", ":")))
for code, files in country_tasks.items(): (SITE / f"{code.lower()}_tasks.json").write_text(json.dumps(files, ensure_ascii=True, separators=(",", ":")))
print("countries:", list((data.get("c") or {}).keys()), {k: len(v["occ"]) for k, v in (data.get("c") or {}).items()})
print("quick site ok; Infinity:", js.count("Infinity"), "| NaN:", js.count("NaN"))
