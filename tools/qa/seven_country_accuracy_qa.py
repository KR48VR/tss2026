"""Accuracy audit for the Australia, UK, Canada, New Zealand and Norway pages, and the Europe table: every headline number on the page must equal the processed
tables and the build stats it was built from."""
import json, re, sys
sys.path.insert(0, "/home/user/tss2026")
import pandas as pd
from onetx.io import load
from onetx.config import PROCESSED
from onetx import countries as cn
html = open("/home/user/tss2026/site/index.html").read()
D = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1).replace("<\\/", "</"))
stats = json.loads((PROCESSED / "build_stats.json").read_text())
problems = []
def check(label, a, b, tol=0.6):
    ok = (a is None and (b is None or (isinstance(b, float) and b != b))) or (a is not None and b is not None and b == b and abs(float(a) - float(b)) <= tol)
    print(("ok  " if ok else "BAD ") + f"{label}: page {a} vs table {b}")
    if not ok: problems.append(label)
for code, spec in cn.SPECS.items():
    c = (D.get("c") or {}).get(code)
    if not c: print("MISSING payload", code); problems.append(code); continue
    p = spec.prefix; cs = stats["countries"][code]
    econ = load(f"{p}economy_exposure"); e = econ[(econ.scope == "economy")].set_index("weighting")
    w = load(f"{p}weights"); t = load(f"{p}weighted")
    ew = next(x for x in c["economy"] if x["scope"] == "economy" and x["w"] == "wage"); ee = next(x for x in c["economy"] if x["scope"] == "economy" and x["w"] == "emp")
    check(f"{code} AI doing by 2035 (wage, central)", ew["db"][9], 100 * e.loc["wage_value", "done_by_2035"])
    check(f"{code} AI involved by 2035 (wage, central)", ew["sb"][9], 100 * e.loc["wage_value", "share_by_2035"])
    check(f"{code} quarter-done year (wage, central)", ew["hq"][1], e.loc["wage_value", "quarter_done_year_central_from_curve"], 0.06)
    check(f"{code} seen in AI use (emp)", ee["so"], 100 * e.loc["employment", "share_weight_observed"])
    check(f"{code} employed total (thousands)", ee["total"], w.employment_thousands.sum(), 1.0)
    check(f"{code} occupation rows", len(c["occ"]), len(w), 0)
    check(f"{code} lead years", c["lead"], cs["lead"]["lead_years"], 0.06)
    if cs["lead"].get("proxy"): print(f"{code}: lead borrowed from {cs['lead']['proxy']}; page says lead_proxy={c.get('lead_proxy')}, adoption_now={c['adoption_now']}, adoption points={len(c['adoption'])}"); problems.extend([f"{code} proxy not marked"] if c.get("lead_proxy") != cs["lead"]["proxy"] or c["adoption"] else [])
    else: check(f"{code} adoption now", c["adoption_now"], cs["lead"]["latest_pct"], 0.06)
    iv = load(f"{p}industry_view"); iv = iv.set_index("industry") if len(iv) else iv
    if not len(iv): print(f"{code}: no industry lens (expected for New Zealand); industries payload:", c["industries"])
    for ind in list(iv.index)[:4]:
        row = next(x for x in c["industries"] if x["code"] == ind)
        check(f"{code} industry {ind} AI use latest", row["ai"], iv.loc[ind, "ai_use_latest_pct"] if pd.notna(iv.loc[ind, "ai_use_latest_pct"]) else None, 0.06)
        check(f"{code} industry {ind} shift", row["shift"], iv.loc[ind, "anchor_shift_years"], 0.06)
        check(f"{code} industry {ind} AI doing 2035", row["db35"], 100 * iv.loc[ind, "done_by_2035"])
    # a sample occupation: the largest
    big = t.sort_values("employment_thousands", ascending=False).iloc[0]
    o = next(x for x in c["occ"] if x["c"] == big.nat_code)
    check(f"{code} largest occupation {big.nat_code} employment", o["emp"], big.employment_thousands, 0.06)
    check(f"{code} largest occupation AI doing 2035", (o["db"] or [None]*15)[9], 100 * big.done_by_2035 if pd.notna(big.done_by_2035) else None)
    check(f"{code} largest occupation quarter-done central", o["hq"][1], big.quarter_done_year_p50 if pd.notna(big.quarter_done_year_p50) else None, 0.06)
    sh = load(f"{p}adoption_shift_national").set_index("nat_code").adoption_shift_years
    check(f"{code} largest occupation shift", o["shift"], sh.get(big.nat_code), 0.06)
    outc = load(f"{p}outcomes").set_index(spec.code_col)
    if big.nat_code in outc.index and o["oc"]:
        check(f"{code} largest occupation employment change %", o["oc"][2], outc.loc[big.nat_code, "emp_change_pct"], 0.06)
    bands = load(f"{p}skill_level_bands").set_index("level")
    for b in c["levels_national"]["bands"][:2]:
        check(f"{code} skill level {b[0]} AI doing 2035", b[6], 100 * bands.loc[b[0], "done_by_2035"])
    check(f"{code} national level rho", c["levels_national"]["rho"], cs["skill_levels_national"]["rho_level_vs_done_2035"], 0.006)
    tasks = json.load(open(f"/home/user/tss2026/site/{code.lower()}_tasks.json"))
    links = load(f"{p}task_onet_task"); n1 = int((links["rank"] == 1).sum())
    check(f"{code} task items in file", sum(len(g[1]) for v in tasks["occ"].values() for g in v["g"]), n1, 0)
    if code == "GBR":
        stds = load("uk_ifate_standards"); check("GBR standards", len(c["standards"]), len(stds), 0)
        rv = load("uk_ifate_route_view"); check("GBR routes", len(c["routes"]), len(rv), 0)
        dl = load("uk_ifate_duty_onet_task"); check("GBR duties in file", sum(len(v["d"]) for v in tasks["std"].values()), len(dl), 0)
    if code == "CAN":
        rl = load("ca_rating_level_exposure"); rc = load("ca_rating_level_correlations").set_index("element_name")
        ix = rl[rl.family == "index"].set_index("band"); LN = c["levels_nat"]
        for b in LN["index"]:
            if b: check(f"CAN OaSIS index band {b[0]} AI doing 2035", b[5], 100 * ix.loc[b[0], "done_by_2035"]); check(f"CAN OaSIS index band {b[0]} n", b[2], ix.loc[b[0], "n_occupations"], 0)
        check("CAN OaSIS index rho", LN["index_rho"], rc.loc[LN["index_name"], "rho_level_vs_done_2035"], 0.006)
        check("CAN OaSIS skills listed", len(LN["skills"]), rl[rl.family == "skill"].element_name.nunique(), 0)
        rt = load("ca_ratings"); titles = load("ca_titles")
        big_k = tasks["occ"][big.nat_code]["k"]; top = rt[(rt.noc_code == big.nat_code) & (rt.family == big_k[0][0]) & (rt.element_name == big_k[0][1])]
        check(f"CAN largest occupation top rating {big_k[0][1]}", big_k[0][2], float(top.level.iloc[0]) if len(top) else None, 0.06)
        check("CAN codes with a skills list", sum(1 for v in tasks["occ"].values() if v.get("k")), rt.noc_code.nunique(), 0)
        check("CAN codes with a description", sum(1 for v in tasks["occ"].values() if v.get("desc")), load("ca_descriptions").noc_code.nunique(), 0)
        S = json.load(open("/home/user/tss2026/site/search.json")); S["nat"].get("CAN") is None and print("skip CAN search (search.json not rebuilt yet)") or check("CAN search alt titles", len(S["nat"]["CAN"]), len(titles[titles.title_type != "principal"].drop_duplicates(["noc_code", "title"])), 400)
        lab = c["outcomes"].get("labels") or {}; print("CAN outcome labels:", lab)
    # compare block
    cc = D["compare"]["countries"][code]
    check(f"{code} compare lead", cc["lead"], cs["lead"]["lead_years"], 0.06)
    dec = (cs.get("decompositions") or {})
    for k, v in dec.items():
        if not k.endswith("wage_value"): continue
        pv = D["compare"]["pairs"][k]["by_year"]["central_2035"]
        check(f"{code} pair {k} gap", 100 * pv["gap"], 100 * v["by_year"]["central_2035"]["gap"], 0.06)
        check(f"{code} pair {k} parts add up", 100 * (pv["job_mix"] + pv["timing"]), 100 * pv["gap"], 0.016)
# Europe at a glance: every country row against eu_economy and the stats
E = D.get("europe")
if E:
    ee = load("eu_economy"); es = stats.get("europe") or {}
    check("Europe countries on page", len(E["countries"]), sum(1 for k, v in es.items() if isinstance(v, dict) and "lead" in v), 0)
    for c in E["countries"][:40]:
        e = ee[(ee.geo == c["geo"]) & (ee.scope == "economy")].set_index("weighting")
        check(f"EU {c['geo']} AI doing by 2035 (wage, central)", c["econ"]["wage"]["db"][9], 100 * e.loc["wage_value", "done_by_2035"])
        check(f"EU {c['geo']} quarter-done (wage, central)", c["econ"]["wage"]["hq"][1], e.loc["wage_value", "quarter_done_year_central_from_curve"], 0.06)
        check(f"EU {c['geo']} lead", c["lead"], es[c["geo"]]["lead"]["lead_years"], 0.06)
        check(f"EU {c['geo']} groups listed", len(c["groups"]), int(load("eu_groups").query("geo == @c['geo']").done_by_2035.notna().sum()), 0)
else:
    print("no Europe payload")
print("\nPROBLEMS:", len(problems), problems)
