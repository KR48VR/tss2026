"""Build a self-contained web page (site/index.html) from the processed tables so that anyone can
browse the results without running Python. `python -m onetx site`."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED, ROOT
from .forecast import YEARS
from .io import load

SITE = ROOT / "site"
TEMPLATE = Path(__file__).with_name("site_template.html")
STATUS_CODE = {"observed_automation": 0, "observed_augmentation": 1, "physical_robotics": 2, "not_observed": 3}
BARRIER_CODE = {"none": 0, "interpersonal": 1, "in_person": 2, "accountability": 3, "handover": 4, "physical": 5}
TIERS = ["seen", "rated", "newly_rated", "mixed", "unproven", "physical"]
TIER_CODE = {t: i for i, t in enumerate(TIERS)}
AUTONOMY_CODE = {"A0": 0, "A1": 1, "A2": 2, "A3": 3}
RUBRIC_BARRIERS = ["none", "physical", "in_person", "relationship", "accountability", "system_access", "judgement", "data_absent"]
RBARRIER_CODE = {b: i for i, b in enumerate(RUBRIC_BARRIERS)}


def _r(x, nd=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), nd) if nd else int(round(float(x)))


def _pct(x):
    return _r(100 * x) if x is not None and not pd.isna(x) else None


def _sb(row):
    return [_pct(row.get(f"share_by_{y}")) for y in YEARS]


def _half(row):
    return [_r(row.get(f"half_year_{q}"), 1) for q in ("p10", "p50", "p90")]


def _sbq(row, q):
    if row is None:
        return None
    vals = [row.get(f"share_by_{y}_{q}") for y in YEARS]
    return [_pct(v) for v in vals] if any(pd.notna(v) for v in vals) else None


def _tiers(row):
    if row is None:
        return None
    vals = [row.get(f"share_tier_{t}") for t in TIERS]
    return [_pct(v) for v in vals] if any(pd.notna(v) for v in vals) else None


def fmt_int(v):
    try:
        return f"{int(round(float(v))):,}"
    except (TypeError, ValueError):
        return "\u2013"


def _pc(v):
    try:
        return f"{round(float(v) * 100)}%"
    except (TypeError, ValueError):
        return "\u2013"


def fmt1(v):
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "\u2013"


def _code(mapping, v):
    return mapping.get(v) if isinstance(v, str) else None


def build_data(companion_url: str | None = None) -> dict:
    stats = json.loads((PROCESSED / "build_stats.json").read_text())
    econ = load("sg_economy_exposure")
    ssocw = load("sg_ssoc2024_weighted")
    fssoc = load("forecast_ssoc2024").set_index("ssoc_code")
    roles = load("frontier_sfw_roles")
    froles = load("forecast_sfw_roles").set_index("role_id")
    role_ssoc = load("xw_sfw_role_ssoc2024")
    kt_scores = load("frontier_sfw_key_tasks")
    kt = load("sfw_key_tasks").set_index("key_task_id")
    ft = load("forecast_tasks").set_index("task_id")
    tasks = load("onet_tasks").set_index("task_id")
    occ = load("frontier_occupations")
    focc = load("forecast_occupations").set_index("onet_soc_code")
    o2s = load("xw_onetsoc_ssoc2024")
    skills = load("frontier_skills").merge(load("forecast_skills").drop(columns=["family", "element_name", "n_occupations", "onet_version"]), on="element_id", how="left")
    act = load("frontier_activities").merge(load("forecast_activities").drop(columns=["name", "level", "n_tasks"]), on="activity_id", how="left")
    sectors = load("sg_sfw_sectors")
    sh_ssoc = load("sg_adoption_shift_ssoc2024").set_index("ssoc_code").adoption_shift_years if (PROCESSED / "sg_adoption_shift_ssoc2024.parquet").exists() else pd.Series(dtype=float)
    sh_role = load("sg_adoption_shift_sfw_roles").set_index("role_id").adoption_shift_years if (PROCESSED / "sg_adoption_shift_sfw_roles.parquet").exists() else pd.Series(dtype=float)
    sh_onet = load("sg_adoption_shift_onet").set_index("onet_soc_code").adoption_shift_years if (PROCESSED / "sg_adoption_shift_onet.parquet").exists() else pd.Series(dtype=float)

    # ---- economy
    economy = []
    for _, r in econ.iterrows():
        economy.append({"scope": r.scope, "level": r.level, "name": r.scope_name, "w": "emp" if r.weighting == "employment" else "wage", "total": _r(r.weight_total / (1 if r.weighting == "employment" else 1000), 1), "oa": _pct(r.observed_automation), "so": _pct(r.share_weight_observed), "sp": _pct(r.share_weight_physical), "sb": _sb(r), "sb10": [_pct(r.get(f"share_by_{y}_p10")) for y in YEARS], "sb90": [_pct(r.get(f"share_by_{y}_p90")) for y in YEARS], "half": [_r(r.get("half_year_fast_from_curve", r.half_year_p10), 1), _r(r.half_year_central_from_curve, 1), _r(r.get("half_year_slow_from_curve", r.half_year_p90), 1)]})

    # ---- SSOC occupations
    occ_title = occ.set_index("onet_soc_code").title
    top_onet = {}
    for code, g in o2s.sort_values("weight_rev", ascending=False).groupby("ssoc_code"):
        top_onet[code] = [[occ_title.get(c, c), _pct(w)] for c, w in zip(g.onet_soc_code.head(4), g.weight_rev.head(4))]
    role_index = {rid: i for i, rid in enumerate(roles.role_id)}
    roles_by_ssoc = role_ssoc.groupby("ssoc_code").role_id.apply(lambda s: [role_index[r] for r in s if r in role_index]).to_dict()
    ssoc_list = []
    for _, r in ssocw.sort_values("ssoc_code").iterrows():
        f = fssoc.loc[r.ssoc_code] if r.ssoc_code in fssoc.index else None
        ssoc_list.append({"c": r.ssoc_code, "t": r.ssoc_title, "mg": r.major_group, "sm": r.submajor_label, "smn": r.submajor_name, "emp": _r(r.employment_thousands, 1), "wage": _r(r.wage_median), "ws": r.wage_source, "oa": _pct(r.observed_automation), "so": _pct(r.share_weight_observed), "sp": _pct(r.share_weight_physical), "ai": _r(r.ai_applicability_score, 2), "sb": _sb(r) if f is not None else None, "sb10": _sbq(f, "p10"), "sb90": _sbq(f, "p90"), "tiers": _tiers(f), "shift": _r(sh_ssoc.get(r.ssoc_code), 1) if r.ssoc_code in sh_ssoc.index else None, "h": _half(r), "bi": _r(f.barrier_index, 2) if f is not None else None, "onet": top_onet.get(r.ssoc_code, []), "roles": roles_by_ssoc.get(r.ssoc_code, [])})

    # ---- Skills Framework roles with key tasks
    ks = kt_scores.merge(kt[["key_task", "cwf"]], left_on="key_task_id", right_index=True, how="left")
    ft_cols = ["status", "disrupt_year_p50", "disrupt_year_p10", "disrupt_year_p90", "barrier_type"] + [c for c in ["evidence_tier", "autonomy_2026", "barrier_2026"] if c in ft.columns]
    ks = ks.merge(ft[ft_cols], left_on="top_task_id", right_index=True, how="left")
    for c in ["evidence_tier", "autonomy_2026", "barrier_2026"]:
        if c not in ks.columns:
            ks[c] = None
    ks["task_text"] = ks.top_task_id.map(tasks.task_statement)
    kt_by_role = {rid: g for rid, g in ks.groupby("role_id")}
    ssoc_by_role = {rid: [[c, s, _r(sc, 0)] for c, s, sc in zip(g.ssoc_code, g.source, g.match_score)] for rid, g in role_ssoc.groupby("role_id")}
    ssoc_title = ssocw.set_index("ssoc_code").ssoc_title
    role_list, key_tasks = [], []
    for _, r in roles.iterrows():
        f = froles.loc[r.role_id] if r.role_id in froles.index else None
        g = kt_by_role.get(r.role_id)
        groups = []
        if g is not None:
            for cwf, gg in g.groupby(g.cwf.fillna(""), sort=False):
                items = [[str(k.key_task)[:150], STATUS_CODE.get(k.status, 3), _r(k.disrupt_year_p50, 1), BARRIER_CODE.get(k.barrier_type, 0), _r(k.top_score, 2), str(k.task_text)[:110] if isinstance(k.task_text, str) else None, _code(TIER_CODE, k.evidence_tier), _code(AUTONOMY_CODE, k.autonomy_2026), _code(RBARRIER_CODE, k.barrier_2026)] for _, k in gg.sort_values("disrupt_year_p50").iterrows()]
                groups.append([str(cwf)[:90], items])
        key_tasks.append(groups)
        role_list.append({"sector": r.sector, "track": r.track if isinstance(r.track, str) else "", "role": r.job_role, "desc": (r.job_role_description or "")[:360] if isinstance(r.job_role_description, str) else "", "ssoc": [[c, s, sc, ssoc_title.get(c, "")] for c, s, sc in ssoc_by_role.get(r.role_id, [])], "oat": _pct(r.observed_automation_via_tasks), "oas": _pct(r.observed_automation_via_ssoc), "sko": _pct(r.share_key_tasks_observed), "skp": _pct(r.share_key_tasks_physical), "nk": _r(r.n_key_tasks), "ms": _r(r.mean_match_score, 2), "sb": _sb(f) if f is not None else None, "sb10": _sbq(f, "p10"), "sb90": _sbq(f, "p90"), "tiers": _tiers(f), "shift": _r(sh_role.get(r.role_id), 1) if r.role_id in sh_role.index else None, "h": _half(f) if f is not None else [None, None, None], "bi": _r(f.barrier_index, 2) if f is not None else None})

    # ---- O*NET occupations
    onet_list = []
    for _, r in occ.sort_values("title").iterrows():
        f = focc.loc[r.onet_soc_code] if r.onet_soc_code in focc.index else None
        onet_list.append({"c": r.onet_soc_code, "t": r.title, "oa": _pct(r.observed_automation), "so": _pct(r.share_weight_observed), "sp": _pct(r.share_weight_physical), "ms": _r(r.ai_applicability_score, 2), "oe": _r(r.openai_exposure_beta_gpt4, 2), "g": r.google_atlas_flag if isinstance(r.google_atlas_flag, str) else None, "np": _r(r.n_providers_positive), "h": _half(f) if f is not None else [None, None, None], "sb": _sb(f) if f is not None else None, "sb10": _sbq(f, "p10"), "sb90": _sbq(f, "p90"), "tiers": _tiers(f), "shift": _r(sh_onet.get(r.onet_soc_code), 1) if r.onet_soc_code in sh_onet.index else None, "gd": bool(r.gdpval_covered)})

    skills_list = [{"f": r.family, "name": r.element_name, "n": _r(r.n_occupations), "oa": _pct(r.via_occupations_observed_automation), "oaa": _pct(r.via_activities_observed_automation), "h": _half(r), "sb": _sb(r), "sb10": _sbq(r, "p10"), "sb90": _sbq(r, "p90")} for _, r in skills.sort_values(["family", "element_name"]).iterrows()]
    gwa_list = [{"name": r["name"], "n": _r(r.n_tasks), "so": _pct(r.share_tasks_observed), "oa": _pct(r.observed_automation), "ms": _r(r.ms_share * 100, 1) if pd.notna(r.ms_share) else None, "h": _half(r), "sb": _sb(r), "sb10": _sbq(r, "p10"), "sb90": _sbq(r, "p90")} for _, r in act[act.level == "gwa"].sort_values("observed_automation", ascending=False).iterrows()]
    sector_list = [{"sector": r.sector, "n": _r(r.n_roles), "oat": _pct(r.observed_automation_via_tasks), "sko": _pct(r.share_key_tasks_observed), "skp": _pct(r.share_key_tasks_physical), "sb30": _pct(r.share_by_2030), "sb35": _pct(r.share_by_2035), "sb30f": _pct(r.get("share_by_2030_p10")), "sb30s": _pct(r.get("share_by_2030_p90")), "sb35f": _pct(r.get("share_by_2035_p10")), "sb35s": _pct(r.get("share_by_2035_p90")), "h": [_r(r.get("half_year_p10_median_role"), 1), _r(r.half_year_p50_median_role, 1), _r(r.half_year_p90_median_role, 1)]} for _, r in sectors.iterrows()]

    fp = stats.get("forecast_params", {})
    val = stats.get("validation") or {}
    lad = dict(val.get("ladder") or {})
    sg_ad = load("sg_firm_adoption_imda") if (PROCESSED / "sg_firm_adoption_imda.parquet").exists() else None
    if sg_ad is not None and len(sg_ad):
        allrows = sg_ad[sg_ad.group == "all"].sort_values("year")
        lad["adoption_now"] = float(allrows.pct.iloc[-1]) / 100.0
        lad["adoption_now_year"] = f"IMDA, {int(allrows.year.iloc[-1])}"
    lad["adoption_half_year"] = (stats.get("adoption_anchor_years") or {}).get("central", 0) - (stats.get("geo_lead_years_central") or 0)
    lad["usage_to"] = "May 2026"
    swo = occ.share_weight_observed[occ.share_weight_observed > 0]
    lad["job_task_share"] = float(swo.median()) if len(swo) else None
    occ_corr = {}
    for name, col in [("anthropic", "aei_job_exposure"), ("microsoft", "ai_applicability_score")]:
        if col in occ.columns:
            d2 = occ[["observed_automation", col]].dropna()
            if len(d2) > 30:
                occ_corr[name] = float(d2.corr().iloc[0, 1])
    def _a(key, sub):
        v = (val.get(key) or {}).get(sub)
        return None if v is None else round(float(v), 2)
    agr = val.get("agreement") or {}
    rep_ = val.get("repeat_2026") or {}
    aging = val.get("aging_of_2023_no") or {}
    sources = [
        {"name": "OpenAI, \"GPTs are GPTs\", human raters", "rung": 1, "what": "rated each task: could AI halve the time?", "unit": "task", "date": "2023", "record": f"{_a('auc_seen','human_2023')} at telling which desk tasks would be seen in use by 2026; {_pc(aging.get('share_seen_by_2026'))} of its \"no\" tasks were seen anyway", "caution": "a \"no\" from 2023 is a flag, not a verdict"},
        {"name": "OpenAI, \"GPTs are GPTs\", GPT-4 ratings", "rung": 1, "what": "the same rubric, rated by GPT-4", "unit": "task", "date": "2023", "record": f"{_a('auc_seen','gpt4_2023')} at seen-in-use; {_a('auc_doing_given_seen','gpt4_2023')} at AI-mostly-doing among seen tasks; agreed with the human raters on {_pc((agr.get('human_vs_gpt4_2023') or {}).get('agree'))} of tasks", "caution": "a model rating its own kind"},
        {"name": "Claude rating, October 2026 (this project)", "rung": 1, "what": "could AI halve the time; could an agent do it end to end; main barrier", "unit": "task", "date": "Oct 2026", "record": f"{_a('auc_seen','exposure_2026')} at seen-in-use; the end-to-end rating scores {_a('auc_doing_given_seen','autonomy_2026')} at AI-mostly-doing, the best of any rating; agrees with the 2023 human raters on {_pc((agr.get('claude2026_vs_human_2023') or {}).get('agree'))}; a repeat run gave the same yes/no on {_pc(rep_.get('exposure_yes_no_same'))}", "caution": "tasks only this rating calls doable are seen in use least, so they wait the longest lag"},
        {"name": "Anthropic Economic Index", "rung": 2, "what": "share of Claude conversations by task; whether AI does or helps", "unit": "task", "date": "Dec 2024 to May 2026", "record": "the map's usage base; the other sources are scored against it, so it cannot be scored against itself", "caution": "reflects who uses Claude: office and software work first"},
        {"name": "Microsoft, \"Working with AI\"", "rung": 2, "what": "Copilot usage by work activity; an applicability score per occupation", "unit": "activity, occupation", "date": "2024 conversations, published 2025", "record": f"occupation scores correlate {round(occ_corr.get('microsoft', float('nan')),2)} with this map's usage-based score" if 'microsoft' in occ_corr else "second opinion at activity level", "caution": "activity level, not task level; a different user base"},
        {"name": "OpenAI, \"How people use ChatGPT\"", "rung": 2, "what": "shares of messages by work activity", "unit": "activity", "date": "Jul 2025", "record": "not testable at task level; used as a second opinion on which activities dominate", "caution": "consumer ChatGPT, mostly non-work use"},
        {"name": "Google ATLAS", "rung": 2, "what": "lists of occupations over- and under-represented in Gemini use", "unit": "occupation", "date": "Jul 2026", "record": f"one of four providers in the agreement count; {fmt_int((stats.get('provider_consensus_counts') or {}).get('3',0)+(stats.get('provider_consensus_counts') or {}).get('4',0))} occupations have three or more providers agreeing", "caution": "lists only, no numbers per task"},
        {"name": "Anthropic occupation exposure", "rung": 2, "what": "occupation-level observed exposure", "unit": "occupation", "date": "2026", "record": f"correlates {round(occ_corr.get('anthropic', float('nan')),2)} with this map's task-weighted score (same data, different method)" if 'anthropic' in occ_corr else "same data, different method", "caution": "shares the usage base's blind spots"},
        {"name": "US Census Business Trends and Outlook Survey", "rung": 4, "what": "share of firms using AI in the last two weeks, every two weeks", "unit": "firm", "date": (stats.get('data_as_of') or {}).get('btos_latest_reading', ''), "record": "a copy frozen at Aug 2025 predicted 15% of firms for Oct 2026; the Census reported 24.6%. Now refreshed on every rebuild", "caution": "the question was reworded in Nov 2025; the fit carries a step"},
        {"name": "IMDA Singapore Digital Economy Reports", "rung": 4, "what": "share of Singapore firms that used AI in the year", "unit": "firm", "date": "2023 to 2025", "record": f"sets Singapore's measured lead over the US ({fmt1(stats.get('geo_lead_years_central'))} years); its sector ordering agrees with the Census sector series (rank correlation {abs(round(((stats.get('sector_adoption') or {}).get('imda_check') or {}).get('spearman_imda2023_vs_us_shift', 0),2))})", "caution": "three yearly points; a looser question than the Census's"},
        {"name": "METR time horizons", "rung": 1, "what": "how long a task AI agents can complete, by model release", "unit": "benchmark", "date": "models to 2026", "record": f"sets the capability doubling time ({fmt_int(fp.get('doubling_days_central'))} days); not testable against usage here", "caution": "software and research tasks, not the whole economy"},
        {"name": "Headlines: \"X% of jobs exposed\"", "rung": 1, "what": "usually a rating applied to occupations", "unit": "occupation", "date": "varies", "record": "use the translator above: a rung-one job claim has historically meant a fraction of that in work actually seen, and less again with AI doing it", "caution": "check whether the report measured use or rated possibility"},
    ]
    meta = {
        "built": date.today().isoformat(), "now": stats.get("now"), "years": YEARS,
        "adoption_anchor": {k: _r(v, 1) for k, v in stats.get("adoption_anchor_years", {}).items()},
        "geo_lead": _r(stats.get("geo_lead_years_central"), 1),
        "delegation_horizon_hours": _r(fp.get("delegation_horizon_now_min", 0) / 60, 1),
        "doubling_days": fp.get("doubling_days_central"),
        "task_status": stats.get("task_status_counts", {}), "barrier_types": stats.get("barrier_type_counts", {}),
        "crosswalk": stats.get("crosswalk_counts", {}), "sg_weights": stats.get("sg_weights", {}),
        "singapore_economy": stats.get("singapore_economy", {}),
        "usage_ratio": stats.get("singapore_usage_ratio", {}),
        "companion_url": companion_url,
        "data_as_of": stats.get("data_as_of", {}),
        "adoption": {"us_half": {k: _r(v, 1) for k, v in stats.get("adoption_anchor_years", {}).items()}, "sg_half": {k: _r(v - (stats.get("geo_lead_years_central") or 0), 1) for k, v in stats.get("adoption_anchor_years", {}).items()}, "lead": _r(stats.get("geo_lead_years_central"), 1), "lead_years_used": (stats.get("singapore_firm_lead") or {}).get("years_used"), "btos_last_pct": (stats.get("btos_adoption_fit") or {}).get("last_pct"), "btos_last_date": (stats.get("btos_adoption_fit") or {}).get("last_ref_end"), "half_old_wording": _r((stats.get("btos_adoption_fit") or {}).get("year_cross_50pct_old_wording"), 1), "sg_own_half": _r((stats.get("singapore_firm_lead") or {}).get("sg_year_cross_50pct"), 1)},
        "evidence": {"tiers": (stats.get("evidence") or {}).get("tiers", {}), "lag_unproven": _r(((stats.get("evidence") or {}).get("evidence_lag_years", {}).get("unproven") or [None, None, None])[1], 1), "lag_mixed": _r(((stats.get("evidence") or {}).get("evidence_lag_years", {}).get("mixed") or [None, None, None])[1], 1), "lag_newly": _r(((stats.get("evidence") or {}).get("evidence_lag_years", {}).get("newly_rated") or [None, None, None])[1], 1), "ratings_2026_used": (stats.get("evidence") or {}).get("ratings_2026_used", False)},
        "sector_shift": (stats.get("sector_adoption") or {}).get("industry_shift_years", {}),
        "ladder": lad, "validation": {k: val.get(k) for k in ["auc_seen", "auc_doing_given_seen", "agreement", "repeat_2026", "aging_of_2023_no", "aging_of_2023_yes"]}, "sources": sources,
        "sector_labels": {"J": "Information & communications", "K": "Financial & insurance services", "M": "Professional services", "OP": "Public administration & education", "L": "Real estate services", "Q": "Health & social services", "N": "Administrative & support services", "R": "Arts, entertainment & recreation", "C": "Manufacturing", "G": "Wholesale & retail trade", "F": "Construction", "SU": "Other community, social & personal services", "ABDE": "Agriculture, mining, utilities", "H": "Transportation & storage", "I": "Accommodation & food services"},
    }
    return {"meta": meta, "economy": economy, "ssoc": ssoc_list, "roles": role_list, "onet": onet_list, "skills": skills_list, "gwa": gwa_list, "sectors": sector_list, "_key_tasks": key_tasks}


def build_site(companion_url: str | None = None) -> Path:
    """Write site/index.html (page + core data) and site/kt.json (key tasks per role, loaded on demand)."""
    SITE.mkdir(parents=True, exist_ok=True)
    data = build_data(companion_url)
    key_tasks = data.pop("_key_tasks")
    js = json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    meaning_js = Path(__file__).with_name("site_meaning.js").read_text().replace("if(typeof module!=='undefined') module.exports=Meaning;", "")
    html = TEMPLATE.read_text().replace("__DATA__", js).replace("__MEANING_JS__", meaning_js)
    out = SITE / "index.html"
    out.write_text(html)
    (SITE / "kt.json").write_text(json.dumps(key_tasks, ensure_ascii=True, separators=(",", ":")))
    from .site_search import build_search_assets

    stats = build_search_assets([s["c"] for s in data["ssoc"]], load("frontier_sfw_roles").role_id.tolist(), key_tasks)
    (SITE / "search_stats.json").write_text(json.dumps(stats))
    return out
