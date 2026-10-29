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
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
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


def _db(row, q=None):
    if row is None:
        return None
    suf = f"_{q}" if q else ""
    vals = [row.get(f"done_by_{y}{suf}") for y in YEARS]
    return [_pct(v) for v in vals] if any(pd.notna(v) for v in vals) else None


def _done_years(row, kind):
    if row is None:
        return [None, None, None]
    return [_r(row.get(f"{kind}_done_year_{q}"), 1) for q in ("p10", "p50", "p90")]


def _tiers(row):
    if row is None:
        return None
    vals = [row.get(f"share_tier_{t}") for t in TIERS]
    return [_pct(v) for v in vals] if any(pd.notna(v) for v in vals) else None


def _outcomes_meta(stats, outsub):
    o = stats.get("outcomes") or {}
    if not o or "tests" not in o:
        return None
    years = [int(c.split("_")[1]) for c in (outsub.columns if outsub is not None else []) if c.startswith("emp_") and c.split("_")[1].isdigit()]
    series = {}
    if outsub is not None:
        for _, r in outsub.iterrows():
            series[str(r.submajor_label).replace(" ", "")] = [_r(r.get(f"emp_{y}"), 1) for y in years]
    return {"tests": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()} for t in o["tests"]], "headline": o.get("headline"), "retrenchment": o.get("retrenchment"), "vacancy_rate": o.get("vacancy_rate"), "resignation_rate": o.get("resignation_rate"),
            "n_wages": o.get("n_ssoc_with_wages_both_years"), "n_vacancies": o.get("n_ssoc_with_vacancies_2025"), "n_minor": o.get("n_minor_groups_with_vacancy_shares"), "emp_years": years, "emp_series": series}


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


def _outcome(outc, code):
    """[vacancies Sep 2025, share of group 2023, share 2025, wage Jun 2024, wage Jun 2025, wage change %, sub-group employment change 2023-25 %, 2019-25 %, occupational group key]"""
    if outc is None or code not in outc.index:
        return None
    o = outc.loc[code]
    return [_r(o.get("vacancies_2025")), _r(o.get("share_2023"), 1), _r(o.get("share_2025"), 1), _r(o.get("wage_2024")), _r(o.get("wage_2025")), _r(o.get("wage_change_pct"), 1), _r(o.get("emp_change_2023_2025_pct"), 1), _r(o.get("emp_change_2019_2025_pct"), 1), o.get("occ_group") if isinstance(o.get("occ_group"), str) else None, str(o.get("submajor_label")) if pd.notna(o.get("submajor_label")) else None]


def _emp_change(outsub, scope):
    if outsub is None or not str(scope).startswith("submajor_"):
        return None
    key = str(scope).replace("submajor_", "").replace(" ", "")
    m = outsub[outsub.submajor_label.str.replace(" ", "") == key]
    if m.empty:
        return None
    r = m.iloc[0]
    return [_r(r.get("emp_change_2023_2025_pct"), 1), _r(r.get("emp_change_2019_2025_pct"), 1)]


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
    outc = load("sg_outcomes_ssoc2024").set_index("ssoc_code") if (PROCESSED / "sg_outcomes_ssoc2024.parquet").exists() else None
    outsub = load("sg_outcomes_submajor") if (PROCESSED / "sg_outcomes_submajor.parquet").exists() else None
    sh_ssoc = load("sg_adoption_shift_ssoc2024").set_index("ssoc_code").adoption_shift_years if (PROCESSED / "sg_adoption_shift_ssoc2024.parquet").exists() else pd.Series(dtype=float)
    sh_role = load("sg_adoption_shift_sfw_roles").set_index("role_id").adoption_shift_years if (PROCESSED / "sg_adoption_shift_sfw_roles.parquet").exists() else pd.Series(dtype=float)
    sh_onet = load("sg_adoption_shift_onet").set_index("onet_soc_code").adoption_shift_years if (PROCESSED / "sg_adoption_shift_onet.parquet").exists() else pd.Series(dtype=float)

    # ---- economy
    def _econ_row(r):
        return {"scope": r.scope, "level": r.level, "name": r.scope_name, "w": "emp" if r.weighting == "employment" else "wage", "total": _r(r.weight_total / (1 if r.weighting == "employment" else 1000), 1), "oa": _pct(r.observed_automation), "so": _pct(r.share_weight_observed), "sp": _pct(r.share_weight_physical), "sb": _sb(r), "sb10": [_pct(r.get(f"share_by_{y}_p10")) for y in YEARS], "sb90": [_pct(r.get(f"share_by_{y}_p90")) for y in YEARS], "half": [_r(r.get("half_year_fast_from_curve", r.half_year_p10), 1), _r(r.half_year_central_from_curve, 1), _r(r.get("half_year_slow_from_curve", r.half_year_p90), 1)],
                "db": _db(r), "db10": _db(r, "p10"), "db90": _db(r, "p90"), "cap": _pct(r.get("ceiling_ai_share")), "hq": [_r(r.get("quarter_done_year_fast_from_curve"), 1), _r(r.get("quarter_done_year_central_from_curve"), 1), _r(r.get("quarter_done_year_slow_from_curve"), 1)], "hd": [_r(r.get("half_done_year_fast_from_curve"), 1), _r(r.get("half_done_year_central_from_curve"), 1), _r(r.get("half_done_year_slow_from_curve"), 1)], "ech": _emp_change(outsub, r.scope)}
    economy = [_econ_row(r) for _, r in econ.iterrows()]

    # ---- the United States as a second workforce (when built)
    us_econ = load("us_economy_exposure") if (PROCESSED / "us_economy_exposure.parquet").exists() else None
    us_w = load("us_onet_weights").set_index("onet_soc_code") if (PROCESSED / "us_onet_weights.parquet").exists() else None
    focc_us = load("forecast_occupations_us").set_index("onet_soc_code") if (PROCESSED / "forecast_occupations_us.parquet").exists() else None

    def _us_match(code):
        """The SSOC code's heaviest O*NET occupation with US employment: [title, employed (thousands), median annual wage USD, quarter-done years (fast, central, slow), AI doing by 2035 (fast, central, slow), AI involved by 2035]."""
        if us_w is None or focc_us is None:
            return None
        # rank by the number of US workers this code accounts for (employment x the share of the O*NET
        # occupation that maps to it), so a large occupation split across several codes beats a small one
        g = o2s[o2s.ssoc_code == code].copy()
        emp = np.array([float(us_w.loc[c].employment_thousands) if c in us_w.index and pd.notna(us_w.loc[c].employment_thousands) else 0.0 for c in g.onet_soc_code], dtype=float)
        g["us_workers"] = emp * g.weight_rev.fillna(0).to_numpy(dtype=float)
        g = g.sort_values("us_workers", ascending=False)
        for c in g.onet_soc_code:
            if c in us_w.index and c in focc_us.index and pd.notna(us_w.loc[c].employment_thousands) and us_w.loc[c].employment_thousands > 0:
                f = focc_us.loc[c]; w = us_w.loc[c]
                return [w.onet_title, _r(w.employment_thousands, 1), _r(w.wage_median), [_r(f.get("quarter_done_year_p10"), 1), _r(f.get("quarter_done_year_p50"), 1), _r(f.get("quarter_done_year_p90"), 1)], [_pct(f.get("done_by_2035_p10")), _pct(f.get("done_by_2035")), _pct(f.get("done_by_2035_p90"))], _pct(f.get("share_by_2035")), _r(w.wage_top_coded) if "wage_top_coded" in w else 0]
        return None

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
        ssoc_list.append({"c": r.ssoc_code, "t": r.ssoc_title, "mg": r.major_group, "sm": r.submajor_label, "smn": r.submajor_name, "emp": _r(r.employment_thousands, 1), "wage": _r(r.wage_median), "ws": r.wage_source, "oa": _pct(r.observed_automation), "so": _pct(r.share_weight_observed), "sp": _pct(r.share_weight_physical), "ai": _r(r.ai_applicability_score, 2), "sb": _sb(r) if f is not None else None, "sb10": _sbq(f, "p10"), "sb90": _sbq(f, "p90"), "tiers": _tiers(f), "shift": _r(sh_ssoc.get(r.ssoc_code), 1) if r.ssoc_code in sh_ssoc.index else None, "h": _half(r), "db": _db(f), "db10": _db(f, "p10"), "db90": _db(f, "p90"), "cap": _pct(f.ceiling_ai_share) if f is not None else None, "hq": _done_years(f, "quarter"), "hd": _done_years(f, "half"), "oc": _outcome(outc, r.ssoc_code), "bi": _r(f.barrier_index, 2) if f is not None else None, "onet": top_onet.get(r.ssoc_code, []), "roles": roles_by_ssoc.get(r.ssoc_code, []), "us": _us_match(r.ssoc_code)})

    # ---- Skills Framework roles with key tasks
    ks = kt_scores.merge(kt[["key_task", "cwf"]], left_on="key_task_id", right_index=True, how="left")
    ft_cols = ["status", "disrupt_year_p50", "disrupt_year_p10", "disrupt_year_p90", "barrier_type"] + [c for c in ["evidence_tier", "autonomy_2026", "barrier_2026", "ai_share"] if c in ft.columns]
    ks = ks.merge(ft[ft_cols], left_on="top_task_id", right_index=True, how="left")
    for c in ["evidence_tier", "autonomy_2026", "barrier_2026", "ai_share"]:
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
                items = [[str(k.key_task)[:150], STATUS_CODE.get(k.status, 3), _r(k.disrupt_year_p50, 1), BARRIER_CODE.get(k.barrier_type, 0), _r(k.top_score, 2), str(k.task_text)[:110] if isinstance(k.task_text, str) else None, _code(TIER_CODE, k.evidence_tier), _code(AUTONOMY_CODE, k.autonomy_2026), _code(RBARRIER_CODE, k.barrier_2026), _pct(k.ai_share)] for _, k in gg.sort_values("disrupt_year_p50").iterrows()]
                groups.append([str(cwf)[:90], items])
        key_tasks.append(groups)
        role_list.append({"sector": r.sector, "track": r.track if isinstance(r.track, str) else "", "role": r.job_role, "desc": (r.job_role_description or "")[:360] if isinstance(r.job_role_description, str) else "", "ssoc": [[c, s, sc, ssoc_title.get(c, "")] for c, s, sc in ssoc_by_role.get(r.role_id, [])], "oat": _pct(r.observed_automation_via_tasks), "oas": _pct(r.observed_automation_via_ssoc), "sko": _pct(r.share_key_tasks_observed), "skp": _pct(r.share_key_tasks_physical), "nk": _r(r.n_key_tasks), "ms": _r(r.mean_match_score, 2), "sb": _sb(f) if f is not None else None, "sb10": _sbq(f, "p10"), "sb90": _sbq(f, "p90"), "tiers": _tiers(f), "shift": _r(sh_role.get(r.role_id), 1) if r.role_id in sh_role.index else None, "h": _half(f) if f is not None else [None, None, None], "db": _db(f), "db10": _db(f, "p10"), "db90": _db(f, "p90"), "cap": _pct(f.ceiling_ai_share) if f is not None else None, "hq": _done_years(f, "quarter"), "hd": _done_years(f, "half"), "bi": _r(f.barrier_index, 2) if f is not None else None})

    # ---- O*NET occupations
    onet_list = []
    for _, r in occ.sort_values("title").iterrows():
        f = focc.loc[r.onet_soc_code] if r.onet_soc_code in focc.index else None
        onet_list.append({"c": r.onet_soc_code, "t": r.title, "oa": _pct(r.observed_automation), "so": _pct(r.share_weight_observed), "sp": _pct(r.share_weight_physical), "ms": _r(r.ai_applicability_score, 2), "oe": _r(r.openai_exposure_beta_gpt4, 2), "g": r.google_atlas_flag if isinstance(r.google_atlas_flag, str) else None, "np": _r(r.n_providers_positive), "h": _half(f) if f is not None else [None, None, None], "sb": _sb(f) if f is not None else None, "sb10": _sbq(f, "p10"), "sb90": _sbq(f, "p90"), "tiers": _tiers(f), "shift": _r(sh_onet.get(r.onet_soc_code), 1) if r.onet_soc_code in sh_onet.index else None, "db": _db(f), "cap": _pct(f.ceiling_ai_share) if f is not None else None, "hq": _done_years(f, "quarter"), "gd": bool(r.gdpval_covered)})

    # ---- United States: occupations with US weights and timing, clusters, competency models, levels
    us_payload, us_tasks = None, None
    if us_econ is not None and us_w is not None and focc_us is not None:
        us_payload, us_tasks = build_us_payload(occ, o2s, us_w, focc_us, us_econ, onet_list, stats, ssoc_list)
    # ---- further countries (Australia, the United Kingdom), when built
    country_payloads, country_tasks = build_country_payloads(onet_list, ssoc_list, stats, us_payload)

    skills_list = [{"f": r.family, "name": r.element_name, "n": _r(r.n_occupations), "oa": _pct(r.via_occupations_observed_automation), "oaa": _pct(r.via_activities_observed_automation), "h": _half(r), "sb": _sb(r), "sb10": _sbq(r, "p10"), "sb90": _sbq(r, "p90"), "db": _db(r), "db10": _db(r, "p10"), "db90": _db(r, "p90"), "hq": _done_years(r, "quarter"), "cap": _pct(r.get("ceiling_ai_share"))} for _, r in skills.sort_values(["family", "element_name"]).iterrows()]
    gwa_list = [{"name": r["name"], "n": _r(r.n_tasks), "so": _pct(r.share_tasks_observed), "oa": _pct(r.observed_automation), "ms": _r(r.ms_share * 100, 1) if pd.notna(r.ms_share) else None, "h": _half(r), "sb": _sb(r), "sb10": _sbq(r, "p10"), "sb90": _sbq(r, "p90"), "db": _db(r), "db10": _db(r, "p10"), "db90": _db(r, "p90"), "hq": _done_years(r, "quarter"), "cap": _pct(r.get("ceiling_ai_share"))} for _, r in act[act.level == "gwa"].sort_values("observed_automation", ascending=False).iterrows()]
    sector_list = [{"sector": r.sector, "n": _r(r.n_roles), "oat": _pct(r.observed_automation_via_tasks), "sko": _pct(r.share_key_tasks_observed), "skp": _pct(r.share_key_tasks_physical), "sb30": _pct(r.share_by_2030), "sb35": _pct(r.share_by_2035), "sb30f": _pct(r.get("share_by_2030_p10")), "sb30s": _pct(r.get("share_by_2030_p90")), "sb35f": _pct(r.get("share_by_2035_p10")), "sb35s": _pct(r.get("share_by_2035_p90")), "h": [_r(r.get("half_year_p10_median_role"), 1), _r(r.half_year_p50_median_role, 1), _r(r.half_year_p90_median_role, 1)], "db30": _pct(r.get("done_by_2030")), "db35": _pct(r.get("done_by_2035")), "db30f": _pct(r.get("done_by_2030_p10")), "db30s": _pct(r.get("done_by_2030_p90")), "db35f": _pct(r.get("done_by_2035_p10")), "db35s": _pct(r.get("done_by_2035_p90")), "cap": _pct(r.get("ceiling_ai_share")), "hq": [_r(r.get("quarter_done_year_p10"), 1), _r(r.get("quarter_done_year_p50"), 1), _r(r.get("quarter_done_year_p90"), 1)]} for _, r in sectors.iterrows()]

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
        {"name": "MOM job vacancies, wages and employment by occupation", "rung": 5, "what": "vacancies by detailed occupation (Sep 2023, 2024, 2025); median gross wages (Jun 2024, Jun 2025); employed residents by occupation sub-group (Jun, yearly)", "unit": "occupation, sub-group", "date": "to Sep 2025", "record": (lambda o: f"the outcomes the map is tested against: {o.get('n_ssoc_with_wages_both_years', 0)} occupations with wages in both years, {o.get('n_minor_groups_with_vacancy_shares', 0)} vacancy lines, 41 sub-groups; so far no clear relation to exposure at occupation level, and the more exposed groups grew faster")(stats.get("outcomes") or {}), "caution": "surveys of firms with 25 or more employees; vacancy shares are within occupational group among listed occupations"},
        {"name": "MOM retrenchment, vacancy and resignation rates by occupational group", "rung": 5, "what": "retrenchments per 1,000 employees, job vacancy rate, resignation rate, for three broad groups", "unit": "occupational group", "date": "2025 (yearly)", "record": "PMET retrenchments per 1,000 rose from 6.1 (2019) to 10.0 (2025) while falling for the other two groups; vacancy rates fell for all groups from their 2022 peak", "caution": "three groups only; cyclical swings dwarf any AI effect so far"},
        {"name": "US BLS Occupational Employment and Wage Statistics", "rung": 5, "what": "employment and median wages by detailed occupation and by industry sector (May 2024, May 2023)", "unit": "occupation, sector", "date": "May 2024", "record": (lambda u: f"the second workforce the map is weighted by ({fmt_int((u.get('weights') or {}).get('employment_thousands_total', 0) / 1000)} million employed); employment change May 2023 to May 2024 across {fmt_int((u.get('outcomes') or {}).get('n_soc', 0))} occupations is the US rung-five test")((stats.get("countries") or {}).get("USA") or {}), "caution": "a different survey, question and year from Singapore's; wages are annual and in US dollars"},
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
        "scenario": {"adoption_mult": list(fp.get("adoption_growth_mult", (1.5, 1.0, 0.6))), "barrier_years": list(fp.get("max_barrier_years", (2.0, 4.0, 6.0))), "lag_mult": [0.5, 1.0, 2.0]},
        "ai_share": stats.get("ai_share", {}), "ladder": lad,
        "outcomes": _outcomes_meta(stats, outsub), "validation": {k: val.get(k) for k in ["auc_seen", "auc_doing_given_seen", "agreement", "repeat_2026", "aging_of_2023_no", "aging_of_2023_yes"]}, "sources": sources,
        "sector_labels": {"J": "Information & communications", "K": "Financial & insurance services", "M": "Professional services", "OP": "Public administration & education", "L": "Real estate services", "Q": "Health & social services", "N": "Administrative & support services", "R": "Arts, entertainment & recreation", "C": "Manufacturing", "G": "Wholesale & retail trade", "F": "Construction", "SU": "Other community, social & personal services", "ABDE": "Agriculture, mining, utilities", "H": "Transportation & storage", "I": "Accommodation & food services"},
    }
    compare = None
    cs = stats.get("countries") or {}
    if us_econ is not None and "USA" in cs and "decomposition" in cs:
        def _rnd(o):
            if isinstance(o, dict):
                return {k: _rnd(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_rnd(v) for v in o]
            if isinstance(o, float):
                return None if np.isnan(o) else round(o, 4)
            return o
        usa = cs["USA"]
        bt = stats.get("btos_adoption_fit") or {}
        compare = {
            "countries": {
                "SGP": {"name": "Singapore", "econ": [e for e in economy if e["level"] in ("economy", "major_group")], "adoption_now": lad.get("adoption_now"), "adoption_now_label": lad.get("adoption_now_year"), "half_firms": _r(lad.get("adoption_half_year"), 1), "lead": _r(stats.get("geo_lead_years_central"), 1), "emp_label": "employed residents, June 2025 (MOM)", "wage_label": "median gross monthly wage, June 2025 (MOM)", "shift_range": [_r(v, 1) for v in ((stats.get("sector_adoption") or {}).get("ssoc_shift_range") or [None, None])]},
                "USA": {"name": "United States", "econ": [_econ_row(r) for _, r in us_econ.iterrows() if r.level in ("economy", "major_group")], "adoption_now": (bt.get("last_pct") / 100.0) if bt.get("last_pct") is not None else None, "adoption_now_label": f"Census BTOS, {bt.get('last_ref_end', '')}", "half_firms": _r((stats.get("adoption_anchor_years") or {}).get("central"), 1), "lead": 0.0, "emp_label": "employed, May 2024 (BLS OEWS)", "wage_label": "median annual wage, May 2024 (BLS OEWS)", "shift_range": [_r(v, 1) for v in (usa.get("shift_range") or [None, None])], "oews_year": (usa.get("weights") or {}).get("oews_year")},
            },
            "decomp": _rnd(cs["decomposition"]),
            "us_outcomes": _rnd(usa.get("outcomes") or {}),
            "us_weights": _rnd(usa.get("weights") or {}),
        }
    if compare is not None and country_payloads:
        for code, cp in country_payloads.items():
            compare["countries"][code] = {"name": cp["name"], "econ": [e for e in cp["economy"] if e["level"] in ("economy", "major_group")], "adoption_now": (cp["adoption_now"] / 100.0) if cp.get("adoption_now") is not None else None, "adoption_now_label": f"{cp['adoption_source']}, {cp.get('adoption_now_text') or _date_label(cp.get('adoption_now_date'))}", "half_firms": None, "lead": cp.get("lead"), "emp_label": cp["emp_label"], "wage_label": cp["wage_label"], "shift_range": [_r(v, 1) for v in (cp.get("shift_range") or [None, None])], "outcomes": cp.get("outcomes") or {}, "major_names": cp.get("major_names")}
        cs = stats.get("countries") or {}
        pairs = {}
        for code in country_payloads:
            for k, v in ((cs.get(code) or {}).get("decompositions") or {}).items():
                pairs[k] = _rnd(v)
        compare["pairs"] = pairs
    return {"meta": meta, "economy": economy, "ssoc": ssoc_list, "roles": role_list, "onet": onet_list, "skills": skills_list, "gwa": gwa_list, "sectors": sector_list, "compare": compare, "us": us_payload, "c": country_payloads or None, "_key_tasks": key_tasks, "_us_tasks": us_tasks, "_country_tasks": country_tasks}


def _exists(name):
    return (PROCESSED / f"{name}.parquet").exists()


def _rnd(o):
    if isinstance(o, dict):
        return {k: _rnd(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_rnd(v) for v in o]
    if isinstance(o, float):
        return None if np.isnan(o) else round(o, 4)
    return o


def _date_label(y):
    """A fractional year such as 2026.76 -> 'Oct 2026'."""
    if y is None:
        return ""
    import math
    yr = int(math.floor(y)); mon = min(12, max(1, int(round((y - yr) * 12 + 0.5))))
    return f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][mon-1]} {yr}"


def build_us_payload(occ, o2s, us_w, focc_us, us_econ, onet_list, stats, ssoc_list):
    """The US side of the page: per-occupation entries (weights, timing, outcomes, clusters, the Singapore
    counterpart), career clusters with their competency models, the skill-level test, and the on-demand
    file with each occupation's tasks, skills with levels, software, related occupations and sectors."""
    cs = (stats.get("countries") or {}).get("USA") or {}
    order = [e["c"] for e in onet_list]
    idx = {c: i for i, c in enumerate(order)}
    clusters = load("us_career_clusters") if _exists("us_career_clusters") else None
    cl_view = load("us_cluster_view") if _exists("us_cluster_view") else None
    levels = load("us_skill_level_exposure") if _exists("us_skill_level_exposure") else None
    lcorr = load("us_skill_level_correlations") if _exists("us_skill_level_correlations") else None
    usoc = load("us_outcomes_soc").set_index("soc_2018") if _exists("us_outcomes_soc") else None
    sh_us = load("us_adoption_shift_onet").set_index("onet_soc_code").adoption_shift_years if _exists("us_adoption_shift_onet") else pd.Series(dtype=float)
    mix = load("us_soc_sector_mix") if _exists("us_soc_sector_mix") else None
    jz = load("onet_job_zones").set_index("onet_soc_code").job_zone
    cl_by_occ = clusters.groupby("onet_soc_code").cluster_code.apply(list).to_dict() if clusters is not None else {}
    cl_names = clusters.drop_duplicates("cluster_code").set_index("cluster_code").cluster.to_dict() if clusters is not None else {}
    ssoc_idx = {e["c"]: i for i, e in enumerate(ssoc_list)}
    # Singapore counterpart: the SSOC code with the largest overlap with this occupation, measured as the
    # share of the occupation that maps to the code (the forward crosswalk weight) times the code's resident
    # employment, so Accountants and Auditors lands on Accountant rather than on the larger but more
    # loosely linked Assistant accountant
    sg_emp = {e["c"]: (e.get("emp") or 0.0) for e in ssoc_list}
    sg_title = {e["c"]: e["t"] for e in ssoc_list}
    occ_title_us = {e["c"]: e["t"] for e in onet_list}
    sg_top = {}
    for c, g in o2s.groupby("onet_soc_code"):
        best, best_k = None, (-1, -1.0)
        for code, wr in zip(g.ssoc_code, g.weight.fillna(0)):
            if code in ssoc_idx:
                k = (_title_overlap(occ_title_us.get(c, ""), sg_title.get(code, "")), float(sg_emp.get(code, 0.0)) * float(wr))
                if k > best_k:
                    best, best_k = ssoc_idx[code], k
        if best is not None:
            sg_top[c] = best
    occ_rows = []
    for e in onet_list:
        c = e["c"]
        w = us_w.loc[c] if c in us_w.index else None
        f = focc_us.loc[c] if c in focc_us.index else None
        o = usoc.loc[w.soc_2018] if (w is not None and usoc is not None and w.soc_2018 in usoc.index) else None
        occ_rows.append({"c": c, "t": e["t"], "soc": w.soc_2018 if w is not None else None, "mg": w.major_group if w is not None else None, "sm": w.submajor_label if w is not None else None, "smn": w.submajor_name if (w is not None and isinstance(w.submajor_name, str)) else None,
                         "jz": _r(jz.get(c)) if pd.notna(jz.get(c, np.nan)) else None, "emp": _r(w.employment_thousands, 1) if w is not None and pd.notna(w.employment_thousands) else None, "wage": _r(w.wage_median) if w is not None and pd.notna(w.wage_median) else None, "wtop": bool(w.wage_top_coded) if w is not None else False, "wsrc": w.wage_source if w is not None and isinstance(w.wage_source, str) else None,
                         "oa": e["oa"], "so": e["so"], "sp": e["sp"], "ms": e["ms"], "oe": e["oe"], "g": e["g"], "np": e["np"], "gd": e["gd"],
                         "sb": _sb(f) if f is not None else None, "sb10": _sbq(f, "p10"), "sb90": _sbq(f, "p90"), "db": _db(f), "db10": _db(f, "p10"), "db90": _db(f, "p90"), "cap": _pct(f.ceiling_ai_share) if f is not None else None, "hq": _done_years(f, "quarter"), "hd": _done_years(f, "half"), "h": _half(f) if f is not None else [None, None, None], "tiers": _tiers(f), "bi": _r(f.barrier_index, 2) if f is not None else None, "shift": _r(sh_us.get(c), 1) if c in sh_us.index else None,
                         "oc": [_r(o.emp_2023), _r(o.emp_2024), _r(o.emp_change_pct, 1), _r(o.wage_2023), _r(o.wage_2024), _r(o.wage_change_pct, 1)] if o is not None else None,
                         "cl": cl_by_occ.get(c, []), "sg": sg_top.get(c)})
    # economy rows (same shape as Singapore's)
    def _row(r):
        return {"scope": r.scope, "level": r.level, "name": r.scope_name, "w": "emp" if r.weighting == "employment" else "wage", "total": _r(r.weight_total / (1 if r.weighting == "employment" else 1000), 1), "oa": _pct(r.observed_automation), "so": _pct(r.share_weight_observed), "sp": _pct(r.share_weight_physical), "sb": _sb(r), "sb10": [_pct(r.get(f"share_by_{y}_p10")) for y in YEARS], "sb90": [_pct(r.get(f"share_by_{y}_p90")) for y in YEARS], "half": [_r(r.get("half_year_fast_from_curve", r.half_year_p10), 1), _r(r.half_year_central_from_curve, 1), _r(r.get("half_year_slow_from_curve", r.half_year_p90), 1)],
                "db": _db(r), "db10": _db(r, "p10"), "db90": _db(r, "p90"), "cap": _pct(r.get("ceiling_ai_share")), "hq": [_r(r.get("quarter_done_year_fast_from_curve"), 1), _r(r.get("quarter_done_year_central_from_curve"), 1), _r(r.get("quarter_done_year_slow_from_curve"), 1)], "hd": [_r(r.get("half_done_year_fast_from_curve"), 1), _r(r.get("half_done_year_central_from_curve"), 1), _r(r.get("half_done_year_slow_from_curve"), 1)], "ech": None}
    economy = [_row(r) for _, r in us_econ.iterrows()]
    for e in economy:
        if e["level"] == "submajor_group" and str(e["name"]).endswith(" nan"):
            e["name"] = e["name"][:-4] + " (no published group title)"
    # clusters with their occupations and the matched competency models
    cl_list, cmc_blocks = [], {}
    if cl_view is not None:
        emp_by = us_w.employment_thousands
        models = load("cmc_models") if _exists("cmc_models") else None
        xw = load("cmc_models_to_clusters") if _exists("cmc_models_to_clusters") else None
        blocks = load("cmc_blocks") if _exists("cmc_blocks") else None
        mb = load("cmc_model_blocks") if _exists("cmc_model_blocks") else None
        kb = load("cmc_key_behaviors") if _exists("cmc_key_behaviors") else None
        tiers = load("cmc_tiers").set_index("tier_id").tier_name.to_dict() if _exists("cmc_tiers") else {}
        for _, r in cl_view.iterrows():
            code = str(r.cluster_code)
            occs = clusters[clusters.cluster_code == code].onet_soc_code
            occs = sorted([c for c in occs if c in idx and pd.notna(emp_by.get(c, np.nan)) and float(emp_by.get(c, 0)) > 0], key=lambda c: -float(emp_by.get(c, 0)))
            model_names = xw[xw.cluster_code == code].model_name.tolist() if xw is not None else []
            cl_list.append({"code": code, "name": r.cluster, "n": _r(r.n_occupations), "emp": _r(r.employment_thousands, 1), "oa": _pct(r.observed_automation), "so": _pct(r.share_weight_observed), "sp": _pct(r.share_weight_physical), "sb30": _pct(r.share_by_2030), "sb35": _pct(r.share_by_2035), "sb35f": _pct(r.get("share_by_2035_p10")), "sb35s": _pct(r.get("share_by_2035_p90")), "db30": _pct(r.done_by_2030), "db30f": _pct(r.get("done_by_2030_p10")), "db30s": _pct(r.get("done_by_2030_p90")), "db35": _pct(r.done_by_2035), "db35f": _pct(r.get("done_by_2035_p10")), "db35s": _pct(r.get("done_by_2035_p90")), "cap": _pct(r.ceiling_ai_share), "hq": [_r(r.get("quarter_done_year_fast_from_curve"), 1), _r(r.get("quarter_done_year_central_from_curve"), 1), _r(r.get("quarter_done_year_slow_from_curve"), 1)], "h": [_r(r.get("half_year_fast_from_curve"), 1), _r(r.get("half_year_central_from_curve"), 1), _r(r.get("half_year_slow_from_curve"), 1)], "occ": [idx[c] for c in occs], "models": model_names})
            if models is not None and model_names:
                out_blocks = []
                for mname in model_names:
                    mid = models[models.model_name == mname].model_id
                    if not len(mid):
                        continue
                    bl = mb[mb.model_id == int(mid.iloc[0])].merge(blocks, on=["tier_id", "block_id"], how="left").sort_values(["tier_id", "block_id"])
                    for _, b in bl.iterrows():
                        t = int(b.tier_id)
                        kbs = kb[kb.block_id == int(b.block_id)].sort_values("kb_sort_id").kb_name.dropna().astype(str).tolist() if (kb is not None and t >= 4) else []
                        kbs = [k for k in kbs if k.strip() and k.strip() != str(b.block_name).strip()][:8]
                        out_blocks.append([mname, t, tiers.get(t, f"Tier {t}"), str(b.block_name), str(b.block_desc)[:220], kbs])
                cmc_blocks[code] = out_blocks
    # the skill-level test
    lv = None
    if levels is not None:
        def band_rows(df):
            out = {}
            for _, r in df.iterrows():
                out[r.band] = [r.band, r.band_name, _r(r.n_occupations), _pct(r.share_weight_observed), _pct(r.observed_automation), _pct(r.done_by_2035), _pct(r.get("done_by_2035_p10")), _pct(r.get("done_by_2035_p90")), [_r(r.get("quarter_done_year_fast_from_curve"), 1), _r(r.get("quarter_done_year_central_from_curve"), 1), _r(r.get("quarter_done_year_slow_from_curve"), 1)]]
            return [out.get(b) for b in ("L", "M", "H")]
        corr = lcorr.set_index("element_name") if lcorr is not None else None
        skills = []
        for name, g in levels[levels.family != "index"].groupby("element_name", sort=True):
            cr = corr.loc[name] if (corr is not None and name in corr.index) else None
            skills.append([name, g.family.iloc[0], _r(cr.rho_level_vs_done_2035, 2) if cr is not None and pd.notna(cr.rho_level_vs_done_2035) else None, _r(cr.rho_level_vs_seen, 2) if cr is not None and pd.notna(cr.rho_level_vs_seen) else None, _r(cr.n_occupations) if cr is not None else None, band_rows(g)])
        ix = levels[levels.family == "index"]
        cr = corr.loc["All essential skills (average level)"] if (corr is not None and "All essential skills (average level)" in corr.index) else None
        lv = {"index": band_rows(ix), "index_rho": _r(cr.rho_level_vs_done_2035, 2) if cr is not None else None, "index_rho_seen": _r(cr.rho_level_vs_seen, 2) if cr is not None else None, "index_n": _r(cr.n_occupations) if cr is not None else None, "skills": skills, "bands": [["L", "low or not needed (level below 2.5)"], ["M", "middle (2.5 to 4.5)"], ["H", "high (4.5 and above)"]]}
    # on-demand file: tasks by work activity, skills with levels and examples, software, related occupations, sectors
    ft = (load("forecast_tasks_us") if _exists("forecast_tasks_us") else load("forecast_tasks")).set_index("task_id")   # US-timed years for US task lists
    tasks = load("onet_tasks")
    td = load("onet_task_dwa")[["task_id", "dwa_id"]].merge(load("onet_activities")[["dwa_id", "gwa_name"]], on="dwa_id", how="left").drop_duplicates("task_id").set_index("task_id").gwa_name
    ratings = load("onet_descriptor_ratings_v31")
    ratings = ratings[ratings.family.isin(["essential_skill", "transferable_skill", "knowledge"]) & (ratings.importance >= 3.0)]
    anchors = onet_level_anchors()
    sw = load("onet_software_skills_v31") if _exists("onet_software_skills_v31") else None
    rel = load("onet_related_occupations")
    rel = rel[rel.relatedness_tier == "Primary-Short"].sort_values("index")
    titles = occ.set_index("onet_soc_code").title
    sec_lab = {"11": "Agriculture", "21": "Mining, oil and gas", "22": "Utilities", "23": "Construction", "31": "Manufacturing", "42": "Wholesale trade", "44": "Retail trade", "48": "Transport and warehousing", "51": "Information", "52": "Finance and insurance", "53": "Real estate", "54": "Professional and technical services", "55": "Management of companies", "56": "Administrative and support services", "61": "Education", "62": "Health care and social assistance", "71": "Arts, entertainment and recreation", "72": "Accommodation and food services", "81": "Other services", "99": "Government"}
    fam_code = {"essential_skill": "E", "transferable_skill": "T", "knowledge": "K"}
    us_tasks = {"occ": {}, "cmc": cmc_blocks}
    t_by = {c: g for c, g in tasks.groupby("onet_soc_code")}
    r_by = {c: g for c, g in ratings.groupby("onet_soc_code")}
    sw_by = {c: g for c, g in sw.groupby("onet_soc_code")} if sw is not None else {}
    rel_by = {c: g for c, g in rel.groupby("onet_soc_code")}
    mix_by = {c: g for c, g in mix.groupby("soc_2018")} if mix is not None else {}
    for e in onet_list:
        c = e["c"]
        g = t_by.get(c)
        groups = []
        if g is not None:
            gg = g.merge(ft[["status", "disrupt_year_p50", "barrier_type", "evidence_tier", "autonomy_2026", "barrier_2026", "ai_share"]], left_on="task_id", right_index=True, how="left")
            gg["gwa"] = gg.task_id.map(td).fillna("Other tasks")
            for gwa, grp in gg.sort_values("importance", ascending=False).groupby("gwa", sort=False):
                items = [[str(k.task_statement)[:170], STATUS_CODE.get(k.status, 3), _r(k.disrupt_year_p50, 1), BARRIER_CODE.get(k.barrier_type, 0), _code(TIER_CODE, k.evidence_tier), _code(AUTONOMY_CODE, k.autonomy_2026), _code(RBARRIER_CODE, k.barrier_2026), _pct(k.ai_share), 1 if k.task_type == "Core" else 0, _r(k.importance, 1)] for _, k in grp.iterrows()]
                groups.append([str(gwa), items])
            groups.sort(key=lambda x: -sum((it[9] or 0) for it in x[1]))
        ks = []
        rg = r_by.get(c)
        if rg is not None:
            for _, k in rg.sort_values(["importance"], ascending=False).head(18).iterrows():
                ks.append([fam_code.get(k.family, "?"), str(k.element_name), _r(k.importance, 1), _r(k.level, 1) if pd.notna(k.level) else None, anchors.get((k.element_id, round(float(k.level)) if pd.notna(k.level) else None))])
        sws = []
        sg_ = sw_by.get(c)
        if sg_ is not None:
            sg_ = sg_.assign(hot=(sg_.hot_technology == "Y").astype(int)).sort_values(["hot"], ascending=False).drop_duplicates("workplace_example").head(15)
            sws = [[str(k.workplace_example), str(k.element_name), int(k.hot)] for _, k in sg_.iterrows()]
        rels = [[str(k.related_onet_soc_code), str(titles.get(k.related_onet_soc_code, "")), idx.get(k.related_onet_soc_code)] for _, k in rel_by.get(c, pd.DataFrame(columns=rel.columns)).head(6).iterrows()]
        secs = []
        w = us_w.loc[c] if c in us_w.index else None
        if w is not None and w.soc_2018 in mix_by:
            mg = mix_by[w.soc_2018].sort_values("share", ascending=False).head(4)
            secs = [[sec_lab.get(str(k.sector), str(k.sector)), _pct(k.share)] for _, k in mg.iterrows()]
        us_tasks["occ"][c] = {"g": groups, "k": ks, "sw": sws, "rel": rels, "sec": secs}
    payload = {"occ": occ_rows, "economy": economy, "clusters": cl_list, "levels": lv, "outcomes": (cs.get("outcomes") or {}), "weights": cs.get("weights") or {}, "shift_range": cs.get("shift_range"),
               "jz_names": {1: "little or no preparation", 2: "some preparation", 3: "medium preparation", 4: "considerable preparation", 5: "extensive preparation"}}
    return payload, us_tasks


def onet_level_anchors() -> dict:
    """(element_id, level rounded) -> the nearest written example on the 0-7 level scale."""
    from . import onet
    try:
        a = onet.load_level_anchors()
    except Exception:
        return {}
    out = {}
    for eid, g in a.groupby("element_id"):
        vals = g.sort_values("anchor_value")
        for lv in range(0, 8):
            nearest = vals.iloc[(vals.anchor_value - lv).abs().argmin()]
            out[(eid, lv)] = f"level {int(nearest.anchor_value)} example: {nearest.anchor_description}"
    return out


def build_site(companion_url: str | None = None) -> Path:
    """Write site/index.html (page + core data) and site/kt.json (key tasks per role, loaded on demand)."""
    SITE.mkdir(parents=True, exist_ok=True)
    data = build_data(companion_url)
    key_tasks = data.pop("_key_tasks")
    us_tasks = data.pop("_us_tasks")
    country_tasks = data.pop("_country_tasks") or {}
    js = json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    meaning_js = Path(__file__).with_name("site_meaning.js").read_text().replace("if(typeof module!=='undefined') module.exports=Meaning;", "")
    html = TEMPLATE.read_text().replace("__DATA__", js).replace("__MEANING_JS__", meaning_js)
    out = SITE / "index.html"
    out.write_text(html)
    (SITE / "kt.json").write_text(json.dumps(key_tasks, ensure_ascii=True, separators=(",", ":")))
    if us_tasks is not None:
        (SITE / "us_tasks.json").write_text(json.dumps(us_tasks, ensure_ascii=True, separators=(",", ":")))
    for code, files in country_tasks.items():
        (SITE / f"{code.lower()}_tasks.json").write_text(json.dumps(files, ensure_ascii=True, separators=(",", ":")))
    from .site_search import build_search_assets

    national = {code: [o["c"] for o in cp["occ"]] for code, cp in (data.get("c") or {}).items()}
    stats = build_search_assets([s["c"] for s in data["ssoc"]], load("frontier_sfw_roles").role_id.tolist(), key_tasks, [o["c"] for o in data["onet"]] if data.get("us") else None, national or None)
    (SITE / "search_stats.json").write_text(json.dumps(stats))
    return out


# ----------------------------------------------------------------------------- further countries (Australia, the United Kingdom)
def _econ_payload_row(r):
    """An economy row in the page's shape (the same keys as Singapore's and the US's)."""
    return {"scope": r.scope, "level": r.level, "name": r.scope_name, "w": "emp" if r.weighting == "employment" else "wage", "total": _r(r.weight_total / (1 if r.weighting == "employment" else 1000), 1), "oa": _pct(r.observed_automation), "so": _pct(r.share_weight_observed), "sp": _pct(r.share_weight_physical), "sb": _sb(r), "sb10": [_pct(r.get(f"share_by_{y}_p10")) for y in YEARS], "sb90": [_pct(r.get(f"share_by_{y}_p90")) for y in YEARS], "half": [_r(r.get("half_year_fast_from_curve", r.half_year_p10), 1), _r(r.half_year_central_from_curve, 1), _r(r.get("half_year_slow_from_curve", r.half_year_p90), 1)],
            "db": _db(r), "db10": _db(r, "p10"), "db90": _db(r, "p90"), "cap": _pct(r.get("ceiling_ai_share")), "hq": [_r(r.get("quarter_done_year_fast_from_curve"), 1), _r(r.get("quarter_done_year_central_from_curve"), 1), _r(r.get("quarter_done_year_slow_from_curve"), 1)], "hd": [_r(r.get("half_done_year_fast_from_curve"), 1), _r(r.get("half_done_year_central_from_curve"), 1), _r(r.get("half_done_year_slow_from_curve"), 1)], "ech": None}


def _levels_payload(levels, lcorr):
    """The O*NET skill-level test in the page's shape (shared by the US and the further countries)."""
    if levels is None:
        return None

    def band_rows(df):
        out = {}
        for _, r in df.iterrows():
            out[r.band] = [r.band, r.band_name, _r(r.n_occupations), _pct(r.share_weight_observed), _pct(r.observed_automation), _pct(r.done_by_2035), _pct(r.get("done_by_2035_p10")), _pct(r.get("done_by_2035_p90")), [_r(r.get("quarter_done_year_fast_from_curve"), 1), _r(r.get("quarter_done_year_central_from_curve"), 1), _r(r.get("quarter_done_year_slow_from_curve"), 1)]]
        return [out.get(b) for b in ("L", "M", "H")]
    corr = lcorr.set_index("element_name") if lcorr is not None else None
    skills = []
    for name, g in levels[levels.family != "index"].groupby("element_name", sort=True):
        cr = corr.loc[name] if (corr is not None and name in corr.index) else None
        skills.append([name, g.family.iloc[0], _r(cr.rho_level_vs_done_2035, 2) if cr is not None and pd.notna(cr.rho_level_vs_done_2035) else None, _r(cr.rho_level_vs_seen, 2) if cr is not None and pd.notna(cr.rho_level_vs_seen) else None, _r(cr.n_occupations) if cr is not None else None, band_rows(g)])
    ix = levels[levels.family == "index"]
    cr = corr.loc["All essential skills (average level)"] if (corr is not None and "All essential skills (average level)" in corr.index) else None
    return {"index": band_rows(ix), "index_rho": _r(cr.rho_level_vs_done_2035, 2) if cr is not None else None, "index_rho_seen": _r(cr.rho_level_vs_seen, 2) if cr is not None else None, "index_n": _r(cr.n_occupations) if cr is not None else None, "skills": skills, "bands": [["L", "low or not needed (level below 2.5)"], ["M", "middle (2.5 to 4.5)"], ["H", "high (4.5 and above)"]]}


_TITLE_STOP = {"and", "or", "of", "the", "in", "for", "to", "a", "an", "other", "all", "general", "nec", "n.e.c.", "not", "elsewhere", "classified", "workers", "worker", "occupations", "occupation", "professionals", "professional", "related", "except", "including", "assistants", "managers", "manager", "technicians", "technician", "specialists", "specialist", "services", "service"}


def _title_overlap(a: str, b: str) -> int:
    """1 when the two titles share a content word (one word a prefix of the other, four letters or more, so
    'sales' meets 'salespersons' and 'nurse' meets 'nurses'), else 0; the tie-break for counterparts."""
    import re as _re
    words = lambda s: [w for w in _re.findall(r"[a-z]+", str(s).lower()) if w not in _TITLE_STOP and len(w) > 3]
    wa, wb = words(a), words(b)
    return 1 if any(x.startswith(y) or y.startswith(x) for x in wa for y in wb) else 0


def _task_item(text, f, score, onet_text, extra=None):
    """One matched task line in the key-task shape the page already renders:
    [text, status, central year, barrier, match score, O*NET task text, tier, autonomy, rubric barrier, AI share, ...extra]."""
    item = [str(text)[:170], STATUS_CODE.get(f.status if f is not None else None, 3), _r(f.disrupt_year_p50, 1) if f is not None else None, BARRIER_CODE.get(f.barrier_type if f is not None else None, 0), _r(score, 2), str(onet_text)[:110] if isinstance(onet_text, str) else None,
            _code(TIER_CODE, f.get("evidence_tier")) if f is not None else None, _code(AUTONOMY_CODE, f.get("autonomy_2026")) if f is not None else None, _code(RBARRIER_CODE, f.get("barrier_2026")) if f is not None else None, _pct(f.get("ai_share")) if f is not None else None]
    return item + (extra or [])


def build_country_payloads(onet_list, ssoc_list, stats, us_payload):
    """Per further country: the page payload and the on-demand task file. Returns ({code: payload}, {code: tasks})."""
    from . import countries as cn

    payloads, files = {}, {}
    for code, spec in cn.SPECS.items():
        p = spec.prefix
        if not (_exists(f"{p}weighted") and _exists(f"{p}economy_exposure")):
            continue
        payloads[code], files[code] = build_country_payload(spec, onet_list, ssoc_list, stats, us_payload)
    return payloads, files


def build_country_payload(spec, onet_list, ssoc_list, stats, us_payload):
    p = spec.prefix
    cs = ((stats.get("countries") or {}).get(spec.code)) or {}
    t = load(f"{p}weighted").sort_values("nat_code")
    econ = load(f"{p}economy_exposure")
    xw = load(f"{p}onet_crosswalk")
    outc = load(f"{p}outcomes").set_index(spec.code_col) if _exists(f"{p}outcomes") else None
    sh = load(f"{p}adoption_shift_national").set_index("nat_code").adoption_shift_years if _exists(f"{p}adoption_shift_national") else pd.Series(dtype=float)
    iv = load(f"{p}industry_view") if _exists(f"{p}industry_view") else None
    bands = load(f"{p}skill_level_bands") if _exists(f"{p}skill_level_bands") else None
    levels = load(f"{p}onet_skill_level_exposure") if _exists(f"{p}onet_skill_level_exposure") else None
    lcorr = load(f"{p}onet_skill_level_correlations") if _exists(f"{p}onet_skill_level_correlations") else None
    tasks = load(f"{p}tasks").rename(columns={spec.code_col: "nat_code"}) if _exists(f"{p}tasks") else None
    links = load(f"{p}task_onet_task") if _exists(f"{p}task_onet_task") else None
    ft = (load(f"forecast_tasks_{p.rstrip('_')}") if _exists(f"forecast_tasks_{p.rstrip('_')}") else load("forecast_tasks")).set_index("task_id")
    onet_tasks = load("onet_tasks").set_index("task_id").task_statement
    occ_title = {e["c"]: e["t"] for e in onet_list}
    us_idx = {e["c"]: i for i, e in enumerate((us_payload or {}).get("occ") or [])}
    us_occ = (us_payload or {}).get("occ") or []
    # the heaviest O*NET occupations in each code (by the share of the code's workers), and counterparts
    top_onet, top_code = {}, {}
    us_emp = {e["c"]: (e.get("emp") or 0.0) for e in us_occ}
    nat_title = dict(zip(t.nat_code, t.nat_title))
    for c, g in xw.sort_values("weight_rev", ascending=False).groupby("nat_code"):
        top_onet[c] = [[occ_title.get(o, o), _pct(w)] for o, w in zip(g.onet_soc_code.head(4), g.weight_rev.head(4))]
        # the counterpart: among the O*NET occupations whose title shares a word with the code's title (when
        # any does), the one whose workers overlap this code the most (US employment times the share of that
        # occupation which maps to the code); so UK "Chefs" lands on Chefs and Head Cooks, not on the much
        # larger food-preparation supervisors the ISCO chain also links it to
        cands = [(_title_overlap(nat_title.get(c, ""), occ_title.get(o, "")), 1 if str(o).endswith(".00") else 0, round(float(us_emp.get(o, 0.0)) * float(w), 1), -i, o) for i, (o, w) in enumerate(zip(g.onet_soc_code, g.weight.fillna(0)))]
        top_code[c] = max(cands)[4]   # a shared title word first, then a base occupation over its specialisations (OEWS splits a code's employment evenly across them), then overlap
    n_tasks = tasks.groupby("nat_code").size().to_dict() if tasks is not None else {}
    std_by_soc = {}
    standards, routes = None, None
    if spec.code == "GBR" and _exists("uk_ifate_standards") and _exists("uk_ifate_standard_soc"):
        stds = load("uk_ifate_standards").reset_index(drop=True)
        ss = load("uk_ifate_standard_soc")
        soc_of = {r: [[c, _r(s, 2)] for c, s in zip(g.soc_code, g.share)] for r, g in ss.groupby("reference")}
        standards = [{"ref": r.reference, "title": r.title, "lvl": _r(r.level), "route": r.route, "url": r.url, "soc": soc_of.get(r.reference, []), "nd": int(len(r.duties)), "nk": int(len(r.knowledges)), "ns": int(len(r.skills)), "nb": int(len(r.behaviours)), "dur": _r(r.typical_duration_months), "m": "title" if (soc_of.get(r.reference) and any(ss[(ss.reference == r.reference)].method == "title_in_coding_index")) else "text"} for r in stds.itertuples()]
        std_index = {s["ref"]: i for i, s in enumerate(standards)}
        for r in ss.itertuples():
            std_by_soc.setdefault(r.soc_code, []).append(std_index[r.reference])
        if _exists("uk_ifate_route_view"):
            rv = load("uk_ifate_route_view")
            routes = [{"route": r.route, "n": _r(r.n_standards), "nsoc": _r(r.n_soc_codes), "emp": _r(r.employment_thousands, 1), "levels": r.levels, "oa": _pct(r.observed_automation), "so": _pct(r.share_weight_observed), "sp": _pct(r.share_weight_physical), "db30": _pct(r.done_by_2030), "db35": _pct(r.done_by_2035), "db35f": _pct(r.done_by_2035_p10), "db35s": _pct(r.done_by_2035_p90), "sb35": _pct(r.share_by_2035), "cap": _pct(r.ceiling_ai_share), "hq": [_r(r.quarter_done_year_p10, 1), _r(r.quarter_done_year_p50, 1), _r(r.quarter_done_year_p90, 1)]} for r in rv.itertuples()]
    yf, yt = (cs.get("outcomes") or {}).get("years") or [None, None]
    occ_rows = []
    for _, r in t.iterrows():
        o = outc.loc[r.nat_code] if (outc is not None and r.nat_code in outc.index) else None
        oc = None
        if o is not None:
            wcols = [c for c in o.index if c.startswith("wage_") and c.split("_")[1].isdigit()]
            wcols = sorted(wcols, key=lambda c: int(c.split("_")[1]))
            oc = [_r(o.get(f"emp_{yf}"), 1), _r(o.get(f"emp_{yt}"), 1), _r(o.get("emp_change_pct"), 1), _r(o.get(wcols[0])) if wcols else None, _r(o.get(wcols[-1])) if wcols else None, _r(o.get("wage_change_pct"), 1)]
        top = top_code.get(r.nat_code)
        ui = us_idx.get(top)
        occ_rows.append({"c": r.nat_code, "t": r.nat_title, "mg": r.major_group, "sm": r.submajor_label, "smn": r.submajor_name if isinstance(r.submajor_name, str) else None, "lvl": _r(r.skill_level, 1) if pd.notna(r.skill_level) else None, "emp": _r(r.employment_thousands, 1), "wage": _r(r.get("wage_median", r.get("wage_weekly"))), "ws": r.wage_source if isinstance(r.wage_source, str) else None,
                         "oa": _pct(r.observed_automation), "so": _pct(r.share_weight_observed), "sp": _pct(r.share_weight_physical), "ai": _r(r.get("ai_applicability_score"), 2), "sb": _sb(r) if pd.notna(r.get("share_by_2035")) else None, "sb10": _sbq(r, "p10"), "sb90": _sbq(r, "p90"), "tiers": _tiers(r), "shift": _r(sh.get(r.nat_code), 1) if r.nat_code in sh.index else None, "h": _half(r),
                         "db": _db(r), "db10": _db(r, "p10"), "db90": _db(r, "p90"), "cap": _pct(r.get("ceiling_ai_share")), "hq": _done_years(r, "quarter"), "hd": _done_years(r, "half"), "oc": oc, "bi": _r(r.get("barrier_index"), 2) if pd.notna(r.get("barrier_index")) else None,
                         "onet": top_onet.get(r.nat_code, []), "nt": int(n_tasks.get(r.nat_code, 0)), "us": ui, "sg": (us_occ[ui].get("sg") if ui is not None else None), "std": std_by_soc.get(r.nat_code, [])})
    economy = [_econ_payload_row(r) for _, r in econ.iterrows()]
    industries = None
    if iv is not None:
        industries = [{"code": r.industry, "name": r.label, "es": _pct(r.employment_share), "ai": _r(r.ai_use_latest_pct, 1) if pd.notna(r.ai_use_latest_pct) else None, "aid": _r(r.ai_use_latest_date, 2) if pd.notna(r.ai_use_latest_date) else None, "ai0": _r(r.ai_use_first_pct, 1) if pd.notna(r.ai_use_first_pct) else None, "aid0": _r(r.ai_use_first_date, 2) if pd.notna(r.ai_use_first_date) else None, "shift": _r(r.anchor_shift_years, 1), "sv": bool(r.surveyed),
                       "oa": _pct(r.get("observed_automation")), "so": _pct(r.get("share_weight_observed")), "sp": _pct(r.get("share_weight_physical")), "db30": _pct(r.get("done_by_2030")), "db35": _pct(r.get("done_by_2035")), "db35f": _pct(r.get("done_by_2035_p10")), "db35s": _pct(r.get("done_by_2035_p90")), "sb35": _pct(r.get("share_by_2035")), "cap": _pct(r.get("ceiling_ai_share")), "hq": [_r(r.get("quarter_done_year_p10"), 1), _r(r.get("quarter_done_year_p50"), 1), _r(r.get("quarter_done_year_p90"), 1)]} for _, r in iv.iterrows()]
    levels_national = None
    if bands is not None:
        bs = cs.get("skill_levels_national") or {}
        levels_national = {"bands": [[_r(r.level), r.level_name, _r(r.n_codes), _pct(r.employment_share), _pct(r.share_weight_observed), _pct(r.observed_automation), _pct(r.done_by_2035), _pct(r.get("done_by_2035_p10")), _pct(r.get("done_by_2035_p90")), [_r(r.get("quarter_done_year_p10"), 1), _r(r.get("quarter_done_year_p50"), 1), _r(r.get("quarter_done_year_p90"), 1)]] for _, r in bands.sort_values("level").iterrows()],
                           "rho": _r(bs.get("rho_level_vs_done_2035"), 2) if bs.get("rho_level_vs_done_2035") is not None else None, "rho_seen": _r(bs.get("rho_level_vs_seen"), 2) if bs.get("rho_level_vs_seen") is not None else None, "n": bs.get("n_codes"), "reading": bs.get("reading"), "max_level": int(max(spec.skill_level_names))}
    lead = cs.get("lead") or {}
    adoption = [[_r(x.get("date"), 2), _r(x.get("pct"), 1), x.get("label")] for x in (lead.get("points") or [])]
    # O*NET occupation (page order) -> the national code holding most of its workers, for task search results
    nat_idx = {r["c"]: i for i, r in enumerate(occ_rows)}
    emp_of = {r["c"]: (r["emp"] or 0.0) for r in occ_rows}
    best = {}
    for o, g in xw.groupby("onet_soc_code"):
        k = max(((_title_overlap(occ_title.get(o, ""), nat_title.get(c, "")), emp_of.get(c, 0.0) * float(w), c) for c, w in zip(g.nat_code, g.weight.fillna(0))), default=None)
        if k and k[2] in nat_idx:
            best[o] = nat_idx[k[2]]
    o2n = [best.get(e["c"]) for e in onet_list]
    adoption_now_text = (lead.get("points") or [{}])[-1].get("label") if spec.code == "AUS" else None
    rlev = load(f"{p}rating_level_exposure") if _exists(f"{p}rating_level_exposure") else None
    rcorr = load(f"{p}rating_level_correlations") if _exists(f"{p}rating_level_correlations") else None
    levels_nat = None
    if rlev is not None and len(rlev):
        lv_r = rlev.rename(columns={"n_occupations": "n_occupations"})
        levels_nat = _levels_payload(lv_r, rcorr)
        idxname = next((n for n in rlev.element_name.unique() if n.startswith("All ")), None)
        if idxname and rcorr is not None and idxname in set(rcorr.element_name):
            cr = rcorr.set_index("element_name").loc[idxname]
            levels_nat["index_rho"] = _r(cr.rho_level_vs_done_2035, 2) if pd.notna(cr.rho_level_vs_done_2035) else None
            levels_nat["index_rho_seen"] = _r(cr.rho_level_vs_seen, 2) if pd.notna(cr.rho_level_vs_seen) else None
            levels_nat["index_n"] = _r(cr.n_occupations)
        levels_nat["index_name"] = idxname
        levels_nat["bands"] = [[b, name] for b, name, lo, hi in spec.rating_bands]
        levels_nat["framework"] = spec.framework_name
    office = {"AUS": "ABS", "GBR": "ONS", "CAN": "OaSIS"}.get(spec.code, spec.name)
    payload = {"code": spec.code, "name": spec.name, "adj": spec.adjective, "adoption_now_text": (f"{adoption_now_text} financial year" if adoption_now_text else None), "office": office, "lvl_desc": spec.level_descending, "levels_nat": levels_nat, "framework": spec.framework_name, "sys": {"AUS": "ANZSCO", "GBR": "SOC 2020", "CAN": "NOC 2021"}.get(spec.code, spec.classification), "cls": spec.classification, "cur": spec.currency, "emp_label": spec.emp_label, "wage_label": spec.wage_label, "wage_period": spec.wage_period, "adoption_source": spec.adoption_source, "industry_label": spec.industry_label, "major_names": spec.major_names,
               "occ": occ_rows, "economy": economy, "industries": industries, "levels_national": levels_national, "levels": _levels_payload(levels, lcorr), "outcomes": cs.get("outcomes") or {}, "weights": cs.get("weights") or {}, "shift_range": cs.get("shift_range"), "lead": _r(lead.get("lead_years"), 1) if lead.get("lead_years") is not None else None, "lead_points_used": lead.get("n_points_used"), "adoption": adoption, "adoption_now": _r(lead.get("latest_pct"), 1) if lead.get("latest_pct") is not None else None, "adoption_now_date": _r(lead.get("latest_date"), 2) if lead.get("latest_date") is not None else None,
               "standards": standards, "routes": routes, "industry_shifts": cs.get("industry_shifts") or {}, "o2n": o2n}
    # on-demand file: the official task list per code, matched to O*NET tasks, on this country's timing
    files = {"occ": {}, "std": {}}
    if tasks is not None and links is not None:
        top_links = links[links["rank"] == 1].set_index(["nat_code", "task_seq"])
        desc = load(f"{p}descriptions").set_index(spec.code_col) if _exists(f"{p}descriptions") else None
        st_desc = load("uk_structure").set_index("soc_code") if (spec.code == "GBR" and _exists("uk_structure")) else None
        nat_ratings = load(f"{p}ratings") if (spec.ratings is not None and _exists(f"{p}ratings")) else None
        rat_by = {c: g for c, g in nat_ratings.groupby(spec.code_col)} if nat_ratings is not None else {}
        for c, g in tasks.groupby("nat_code"):
            items = []
            for r in g.sort_values("task_seq").itertuples():
                key = (c, int(r.task_seq))
                if key not in top_links.index:
                    continue
                lk = top_links.loc[key]
                f = ft.loc[lk.task_id] if lk.task_id in ft.index else None
                items.append(_task_item(r.task, f, lk.score, onet_tasks.get(lk.task_id), [1 if bool(lk.in_mapped_occ) else 0]))
            items.sort(key=lambda it: (it[2] is None, it[2] or 9999))
            entry = {"g": [["Official tasks", items]]}
            if desc is not None and c in desc.index:
                entry["desc"] = str(desc.loc[c].description)[:600]
                entry["skill"] = str(desc.loc[c].skill_text)[:400]
            if st_desc is not None and c in st_desc.index:
                entry["desc"] = str(st_desc.loc[c].description or "")[:600]
                entry["entry"] = str(st_desc.loc[c].entry_routes or "")[:500]
            if c in rat_by:
                g = rat_by[c]
                fam_order = {"skill": 0, "work_activity": 1, "knowledge": 2}
                g = g[g.level > 0].assign(_f=g.family.map(fam_order)).sort_values(["_f", "level"], ascending=[True, False])
                entry["k"] = [[str(k.family), str(k.element_name), _r(k.level, 1)] for _, k in g.head(24).iterrows()]
            files["occ"][c] = entry
    if standards is not None and _exists("uk_ifate_duty_onet_task"):
        stds = load("uk_ifate_standards").set_index("reference")
        dl = load("uk_ifate_duty_onet_task").set_index(["reference", "duty_seq"])
        for s in standards:
            r = stds.loc[s["ref"]]
            duties = []
            for i, d in enumerate(r.duties):
                key = (s["ref"], i)
                if key not in dl.index:
                    continue
                lk = dl.loc[key]
                f = ft.loc[lk.task_id] if lk.task_id in ft.index else None
                duties.append(_task_item(d, f, lk.score, onet_tasks.get(lk.task_id), [1 if (i < len(r.duty_core) and bool(r.duty_core[i])) else 0]))
            duties.sort(key=lambda it: (it[2] is None, it[2] or 9999))
            files["std"][s["ref"]] = {"d": duties, "k": [str(x)[:220] for x in r.knowledges], "s": [str(x)[:220] for x in r.skills], "b": [str(x)[:220] for x in r.behaviours], "sum": str(r.summary)[:600]}
    return payload, files
