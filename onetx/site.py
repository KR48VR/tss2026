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
        ssoc_list.append({"c": r.ssoc_code, "t": r.ssoc_title, "mg": r.major_group, "sm": r.submajor_label, "smn": r.submajor_name, "emp": _r(r.employment_thousands, 1), "wage": _r(r.wage_median), "ws": r.wage_source, "oa": _pct(r.observed_automation), "so": _pct(r.share_weight_observed), "sp": _pct(r.share_weight_physical), "ai": _r(r.ai_applicability_score, 2), "sb": _sb(r) if f is not None else None, "h": _half(r), "bi": _r(f.barrier_index, 2) if f is not None else None, "onet": top_onet.get(r.ssoc_code, []), "roles": roles_by_ssoc.get(r.ssoc_code, [])})

    # ---- Skills Framework roles with key tasks
    ks = kt_scores.merge(kt[["key_task", "cwf"]], left_on="key_task_id", right_index=True, how="left")
    ks = ks.merge(ft[["status", "disrupt_year_p50", "disrupt_year_p10", "disrupt_year_p90", "barrier_type"]], left_on="top_task_id", right_index=True, how="left")
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
                items = [[str(k.key_task)[:150], STATUS_CODE.get(k.status, 3), _r(k.disrupt_year_p50, 1), BARRIER_CODE.get(k.barrier_type, 0), _r(k.top_score, 2), str(k.task_text)[:110] if isinstance(k.task_text, str) else None] for _, k in gg.sort_values("disrupt_year_p50").iterrows()]
                groups.append([str(cwf)[:90], items])
        key_tasks.append(groups)
        role_list.append({"sector": r.sector, "track": r.track if isinstance(r.track, str) else "", "role": r.job_role, "desc": (r.job_role_description or "")[:360] if isinstance(r.job_role_description, str) else "", "ssoc": [[c, s, sc, ssoc_title.get(c, "")] for c, s, sc in ssoc_by_role.get(r.role_id, [])], "oat": _pct(r.observed_automation_via_tasks), "oas": _pct(r.observed_automation_via_ssoc), "sko": _pct(r.share_key_tasks_observed), "skp": _pct(r.share_key_tasks_physical), "nk": _r(r.n_key_tasks), "ms": _r(r.mean_match_score, 2), "sb": _sb(f) if f is not None else None, "h": _half(f) if f is not None else [None, None, None], "bi": _r(f.barrier_index, 2) if f is not None else None})

    # ---- O*NET occupations
    onet_list = []
    for _, r in occ.sort_values("title").iterrows():
        f = focc.loc[r.onet_soc_code] if r.onet_soc_code in focc.index else None
        onet_list.append({"c": r.onet_soc_code, "t": r.title, "oa": _pct(r.observed_automation), "so": _pct(r.share_weight_observed), "sp": _pct(r.share_weight_physical), "ms": _r(r.ai_applicability_score, 2), "oe": _r(r.openai_exposure_beta_gpt4, 2), "g": r.google_atlas_flag if isinstance(r.google_atlas_flag, str) else None, "np": _r(r.n_providers_positive), "h": _half(f) if f is not None else [None, None, None], "sb": _sb(f) if f is not None else None, "gd": bool(r.gdpval_covered)})

    skills_list = [{"f": r.family, "name": r.element_name, "n": _r(r.n_occupations), "oa": _pct(r.via_occupations_observed_automation), "oaa": _pct(r.via_activities_observed_automation), "h": _half(r), "sb": _sb(r)} for _, r in skills.sort_values(["family", "element_name"]).iterrows()]
    gwa_list = [{"name": r["name"], "n": _r(r.n_tasks), "so": _pct(r.share_tasks_observed), "oa": _pct(r.observed_automation), "ms": _r(r.ms_share * 100, 1) if pd.notna(r.ms_share) else None, "h": _half(r), "sb": _sb(r)} for _, r in act[act.level == "gwa"].sort_values("observed_automation", ascending=False).iterrows()]
    sector_list = [{"sector": r.sector, "n": _r(r.n_roles), "oat": _pct(r.observed_automation_via_tasks), "sko": _pct(r.share_key_tasks_observed), "skp": _pct(r.share_key_tasks_physical), "sb30": _pct(r.share_by_2030), "sb35": _pct(r.share_by_2035), "h50": _r(r.half_year_p50_median_role, 1), "h90": _r(r.half_year_p90_median_role, 1)} for _, r in sectors.iterrows()]

    fp = stats.get("forecast_params", {})
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
    }
    return {"meta": meta, "economy": economy, "ssoc": ssoc_list, "roles": role_list, "onet": onet_list, "skills": skills_list, "gwa": gwa_list, "sectors": sector_list, "_key_tasks": key_tasks}


def build_site(companion_url: str | None = None) -> Path:
    """Write site/index.html (page + core data) and site/kt.json (key tasks per role, loaded on demand)."""
    SITE.mkdir(parents=True, exist_ok=True)
    data = build_data(companion_url)
    key_tasks = data.pop("_key_tasks")
    js = json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("__DATA__", js)
    out = SITE / "index.html"
    out.write_text(html)
    (SITE / "kt.json").write_text(json.dumps(key_tasks, ensure_ascii=False, separators=(",", ":")))
    return out
