"""End-to-end pipeline: raw sources -> processed parquet tables -> DuckDB views -> summary report."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from . import ai_signals as ai
from . import crosswalks as xw
from . import forecast as fcm
from . import frontier as fr
from . import match
from . import onet
from . import singapore as sg
from . import singapore_economy as se
from . import singapore_labour as sl
from .config import AEI_VINTAGES, PROCESSED, ensure_dirs
from .io import register_duckdb, save, write_json


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build(forecast_params: fcm.ForecastParams | None = None, now: str = "2026-10-01") -> dict:
    ensure_dirs()
    t0 = time.time()
    stats: dict = {"now": now}

    # ---- O*NET core (30.1 classic ids; 31.0 extras)
    occ = onet.load_occupations(); save(occ, "onet_occupations"); _log("O*NET occupations", t0)
    tasks = onet.load_tasks(); save(tasks, "onet_tasks")
    save(onet.load_task_dwa(), "onet_task_dwa")
    save(onet.load_dwa_hierarchy(), "onet_activities")
    save(onet.load_dwa_hierarchy("31.0"), "onet_activities_v31")
    save(onet.load_descriptor_ratings("30.1"), "onet_descriptor_ratings")
    save(onet.load_descriptor_ratings("31.0"), "onet_descriptor_ratings_v31")
    save(onet.load_descriptor_to_work_activity("30.1"), "onet_descriptor_to_gwa")
    save(onet.load_descriptor_to_work_activity("31.0"), "onet_descriptor_to_gwa_v31")
    save(onet.load_technology_skills("30.1"), "onet_technology_skills")
    save(onet.load_technology_skills("31.0"), "onet_software_skills_v31")
    save(onet.load_content_model("30.1"), "onet_content_model")
    save(onet.load_content_model("31.0"), "onet_content_model_v31")
    save(onet.load_job_zones(), "onet_job_zones")
    save(onet.load_alternate_titles(), "onet_alternate_titles")
    save(onet.load_related_occupations(), "onet_related_occupations")
    save(onet.load_emerging_tasks("31.0"), "onet_emerging_tasks_v31")
    _log("O*NET tables", t0)

    # ---- Singapore
    for ed in (2024, 2020):
        save(sg.load_ssoc_structure(ed), f"ssoc{ed}_structure")
        save(sg.load_ssoc_definitions(ed), f"ssoc{ed}_definitions")
        save(sg.load_ssoc_tasks(ed), f"ssoc{ed}_tasks")
        save(sg.load_ssoc_alpha_index(ed), f"ssoc{ed}_alpha_index")
    save(sg.load_ssoc_2024_2020(), "xw_ssoc2024_ssoc2020")
    save(sg.load_sfw_roles(), "sfw_roles")
    kt = sg.load_sfw_key_tasks(); save(kt, "sfw_key_tasks")
    save(sg.load_sfw_role_tscs(), "sfw_role_tscs")
    save(sg.load_sfw_tsc_key(), "sfw_tsc_key")
    save(sg.load_sfw_tsc_ka(), "sfw_tsc_knowledge_abilities")
    us = sg.load_unique_skills(); save(us, "sfw_unique_skills")
    save(sg.load_tsc_to_unique_skill(), "sfw_tsc_to_unique_skill")
    _log("Singapore tables", t0)

    # ---- occupation crosswalks
    save(xw.onetsoc_to_soc2018(), "xw_onetsoc_soc2018")
    o2i = xw.onetsoc_to_isco08(); save(o2i, "xw_onetsoc_isco08")
    save(xw.isco08_to_ssoc(2024), "xw_isco08_ssoc2024")
    save(xw.isco08_to_ssoc(2020), "xw_isco08_ssoc2020")
    o2s = xw.onetsoc_to_ssoc(2024); save(o2s, "xw_onetsoc_ssoc2024")
    save(xw.onetsoc_to_ssoc(2020), "xw_onetsoc_ssoc2020")
    role_ssoc = match.build_role_ssoc_crosswalk(2024); save(role_ssoc, "xw_sfw_role_ssoc2024")
    stats["role_text_matcher_eval"] = match.evaluate_role_text_matcher(2024)
    stats["crosswalk_counts"] = {"onet_occupations": int(occ.shape[0]), "onet_isco_pairs": int(o2i.shape[0]), "onet_ssoc_pairs": int(o2s.shape[0]), "ssoc_covered": int(o2s.ssoc_code.nunique()), "sfw_roles": int(role_ssoc.role_id.nunique()), "sfw_roles_from_singstat": int(role_ssoc[role_ssoc.source == "singstat_index"].role_id.nunique())}
    _log("occupation crosswalks", t0)

    # ---- text crosswalks
    kt_links, kt_dwa = match.match_sfw_key_tasks(role_ssoc); save(kt_links, "xw_sfw_keytask_onet_task"); save(kt_dwa, "xw_sfw_keytask_onet_dwa")
    _log("key task -> O*NET task matches", t0)
    us_iwa, us_desc = match.match_unique_skills(); save(us_iwa, "xw_sfw_skill_onet_iwa"); save(us_desc, "xw_sfw_skill_onet_descriptor")
    ssoc_task_links = match.match_ssoc_tasks(2024); save(ssoc_task_links, "xw_ssoc2024_task_onet_task")
    _log("skill and SSOC-task matches", t0)

    # ---- AI signals
    aei_long = ai.load_aei_task_metrics(tasks); save(aei_long, "ai_aei_task_metrics_long")
    aei_wide = ai.aei_task_wide(aei_long); save(aei_wide, "ai_aei_task_metrics")
    save(ai.load_aei_task_penetration(tasks), "ai_aei_task_penetration")
    save(ai.load_aei_job_exposure(), "ai_aei_job_exposure")
    save(ai.load_robot_exposure_tasks(), "ai_robot_exposure_tasks")
    save(ai.load_robot_exposure_occupations(), "ai_robot_exposure_occupations")
    save(ai.load_microsoft_iwa(), "ai_microsoft_iwa")
    save(ai.load_microsoft_soc(), "ai_microsoft_soc")
    save(ai.load_gdpval(), "ai_gdpval_tasks")
    metr = ai.compute_metr_horizons(); save(metr, "ai_metr_horizons")
    trend = ai.metr_frontier_trend(metr); stats["metr_trend"] = trend
    save(ai.load_openai_gpts_exposure(), "ai_openai_gpts_exposure")
    save(ai.load_openai_gwa_shares(), "ai_openai_chatgpt_gwa_shares")
    save(ai.load_google_atlas_occupations(), "ai_google_atlas_occupations")
    save(ai.load_microsoft_diffusion_countries(), "ai_microsoft_diffusion_countries")
    btos = ai.load_btos_ai_adoption(); save(btos, "ai_btos_firm_adoption")
    stats["btos_adoption_fit"] = ai.fit_adoption_logistic(btos)
    save(ai.load_btos_sector_adoption(), "ai_btos_sector_adoption")
    save(ai.load_singapore_firm_adoption(), "sg_firm_adoption_imda")
    save(ai.load_openai_task_labels(), "ai_openai_task_labels")
    _log("AI signals", t0)

    # ---- Singapore wages and employment
    save(sl.match_ows_to_ssoc(2024), "sg_ows_wages_ssoc2024")
    save(sl.load_ows2025_wages(), "sg_ows2025_wages")
    save(sl.load_lfs_employment_major(), "sg_lfs_employment_major")
    d8 = sl.load_lfs_employment_submajor(2025); d8["submajor_members"] = d8.submajor_members.map(",".join); save(d8, "sg_lfs_employment_submajor")
    ssoc_w = sl.build_ssoc_weights(2024); save(ssoc_w, "sg_ssoc2024_weights")
    stats["sg_weights"] = {"ssoc_with_ows_wage": int(ssoc_w.in_ows.sum()), "wage_sources": ssoc_w.wage_source.value_counts().to_dict(), "employment_thousands_total": float(ssoc_w.employment_thousands.sum()), "employment_method": str(ssoc_w.employment_method.iloc[0])}
    _log("Singapore labour tables", t0)

    # ---- frontier + roll-ups
    tf = fr.build_task_frontier(); save(tf, "frontier_tasks")
    barriers = fr.task_barriers(tf); save(barriers, "frontier_task_barriers")
    act = fr.rollup_activities(tf); save(act, "frontier_activities")
    occf = fr.occupation_consensus(fr.rollup_occupations(tf)); save(occf, "frontier_occupations")
    stats["provider_consensus_counts"] = occf.n_providers_positive.value_counts().sort_index().to_dict()
    save(fr.rollup_skills(occf, act, "30.1"), "frontier_skills")
    save(fr.rollup_skills(occf, act, "31.0"), "frontier_skills_v31")
    ssoc_scores = fr.push_to_ssoc(occf, o2s, 2024); save(ssoc_scores, "frontier_ssoc2024")
    roles_scored, kt_scored = fr.push_to_sfw(role_ssoc, ssoc_scores, kt_links, tf); save(roles_scored, "frontier_sfw_roles"); save(kt_scored, "frontier_sfw_key_tasks")
    stats["task_status_counts"] = tf.status.value_counts().to_dict()
    stats["barrier_type_counts"] = barriers.barrier_type.value_counts().to_dict()
    _log("frontier scores", t0)

    # ---- forecast
    stats.update(build_forecasts(now=now, forecast_params=forecast_params, metr_trend=trend))
    _log("forecasts", t0)

    register_duckdb()
    write_json(stats, PROCESSED / "build_stats.json")
    _log("DuckDB views + stats", t0)
    return stats


def build_forecasts(now: str = "2026-10-01", forecast_params: fcm.ForecastParams | None = None, metr_trend: dict | None = None) -> dict:
    """Re-run only the timing model from the processed frontier tables (fast; use it to try parameters)."""
    from .io import load

    tf = load("frontier_tasks"); occ = load("onet_occupations"); o2s = load("xw_onetsoc_ssoc2024"); kt_links = load("xw_sfw_keytask_onet_task")
    kt = load("sfw_key_tasks"); act = load("frontier_activities"); occf = load("frontier_occupations")
    barriers = load("frontier_task_barriers") if (PROCESSED / "frontier_task_barriers.parquet").exists() else fr.task_barriers(tf)
    if metr_trend is None:
        metr_trend = ai.metr_frontier_trend(load("ai_metr_horizons"))
    params = forecast_params or fcm.ForecastParams(doubling_days_central=round(metr_trend["doubling_days"]), log2_h50_at_anchor=round(metr_trend["log2_horizon_at_2024_01_01"], 3))
    params = fcm.calibrate(params, tf)
    btos = ai.load_btos_ai_adoption()
    btos_fit = ai.fit_adoption_logistic(btos)
    ratio = fcm.singapore_usage_ratio()
    sg_lead = ai.singapore_firm_lead(btos_fit, ai.load_singapore_firm_adoption())
    params = fcm.calibrate_adoption(params, btos_fit, ratio.get("geometric_mean"), sg_lead.get("lead_years"))
    sector_levels = ai.fit_sector_levels(ai.load_btos_sector_adoption(), btos_fit); save(sector_levels, "ai_btos_sector_levels")
    # capability evidence tiers: 2023 task ratings (human + GPT-4) and the 2026 Claude ratings when the run is at least 90% complete
    from . import rate as rt
    labels = load("ai_openai_task_labels") if (PROCESSED / "ai_openai_task_labels.parquet").exists() else None
    ratings = rt.load_ratings()
    ratings = ratings if len(ratings) >= 0.9 * len(tf) else None
    tiers = fcm.evidence_tiers(tf, labels, ratings); save(tiers, "frontier_task_evidence")
    params, ev_stats = fcm.calibrate_evidence(params, load("ai_aei_task_metrics_long"), tiers, tf)
    params, ai_share_stats = fcm.calibrate_ai_share(params, tf, tiers)
    # sector-specific adoption: industry shifts from the Census sector series, carried to SSOC codes, O*NET occupations and roles
    from . import sector_adoption as sa
    shifts = sa.build_shifts(sector_levels)
    save(shifts["industry"], "ai_adoption_shift_industry"); save(shifts["ssoc"], "sg_adoption_shift_ssoc2024"); save(shifts["onet"], "sg_adoption_shift_onet"); save(shifts["role"], "sg_adoption_shift_sfw_roles"); save(shifts["mix"], "sg_ssoc2024_industry_mix")
    fc = fcm.forecast_tasks(tf, barriers, params, now=now, tiers=tiers, occ_shift=shifts["onet"]); save(fc, "forecast_tasks")
    # track record of the ratings against observed use, and the ladder conversions for the claim translator
    from . import validation as val
    rep_path = rt.REF_DIR / f"ai_task_ratings_{rt.rubric()[0]}_repeat.csv"
    repeat = pd.read_csv(rep_path, dtype=str) if rep_path.exists() else None
    validation = val.rating_validation(tf, labels, rt.load_ratings() if len(rt.load_ratings()) else None, repeat)
    now_y = fcm._frac_year(pd.Timestamp(now))
    stats = {"forecast_params": {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in fcm.asdict(params).items()}, "adoption_anchor_years": {"fast": fcm.adoption_cross_year(params, 0.5, params.adoption_growth_mult[0]), "central": fcm.adoption_cross_year(params, 0.5, 1.0), "slow": fcm.adoption_cross_year(params, 0.5, params.adoption_growth_mult[2])}, "geo_lead_years_central": fcm.geo_lead_years(params), "singapore_usage_ratio": ratio, "singapore_firm_lead": sg_lead, "btos_adoption_fit": btos_fit, "validation": validation, "ai_share": ai_share_stats, "sector_adoption": {"industry_shift_years": {r.industry: round(float(r.anchor_shift_years), 2) for r in shifts["industry"].itertuples()}, "ssoc_shift_range": [float(shifts["ssoc"].adoption_shift_years.min()), float(shifts["ssoc"].adoption_shift_years.max())], "imda_check": sa.check_against_imda(shifts["industry"], ai.load_singapore_firm_adoption())}, "evidence": {**ev_stats, "tiers": tiers.evidence_tier.value_counts().to_dict(), "ratings_2026_used": ratings is not None, "ratings_2026_rows": int(len(rt.load_ratings()))},
             "data_as_of": {"aei_latest_vintage": max(v["vintage"] for v in AEI_VINTAGES) if AEI_VINTAGES and isinstance(AEI_VINTAGES[0], dict) else str(tf.date.max())[:10] if "date" in tf else None, "btos_latest_reading": btos_fit["last_ref_end"], "imda_latest_year": int(ai.load_singapore_firm_adoption().year.max()), "ratings_2023": "2023", "ratings_2026": (str(rt.load_ratings().rated_at.max())[:10] if len(rt.load_ratings()) else None), "now": now, "build_date": pd.Timestamp.today().strftime("%Y-%m-%d")}}
    occ_curve = fcm.disruption_curve(fc, "onet_soc_code").merge(occ[["onet_soc_code", "title"]], on="onet_soc_code", how="left"); save(occ_curve, "forecast_occupations")
    f2s = fc.merge(o2s[["onet_soc_code", "ssoc_code", "ssoc_title", "weight_rev"]], on="onet_soc_code")
    f2s["w"] = f2s.task_weight * f2s.weight_rev
    f2s = fcm.reshift_years(f2s, f2s.ssoc_code.map(shifts["ssoc"].set_index("ssoc_code").adoption_shift_years), params, now_y)
    ssoc_curve = fcm.disruption_curve(f2s, "ssoc_code", weight="w").merge(f2s[["ssoc_code", "ssoc_title"]].drop_duplicates("ssoc_code"), on="ssoc_code", how="left"); save(ssoc_curve, "forecast_ssoc2024")
    k1 = kt_links[kt_links["rank"] == 1][["key_task_id", "task_id", "score"]].merge(kt[["key_task_id", "role_id"]], on="key_task_id").merge(fc.drop(columns=["task_weight"]), on="task_id")
    k1["task_weight"] = 1.0
    k1 = fcm.reshift_years(k1, k1.role_id.map(shifts["role"].set_index("role_id").adoption_shift_years), params, now_y)
    role_curve = fcm.disruption_curve(k1, "role_id").merge(sg.load_sfw_roles()[["role_id", "sector", "track", "job_role"]], on="role_id", how="left"); save(role_curve, "forecast_sfw_roles")
    td = onet.load_task_dwa()[["task_id", "dwa_id"]].merge(onet.load_dwa_hierarchy()[["dwa_id", "iwa_id", "gwa_id"]], on="dwa_id")
    fa = fc.merge(td, on="task_id")
    frames = []
    for lvl in ["dwa_id", "iwa_id", "gwa_id"]:
        c = fcm.disruption_curve(fa, lvl).rename(columns={lvl: "activity_id"}); c["level"] = lvl[:3]; frames.append(c)
    act_curve = pd.concat(frames, ignore_index=True).merge(act[["activity_id", "name"]].drop_duplicates("activity_id"), on="activity_id", how="left"); save(act_curve, "forecast_activities")
    save(fcm.skill_curves(fc, occf), "forecast_skills")
    stats["forecast_buckets"] = fc.timing_bucket.value_counts().to_dict()
    # Singapore economy: employment- and wage-weighted views
    ssoc_w = load("sg_ssoc2024_weights") if (PROCESSED / "sg_ssoc2024_weights.parquet").exists() else sl.build_ssoc_weights(2024)
    swt = se.ssoc_weighted_table(load("frontier_ssoc2024"), ssoc_curve, ssoc_w); save(swt, "sg_ssoc2024_weighted")
    econ = se.economy_exposure(swt); save(econ, "sg_economy_exposure")
    save(se.sector_view(load("frontier_sfw_roles"), role_curve), "sg_sfw_sectors")
    # rung five: what has happened to jobs so far, and whether the map's signals relate to it yet
    from . import outcomes as oc
    try:
        outc = oc.build_outcomes()
        save(outc["ssoc"], "sg_outcomes_ssoc2024"); save(outc["submajor"], "sg_outcomes_submajor"); save(outc["jv_shares_minor"], "sg_jv_shares_minor")
        save(outc["retrenchment"], "sg_retrenchment_by_group"); save(outc["vacancy_rate"], "sg_vacancy_rate_by_group"); save(outc["resignation_rate"], "sg_resignation_rate_by_group")
        stats["outcomes"] = oc.outcome_tests(outc, ssoc_curve, load("frontier_ssoc2024"), econ)
        stats["data_as_of"].update({"mom_vacancies": "Sep 2025", "mom_employment_by_occupation": "Jun 2025", "mom_retrenchment": int(outc["retrenchment"].year.max()) if len(outc["retrenchment"]) else None, "mom_wages_change": "Jun 2024 to Jun 2025"})
    except FileNotFoundError as e:
        stats["outcomes"] = {"error": f"outcome files missing: {e}"}
    e = econ[(econ.scope == "economy")].set_index("weighting")
    stats["singapore_economy"] = {w: {"observed_automation": float(e.loc[w, "observed_automation"]), "share_weight_observed": float(e.loc[w, "share_weight_observed"]), "share_by_2030": float(e.loc[w, "share_by_2030"]), "share_by_2035": float(e.loc[w, "share_by_2035"]), "half_year_central": float(e.loc[w, "half_year_central_from_curve"])} for w in ["employment", "wage_value"]}
    register_duckdb()
    return stats
