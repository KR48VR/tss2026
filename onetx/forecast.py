"""Disruption-timing model. Transparent, parameterised, and reported as ranges.

For each O*NET task we answer two questions and add an explicit adoption lag.

1. WHEN CAN AI DO IT?  (capability)
   Non-physical tasks. Today's AI is already used to *do* (not just help with) tasks of an hour or
   two; its "delegation horizon" moves with the capability frontier. We calibrate the horizon on the
   Anthropic data: among observed tasks, the share of use that is automation-style falls with the task's
   human-only length, and the length at which that share crosses 50% is today's effective delegation
   horizon (a few hours). We then grow that horizon at METR's measured doubling rate (about 3.5 months
   since 2024, fast/slow variants) and call a task "capable" when the horizon reaches its length.
   A task already observed being automated is capable now, whatever the length model says.
   Physical tasks. Anthropic's robot-exposure tier (E1 purpose-built cell ... E3 unstructured setting)
   is turned into a years-from-now offset with wide ranges. E0 = no cited system can do it: beyond
   horizon.

2. HOW LONG UNTIL ORGANISATIONS LET IT?  (adoption lag)
   Firm-level AI adoption has been climbing from single digits since 2023; a central lag of 3 years
   from "capable" to "widely adopted" with a 1.5-6 year range is our assumption. The lag is shortened
   where the task is already observed being delegated, lengthened where usage data say people can
   rarely hand the task over (human_only_ability >= 95%) and for tasks in high-contact occupations
   (O*NET work context: physical proximity and face-to-face discussion) that are not yet observed.

Outputs per task: disrupt_year_p10 / p50 / p90 (fast / central / slow) plus the components. Roll-ups
report, for each occupation / SSOC code / Skills Framework role / activity / skill, the share of task
weight whose central year is at or before each calendar year, and the first year at which that share
passes 50% ("half-way year").

"Disruptable" means: the task can be done by AI at workplace reliability and typical organisations
have had time to adopt it. It is not a forecast of job loss: tasks change, roles re-bundle, demand can
rise. Every number is a planning signal with a range.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import onet


@dataclass
class ForecastParams:
    anchor_date: str = "2024-01-01"
    log2_h50_at_anchor: float = 1.91          # METR fit: 50% horizon ~ 3.8 min at 2024-01-01 (benchmark tasks)
    doubling_days_central: float = 106.0      # METR frontier doubling since 2024 (our fit)
    doubling_days_fast: float = 85.0
    doubling_days_slow: float = 212.0         # METR's long-run 2019-2025 figure (~7 months)
    # effective real-world delegation horizon today (minutes) and the logistic slope of
    # automation-share against log2(minutes); both overwritten by calibrate() from the data
    delegation_horizon_now_min: float = 120.0
    delegation_slope: float = -0.25
    adoption_lag_years: tuple = (1.5, 3.0, 6.0)   # fast / central / slow
    observed_lag_bonus_years: float = 1.0     # subtract where usage already observed with automation
    hard_to_delegate_penalty_years: float = 1.0   # add where human_only_ability_pct >= threshold
    hard_to_delegate_threshold: float = 95.0
    in_person_penalty_years: float = 2.0      # add (scaled by in_person_index) for not-yet-observed tasks
    robot_tier_years: dict = field(default_factory=lambda: {1: (2, 4, 8), 2: (4, 7, 12), 3: (8, 12, 20), 0: (np.nan, np.nan, np.nan)})
    horizon_cap_year: float = 2045.0


def calibrate(p: ForecastParams, tf: pd.DataFrame) -> ForecastParams:
    """Fit automation_share ~ sigmoid(a + b log2 minutes) on observed non-physical tasks; the 50% point
    is today's delegation horizon."""
    d = tf[(~tf.physical) & (tf.usage_latest > 0) & tf.human_only_hours.notna() & tf.automation_share.notna()].copy()
    x = np.log2(d.human_only_hours.values * 60.0)
    y = d.automation_share.clip(1e-3, 1 - 1e-3).values
    logit = np.log(y / (1 - y))
    b, a = np.polyfit(x, logit, 1)
    p = ForecastParams(**{**asdict(p)})
    if b < 0:
        p.delegation_slope = float(b)
        p.delegation_horizon_now_min = float(2 ** (-a / b))
    return p


def in_person_index(version: str = "30.1") -> pd.Series:
    """0-1 index per occupation from O*NET work context: percentile rank of the mean of Physical
    Proximity, Face-to-Face Discussions and Contact With Others (context scale CX, 1-5). Nearly every
    job scores high on these in absolute terms, so the rank is what discriminates."""
    try:
        wc = onet._read(version, "Work Context")
    except FileNotFoundError:
        return pd.Series(dtype=float)
    wc.columns = [onet._snake(c) for c in wc.columns]
    wc = wc[(wc.scale_id == "CX") & wc.element_name.str.contains("Physical Proximity|Face-to-Face Discussions|Contact With Others", regex=True)]
    wc["data_value"] = pd.to_numeric(wc.data_value, errors="coerce")
    s = wc.groupby("onet_soc_code").data_value.mean()
    # percentile rank across occupations: 1 = the most in-person job, 0 = the least
    return s.rank(pct=True)


def impute_task_minutes(tf: pd.DataFrame) -> pd.DataFrame:
    """human_only_minutes with imputation chain: task (AEI) -> DWA median -> IWA median -> GWA median -> Job Zone median."""
    t = tf[["task_id", "onet_soc_code", "human_only_hours"]].copy()
    t["minutes"] = t.human_only_hours * 60.0
    t["minutes_source"] = np.where(t.minutes.notna(), "aei_task", None)
    td = onet.load_task_dwa()[["task_id", "dwa_id"]]
    h = onet.load_dwa_hierarchy()[["dwa_id", "iwa_id", "gwa_id"]]
    m = t.merge(td, on="task_id", how="left").merge(h, on="dwa_id", how="left")
    for lvl in ["dwa_id", "iwa_id", "gwa_id"]:
        means = m.dropna(subset=["minutes"]).groupby(lvl).minutes.median()
        fill = m.assign(f=m[lvl].map(means)).groupby("task_id").f.mean()
        need = t.minutes.isna()
        t.loc[need, "minutes"] = t.loc[need, "task_id"].map(fill)
        t.loc[need & t.minutes.notna(), "minutes_source"] = lvl.replace("_id", "_median")
    jz = onet.load_job_zones().set_index("onet_soc_code").job_zone
    t["job_zone"] = t.onet_soc_code.map(jz)
    jz_med = t.dropna(subset=["minutes"]).groupby("job_zone").minutes.median()
    need = t.minutes.isna()
    t.loc[need, "minutes"] = t.loc[need, "job_zone"].map(jz_med)
    t.loc[need & t.minutes.notna(), "minutes_source"] = "job_zone_median"
    t["minutes"] = t["minutes"].fillna(float(t.minutes.median()))
    t["minutes_source"] = t["minutes_source"].fillna("global_median")
    return t[["task_id", "minutes", "minutes_source"]]


def _frac_year(ts: pd.Timestamp) -> float:
    return ts.year + (ts.dayofyear - 1) / 365.25


def forecast_tasks(tf: pd.DataFrame, params: ForecastParams | None = None, now: str = "2026-10-01") -> pd.DataFrame:
    p = params or ForecastParams()
    now_y = _frac_year(pd.Timestamp(now))
    mins = impute_task_minutes(tf)
    d = tf[["task_id", "onet_soc_code", "task_weight", "physical", "robot_tier", "status", "usage_latest", "automation_share", "observed_automation", "human_only_ability_pct", "usage_index", "usage_trend"]].merge(mins, on="task_id")
    ipi = in_person_index()
    d["in_person_index"] = d.onet_soc_code.map(ipi).fillna(float(ipi.median()) if len(ipi) else 0.5)

    # doublings needed for the delegation horizon to reach the task's length (negative = already there)
    doublings = np.log2(d.minutes.values / p.delegation_horizon_now_min)
    cap = {}
    for name, dd in [("fast", p.doubling_days_fast), ("central", p.doubling_days_central), ("slow", p.doubling_days_slow)]:
        cap[name] = now_y + np.maximum(doublings, 0.0) * dd / 365.25
    for name, i in [("fast", 0), ("central", 1), ("slow", 2)]:
        off = d.robot_tier.map(lambda t: p.robot_tier_years.get(int(t), (np.nan, np.nan, np.nan))[i] if pd.notna(t) else np.nan).astype(float).values
        cap[name] = np.where(d.physical.values, now_y + off, cap[name])
    observed = (d.status == "observed_automation").values
    for name in cap:
        cap[name] = np.where(observed & ~d.physical.values, now_y, cap[name])

    lag_fast, lag_central, lag_slow = p.adoption_lag_years
    bonus = np.where(observed, p.observed_lag_bonus_years, 0.0)
    hard = (d.human_only_ability_pct.fillna(0) >= p.hard_to_delegate_threshold).values
    not_obs = (d.status == "not_observed").values
    penalty = np.where(hard, p.hard_to_delegate_penalty_years, 0.0) + np.where(not_obs, p.in_person_penalty_years * d.in_person_index.values, 0.0)
    out = d.copy()
    out["capable_year_fast"], out["capable_year_central"], out["capable_year_slow"] = cap["fast"], cap["central"], cap["slow"]
    out["adoption_lag_central"] = np.maximum(lag_central - bonus, 0.5) + penalty
    out["disrupt_year_p10"] = out.capable_year_fast + np.maximum(lag_fast - bonus, 0.25) + 0.5 * penalty
    out["disrupt_year_p50"] = out.capable_year_central + out.adoption_lag_central
    out["disrupt_year_p90"] = out.capable_year_slow + np.maximum(lag_slow - bonus, 1.0) + 1.5 * penalty
    for c in ["capable_year_fast", "capable_year_central", "capable_year_slow", "disrupt_year_p10", "disrupt_year_p50", "disrupt_year_p90"]:
        out[c] = out[c].where(out[c] <= p.horizon_cap_year, np.nan)
    out["beyond_horizon"] = out.disrupt_year_p50.isna()
    out["timing_bucket"] = pd.cut(out.disrupt_year_p50, bins=[0, 2028, 2030, 2033, 2037, 2045], labels=["by_2028", "2029_2030", "2031_2033", "2034_2037", "2038_2045"]).astype(object)
    out.loc[out.beyond_horizon, "timing_bucket"] = "beyond_2045_or_unknown"
    out["already_observed"] = observed
    out["params"] = str(asdict(p))
    return out


YEARS = list(range(2026, 2041))


def disruption_curve(fc: pd.DataFrame, by: str, weight: str = "task_weight", col: str = "disrupt_year_p50") -> pd.DataFrame:
    """Per group: share of weight whose year <= Y for each Y; the half-way year for p10/p50/p90."""
    d = fc.dropna(subset=[by]).copy()
    d["w"] = d[weight].fillna(0)
    g = d.groupby(by)
    tot = g.w.sum()
    out = pd.DataFrame({"n_tasks": g.task_id.nunique(), "weight_total": tot})
    for y in YEARS:
        out[f"share_by_{y}"] = d.assign(_x=d.w * (d[col] <= y)).groupby(by)._x.sum() / tot
    out["share_beyond_horizon"] = d.assign(_x=d.w * d[col].isna()).groupby(by)._x.sum() / tot
    out["share_already_observed"] = d.assign(_x=d.w * d["already_observed"]).groupby(by)._x.sum() / tot if "already_observed" in d else np.nan

    def half(colname):
        res = {}
        for k, grp in d.groupby(by):
            grp = grp.sort_values(colname)
            cs = grp.w.cumsum() / grp.w.sum()
            hit = grp[(cs >= 0.5) & grp[colname].notna()]
            res[k] = float(hit[colname].iloc[0]) if len(hit) else np.nan
        return pd.Series(res)

    out["half_year_p10"] = half("disrupt_year_p10")
    out["half_year_p50"] = half("disrupt_year_p50")
    out["half_year_p90"] = half("disrupt_year_p90")
    return out.reset_index()


def skill_curves(fc: pd.DataFrame, occ_scores: pd.DataFrame, version: str = "30.1", min_importance: float = 3.0) -> pd.DataFrame:
    """Skills / knowledge / abilities: importance-weighted average of the occupation curves across the
    occupations where the element matters (importance >= min_importance)."""
    oc = disruption_curve(fc, "onet_soc_code")
    r = onet.load_descriptor_ratings(version)
    r = r[r.importance >= min_importance].merge(oc, on="onet_soc_code", how="inner")
    share_cols = [c for c in oc.columns if c.startswith("share_by_")] + ["share_beyond_horizon", "share_already_observed"]
    rows = []
    for (fam, eid, name), g in r.groupby(["family", "element_id", "element_name"]):
        w = g.importance.values
        row = {"family": fam, "element_id": eid, "element_name": name, "n_occupations": int(g.onet_soc_code.nunique())}
        for c in share_cols:
            row[c] = float(np.average(g[c].fillna(0).values, weights=w))
        for q in ["p10", "p50", "p90"]:
            v = g[f"half_year_{q}"]
            row[f"half_year_{q}"] = float(np.average(v.fillna(2046).values, weights=w))
        rows.append(row)
    out = pd.DataFrame(rows)
    for q in ["p10", "p50", "p90"]:
        out[f"half_year_{q}"] = out[f"half_year_{q}"].where(out[f"half_year_{q}"] <= 2045, np.nan)
    out["onet_version"] = version
    return out
