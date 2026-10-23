"""Disruption-timing model. Transparent, parameterised, and reported as ranges.

For each O*NET task: when could AI do it (capability), when do organisations let it (adoption), and
what else holds it back (barriers). Each piece is anchored on published data and has a fast and a
slow case.

1. CAPABILITY
   Non-physical tasks. AI is already used to *do* (not just help with) tasks of a few hours. We
   calibrate today's "delegation horizon" on the Anthropic data (the task length at which the
   automation share crosses 50%, about 4 hours) and grow it at METR's measured frontier doubling
   time (about 3.5 months since 2024; fast 85 days, slow 212 days). A task is capable when the horizon
   reaches its length; a task already observed being automated is capable now.
   Physical tasks. Anthropic's robot-exposure tier sets years-from-now to capability.

2. ADOPTION
   The US Census BTOS series (share of firms using AI in production, biweekly since Sept 2023) is fitted
   with a logistic curve. Its crossing of 50% of firms (central: 2030, fitted) is the anchor date at
   which a typical, barrier-free task that is capable today would be in majority use. Faster or slower
   growth (x1.5 / x0.6) give the fast and slow cases. Singapore runs ahead of the US on individual AI
   use (Microsoft: 64% vs 33% of working-age people; Anthropic per-capita index about 1.5x), which we
   convert into a lead of about a year on the same curve; set `geo` to "USA" to switch it off.

3. BARRIERS
   The per-task barrier index (frontier.task_barriers: interpersonal work, in-person work,
   accountability, how often people keep the task themselves) delays adoption by up to
   `max_barrier_years` (central 4). Tasks already observed being automated get a one-year credit.

disrupt_year = adoption anchor + max(0, capability year - now) + barrier delay - observed credit - geo lead

Outputs per task: capable_year_{fast,central,slow}, disrupt_year_{p10,p50,p90}, timing_bucket and the
components. Roll-ups report, per occupation / SSOC / role / activity / skill, the share of task weight
whose central year is at or before each calendar year and the first year at which that share passes
50% ("half-way year").

"Disruptable" means: AI can do the task at workplace reliability and typical organisations have had
time to adopt it. It is not a forecast of job loss: tasks change, roles re-bundle, demand can rise.
Every number is a planning signal with a range.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import onet


@dataclass
class ForecastParams:
    # capability
    anchor_date: str = "2024-01-01"
    log2_h50_at_anchor: float = 1.91
    doubling_days_central: float = 106.0
    doubling_days_fast: float = 85.0
    doubling_days_slow: float = 212.0
    delegation_horizon_now_min: float = 260.0   # overwritten by calibrate()
    delegation_slope: float = -0.29             # overwritten by calibrate()
    robot_tier_years: dict = field(default_factory=lambda: {1: (2, 4, 8), 2: (4, 7, 12), 3: (8, 12, 20), 0: (np.nan, np.nan, np.nan)})
    # adoption (BTOS logistic; overwritten by calibrate_adoption())
    adoption_saturation: float = 0.9
    adoption_growth_per_year: float = 0.505
    adoption_intercept: float = -2.99
    adoption_growth_mult: tuple = (1.5, 1.0, 0.6)   # fast / central / slow
    geo: str = "SGP"
    geo_usage_ratio_vs_us: float = 1.7              # Singapore individual AI use relative to the US (Microsoft x Anthropic, geometric mean); fallback for the lead
    geo_lead_years_measured: float | None = None     # Singapore firms' lead over US firms at matched adoption levels (IMDA vs Census); used when set
    # capability evidence: years before a desk task with no usage evidence is treated as capable, by
    # evidence tier (overwritten by calibrate_evidence(): measured from how far each tier lags the
    # best-rated tasks in being seen in actual use)
    evidence_lag_years: dict = field(default_factory=lambda: {"mixed": (0.7, 1.4, 2.8), "unproven": (1.1, 2.2, 4.4), "newly_rated": (0.0, 0.0, 0.0)})
    # barriers
    max_barrier_years: tuple = (2.0, 4.0, 6.0)      # fast / central / slow
    observed_credit_years: float = 1.0
    horizon_cap_year: float = 2045.0


def calibrate(p: ForecastParams, tf: pd.DataFrame) -> ForecastParams:
    """Delegation horizon: fit automation_share ~ sigmoid(a + b log2 minutes) on observed non-physical tasks."""
    d = tf[(~tf.physical) & (tf.usage_latest > 0) & tf.human_only_hours.notna() & tf.automation_share.notna()].copy()
    x = np.log2(d.human_only_hours.values * 60.0)
    y = d.automation_share.clip(1e-3, 1 - 1e-3).values
    b, a = np.polyfit(x, np.log(y / (1 - y)), 1)
    p = ForecastParams(**asdict(p))
    if b < 0:
        p.delegation_slope = float(b)
        p.delegation_horizon_now_min = float(2 ** (-a / b))
    return p


def calibrate_adoption(p: ForecastParams, btos_fit: dict | None = None, geo_ratio: float | None = None, geo_lead: float | None = None) -> ForecastParams:
    p = ForecastParams(**asdict(p))
    if btos_fit:
        p.adoption_saturation = float(btos_fit["saturation"])
        p.adoption_growth_per_year = float(btos_fit["growth_per_year"])
        p.adoption_intercept = float(btos_fit["intercept"])
    if geo_ratio:
        p.geo_usage_ratio_vs_us = float(geo_ratio)
    if geo_lead is not None:
        p.geo_lead_years_measured = float(geo_lead)
    return p


def singapore_usage_ratio() -> dict:
    """Singapore vs US individual AI use: Microsoft diffusion (share of 15-64 using gen-AI) and the
    Anthropic per-capita usage index (Apr-May 2026). Returns both and their geometric mean."""
    from . import ai_signals as ai

    out = {}
    try:
        ms = ai.load_microsoft_diffusion_countries().set_index("economy").ai_user_share_2026q2_pct
        out["microsoft_ratio"] = float(ms["Singapore"] / ms["United States"])
    except Exception:
        pass
    try:
        df = pd.read_csv(ai.AEI / "release_2026_06_26" / "data" / "aei_claude_ai_2026-06-26.csv", dtype=str, usecols=["geo_id", "category_name", "metric_id", "value"])
        df = df[(df.category_name == "overall") & (df.metric_id == "usage_per_capita_index")]
        df["value"] = pd.to_numeric(df.value)
        g = df.groupby("geo_id").value.mean()
        out["anthropic_ratio"] = float(g["SGP"] / g["USA"])
    except Exception:
        pass
    vals = [v for v in out.values() if v and v > 0]
    out["geometric_mean"] = float(np.exp(np.mean(np.log(vals)))) if vals else None
    return out


def adoption_cross_year(p: ForecastParams, level: float = 0.5, mult: float = 1.0) -> float:
    K = p.adoption_saturation
    return 2024.0 + (np.log(level / (K - level)) - p.adoption_intercept) / (p.adoption_growth_per_year * mult)


def geo_lead_years(p: ForecastParams, mult: float = 1.0) -> float:
    """Years Singapore runs ahead of the US curve: the measured firm-adoption lead when available,
    else inferred from the individual-usage ratio through the growth rate."""
    if p.geo.upper() in ("US", "USA"):
        return 0.0
    if p.geo_lead_years_measured is not None:
        return float(p.geo_lead_years_measured)
    if not p.geo_usage_ratio_vs_us:
        return 0.0
    return float(np.log(p.geo_usage_ratio_vs_us) / (p.adoption_growth_per_year * mult))


EVIDENCE_TIERS = ["seen", "rated", "newly_rated", "mixed", "unproven", "physical"]


def evidence_tiers(tf: pd.DataFrame, labels: pd.DataFrame | None, ratings: pd.DataFrame | None = None) -> pd.DataFrame:
    """One evidence tier per task, from strongest to weakest:
      seen         AI use on this task has been published in at least one Economic Index release
      rated        not seen, but rated doable by the 2023 raters (human or GPT-4) and, when a 2026 rating
                   exists, by the 2026 rating too
      newly_rated  not seen; the 2023 raters said no but the 2026 rating says yes (the frontier moved)
      mixed        not seen; the raters disagree (one 2023 rater yes / without 2026 ratings: one of two
                   2023 raters yes; with them: 2026 says no but a 2023 rater said yes)
      unproven     not seen; every rater said no
      physical     physical task (robot track; the ratings exclude robots)
    'Yes' means exposure E1 or E2: AI could cut the time by at least half."""
    d = tf[["task_id", "physical", "usage_n_vintages"]].copy()
    d["task_id"] = d.task_id.astype(str)
    if labels is not None and len(labels):
        lab = labels.copy(); lab["task_id"] = lab.task_id.astype(str)
        d = d.merge(lab[["task_id", "exposure_human_2023", "exposure_gpt4_2023", "automation_gpt4_2023"]], on="task_id", how="left")
    else:
        d["exposure_human_2023"] = np.nan; d["exposure_gpt4_2023"] = np.nan; d["automation_gpt4_2023"] = np.nan
    h_yes = d.exposure_human_2023.isin(["E1", "E2"]); g_yes = d.exposure_gpt4_2023.isin(["E1", "E2"])
    has23 = d.exposure_human_2023.notna() | d.exposure_gpt4_2023.notna()
    yes23_any = h_yes | g_yes
    yes23_both = h_yes & g_yes
    has26 = pd.Series(False, index=d.index)
    if ratings is not None and len(ratings):
        r = ratings.copy(); r["task_id"] = r.task_id.astype(str)
        d = d.merge(r[["task_id", "exposure", "autonomy", "barrier"]].rename(columns={"exposure": "exposure_2026", "autonomy": "autonomy_2026", "barrier": "barrier_2026"}), on="task_id", how="left")
        has26 = d.exposure_2026.notna()
    else:
        d["exposure_2026"] = np.nan; d["autonomy_2026"] = np.nan; d["barrier_2026"] = np.nan
    yes26 = d.exposure_2026.isin(["E1", "E2"])
    seen = d.usage_n_vintages.fillna(0) >= 1
    tier = np.select(
        [d.physical, seen,
         has26 & yes26 & yes23_any, has26 & yes26 & ~yes23_any, has26 & ~yes26 & yes23_any, has26 & ~yes26 & ~yes23_any,
         ~has23, yes23_both, yes23_any],
        ["physical", "seen", "rated", "newly_rated", "mixed", "unproven", "mixed", "rated", "mixed"], default="unproven")
    d["evidence_tier"] = tier
    d["rated_yes_2023"] = np.where(has23, np.where(yes23_both, "both", np.where(yes23_any, "one", "none")), "no_rating")
    d["rated_yes_2026"] = np.where(has26, np.where(yes26, "yes", "no"), "no_rating")
    return d.drop(columns=["physical", "usage_n_vintages"])


def calibrate_evidence(p: ForecastParams, usage_long: pd.DataFrame, tiers: pd.DataFrame, tf: pd.DataFrame, source: str = "claude_ai") -> tuple[ForecastParams, dict]:
    """Measure how far each evidence tier lags the 'rated' tier in being seen in actual use, release by
    release: for desk tasks, the share ever seen by each release is tracked per tier; the rated tier's
    coverage grows at g points a year; a tier whose coverage today equals what the rated tier had
    (today's rated coverage - its own) / g years ago lags by that many years. Central = that lag, fast =
    half, slow = double. Tiers are assigned ignoring the 'seen' split (a task's rating group), so the
    comparison is between rating groups."""
    d = tf[["task_id", "physical"]].copy(); d["task_id"] = d.task_id.astype(str)
    t = tiers.copy(); t["task_id"] = t.task_id.astype(str)
    # rating group = the tier the task would have without usage evidence
    g26 = t.rated_yes_2026; g23 = t.rated_yes_2023
    group = np.where(g26 == "no_rating",
                     np.where(g23 == "both", "rated", np.where(g23 == "one", "mixed", np.where(g23 == "none", "unproven", "mixed"))),
                     np.where(g26 == "yes", np.where(g23.isin(["both", "one"]), "rated", "newly_rated"), np.where(g23.isin(["both", "one"]), "mixed", "unproven")))
    t["group"] = group
    d = d.merge(t[["task_id", "group"]], on="task_id", how="left")
    d = d[~d.physical]
    u = usage_long[(usage_long.metric == "usage_pct") & (usage_long.source == source) & (usage_long.value > 0)].copy()
    u["task_id"] = u.task_id.astype(str)
    vint = sorted(u.vintage.unique())
    ever = set(); cov = []
    for v in vint:
        ever |= set(u[u.vintage == v].task_id)
        s = d.task_id.isin(ever)
        row = {"vintage": v, "mid_date": str(u[u.vintage == v].mid_date.iloc[0])[:10]}
        for grp in ["rated", "newly_rated", "mixed", "unproven"]:
            m = (d.group == grp).values
            row[grp] = float(s[m].mean()) if m.any() else np.nan
        cov.append(row)
    cov = pd.DataFrame(cov)
    cov["t"] = (pd.to_datetime(cov.mid_date) - pd.Timestamp("2025-01-01")).dt.days / 365.25
    out = {"coverage_by_release": cov.round(4).to_dict("records")}
    p = ForecastParams(**asdict(p)); lags = dict(p.evidence_lag_years)
    if len(cov) >= 2 and cov.rated.notna().all():
        g = float(np.polyfit(cov.t.values, cov.rated.values, 1)[0])  # points of coverage per year
        out["rated_coverage_growth_per_year"] = g
        if g > 0:
            for grp in ["newly_rated", "mixed", "unproven"]:
                c = cov[grp].iloc[-1]
                if pd.notna(c):
                    lag = max(0.0, (float(cov.rated.iloc[-1]) - float(c)) / g)
                    lags[grp] = (round(lag * 0.5, 2), round(lag, 2), round(lag * 2.0, 2))
    p.evidence_lag_years = lags
    out["evidence_lag_years"] = lags
    return p, out


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


def forecast_tasks(tf: pd.DataFrame, barriers: pd.DataFrame, params: ForecastParams | None = None, now: str = "2026-10-01", tiers: pd.DataFrame | None = None, occ_shift: pd.DataFrame | None = None) -> pd.DataFrame:
    """occ_shift: optional (onet_soc_code, adoption_shift_years) = years the occupation's industries run
    ahead (negative) or behind the national adoption curve (sector_adoption.py); divided by the growth
    multiplier in the fast and slow cases."""
    p = params or ForecastParams()
    now_y = _frac_year(pd.Timestamp(now))
    mins = impute_task_minutes(tf)
    d = tf[["task_id", "onet_soc_code", "task_weight", "physical", "robot_tier", "status", "usage_latest", "automation_share", "observed_automation", "human_only_ability_pct", "usage_index", "usage_trend"]].merge(mins, on="task_id").merge(barriers[["task_id", "barrier_index", "barrier_type", "b_interpersonal", "b_in_person", "b_accountability", "b_handover"]], on="task_id", how="left")
    d["barrier_index"] = d.barrier_index.fillna(float(d.barrier_index.median()))
    if tiers is not None:
        t = tiers.copy(); t["task_id"] = t.task_id.astype(d.task_id.dtype) if d.task_id.dtype != object else t.task_id.astype(str)
        d = d.merge(t[["task_id", "evidence_tier", "exposure_2026", "autonomy_2026", "barrier_2026"]], on="task_id", how="left")
        d["evidence_tier"] = d.evidence_tier.fillna("mixed")
    else:
        d["evidence_tier"] = np.where(d.physical, "physical", np.where(d.usage_latest > 0, "seen", "mixed"))
        d["exposure_2026"] = np.nan; d["autonomy_2026"] = np.nan; d["barrier_2026"] = np.nan

    # 1. capability
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
    # evidence lag: a desk task nobody has seen AI do, and that raters say AI cannot do, is not treated
    # as capable today; it waits the measured lag of its evidence tier (never "AI cannot")
    for name, i in [("fast", 0), ("central", 1), ("slow", 2)]:
        lag = d.evidence_tier.map(lambda tname: p.evidence_lag_years.get(tname, (0.0, 0.0, 0.0))[i]).astype(float).values
        cap[name] = np.where(~d.physical.values & ~observed, np.maximum(cap[name], now_y + lag), cap[name])

    # 2. adoption anchor and geo lead, 3. barriers
    out = d.copy()
    anchors, leads = {}, {}
    for name, mult in zip(["fast", "central", "slow"], p.adoption_growth_mult):
        anchors[name] = adoption_cross_year(p, 0.5, mult)
        leads[name] = geo_lead_years(p, mult)
    credit = np.where(observed, p.observed_credit_years, 0.0)
    if occ_shift is not None and len(occ_shift):
        shift = d.onet_soc_code.map(occ_shift.set_index("onet_soc_code").adoption_shift_years).fillna(0.0).values
    else:
        shift = np.zeros(len(d))
    out["adoption_shift_years"] = shift
    for (name, q, mb), mult in zip([("fast", "p10", p.max_barrier_years[0]), ("central", "p50", p.max_barrier_years[1]), ("slow", "p90", p.max_barrier_years[2])], p.adoption_growth_mult):
        barrier_years = d.barrier_index.values * mb
        y = anchors[name] - leads[name] + shift / mult + np.maximum(cap[name] - now_y, 0.0) + barrier_years - credit
        y = np.maximum(y, now_y + 0.5)
        out[f"capable_year_{name}"] = cap[name]
        out[f"barrier_years_{name}"] = barrier_years
        out[f"disrupt_year_{q}"] = y
    out["adoption_anchor_year_central"] = anchors["central"]
    out["geo_lead_years_central"] = leads["central"]
    for c in ["capable_year_fast", "capable_year_central", "capable_year_slow", "disrupt_year_p10", "disrupt_year_p50", "disrupt_year_p90"]:
        out[c] = out[c].where(out[c] <= p.horizon_cap_year, np.nan)
    out["beyond_horizon"] = out.disrupt_year_p50.isna()
    out["timing_bucket"] = pd.cut(out.disrupt_year_p50, bins=[0, 2028, 2030, 2033, 2037, 2045], labels=["by_2028", "2029_2030", "2031_2033", "2034_2037", "2038_2045"]).astype(object)
    out.loc[out.beyond_horizon, "timing_bucket"] = "beyond_2045_or_unknown"
    out["already_observed"] = observed
    out["params"] = str(asdict(p))
    return out


YEARS = list(range(2026, 2041))


def reshift_years(df: pd.DataFrame, new_shift: pd.Series, p: ForecastParams, now_y: float) -> pd.DataFrame:
    """Move task disruption years from their occupation's adoption shift to another entity's shift
    (an SSOC code's or a Skills Framework role's industry), scenario by scenario."""
    d = df.copy()
    delta = (new_shift.fillna(0.0).values - d.adoption_shift_years.fillna(0.0).values)
    for q, mult in zip(["p10", "p50", "p90"], p.adoption_growth_mult):
        col = f"disrupt_year_{q}"
        y = d[col].values + delta / mult
        d[col] = np.where(np.isnan(d[col].values), np.nan, np.maximum(y, now_y + 0.5))
        d[col] = d[col].where(d[col] <= p.horizon_cap_year, np.nan)
    return d


def disruption_curve(fc: pd.DataFrame, by: str, weight: str = "task_weight", col: str = "disrupt_year_p50") -> pd.DataFrame:
    """Per group: share of weight whose year <= Y for each Y; the half-way year for p10/p50/p90."""
    d = fc.dropna(subset=[by]).copy()
    d["w"] = d[weight].fillna(0)
    g = d.groupby(by)
    tot = g.w.sum()
    out = pd.DataFrame({"n_tasks": g.task_id.nunique(), "weight_total": tot})
    for y in YEARS:
        out[f"share_by_{y}"] = d.assign(_x=d.w * (d[col] <= y)).groupby(by)._x.sum() / tot
    # the same curves under the fast (p10) and slow (p90) cases, for honest ranges on aggregates
    for q in ["p10", "p90"]:
        qc = col.replace("p50", q)
        if qc in d.columns and qc != col:
            for y in YEARS:
                out[f"share_by_{y}_{q}"] = d.assign(_x=d.w * (d[qc] <= y)).groupby(by)._x.sum() / tot
    out["share_beyond_horizon"] = d.assign(_x=d.w * d[col].isna()).groupby(by)._x.sum() / tot
    out["share_already_observed"] = d.assign(_x=d.w * d["already_observed"]).groupby(by)._x.sum() / tot if "already_observed" in d else np.nan
    if "barrier_index" in d:
        out["barrier_index"] = d.assign(_x=d.w * d.barrier_index).groupby(by)._x.sum() / tot
    if "evidence_tier" in d:
        for tname in EVIDENCE_TIERS:
            out[f"share_tier_{tname}"] = d.assign(_x=d.w * (d.evidence_tier == tname)).groupby(by)._x.sum() / tot

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
