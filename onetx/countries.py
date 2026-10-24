"""Further countries on the same map. Everything a country needs is listed in one place (a CountrySpec),
and the same steps run for each:

  1. a crosswalk from O*NET occupations to the national classification through ISCO-08
     (O*NET -> ISCO from crosswalks.py; ISCO -> ANZSCO from the ABS table, ISCO -> SOC 2020 from the
     ONS coding index), with `weight` summing to 1 per O*NET occupation and `weight_rev` to 1 per
     national code
  2. national weights (employment and pay per code) and their O*NET-level mirror
  3. the country's own firm-adoption series against the fitted US curve: a lead in years
     (ai_signals.singapore_firm_lead, generalised) and each industry's level against the national one,
     carried to occupations through the country's occupation-by-industry mix
  4. the timing model re-run with that lead and those shifts (forecast.forecast_tasks), rolled up to
     national codes and to the whole economy, and crossed with the other countries' weights for the
     job-mix / timing decomposition
  5. rung five: employment and pay change by national code against exposure
  6. the national skill levels (ANZSCO 1-5, SOC 1-4) against AI, beside the O*NET skill-level test
  7. the official task list per national code matched to O*NET tasks, so each code's page shows when AI
     could do each part of what the statistics office says the job involves; for the UK also the
     apprenticeship standards' duties
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import au_labour as au
from . import forecast as fcm
from . import io
from . import uk_labour as uk
from . import us_economy as ue
from . import singapore_economy as se
from .ai_signals import _logit
from .forecast import YEARS
from .frontier import _weighted_mean
from .match import _rerank_with_bonus
from .outcomes import _reading, _spearman


@dataclass
class CountrySpec:
    code: str                      # ISO3, used in ForecastParams.geo and the page
    name: str
    adjective: str
    prefix: str                    # processed-table prefix: au_, uk_
    code_col: str                  # national code column: anzsco_code, soc_code
    title_col: str
    classification: str            # plain-words name of the national classification
    major_names: dict
    skill_level_names: dict
    industry_names: dict           # industry code -> label (the sector lens)
    industry_label: str
    currency: str
    wage_label: str
    emp_label: str
    wage_period: str
    adoption_source: str
    group_col: str                 # the level at which the occupation-by-industry mix is published
    weights: Callable[[], pd.DataFrame]
    outcomes: Callable[[], pd.DataFrame]
    isco_link: Callable[[set], pd.DataFrame]
    industry_mix: Callable[[], pd.DataFrame]     # columns: group_col, industry, employed
    ai_points: Callable[[], pd.DataFrame]        # columns: date, group_type (all/industry/size), industry, label, pct
    tasks: Callable[[], pd.DataFrame]            # columns: code_col, task_seq, task
    outcome_labels: dict = field(default_factory=dict)
    min_lead_year: float = 2024.0


def _au_mix():
    d = au.load_industry_by_major(2025)
    return d.rename(columns={"employed_thousands": "employed"})[["major_group", "industry", "employed"]]


def _uk_mix():
    d = uk.load_census_occupation_industry()
    return d.rename(columns={"section": "industry"})[["minor", "industry", "employed"]]


def _au_isco(unit_codes):
    return au.load_isco_to_anzsco(unit_codes).rename(columns={"anzsco_code": "nat_code"})[["isco_code", "nat_code", "weight"]]


def _uk_isco(unit_codes):
    d = uk.load_soc_to_isco()
    d = d[d.soc_code.isin(unit_codes)] if unit_codes else d
    d = d.drop(columns=["weight"]).rename(columns={"soc_code": "nat_code", "weight_rev": "weight"})[["isco_code", "nat_code", "weight"]]
    d["weight"] = d.weight / d.groupby("isco_code").weight.transform("sum")
    return d


SPECS: dict[str, CountrySpec] = {
    "AUS": CountrySpec(code="AUS", name="Australia", adjective="Australian", prefix="au_", code_col="anzsco_code", title_col="anzsco_title", classification="ANZSCO unit group",
                       major_names=au.MAJOR_GROUP_NAMES, skill_level_names=au.SKILL_LEVEL_NAMES, industry_names=au.DIVISIONS, industry_label="ANZSIC industry division", currency="A$",
                       wage_label="average weekly total cash earnings, May 2025 (ABS Employee Earnings and Hours)", emp_label="employed persons, 2025 average (ABS Labour Force)", wage_period="May 2025",
                       adoption_source="ABS Characteristics of Australian Business", group_col="major_group",
                       weights=au.build_anzsco_weights, outcomes=au.employment_change, isco_link=_au_isco, industry_mix=_au_mix, ai_points=au.load_cab_ai_use, tasks=au.load_anzsco_tasks,
                       outcome_labels={"emp": "employment change, 2024 to 2025 (ABS Labour Force, yearly averages)", "wage": "average weekly earnings change, May 2023 to May 2025 (ABS)"}),
    "GBR": CountrySpec(code="GBR", name="United Kingdom", adjective="UK", prefix="uk_", code_col="soc_code", title_col="soc_title", classification="SOC 2020 unit group",
                       major_names=uk.MAJOR_GROUP_NAMES, skill_level_names=uk.SKILL_LEVEL_NAMES, industry_names=uk.SECTIONS, industry_label="SIC industry section", currency="£",
                       wage_label="median gross annual pay, April 2025 (ONS ASHE, provisional)", emp_label="employee jobs, April 2025 (ONS ASHE)", wage_period="April 2025",
                       adoption_source="ONS Business Insights and Conditions Survey", group_col="minor",
                       weights=uk.build_soc_weights, outcomes=uk.employment_change, isco_link=_uk_isco, industry_mix=_uk_mix, ai_points=uk.load_bics_ai_use, tasks=uk.load_soc_tasks,
                       outcome_labels={"emp": "employee jobs change, April 2024 to April 2025 (ONS ASHE)", "wage": "median gross annual pay change, April 2024 to April 2025 (ONS ASHE)"}),
}


# ----------------------------------------------------------------------------- 1. crosswalk and 2. weights
def national_crosswalk(spec: CountrySpec, unit_codes: set[str] | None = None) -> pd.DataFrame:
    """Compose O*NET -> ISCO-08 with ISCO-08 -> national code. Columns: onet_soc_code, nat_code, weight
    (per O*NET occupation), weight_rev (per national code), n_isco_paths."""
    a = io.load("xw_onetsoc_isco08")[["onet_soc_code", "isco_code", "weight"]].rename(columns={"weight": "w1"})
    b = spec.isco_link(unit_codes).rename(columns={"weight": "w2"})
    m = a.merge(b, on="isco_code", how="inner")
    m["path_w"] = m.w1 * m.w2
    g = m.groupby(["onet_soc_code", "nat_code"]).agg(path_w=("path_w", "sum"), n_isco_paths=("isco_code", "nunique")).reset_index()
    g["weight"] = g.path_w / g.groupby("onet_soc_code").path_w.transform("sum")
    g["weight_rev"] = g.path_w / g.groupby("nat_code").path_w.transform("sum")
    g["country"] = spec.code
    return g.drop(columns="path_w").sort_values(["onet_soc_code", "weight"], ascending=[True, False]).reset_index(drop=True)


def onet_weights_from_national(xw: pd.DataFrame, natw: pd.DataFrame, spec: CountrySpec) -> pd.DataFrame:
    """O*NET-level mirror of the national weights: each national code's employment is shared over the
    O*NET occupations it maps to by weight_rev; pay is the employment-weighted mean. The occupation's
    group labels come from the national code carrying most of its workers."""
    n = natw.rename(columns={spec.code_col: "nat_code"})
    m = xw.merge(n[["nat_code", "employment_thousands", "wage_value", "major_group", "submajor_label", "submajor_name", "skill_level"]], on="nat_code", how="inner")
    m["emp"] = m.employment_thousands * m.weight_rev
    m["wv"] = m.wage_value * m.weight_rev
    g = m.groupby("onet_soc_code").agg(employment_thousands=("emp", "sum"), wage_value=("wv", "sum"), n_nat_codes=("nat_code", "nunique")).reset_index()
    top = m.sort_values("emp", ascending=False).drop_duplicates("onet_soc_code").set_index("onet_soc_code")
    g["major_group"] = g.onet_soc_code.map(top.major_group)
    g["submajor_label"] = g.onet_soc_code.map(top.submajor_label)
    g["submajor_name"] = g.onet_soc_code.map(top.submajor_name)
    g["skill_level"] = g.onet_soc_code.map(top.skill_level)
    g["nat_code_top"] = g.onet_soc_code.map(top.nat_code)
    g["wage_median"] = g.wage_value / g.employment_thousands.where(g.employment_thousands > 0)
    g["employment_method"] = "national_code_employment_split_by_crosswalk_weight"
    g["country"] = spec.code
    return g


def push_frontier(occf: pd.DataFrame, xw: pd.DataFrame) -> pd.DataFrame:
    """National-code frontier scores = weight_rev-weighted means over the O*NET occupations mapping in."""
    cols = [c for c in ["observed_automation", "usage_index", "automation_share", "share_weight_observed", "share_weight_physical", "aei_job_exposure", "ai_applicability_score", "robot_exposure_index"] if c in occf.columns]
    m = xw.merge(occf[["onet_soc_code", *cols]], on="onet_soc_code", how="inner")
    out = pd.DataFrame({"n_onet_occupations": m.groupby("nat_code").onet_soc_code.nunique()})
    for c in cols:
        out[c] = _weighted_mean(m, "nat_code", c, "weight_rev")
    return out.reset_index()


def national_curve(fc: pd.DataFrame, xw: pd.DataFrame, nat_shift: pd.Series | None, params: fcm.ForecastParams, now_y: float) -> pd.DataFrame:
    """Disruption curve per national code from the task forecast: tasks weighted by task weight times the
    crosswalk share, moved from their occupation's industry shift to the national code's own."""
    f = fc.merge(xw[["onet_soc_code", "nat_code", "weight_rev"]], on="onet_soc_code")
    f["w"] = f.task_weight * f.weight_rev
    if nat_shift is not None and len(nat_shift):
        f = fcm.reshift_years(f, f.nat_code.map(nat_shift), params, now_y)
    return fcm.disruption_curve(f, "nat_code", weight="w")


def weighted_table(natw: pd.DataFrame, frontier_nat: pd.DataFrame, curve_nat: pd.DataFrame, spec: CountrySpec) -> pd.DataFrame:
    """One row per national code: weights, frontier scores and timing, shaped for economy_exposure."""
    share_cols = [c for c in curve_nat.columns if c.startswith("share_by_") or c.startswith("done_by_") or c.startswith("half_year_") or c.startswith("quarter_done_year_") or c.startswith("half_done_year_") or c in ("share_beyond_horizon", "share_already_observed", "ceiling_ai_share", "barrier_index", "n_tasks")]
    t = natw.rename(columns={spec.code_col: "nat_code", spec.title_col: "nat_title"}).merge(frontier_nat, on="nat_code", how="left").merge(curve_nat[["nat_code", *share_cols]], on="nat_code", how="left")
    t["major_group_name"] = t.major_group.map(spec.major_names)
    t["has_scores"] = t.observed_automation.notna()
    t["country"] = spec.code
    return t


def economy(t: pd.DataFrame, spec: CountrySpec) -> pd.DataFrame:
    out = se.economy_exposure(t)
    out["country"] = spec.code
    out.loc[out.scope == "economy", "scope_name"] = f"Whole economy ({spec.name})"
    maj = out.level == "major_group"
    out.loc[maj, "scope_name"] = out.loc[maj, "scope"].str[-1].map(spec.major_names).fillna(out.loc[maj, "scope_name"])
    return out


# ----------------------------------------------------------------------------- 3. the country's own adoption: lead and industry shifts
def firm_lead(fit: dict, points: pd.DataFrame, min_year: float = 2024.0) -> dict:
    """Years the country's firms run ahead of (positive) or behind (negative) US firms at matched adoption
    levels: for each national point the date at which the fitted US curve reaches the same share is read
    off, and the lead is the mean gap over points dated from `min_year` (the generative-AI years)."""
    K, a, b = fit["saturation"], fit["intercept"], fit["growth_per_year"]
    pts = points[points.group_type == "all"].dropna(subset=["pct", "date"]).sort_values("date")
    rows = []
    for r in pts.itertuples():
        p = min(max(float(r.pct) / 100.0, 0.002), K - 0.002)
        us_year = 2024.0 + (float(_logit(np.array(p), K)) - a) / b
        rows.append({"date": float(r.date), "label": str(getattr(r, "period", "") or getattr(r, "wave", "")), "pct": float(r.pct), "us_curve_reaches_same_share": us_year, "lead_years": us_year - float(r.date)})
    use = [x for x in rows if x["date"] >= min_year]
    lead = float(np.mean([x["lead_years"] for x in use])) if use else None
    own = {}
    if len(pts) >= 2:
        t = pts.date.values.astype(float) - 2024.0
        yy = _logit(np.clip(pts.pct.values / 100.0, 0.002, K - 0.002), K)
        bs, as_ = np.polyfit(t, yy, 1)
        own = {"own_growth_per_year": float(bs), "own_year_cross_50pct": float(2024.0 + (float(_logit(np.array(0.5), K)) - as_) / bs) if bs > 0 else None, "own_n_points": int(len(pts))}
    return {"points": rows, "lead_years": lead, "n_points_used": len(use), "latest_pct": float(pts.pct.iloc[-1]) if len(pts) else None, "latest_date": float(pts.date.iloc[-1]) if len(pts) else None, **own}


def industry_offsets(points: pd.DataFrame, fit: dict, min_national_pct: float = 3.0) -> pd.DataFrame:
    """Each industry's adoption level against the country's own national level at the same date, in
    logit units averaged over the survey dates where the national share is at least `min_national_pct`
    (earlier readings of 0-1% are too small to carry a level). Converted to years with the US growth
    rate: shift = -offset / b (negative = earlier than the national curve)."""
    K, b = fit["saturation"], fit["growth_per_year"]
    nat = points[points.group_type == "all"].dropna(subset=["pct"]).set_index("date").pct
    d = points[(points.group_type == "industry")].dropna(subset=["pct"]).copy()
    d["national_pct"] = d.date.map(nat)
    d = d[d.national_pct >= min_national_pct]
    if d.empty:
        return pd.DataFrame(columns=["industry", "label", "n_points", "latest_date", "latest_pct", "national_pct_latest", "level_offset", "anchor_shift_years"])
    clip = lambda p: np.clip(p / 100.0, 0.002, K - 0.002)
    d["resid"] = _logit(clip(d.pct.values), K) - _logit(clip(d.national_pct.values), K)
    g = d.sort_values("date").groupby("industry")
    out = pd.DataFrame({"label": g.label.first(), "n_points": g.size(), "latest_date": g.date.last(), "latest_pct": g.pct.last(), "national_pct_latest": g.national_pct.last(), "level_offset": g.resid.mean()}).reset_index()
    out["anchor_shift_years"] = -out.level_offset / b
    return out.sort_values("anchor_shift_years").reset_index(drop=True)


def occupation_shifts(mix: pd.DataFrame, ind_shift: pd.DataFrame, natw: pd.DataFrame, xw: pd.DataFrame, spec: CountrySpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(group shifts, national-code shifts, O*NET shifts). A group's shift is its industry mix times the
    industry shifts (industries without a survey reading sit at the national level, 0); every code in the
    group inherits it; an O*NET occupation takes the weight_rev-weighted mean of its codes."""
    gcol = spec.group_col
    m = mix.merge(ind_shift[["industry", "anchor_shift_years"]], on="industry", how="left")
    m["anchor_shift_years"] = m.anchor_shift_years.fillna(0.0)
    m["share"] = m.employed / m.groupby(gcol).employed.transform("sum")
    grp = m.assign(_x=m.share * m.anchor_shift_years).groupby(gcol).agg(adoption_shift_years=("_x", "sum"), employed=("employed", "sum")).reset_index()
    codes = natw[[spec.code_col]].rename(columns={spec.code_col: "nat_code"}).copy()
    key = codes.nat_code.str[0] if gcol == "major_group" else codes.nat_code.str[:3]
    codes["group"] = key
    codes["adoption_shift_years"] = codes.group.map(grp.set_index(gcol).adoption_shift_years).fillna(0.0)
    o = xw[["onet_soc_code", "nat_code", "weight_rev"]].merge(codes[["nat_code", "adoption_shift_years"]], on="nat_code", how="left")
    o["adoption_shift_years"] = o.adoption_shift_years.fillna(0.0)
    o["_x"] = o.weight_rev * o.adoption_shift_years
    onet = o.groupby("onet_soc_code").agg(_x=("_x", "sum"), w=("weight_rev", "sum")).reset_index()
    onet["adoption_shift_years"] = np.where(onet.w > 0, onet._x / onet.w, 0.0)
    return grp, codes.drop(columns="group"), onet[["onet_soc_code", "adoption_shift_years"]]


# ----------------------------------------------------------------------------- 5. rung five and 6. skill levels
def outcome_tests(change: pd.DataFrame, t: pd.DataFrame, spec: CountrySpec) -> dict:
    """Does employment or pay change by national code relate to exposure? Rank correlations plus the
    employment-weighted change by exposure tercile (a negative rho is the sign disruption would show)."""
    c = change.rename(columns={spec.code_col: "nat_code"})
    d = c.merge(t[["nat_code", "employment_thousands", "done_by_2035", "share_by_2035", "share_weight_observed", "observed_automation"]], on="nat_code", how="inner")
    yf, yt = int(c.year_from.iloc[0]), int(c.year_to.iloc[0])
    tests = []
    for outcome, label in [("emp_change_pct", spec.outcome_labels.get("emp", f"employment change {yf} to {yt}")), ("wage_change_pct", spec.outcome_labels.get("wage", f"pay change {yf} to {yt}"))]:
        for sig, sig_label in [("done_by_2035", "AI doing by 2035 (central)"), ("share_weight_observed", "share of work seen in AI use"), ("observed_automation", "AI already doing the work")]:
            dd = d.dropna(subset=[outcome, sig])
            dd = dd[np.isfinite(dd[outcome])]
            rho, n = _spearman(dd[outcome], dd[sig])
            tests.append({"level": f"detailed occupation ({spec.classification})", "outcome": label, "signal": sig_label, "rho": rho, "n": n, "reading": _reading(rho, n, -1)})
    occ_rhos = [x["rho"] for x in tests if x["rho"] is not None]
    dd = d.dropna(subset=["done_by_2035", f"emp_{yf}", f"emp_{yt}"])
    dd = dd[(dd[f"emp_{yf}"] > 0)]
    terc = {}
    if len(dd) >= 9:
        dd = dd.assign(tercile=pd.qcut(dd.done_by_2035.rank(method="first"), 3, labels=["least exposed third", "middle third", "most exposed third"]))
        terc = {str(k): float(v) for k, v in dd.groupby("tercile", observed=True).apply(lambda g: 100.0 * (g[f"emp_{yt}"].sum() / g[f"emp_{yf}"].sum() - 1.0), include_groups=False).items()}
    return {"tests": tests, "headline": {"occupation": {"median_signed_rho": float(np.median(occ_rhos)) if occ_rhos else None, "n_tests": len(occ_rhos)}}, "n_codes": int(len(d)), "years": [yf, yt], "wage_years": str(c.wage_years.iloc[0]) if "wage_years" in c else f"{yf}-{yt}",
            "total_employment_change_pct": float(100.0 * (dd[f"emp_{yt}"].sum() / dd[f"emp_{yf}"].sum() - 1.0)) if len(dd) else None, "emp_change_by_exposure_tercile": terc, "labels": spec.outcome_labels}


def skill_level_bands(t: pd.DataFrame, spec: CountrySpec) -> tuple[pd.DataFrame, dict]:
    """Exposure by the national skill level (employment-weighted), and the rank correlation between a
    code's level and 'AI doing by 2035' (positive = AI reaches the higher-skilled work first)."""
    d = t.dropna(subset=["skill_level"]).copy()
    d["band"] = d.skill_level.round().astype(int)
    rows = []
    for lvl, g in d.groupby("band"):
        if g.employment_thousands.sum() <= 0:
            continue
        row = {"level": int(lvl), "level_name": spec.skill_level_names.get(int(lvl), f"level {lvl}"), "n_codes": int(len(g)), "employment_thousands": float(g.employment_thousands.sum()), "employment_share": float(g.employment_thousands.sum() / d.employment_thousands.sum())}
        for c in ["observed_automation", "share_weight_observed", "share_weight_physical", "done_by_2035", "done_by_2035_p10", "done_by_2035_p90", "share_by_2035", "done_by_2030", "ceiling_ai_share", "quarter_done_year_p10", "quarter_done_year_p50", "quarter_done_year_p90"]:
            if c in g.columns:
                row[c] = se._wavg(g, c, "employment_thousands")
        rows.append(row)
    dd = d.dropna(subset=["done_by_2035"])
    rho, n = _spearman(dd.skill_level, dd.done_by_2035)
    rho_seen, _ = _spearman(dd.skill_level, dd.share_weight_observed)
    # ANZSCO counts down (1 = highest skill), SOC counts up (4 = highest): read the sign accordingly
    higher_first = (rho is not None) and ((rho < -0.15) if spec.code == "AUS" else (rho > 0.15))
    lower_first = (rho is not None) and ((rho > 0.15) if spec.code == "AUS" else (rho < -0.15))
    reading = "no clear pattern across skill levels" if rho is None or (not higher_first and not lower_first) else ("AI reaches the higher-skilled work first" if higher_first else "AI reaches the lower-skilled work first")
    return pd.DataFrame(rows), {"rho_level_vs_done_2035": rho, "rho_level_vs_seen": rho_seen, "n_codes": n, "reading": reading}


def industry_view(mix: pd.DataFrame, ind_shift: pd.DataFrame, points: pd.DataFrame, t: pd.DataFrame, spec: CountrySpec) -> pd.DataFrame:
    """The sector lens: per industry, its share of employment, the firm AI-use readings, its shift against
    the national curve, and exposure through its occupation mix (group-level, employment-weighted)."""
    gcol = spec.group_col
    key = t.nat_code.str[0] if gcol == "major_group" else t.nat_code.str[:3]
    cols = ["observed_automation", "share_weight_observed", "share_weight_physical", "done_by_2030", "done_by_2035", "done_by_2035_p10", "done_by_2035_p90", "share_by_2035", "ceiling_ai_share", "quarter_done_year_p10", "quarter_done_year_p50", "quarter_done_year_p90"]
    cols = [c for c in cols if c in t.columns]
    grp = pd.DataFrame({c: t.assign(_g=key).groupby("_g").apply(lambda g: se._wavg(g, c, "employment_thousands"), include_groups=False) for c in cols})
    m = mix.copy()
    m["_g"] = m[gcol].astype(str)
    m["share_in_industry"] = m.employed / m.groupby("industry").employed.transform("sum")
    tot = m.employed.sum()
    rows = []
    pts = points[points.group_type == "industry"].dropna(subset=["pct"]).sort_values("date")
    for ind, g in m.groupby("industry"):
        row = {"industry": ind, "label": spec.industry_names.get(ind, ind), "employment_share": float(g.employed.sum() / tot) if tot else np.nan}
        gg = g.merge(grp, left_on="_g", right_index=True, how="left")
        for c in cols:
            ok = gg[c].notna()
            row[c] = float(np.average(gg.loc[ok, c], weights=gg.loc[ok, "share_in_industry"])) if ok.any() and gg.loc[ok, "share_in_industry"].sum() > 0 else np.nan
        row["share_of_groups_scored"] = float(gg.loc[gg[cols[0]].notna(), "share_in_industry"].sum()) if cols else np.nan
        p = pts[pts.industry == ind]
        row["ai_use_latest_pct"] = float(p.pct.iloc[-1]) if len(p) else np.nan
        row["ai_use_latest_date"] = float(p.date.iloc[-1]) if len(p) else np.nan
        row["ai_use_first_pct"] = float(p.pct.iloc[0]) if len(p) > 1 else np.nan
        row["ai_use_first_date"] = float(p.date.iloc[0]) if len(p) > 1 else np.nan
        s = ind_shift[ind_shift.industry == ind]
        row["anchor_shift_years"] = float(s.anchor_shift_years.iloc[0]) if len(s) else 0.0
        row["surveyed"] = bool(len(s))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("employment_share", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- 7. the official task lists against O*NET tasks
def match_tasks(tasks: pd.DataFrame, xw: pd.DataFrame, spec: CountrySpec, k_global: int = 20, k_out: int = 3, bonus: float = 0.10) -> pd.DataFrame:
    """Each official task item -> its closest O*NET tasks (hybrid text similarity), with a bonus for
    tasks of the O*NET occupations the code maps to, as match.match_ssoc_tasks does for Singapore."""
    from . import onet
    from .embed import HybridMatcher

    ot = onet.load_tasks()
    neigh = xw.groupby("nat_code").onet_soc_code.apply(set).to_dict()
    t = tasks.rename(columns={spec.code_col: "nat_code"}).reset_index(drop=True)
    m = HybridMatcher(ot.task_statement.tolist())
    idx, sc = m.query(t.task.tolist(), k=k_global)
    allowed = [neigh.get(c, set()) for c in t.nat_code]
    ridx, rsc, rin = _rerank_with_bonus(idx, sc, allowed, ot.onet_soc_code.values, bonus=bonus, k_out=k_out)
    rows = [(t.nat_code.iat[i], int(t.task_seq.iat[i]), r + 1, int(ot.task_id.iat[ridx[i, r]]), ot.onet_soc_code.iat[ridx[i, r]], float(rsc[i, r]), bool(rin[i, r])) for i in range(len(t)) for r in range(k_out)]
    out = pd.DataFrame(rows, columns=["nat_code", "task_seq", "rank", "task_id", "onet_soc_code", "score", "in_mapped_occ"])
    out["country"] = spec.code
    return out


def match_duties(standards: pd.DataFrame, std_soc: pd.DataFrame, xw: pd.DataFrame, k_global: int = 20, bonus: float = 0.10) -> pd.DataFrame:
    """UK apprenticeship duties -> the closest O*NET task, with the neighbourhood bonus for the O*NET
    occupations of the standard's SOC codes."""
    from . import onet
    from .embed import HybridMatcher

    ot = onet.load_tasks()
    soc_neigh = xw.groupby("nat_code").onet_soc_code.apply(set).to_dict()
    std_neigh = {}
    for r in std_soc.itertuples():
        std_neigh.setdefault(r.reference, set()).update(soc_neigh.get(r.soc_code, set()))
    rows_q = [(s.reference, i, d) for s in standards.itertuples() for i, d in enumerate(s.duties) if d]
    if not rows_q:
        return pd.DataFrame(columns=["reference", "duty_seq", "task_id", "onet_soc_code", "score", "in_mapped_occ"])
    m = HybridMatcher(ot.task_statement.tolist())
    idx, sc = m.query([q[2][:600] for q in rows_q], k=k_global)
    allowed = [std_neigh.get(q[0], set()) for q in rows_q]
    ridx, rsc, rin = _rerank_with_bonus(idx, sc, allowed, ot.onet_soc_code.values, bonus=bonus, k_out=1)
    return pd.DataFrame([(rows_q[i][0], rows_q[i][1], int(ot.task_id.iat[ridx[i, 0]]), ot.onet_soc_code.iat[ridx[i, 0]], float(rsc[i, 0]), bool(rin[i, 0])) for i in range(len(rows_q))], columns=["reference", "duty_seq", "task_id", "onet_soc_code", "score", "in_mapped_occ"])


# ----------------------------------------------------------------------------- orchestration
def build_base(spec: CountrySpec) -> dict:
    """The slow, model-independent tables for one country: weights, crosswalk, task matches, industry mix,
    adoption readings (and for the UK the apprenticeship standards). `python -m onetx countries`."""
    p = spec.prefix
    natw = spec.weights(); io.save(natw, f"{p}weights")
    codes = set(natw[spec.code_col].astype(str))
    xw = national_crosswalk(spec, codes); io.save(xw, f"{p}onet_crosswalk")
    io.save(onet_weights_from_national(xw, natw, spec), f"{p}onet_weights")
    io.save(spec.industry_mix(), f"{p}industry_mix")
    io.save(spec.ai_points(), f"{p}ai_adoption")
    tasks = spec.tasks(); io.save(tasks, f"{p}tasks")
    links = match_tasks(tasks, xw, spec); io.save(links, f"{p}task_onet_task")
    stats = {"codes": int(len(natw)), "codes_linked_to_onet": int(xw.nat_code.nunique()), "onet_occupations_linked": int(xw.onet_soc_code.nunique()), "employment_thousands_total": float(natw.employment_thousands.sum()),
             "wage_sources": natw.wage_source.value_counts(dropna=False).to_dict(), "official_tasks": int(len(tasks)), "codes_with_tasks": int(tasks[spec.code_col].nunique()), "task_match_mean_score": float(links[links["rank"] == 1].score.mean()), "task_match_in_mapped_occ_share": float(links[links["rank"] == 1].in_mapped_occ.mean())}
    if spec.code == "AUS":
        io.save(au.load_anzsco_pages().drop_duplicates("anzsco_code")[["anzsco_code", "description", "skill_text"]], "au_descriptions")
        io.save(au.load_anzsco_structure(), "au_structure")
    if spec.code == "GBR":
        st = uk.load_soc_structure(); io.save(st, "uk_structure")
        stds = uk.load_ifate_standards(); io.save(stds, "uk_ifate_standards")
        ss = uk.match_standards_to_soc(stds, st); io.save(ss, "uk_ifate_standard_soc")
        io.save(match_duties(stds, ss, xw), "uk_ifate_duty_onet_task")
        stats["apprenticeship_standards"] = {"approved": int(len(stds)), "matched_by_title": int(ss[ss.method == "title_in_coding_index"].reference.nunique()), "matched_by_text": int(ss[ss.method == "text_match"].reference.nunique()), "routes": int(stds.route.nunique()), "duties": int(stds.duties.map(len).sum())}
    return stats


def build_country(spec: CountrySpec, tf, barriers, params, now, tiers, occ, occf, btos_fit, o2s, ssoc_w, frontier_ssoc, econ_sg, fc_sg, us_w, econ_us, fc_us, others: dict | None = None) -> tuple[dict, dict]:
    """The timing run and every view for one country. Returns (stats block, context for cross runs)."""
    p = spec.prefix
    now_y = fcm._frac_year(pd.Timestamp(now))
    natw = io.load(f"{p}weights"); xw = io.load(f"{p}onet_crosswalk"); mix = io.load(f"{p}industry_mix"); points = io.load(f"{p}ai_adoption"); onet_w = io.load(f"{p}onet_weights")
    # 3. lead and sector shifts from the country's own survey
    lead = firm_lead(btos_fit, points, spec.min_lead_year)
    ind = industry_offsets(points, btos_fit); io.save(ind, f"{p}adoption_shift_industry")
    grp_shift, nat_shift, onet_shift = occupation_shifts(mix, ind, natw, xw, spec)
    io.save(nat_shift, f"{p}adoption_shift_national"); io.save(onet_shift, f"{p}adoption_shift_onet")
    # 4. the timing model on this country's lead and shifts
    params_c = fcm.ForecastParams(**{**fcm.asdict(params), "geo": spec.code, "geo_lead_years_measured": lead["lead_years"] if lead["lead_years"] is not None else 0.0})
    fc = fcm.forecast_tasks(tf, barriers, params_c, now=now, tiers=tiers, occ_shift=onet_shift); io.save(fc, f"forecast_tasks_{p.rstrip('_')}")
    occ_curve = fcm.disruption_curve(fc, "onet_soc_code").merge(occ[["onet_soc_code", "title"]], on="onet_soc_code", how="left"); io.save(occ_curve, f"forecast_occupations_{p.rstrip('_')}")
    ns = nat_shift.set_index("nat_code").adoption_shift_years
    curve = national_curve(fc, xw, ns, params_c, now_y); io.save(curve, f"forecast_{p}national")
    fr_nat = push_frontier(occf, xw); io.save(fr_nat, f"{p}frontier_national")
    t = weighted_table(natw, fr_nat, curve, spec); io.save(t, f"{p}weighted")
    econ = economy(t, spec); io.save(econ, f"{p}economy_exposure")
    # cross runs: this country's weights on the others' timing and theirs on this one
    def own_weights_on(fc_other):
        return economy(weighted_table(natw, fr_nat, national_curve(fc_other, xw, None, params_c, now_y), spec), spec)
    f2s = fc.merge(o2s[["onet_soc_code", "ssoc_code", "ssoc_title", "weight_rev"]], on="onet_soc_code"); f2s["w"] = f2s.task_weight * f2s.weight_rev
    e_sgw_ct = se.economy_exposure(se.ssoc_weighted_table(frontier_ssoc, fcm.disruption_curve(f2s, "ssoc_code", weight="w"), ssoc_w))
    e_usw_ct = ue.economy_exposure(ue.onet_weighted_table(occf, fcm.disruption_curve(fc, "onet_soc_code"), us_w), "USA")
    decomps = {}
    for w in ["wage_value", "employment"]:
        decomps[f"SGP_{spec.code}_{w}"] = pair_decomposition(econ_sg, econ, own_weights_on(fc_sg), e_sgw_ct, w, "SGP", spec.code)
        decomps[f"USA_{spec.code}_{w}"] = pair_decomposition(econ_us, econ, own_weights_on(fc_us), e_usw_ct, w, "USA", spec.code)
    for oc, octx in (others or {}).items():
        for w in ["wage_value", "employment"]:
            e_ow_ct = economy(weighted_table(octx["natw"], octx["fr_nat"], national_curve(fc, octx["xw"], None, params_c, now_y), octx["spec"]), octx["spec"])
            decomps[f"{oc}_{spec.code}_{w}"] = pair_decomposition(octx["econ"], econ, own_weights_on(octx["fc"]), e_ow_ct, w, oc, spec.code)
    # 5. rung five, 6. skill levels, the sector lens
    try:
        change = spec.outcomes(); io.save(change, f"{p}outcomes")
        out_stats = outcome_tests(change, t, spec)
    except FileNotFoundError as e:
        out_stats = {"error": str(e)}
    bands, band_stats = skill_level_bands(t, spec); io.save(bands, f"{p}skill_level_bands")
    lv = ue.skill_level_exposure(onet_w, io.load("onet_descriptor_ratings_v31"), occf, occ_curve)
    io.save(lv["bands"], f"{p}onet_skill_level_exposure"); io.save(lv["correlations"], f"{p}onet_skill_level_correlations")
    iv = industry_view(mix, ind, points, t, spec); io.save(iv, f"{p}industry_view")
    route_stats = {}
    if spec.code == "GBR" and io.exists("uk_ifate_standards"):
        rv = route_view(io.load("uk_ifate_standards"), io.load("uk_ifate_standard_soc"), t); io.save(rv, "uk_ifate_route_view")
        route_stats = {"routes": int(len(rv)), "earliest": rv.iloc[0].route if len(rv) else None, "latest": rv.iloc[-1].route if len(rv) else None}
    e = econ[econ.scope == "economy"].set_index("weighting")
    stats = {"name": spec.name, "classification": spec.classification, "lead": lead, "geo_lead_years": lead["lead_years"], "industry_shifts": {r.industry: round(float(r.anchor_shift_years), 2) for r in ind.itertuples()},
             "shift_range": [float(nat_shift.adoption_shift_years.min()), float(nat_shift.adoption_shift_years.max())] if len(nat_shift) else [0.0, 0.0],
             "weights": {"employment_thousands_total": float(natw.employment_thousands.sum()), "wage_value_total": float(natw.wage_value.sum()), "codes": int(len(natw)), "wage_sources": natw.wage_source.value_counts(dropna=False).to_dict(), "edition": int(natw.edition.iloc[0]) if "edition" in natw else None},
             "economy": {w: {"observed_automation": float(e.loc[w, "observed_automation"]), "share_weight_observed": float(e.loc[w, "share_weight_observed"]), "done_by_2035": float(e.loc[w, "done_by_2035"]), "share_by_2035": float(e.loc[w, "share_by_2035"]), "quarter_done_year_central": float(e.loc[w, "quarter_done_year_central_from_curve"]) if pd.notna(e.loc[w, "quarter_done_year_central_from_curve"]) else None} for w in ["employment", "wage_value"] if w in e.index},
             "outcomes": out_stats, "skill_levels_national": band_stats, "skill_levels_onet": {"index_rho": float(lv["correlations"].set_index("element_name").loc["All essential skills (average level)", "rho_level_vs_done_2035"]) if "All essential skills (average level)" in set(lv["correlations"].element_name) else None},
             "apprenticeship_routes": route_stats, "decompositions": decomps}
    ctx = {"spec": spec, "natw": natw, "xw": xw, "fr_nat": fr_nat, "econ": econ, "fc": fc, "params": params_c}
    return stats, ctx


def route_view(standards: pd.DataFrame, std_soc: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """The 15 apprenticeship routes as a sector lens: each route's standards, the SOC codes they lead to,
    and the exposure of those codes weighted by employment times the standard's share in the code."""
    m = std_soc.merge(standards[["reference", "route", "title", "level"]], on="reference", how="left").merge(t[["nat_code", "employment_thousands", "observed_automation", "share_weight_observed", "share_weight_physical", "done_by_2030", "done_by_2035", "done_by_2035_p10", "done_by_2035_p90", "share_by_2035", "ceiling_ai_share", "quarter_done_year_p10", "quarter_done_year_p50", "quarter_done_year_p90"]], left_on="soc_code", right_on="nat_code", how="left")
    m["w"] = m.share * m.employment_thousands.fillna(0.0)
    rows = []
    for route, g in m.groupby("route"):
        row = {"route": route, "n_standards": int(g.reference.nunique()), "n_soc_codes": int(g.soc_code.nunique()), "employment_thousands": float(g.drop_duplicates("soc_code").employment_thousands.sum()), "levels": ", ".join(str(int(x)) for x in sorted(g.level.dropna().unique()))}
        for c in ["observed_automation", "share_weight_observed", "share_weight_physical", "done_by_2030", "done_by_2035", "done_by_2035_p10", "done_by_2035_p90", "share_by_2035", "ceiling_ai_share", "quarter_done_year_p10", "quarter_done_year_p50", "quarter_done_year_p90"]:
            row[c] = se._wavg(g, c, "w")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("quarter_done_year_p50").reset_index(drop=True)


def pair_decomposition(e_a: pd.DataFrame, e_b: pd.DataFrame, e_aw_bt: pd.DataFrame, e_bw_at: pd.DataFrame, weighting: str, a: str, b: str) -> dict:
    """Split the A-minus-B gap in 'AI doing by year' into a job-mix part and a timing part (the mean of
    the two orderings, which add up exactly), for the central, fast and slow cases."""
    def row(e):
        return e[(e.scope == "economy") & (e.weighting == weighting)].iloc[0]
    ra, rb, x, y = row(e_a), row(e_b), row(e_aw_bt), row(e_bw_at)
    out = {"weighting": weighting, "a": a, "b": b, "by_year": {}}
    for q, suf in [("central", ""), ("fast", "_p10"), ("slow", "_p90")]:
        for yv in (2030, 2035, 2040):
            c = f"done_by_{yv}{suf}"
            gap = float(ra[c] - rb[c])
            jobmix = 0.5 * (float(x[c] - rb[c]) + float(ra[c] - y[c]))
            out["by_year"][f"{q}_{yv}"] = {a: float(ra[c]), b: float(rb[c]), "gap": gap, "job_mix": jobmix, "timing": gap - jobmix, "a_weights_b_timing": float(x[c]), "b_weights_a_timing": float(y[c])}
    for q in ("central", "fast", "slow"):
        c = f"quarter_done_year_{q}_from_curve"
        if c in ra and c in rb:
            out[f"quarter_done_year_{q}"] = {a: float(ra[c]) if pd.notna(ra[c]) else None, b: float(rb[c]) if pd.notna(rb[c]) else None}
    return out
