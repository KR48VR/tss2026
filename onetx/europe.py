"""Europe at a glance: every country in Eurostat's enterprise AI-use survey, run through the same timing model
at the resolution Eurostat publishes (ISCO-08 sub-major groups, two digits), without job pages.

Inputs (`data/raw/eurostat/`, JSON-stat from the Eurostat dissemination API, `fetch.SOURCES`):
  isoc_eb_ai.json     enterprises (10 or more persons employed) using at least one AI technology, % of
                      enterprises, by country, 2021, 2023, 2024, 2025 (the national readings)
  isoc_eb_ain2.json   the same by NACE activity (the industry readings)
  lfsa_egai2d.json    employed persons (thousands) by ISCO-08 two-digit occupation, 15-64, 2024 and 2025 (EU-LFS)
  earn_ses22_14.json  mean hourly earnings by ISCO-08 major group, 2022 (Structure of Earnings Survey), in euro
  lfsa_eisn2.json     employed persons by ISCO major group and NACE section, 2024 (the occupation-by-industry mix)

For each country the steps are the ones `countries.build_country` runs for a full workforce, through a country
record built on the fly: the O*NET -> ISCO crosswalk rolled up to two digits, the lead read off the US curve
from the country's own readings, industry shifts from the activity readings carried through the major-group
mix, the timing run, and the economy, group and industry views. Pay is the major group's mean hourly earnings
(the finest the SES publishes by occupation), so wage-bill shares are coarse; the employment-weighted figures
are the better read. Nothing here replaces a full country record (Norway has one); it shows the same model on
36 workforces at once.
"""
from __future__ import annotations

import pandas as pd

from . import countries as cn
from . import forecast as fcm
from . import io
from .config import RAW
from .no_labour import jsonstat_frame, _skill_level, MAJOR_GROUP_NAMES, SKILL_LEVEL_NAMES

EUROSTAT = RAW / "eurostat"
AGGREGATES = {"EU27_2020": "European Union (27)", "EA": "Euro area"}
SURVEY_DATE_IN_YEAR = 0.4   # the enterprise ICT survey is collected around the second quarter of the reference year
NACE_SECTIONS = {"A": "Agriculture, forestry and fishing", "B": "Mining and quarrying", "C": "Manufacturing", "D": "Electricity and gas", "E": "Water and waste", "F": "Construction",
                 "G": "Wholesale and retail trade", "H": "Transportation and storage", "I": "Accommodation and food service", "J": "Information and communication", "K": "Financial and insurance",
                 "L": "Real estate", "M": "Professional, scientific and technical", "N": "Administrative and support services", "O": "Public administration", "P": "Education",
                 "Q": "Human health and social work", "R": "Arts, entertainment and recreation", "S": "Other services", "T": "Households as employers", "U": "Extraterritorial organisations"}
# activity codes in the AI survey that name one NACE section, or a pair of sections read as both
_ACTIVITY_TO_SECTIONS = {"C": ["C"], "D_E": ["D", "E"], "F": ["F"], "G": ["G"], "H": ["H"], "I": ["I"], "J": ["J"], "L": ["L"], "M": ["M"], "N": ["N"], "L_M": ["L", "M"]}


def _frame(name: str) -> pd.DataFrame:
    return jsonstat_frame(EUROSTAT / f"{name}.json")


def country_names() -> dict:
    d = _frame("isoc_eb_ai").drop_duplicates("geo")
    names = dict(zip(d.geo, d.label_geo))
    names.update(AGGREGATES)
    return names


def load_ai_use() -> pd.DataFrame:
    """National readings: geo, date, period, pct, group_type='all'."""
    d = _frame("isoc_eb_ai").dropna(subset=["value"])
    d["year"] = d.time.astype(int)
    return pd.DataFrame({"geo": d.geo, "date": d.year + SURVEY_DATE_IN_YEAR, "period": d.time, "pct": d.value, "group_type": "all", "industry": None, "label": "All enterprises with 10 or more persons employed"})


def load_ai_use_by_activity() -> pd.DataFrame:
    """Industry readings keyed to NACE sections: geo, date, period, pct, group_type='industry', industry, label."""
    d = _frame("isoc_eb_ain2").dropna(subset=["value"])
    rows = []
    for r in d.itertuples():
        for sec in _ACTIVITY_TO_SECTIONS.get(r.nace_r2, []):
            rows.append({"geo": r.geo, "date": int(r.time) + SURVEY_DATE_IN_YEAR, "period": r.time, "pct": float(r.value), "group_type": "industry", "industry": sec, "label": NACE_SECTIONS[sec]})
    out = pd.DataFrame(rows)
    # a section quoted by two activity codes (L in 'L' and 'L_M') keeps the single-section reading, listed first
    return out.drop_duplicates(["geo", "date", "industry"], keep="first").reset_index(drop=True)


def ai_points(geo: str) -> pd.DataFrame:
    a = load_ai_use(); b = load_ai_use_by_activity()
    return pd.concat([a[a.geo == geo], b[b.geo == geo]], ignore_index=True).drop(columns="geo")


def load_employment_2d() -> pd.DataFrame:
    """Employed persons (thousands) by two-digit ISCO-08 group, the latest year each country has."""
    d = _frame("lfsa_egai2d").dropna(subset=["value"])
    d = d[d.isco08.str.fullmatch(r"OC\d\d")].copy()
    d["isco2_code"] = d.isco08.str[2:]
    d["isco2_title"] = d.label_isco08
    d["year"] = d.time.astype(int)
    latest = d.groupby("geo").year.transform("max")
    d = d[d.year == latest]
    return d[["geo", "isco2_code", "isco2_title", "year", "value"]].rename(columns={"value": "employment_thousands"}).reset_index(drop=True)


def load_earnings_1d() -> pd.DataFrame:
    """Mean hourly earnings (euro, 2022) by ISCO major group: geo, major_group, mean_hourly_eur."""
    frames = []
    for name, year in (("earn_ses22_14", 2022), ("earn_ses18_14", 2018)):
        if not (EUROSTAT / f"{name}.json").exists():
            continue
        d = _frame(name).dropna(subset=["value"])
        d = d[d.isco08.str.fullmatch(r"OC\d")].copy()
        d["major_group"] = d.isco08.str[-1]
        # all enterprises where published, else those with 10 or more employees (France and others publish only the latter)
        tot = d[d.sizeclas == "TOTAL"]; ge = d[(d.sizeclas == "GE10") & ~d.geo.isin(set(tot.geo))]
        d = pd.concat([tot, ge], ignore_index=True)
        # euro where published, else the national currency (only shares of each country's own wage bill are used, so the unit does not matter)
        eur = d[d.unit == "EUR"]; nac = d[(d.unit == "NAC") & ~d.geo.isin(set(eur.geo))]
        d = pd.concat([eur, nac], ignore_index=True); d["year"] = year
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    # the 2022 survey where a country published it by major group, else 2018
    keep = d.groupby("geo").year.transform("max")
    d = d[d.year == keep]
    return d[["geo", "major_group", "value", "unit", "year"]].rename(columns={"value": "mean_hourly_eur"}).reset_index(drop=True)


def load_mix() -> pd.DataFrame:
    """Employed persons by ISCO major group and NACE section (EU-LFS, 2024, 15 and over): geo, major_group, industry, employed."""
    d = _frame("lfsa_eisn2").dropna(subset=["value"])
    d = d[(d.age == "Y_GE15") & d.isco08.str.fullmatch(r"OC\d") & d.nace_r2.str.fullmatch(r"[A-U]")].copy()
    d["major_group"] = d.isco08.str[-1]
    return d[["geo", "major_group", "nace_r2", "value"]].rename(columns={"nace_r2": "industry", "value": "employed"}).reset_index(drop=True)


def weights(geo: str, emp: pd.DataFrame | None = None, earn: pd.DataFrame | None = None) -> pd.DataFrame:
    """The country's two-digit table in the shape the shared layer expects."""
    emp = load_employment_2d() if emp is None else emp
    earn = load_earnings_1d() if earn is None else earn
    w = emp[emp.geo == geo].drop(columns="geo").copy()
    w["major_group"] = w.isco2_code.str[0]
    w["submajor_label"] = w.isco2_code
    w["submajor_name"] = w.isco2_title
    w["skill_level"] = w.isco2_code.map(_skill_level)
    e = earn[earn.geo == geo].set_index("major_group").mean_hourly_eur
    w["wage_hourly"] = w.major_group.map(e)
    yr = earn[earn.geo == geo].year.iloc[0] if (earn.geo == geo).any() else None
    w["wage_source"] = f"ses_{yr}_major_group_mean_hourly" if yr else "none"
    if w.wage_hourly.isna().all():   # no earnings survey for this country: employment alone carries the wage bill
        w["wage_hourly"] = 1.0
        w["wage_source"] = "no_earnings_survey_equal_pay"
    else:
        w["wage_hourly"] = w.wage_hourly.fillna(w.wage_hourly.mean())
    w["wage_value"] = w.employment_thousands * w.wage_hourly
    w["employment_method"] = f"eu_lfs_{int(w.year.iloc[0]) if len(w) else ''}"
    return w.drop(columns="year").reset_index(drop=True)


def _isco_to_2d(unit_codes: set[str] | None = None) -> pd.DataFrame:
    a = io.load("xw_onetsoc_isco08")[["isco_code"]].drop_duplicates()
    a["nat_code"] = a.isco_code.astype(str).str[:2]
    a = a[a.nat_code.isin(unit_codes)] if unit_codes else a
    a["weight"] = 1.0
    return a[["isco_code", "nat_code", "weight"]].reset_index(drop=True)


def eu_spec(geo: str, name: str) -> cn.CountrySpec:
    """A country record for one Eurostat country at two-digit resolution."""
    return cn.CountrySpec(code=geo, name=name, adjective=name, prefix=f"eu_{geo.lower()}_", code_col="isco2_code", title_col="isco2_title", classification="ISCO-08 sub-major group",
                          major_names=MAJOR_GROUP_NAMES, skill_level_names=SKILL_LEVEL_NAMES, industry_names=NACE_SECTIONS, industry_label="NACE section", currency="€",
                          wage_label="mean hourly earnings of the major group, 2022 (Eurostat SES)", emp_label="employed persons (EU-LFS)", wage_period="2022",
                          adoption_source="Eurostat survey of ICT usage in enterprises", group_col="major_group", group_len=1,
                          weights=lambda: weights(geo), outcomes=lambda: pd.DataFrame(), isco_link=_isco_to_2d, industry_mix=lambda: load_mix().query("geo == @geo").drop(columns="geo"),
                          ai_points=lambda: ai_points(geo), tasks=lambda: pd.DataFrame(columns=["isco2_code", "task_seq", "task"]))


def build_europe(tf, barriers, params, now, tiers, occ, occf, btos_fit, geos: list | None = None, log=print) -> dict:
    """Run the model for every Eurostat country with AI-use readings and two-digit employment. Saves
    eu_economy, eu_groups (one row per country and sub-major group), eu_industry_view and returns the stats
    (per country: name, lead, readings, economy figures, counts)."""
    names = country_names()
    emp = load_employment_2d(); earn = load_earnings_1d(); mix_all = load_mix(); ai = load_ai_use()
    have = sorted(set(ai.geo) & set(emp.geo))
    geos = [g for g in (geos or have) if g in have]
    now_y = fcm._frac_year(pd.Timestamp(now))
    econ_rows, group_rows, ind_rows, stats = [], [], [], {}
    for geo in geos:
        spec = eu_spec(geo, names.get(geo, geo))
        natw = weights(geo, emp, earn)
        if natw.employment_thousands.sum() <= 0:
            continue
        xw = cn.national_crosswalk(spec, set(natw.isco2_code))
        points = ai_points(geo)
        lead = cn.firm_lead(btos_fit, points, spec.min_lead_year)
        ind = cn.industry_offsets(points, btos_fit)
        mix = mix_all[mix_all.geo == geo].drop(columns="geo")
        grp_shift, nat_shift, onet_shift = cn.occupation_shifts(mix, ind, natw, xw, spec)
        params_c = fcm.ForecastParams(**{**fcm.asdict(params), "geo": geo, "geo_lead_years_measured": lead["lead_years"] if lead["lead_years"] is not None else 0.0})
        fc = fcm.forecast_tasks(tf, barriers, params_c, now=now, tiers=tiers, occ_shift=onet_shift)
        ns = nat_shift.set_index("nat_code").adoption_shift_years
        curve = cn.national_curve(fc, xw, ns, params_c, now_y)
        fr_nat = cn.push_frontier(occf, xw)
        t = cn.weighted_table(natw, fr_nat, curve, spec)
        econ = cn.economy(t, spec)
        iv = cn.industry_view(mix, ind, points, t, spec)
        econ["geo"] = geo; t["geo"] = geo; iv["geo"] = geo
        econ_rows.append(econ); group_rows.append(t); ind_rows.append(iv)
        e = econ[econ.scope == "economy"].set_index("weighting")
        stats[geo] = {"name": names.get(geo, geo), "lead": lead, "shift_range": [float(nat_shift.adoption_shift_years.min()), float(nat_shift.adoption_shift_years.max())] if len(nat_shift) else [0.0, 0.0],
                      "employment_thousands_total": float(natw.employment_thousands.sum()), "codes": int(len(natw)), "wage_source": natw.wage_source.iloc[0], "employment_year": natw.employment_method.iloc[0],
                      "industries_surveyed": int((iv.ai_use_latest_pct.notna()).sum()) if "ai_use_latest_pct" in iv.columns else 0,
                      "economy": {w: {k: float(e.loc[w, k]) for k in ("observed_automation", "share_weight_observed", "done_by_2035", "share_by_2035") if k in e.columns} | {"quarter_done_year_central": float(e.loc[w, "quarter_done_year_central_from_curve"]) if "quarter_done_year_central_from_curve" in e.columns and pd.notna(e.loc[w, "quarter_done_year_central_from_curve"]) else None} for w in e.index}}
        log(f"ok    Europe {geo} {names.get(geo, geo)}: lead {lead['lead_years']}, AI doing by 2035 (wage) {stats[geo]['economy'].get('wage_value', {}).get('done_by_2035')}")
    if econ_rows:
        io.save(pd.concat(econ_rows, ignore_index=True), "eu_economy")
        io.save(pd.concat(group_rows, ignore_index=True), "eu_groups")
        io.save(pd.concat(ind_rows, ignore_index=True), "eu_industry_view")
    return stats
