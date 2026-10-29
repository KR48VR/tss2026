"""Norway: a workforce on STYRK-08, Statistics Norway's edition of ISCO-08 (the same four-digit codes), from
the open SSB API (data.ssb.no, JSON-stat 2.0).

Inputs (`data/raw/ssb/`):
  11658_employees_earnings_by_occupation.json   table 11658: employees, jobs, full-time equivalents, average and
                                                median monthly earnings by occupation (1- to 4-digit STYRK-08),
                                                both sexes, all ages, second quarters of 2024, 2025 and 2026
  13265_ai_use_by_industry_2021_2025.json       table 13265: share of enterprises (10+ employed) using at least one
                                                AI technology, by industry (SIC2007 groupings) and size, 2021-2025
  14942_ai_use_by_industry_2026.json            table 14942: the same for 2026 on SIC2025 groupings
plus Eurostat's occupation-by-activity table for Norway (`data/raw/eurostat/lfsa_eisn2.json`, ISCO major group by
NACE section) for the mix that carries industry shifts to occupations.

Norway publishes no official task list per occupation, so its job pages carry O*NET's tasks through the ISCO
link only.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW

SSB = RAW / "ssb"
EUROSTAT = RAW / "eurostat"

MAJOR_GROUP_NAMES = {"0": "Armed forces occupations", "1": "Managers", "2": "Professionals", "3": "Technicians and associate professionals", "4": "Clerical support workers",
                     "5": "Service and sales workers", "6": "Skilled agricultural, forestry and fishery workers", "7": "Craft and related trades workers",
                     "8": "Plant and machine operators and assemblers", "9": "Elementary occupations"}
# ISCO-08 skill levels: 1 simple routine tasks; 2 operating machinery, clerical and service work; 3 complex technical tasks; 4 complex problem-solving (degree level)
SKILL_LEVEL_NAMES = {1: "simple physical or manual tasks (primary education)", 2: "operating machinery, clerical, sales and service work (secondary education)",
                     3: "complex technical and practical tasks (post-secondary education)", 4: "complex problem-solving, decision-making and creative work (degree level)"}
# the industry groupings SSB uses in its ICT-use survey, keyed to NACE sections for the occupation mix
INDUSTRIES = {"C-E": "Manufacturing, electricity, water and waste", "F": "Construction", "G": "Wholesale and retail trade", "H": "Transportation and storage",
              "I": "Accommodation and food service", "J": "Information and communication", "K": "Financial and insurance activities", "L-N": "Real estate, professional, administrative and other business services"}
_SIC_TO_KEY = {"10-39": "C-E", "41-43": "F", "45": "G", "46": "G", "47": "G", "46-47": "G", "49-53": "H", "55-56": "I", "58-63": "J", "64-66": "K", "68-74+77-82+95.1": "L-N", "68-75+77-82+95.1": "L-N"}
_SECTION_TO_KEY = {"C": "C-E", "D": "C-E", "E": "C-E", "F": "F", "G": "G", "H": "H", "I": "I", "J": "J", "K": "K", "L": "L-N", "M": "L-N", "N": "L-N"}


def jsonstat_frame(path: Path) -> pd.DataFrame:
    """A JSON-stat 2.0 dataset as a long table: one column per dimension (codes), `label_<dim>` for the
    labels, and `value`."""
    d = json.loads(Path(path).read_text())
    dims = d["id"]
    cats = []
    for k in dims:
        c = d["dimension"][k]["category"]
        order = sorted(c["index"], key=lambda x: c["index"][x]) if isinstance(c["index"], dict) else list(c["index"])
        cats.append([(code, c.get("label", {}).get(code, code)) for code in order])
    rows = list(itertools.product(*cats))
    vals = d["value"]
    if isinstance(vals, dict):
        v = [vals.get(str(i)) for i in range(len(rows))]
    else:
        v = vals
    out = pd.DataFrame({k: [r[i][0] for r in rows] for i, k in enumerate(dims)})
    for i, k in enumerate(dims):
        out[f"label_{k}"] = [r[i][1] for r in rows]
    out["value"] = pd.to_numeric(pd.Series(v), errors="coerce")
    return out


def load_occupation_table() -> pd.DataFrame:
    """Table 11658 in long form: styrk_code, title, quarter, measure (employees, jobs, fte, mean_monthly,
    median_monthly, mean_basic_monthly), value."""
    df = jsonstat_frame(SSB / "11658_employees_earnings_by_occupation.json")
    m = {"Lonsstakere": "employees", "AntArbForhold": "jobs", "HeltidsEkvMnd": "fte", "GjMdTotal": "mean_monthly", "MedianMndLonn": "median_monthly", "GjAvtaltMdlonn": "mean_basic_monthly"}
    df["measure"] = df.ContentsCode.map(m)
    return df.rename(columns={"Yrke": "styrk_code", "label_Yrke": "title", "Tid": "quarter"})[["styrk_code", "title", "quarter", "measure", "value"]]


def _skill_level(code: str) -> float:
    m, sm = code[0], code[:2]
    if m == "1":
        return 3.0 if sm == "14" else 4.0
    if m == "0":
        return {"01": 4.0, "02": 2.0, "03": 1.0}.get(sm, 2.0)
    return {"2": 4.0, "3": 3.0, "4": 2.0, "5": 2.0, "6": 2.0, "7": 2.0, "8": 2.0, "9": 1.0}[m]


def build_styrk_weights(quarter: str = "2026K2") -> pd.DataFrame:
    """Per four-digit occupation: employees (thousands) and median monthly earnings (NOK) in the quarter,
    the 3- or 2-digit median where the four-digit cell is suppressed, their product as the wage weight, the
    ISCO skill level and group names."""
    t = load_occupation_table()
    q = t[t.quarter == quarter]
    wide = q.pivot_table(index=["styrk_code", "title"], columns="measure", values="value", aggfunc="first").reset_index()
    four = wide[wide.styrk_code.str.fullmatch(r"\d{4}")].copy()
    four = four[four.employees.fillna(0) > 0]
    med3 = wide[wide.styrk_code.str.fullmatch(r"\d{3}")].set_index("styrk_code").median_monthly
    med2 = wide[wide.styrk_code.str.fullmatch(r"\d{2}")].set_index("styrk_code").median_monthly
    df = pd.DataFrame({"styrk_code": four.styrk_code.values, "styrk_title": four.title.values, "employment_thousands": four.employees.values / 1000.0})
    df["wage_monthly"] = four.median_monthly.values
    df["wage_source"] = f"ssb_11658_{quarter}_median_monthly"
    for fallback, label in ((df.styrk_code.str[:3].map(med3), "minor_group_median"), (df.styrk_code.str[:2].map(med2), "submajor_group_median")):
        miss = df.wage_monthly.isna() & fallback.notna()
        df.loc[miss, "wage_monthly"] = fallback[miss]
        df.loc[miss, "wage_source"] = label
    df["major_group"] = df.styrk_code.str[0]
    df["submajor_label"] = df.styrk_code.str[:2]
    names2 = wide[wide.styrk_code.str.fullmatch(r"\d{2}")].set_index("styrk_code").title
    df["submajor_name"] = df.submajor_label.map(names2)
    df["skill_level"] = df.styrk_code.map(_skill_level)
    df["wage_value"] = df.employment_thousands * df.wage_monthly
    df["employment_method"] = f"register_{quarter}"
    df["edition"] = 2026
    return df.reset_index(drop=True)


def load_isco_link(unit_codes: set[str] | None = None) -> pd.DataFrame:
    """ISCO-08 unit group -> STYRK-08 code: the same four digits, weight 1, for the codes the table carries."""
    codes = sorted(unit_codes) if unit_codes else sorted(build_styrk_weights().styrk_code)
    return pd.DataFrame({"isco_code": codes, "nat_code": codes, "weight": 1.0})


def employment_change(q_from: str = "2025K2", q_to: str = "2026K2") -> pd.DataFrame:
    """Rung five for Norway: employees and median monthly earnings per four-digit occupation, second quarter
    to second quarter, from the same register table."""
    t = load_occupation_table()
    four = t[t.styrk_code.str.fullmatch(r"\d{4}")]
    emp = four[four.measure == "employees"].pivot_table(index="styrk_code", columns="quarter", values="value", aggfunc="first")
    wage = four[four.measure == "median_monthly"].pivot_table(index="styrk_code", columns="quarter", values="value", aggfunc="first")
    yf, yt = q_from[:4], q_to[:4]
    df = pd.DataFrame({"styrk_code": emp.index, f"emp_{yf}": emp[q_from].values, f"emp_{yt}": emp[q_to].values})
    df["emp_change_pct"] = 100.0 * (df[f"emp_{yt}"] / df[f"emp_{yf}"].where(df[f"emp_{yf}"] > 0) - 1.0)
    df[f"wage_{yf}"] = df.styrk_code.map(wage[q_from]); df[f"wage_{yt}"] = df.styrk_code.map(wage[q_to])
    df["wage_change_pct"] = 100.0 * (df[f"wage_{yt}"] / df[f"wage_{yf}"].where(df[f"wage_{yf}"] > 0) - 1.0)
    df["year_from"], df["year_to"] = int(yf), int(yt)
    df["wage_years"] = f"{yf}-{yt}"
    return df.reset_index(drop=True)


def outcome_units() -> pd.DataFrame:
    w = build_styrk_weights()
    return pd.DataFrame({"styrk_code": w.styrk_code.values, "emp_unit": "people", "wage_unit": "month"})


def load_ai_use() -> pd.DataFrame:
    """Share of enterprises (10 or more employed) using at least one AI technology: nationally (all industries
    excluding finance, the survey's own total) and by industry grouping, 2021, 2023, 2024, 2025 (SIC2007) and
    2026 (SIC2025). A survey year is dated at its second quarter."""
    rows = []
    for f, nace_col in (("13265_ai_use_by_industry_2021_2025.json", "NACE2007"), ("14942_ai_use_by_industry_2026.json", "NACE2025")):
        d = jsonstat_frame(SSB / f)
        d = d[(d.SyssGrpIKT == "00") & (d.ContentsCode == "BrukarKunstigIntelli")].dropna(subset=["value"])
        for r in d.itertuples():
            ind = getattr(r, nace_col)
            year = int(r.Tid)
            if ind in ("Total-K", "_T"):
                rows.append({"date": year + 0.4, "period": str(year), "pct": float(r.value), "group_type": "all", "industry": None, "label": "All industries (10+ employed, excluding finance)"})
            elif ind in _SIC_TO_KEY:
                rows.append({"date": year + 0.4, "period": str(year), "pct": float(r.value), "group_type": "industry", "industry": _SIC_TO_KEY[ind], "label": INDUSTRIES[_SIC_TO_KEY[ind]]})
    df = pd.DataFrame(rows)
    # the trade groups 45, 46 and 47 are published separately before 2026: one reading per year for G, the mean of the three
    g = df[(df.group_type == "industry") & (df.industry == "G")].groupby(["date", "period", "group_type", "industry", "label"], dropna=False).pct.mean().reset_index()
    df = pd.concat([df[~((df.group_type == "industry") & (df.industry == "G"))], g], ignore_index=True)
    return df.sort_values(["group_type", "industry", "date"]).reset_index(drop=True)


def load_industry_mix() -> pd.DataFrame:
    """Employed persons by ISCO major group and the survey's industry groupings, from Eurostat's occupation by
    economic activity table for Norway (EU-LFS, 2024, 15 and over)."""
    d = jsonstat_frame(EUROSTAT / "lfsa_eisn2.json")
    d = d[(d.geo == "NO") & (d.age == "Y_GE15") & d.isco08.str.fullmatch(r"OC\d") & d.nace_r2.str.fullmatch(r"[A-U]")].dropna(subset=["value"])
    d["major_group"] = d.isco08.str[-1]
    d["industry"] = d.nace_r2.map(_SECTION_TO_KEY)
    d = d.dropna(subset=["industry"])
    return d.groupby(["major_group", "industry"]).value.sum().reset_index().rename(columns={"value": "employed"})


def load_titles() -> pd.DataFrame:
    """The four-digit occupation titles (English) as the only title index Norway publishes."""
    w = build_styrk_weights()
    return pd.DataFrame({"styrk_code": w.styrk_code, "title": w.styrk_title, "category": "principal"})


def no_tasks() -> pd.DataFrame:
    return pd.DataFrame(columns=["styrk_code", "task_seq", "task"])
