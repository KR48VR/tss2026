"""Canada: the fifth workforce. Census employment and income by NOC 2021 unit group, Job Bank wages, the
Statistics Canada concordances that link NOC to the US SOC (and so to O*NET without going through ISCO),
the Canadian Survey on Business Conditions' yearly AI question, the Census occupation-by-industry
table, job vacancies by unit group, and ESDC's OaSIS (Occupational and Skills Information System): a
Canadian O*NET with main duties, example titles and skill, knowledge and work-activity ratings per
occupation, the Skills Framework stand-in for Canada.

Sources (data/raw/statcan/ and data/raw/esdc/, see docs/DATA_SOURCES.md):
  noc2021_structure.csv            NOC 2021 V1.0: broad category, major, sub-major, minor and unit groups
  soc2018us-noc2016v1_3-eng.csv    Statistics Canada concordance SOC 2018 (US) -> NOC 2016 V1.3
  noc2016v1_3-noc2021v1_0-eng.csv  concordance NOC 2016 V1.3 -> NOC 2021 V1.0
  api/api_98100449_employed.csv    Census 2021: employed persons by unit group (Canada, all ages and genders),
                                   pulled cell by cell through the WDS API (the full table is 1.8 GB)
  api/api_98100586_income.csv      Census 2021: median and average employment income (2020) by unit group,
                                   all workers and full-year full-time workers
  api/api_98100594_occ_industry.csv Census 2021: employed persons by minor group (4-digit) and NAICS sector
  api/api_14100444_vacancies.csv   Job Vacancy and Wage Survey: vacancies and offered wage by unit group,
                                   the latest nine quarters
  33100825/33101004/33101167.csv   Canadian Survey on Business Conditions, second quarters of 2024, 2025 and
                                   2026: share of businesses that used AI in producing goods or delivering
                                   services over the last 12 months, by NAICS sector and size
  14100416.csv                     Labour Force Survey: employment by NOC group (2-3 digits), annual
  ../esdc/wages/wages20XX.csv      Job Bank wages by NOC 2021 (low, median, high; hourly unless flagged)
  ../esdc/oasis/*.csv              OaSIS 2025 v1.0: labels, lead statements, main duties, example titles,
                                   skills (0-5), knowledge (0-3), work activities (0-5)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW

STATCAN = RAW / "statcan"
ESDC = RAW / "esdc"

BROAD_NAMES = {"0": "Legislative and senior management", "1": "Business, finance and administration", "2": "Natural and applied sciences", "3": "Health", "4": "Education, law and social, community and government services",
               "5": "Art, culture, recreation and sport", "6": "Sales and service", "7": "Trades, transport and equipment operators", "8": "Natural resources, agriculture and related production", "9": "Manufacturing and utilities"}
# TEER: the second digit of a NOC 2021 code, the training, education, experience and responsibilities the job needs
TEER_NAMES = {0: "management", 1: "university degree", 2: "college diploma, apprenticeship of two years or more, or supervisory work", 3: "college diploma under two years, apprenticeship under two years, or over six months' training",
              4: "secondary school diploma or several weeks' training", 5: "short work demonstration, no formal education"}
NAICS_SECTORS = {"11": "Agriculture, forestry, fishing and hunting", "21": "Mining, quarrying, and oil and gas extraction", "22": "Utilities", "23": "Construction", "31": "Manufacturing", "41": "Wholesale trade", "44": "Retail trade",
                 "48": "Transportation and warehousing", "51": "Information and cultural industries", "52": "Finance and insurance", "53": "Real estate and rental and leasing", "54": "Professional, scientific and technical services",
                 "55": "Management of companies and enterprises", "56": "Administrative and support, waste management and remediation services", "61": "Educational services", "62": "Health care and social assistance",
                 "71": "Arts, entertainment and recreation", "72": "Accommodation and food services", "81": "Other services (except public administration)", "91": "Public administration"}


def _read_csv_any(path: Path) -> pd.DataFrame:
    """Statistics Canada's concordance files are Windows-1252; the newer ones UTF-8 with a BOM."""
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, encoding="latin-1", encoding_errors="replace")


def _sector(label: str) -> str | None:
    m = re.search(r"\[(\d{2})", str(label)) or re.match(r"^(\d{2})", str(label).strip())
    return m.group(1) if m and m.group(1) in NAICS_SECTORS else None


# ----------------------------------------------------------------------------- classification and concordances
def load_noc_structure() -> pd.DataFrame:
    """One row per NOC 2021 unit group (516) with its group names and TEER (the national skill level)."""
    s = pd.read_csv(STATCAN / "noc2021_structure.csv", dtype=str, encoding="utf-8-sig")
    s.columns = ["level", "structure", "code", "title", "definition"]
    names = dict(zip(s.code, s.title))
    u = s[s.level == "5"].copy()
    u = u.rename(columns={"code": "noc_code", "title": "noc_title"})
    u["major_group"] = u.noc_code.str[0]
    u["submajor"] = u.noc_code.str[:2]
    u["minor"] = u.noc_code.str[:4]
    u["submajor_name"] = u.submajor.map(names)
    u["minor_name"] = u.minor.map(names)
    u["major_group_name"] = u.major_group.map(BROAD_NAMES)
    u["teer"] = u.noc_code.str[1].astype(int)
    u["skill_level"] = u.teer.astype(float)
    return u[["noc_code", "noc_title", "major_group", "major_group_name", "submajor", "submajor_name", "minor", "minor_name", "teer", "skill_level", "definition"]].reset_index(drop=True)


def load_onet_to_noc(unit_codes: set[str] | None = None) -> pd.DataFrame:
    """O*NET-SOC 2019 -> NOC 2021 without ISCO: the O*NET code's SOC 2018 prefix -> NOC 2016 V1.3 (Statistics
    Canada concordance, equal split over the NOC codes a SOC code maps to) -> NOC 2021 V1.0 (equal split
    again where a 2016 code was divided). Columns: onet_soc_code, nat_code, weight (sums to 1 per O*NET code)."""
    from . import crosswalks as xw

    a = _read_csv_any(STATCAN / "soc2018us-noc2016v1_3-eng.csv")
    a.columns = ["soc_2018", "soc_title", "partial", "noc2016", "noc2016_title", "notes"]
    a = a.dropna(subset=["soc_2018", "noc2016"])
    a["soc_2018"] = a.soc_2018.str.strip(); a["noc2016"] = a.noc2016.str.strip().str.zfill(4)
    a = a.drop_duplicates(["soc_2018", "noc2016"])
    a["w1"] = 1.0 / a.groupby("soc_2018").noc2016.transform("count")
    b = _read_csv_any(STATCAN / "noc2016v1_3-noc2021v1_0-eng.csv")
    b = b.iloc[:, [0, 3]]; b.columns = ["noc2016", "noc2021"]
    b = b.dropna(); b["noc2016"] = b.noc2016.str.strip().str.zfill(4); b["noc2021"] = b.noc2021.str.strip().str.zfill(5)
    b = b.drop_duplicates()
    b["w2"] = 1.0 / b.groupby("noc2016").noc2021.transform("count")
    o = xw.onetsoc_to_soc2018()[["onet_soc_code", "soc_2018"]]
    m = o.merge(a[["soc_2018", "noc2016", "w1"]], on="soc_2018").merge(b, on="noc2016")
    if unit_codes:
        m = m[m.noc2021.isin(set(unit_codes))]
    m["w"] = m.w1 * m.w2
    g = m.groupby(["onet_soc_code", "noc2021"]).w.sum().reset_index().rename(columns={"noc2021": "nat_code"})
    g["weight"] = g.w / g.groupby("onet_soc_code").w.transform("sum")
    return g[["onet_soc_code", "nat_code", "weight"]]


# ----------------------------------------------------------------------------- the API pulls (Census and vacancies)
def _members(pid: int) -> list[dict]:
    return json.loads((STATCAN / "api" / f"members_{pid}.json").read_text())


def _coord_table(pid: int, csv_name: str) -> pd.DataFrame:
    d = pd.read_csv(STATCAN / "api" / csv_name, dtype=str)
    d["value"] = pd.to_numeric(d.value, errors="coerce")
    parts = d.coordinate.str.split(".", expand=True)
    for i in range(parts.shape[1]):
        d[f"m{i}"] = parts[i].astype(int)
    return d


def _code_from_name(name: str, digits: int) -> str | None:
    m = re.match(rf"^(\d{{{digits}}})\s", str(name))
    return m.group(1) if m else None


def load_census_employment() -> pd.DataFrame:
    """Employed persons (thousands) by NOC 2021 unit group, Census 2021, Canada."""
    d = _coord_table(98100449, "api_98100449_employed.csv")
    mem = {m["id"]: m["name"] for m in _members(98100449)[4]["members"]}
    d["name"] = d.m4.map(mem)
    d["noc_code"] = d.name.map(lambda s: _code_from_name(s, 5))
    out = d.dropna(subset=["noc_code"])[["noc_code", "value"]].rename(columns={"value": "employed"})
    out["employment_thousands"] = out.employed / 1000.0
    return out.reset_index(drop=True)


def load_census_income() -> pd.DataFrame:
    """Median and average employment income (2020, C$) by unit group: all workers and full-year full-time."""
    d = _coord_table(98100586, "api_98100586_income.csv")
    dims = _members(98100586)
    occ = {m["id"]: m["name"] for m in dims[6]["members"]}
    d["noc_code"] = d.m6.map(occ).map(lambda s: _code_from_name(s, 5))
    d = d.dropna(subset=["noc_code"])
    d["measure"] = np.where(d.m7 == 2, "median", "average")
    d["workers"] = np.where(d.m4 == 1, "all", "fyft")
    w = d.pivot_table(index="noc_code", columns=["workers", "measure"], values="value", aggfunc="first")
    w.columns = [f"income_{a}_{b}" for a, b in w.columns]
    return w.reset_index()


def load_industry_by_minor() -> pd.DataFrame:
    """Employed persons by NAICS sector (2-digit) and NOC 2021 minor group (4-digit), Census 2021."""
    d = _coord_table(98100594, "api_98100594_occ_industry.csv")
    dims = _members(98100594)
    ind = {m["id"]: m["name"] for m in dims[1]["members"]}
    occ = {m["id"]: m["name"] for m in dims[5]["members"]}
    d["industry"] = d.m1.map(ind).map(_sector)
    d["minor"] = d.m5.map(occ).map(lambda s: _code_from_name(s, 4))
    d = d.dropna(subset=["industry", "minor"])
    return d.groupby(["minor", "industry"]).value.sum().reset_index().rename(columns={"value": "employed"})


def load_vacancies() -> pd.DataFrame:
    """Job vacancies and the average offered hourly wage by unit group, Canada, the latest nine quarters.
    The vacancy table names occupations without codes; unit groups are matched by title to NOC 2021."""
    d = _coord_table(14100444, "api_14100444_vacancies.csv")
    mems = _members(14100444)[1]["members"]
    parents = {m["parent"] for m in mems if m["parent"]}
    leaf = {m["id"]: m["name"].strip() for m in mems if m["id"] not in parents}
    title_to_code = dict(zip(load_noc_structure().noc_title.str.strip(), load_noc_structure().noc_code))
    d["noc_code"] = d.m1.map(leaf).map(title_to_code)
    d = d.dropna(subset=["noc_code"])
    d["stat"] = np.where(d.m2 == 1, "vacancies", "offered_wage")
    d["quarter"] = pd.to_datetime(d.refPer)
    return d[["noc_code", "stat", "quarter", "value"]].reset_index(drop=True)


# ----------------------------------------------------------------------------- pay, weights, outcomes
def load_jobbank_wages(year: int = 2025) -> pd.DataFrame:
    """Job Bank national wages by NOC 2021: median hourly wage (annual where flagged), the source survey."""
    w = pd.read_csv(ESDC / "wages" / f"wages{year}.csv", encoding="utf-8-sig", dtype=str)
    n = w[w.prov == "NAT"].copy()
    n["noc_code"] = n.NOC_CNP.str.replace("NOC_", "", regex=False).str.zfill(5)
    for c in ["Low_Wage_Salaire_Minium", "Median_Wage_Salaire_Median", "High_Wage_Salaire_Maximal"]:
        n[c] = pd.to_numeric(n[c], errors="coerce")
    n["annual"] = n.Annual_Wage_Flag_Salaire_annuel.astype(str) == "1"
    out = n[["noc_code", "NOC_Title_eng", "Median_Wage_Salaire_Median", "Low_Wage_Salaire_Minium", "High_Wage_Salaire_Maximal", "annual", "Data_Source_E", "Reference_Period"]].copy()
    out.columns = ["noc_code", "title", "median_wage", "low_wage", "high_wage", "annual", "source", "reference_period"]
    out["year"] = year
    return out.reset_index(drop=True)


def build_noc_weights() -> pd.DataFrame:
    """Per NOC 2021 unit group: Census 2021 employment (thousands), median employment income of full-year
    full-time workers (2020 C$; all workers, then group medians where suppressed), their product as the
    wage weight, TEER as the skill level, and the group names."""
    s = load_noc_structure()
    emp = load_census_employment().set_index("noc_code").employment_thousands
    inc = load_census_income().set_index("noc_code")
    df = s[["noc_code", "noc_title", "major_group", "submajor", "submajor_name", "minor", "teer", "skill_level"]].copy().rename(columns={"submajor": "submajor_label"})
    df["employment_thousands"] = df.noc_code.map(emp).fillna(0.0)
    df["wage_median"] = df.noc_code.map(inc.get("income_fyft_median"))
    df["wage_source"] = np.where(df.wage_median.notna(), "census_2021_income_full_year_full_time", None)
    m = df.wage_median.isna()
    df.loc[m, "wage_median"] = df.loc[m, "noc_code"].map(inc.get("income_all_median"))
    df.loc[m & df.wage_median.notna(), "wage_source"] = "census_2021_income_all_workers"
    pub = df.dropna(subset=["wage_median"])
    for lvl, n in [("minor", 4), ("submajor_label", 2), ("major_group", 1)]:
        grp = pub.groupby(pub.noc_code.str[:n]).wage_median.median()
        m = df.wage_median.isna()
        df.loc[m, "wage_median"] = df.loc[m, lvl].map(grp)
        df.loc[m & df.wage_median.notna(), "wage_source"] = f"{lvl.replace('_label', '')}_group_median"
    df["wage_value"] = df.employment_thousands * df.wage_median
    df["employment_method"] = "census_2021_employed_by_unit_group"
    df["edition"] = 2021
    return df.drop(columns=["minor"]).reset_index(drop=True)


def employment_change() -> pd.DataFrame:
    """Rung five for Canada: job vacancies in the latest four quarters against the four before, per unit
    group (Job Vacancy and Wage Survey), and the change in the Job Bank median wage between the 2024 and
    2025 releases. The vacancy series stands in the 'emp' columns the outcome test expects."""
    v = load_vacancies()
    vac = v[v.stat == "vacancies"].sort_values("quarter")
    quarters = sorted(vac.quarter.unique())
    last4, prev4 = quarters[-4:], quarters[-8:-4]
    yt, yf = int(pd.Timestamp(last4[-1]).year), int(pd.Timestamp(prev4[-1]).year)
    a = vac[vac.quarter.isin(prev4)].groupby("noc_code").value.sum()
    b = vac[vac.quarter.isin(last4)].groupby("noc_code").value.sum()
    df = pd.DataFrame({f"emp_{yf}": a, f"emp_{yt}": b}).reset_index()
    df["emp_change_pct"] = 100.0 * (df[f"emp_{yt}"] / df[f"emp_{yf}"].where(df[f"emp_{yf}"] > 0) - 1.0)
    w0 = load_jobbank_wages(2024).set_index("noc_code"); w1 = load_jobbank_wages(2025).set_index("noc_code")
    same_unit = (w0.annual.reindex(df.noc_code).values == w1.annual.reindex(df.noc_code).values)
    df["wage_2024"] = df.noc_code.map(w0.median_wage); df["wage_2025"] = df.noc_code.map(w1.median_wage)
    df.loc[~same_unit, ["wage_2024", "wage_2025"]] = np.nan
    df["wage_change_pct"] = 100.0 * (df.wage_2025 / df.wage_2024.where(df.wage_2024 > 0) - 1.0)
    df["year_from"], df["year_to"] = yf, yt
    df["wage_years"] = "2024-2025"
    df["vacancy_quarters_to"] = str(pd.Timestamp(last4[-1]).date())
    df["emp_unit"] = "vacancies"
    df["wage_unit"] = df.noc_code.map(w1.annual.map({True: "year", False: "hour"}))
    return df


def outcome_units() -> pd.DataFrame:
    """Units for the page's outcome cells: the 'employment' figures are job vacancies (four-quarter sums,
    counts rather than thousands) and Job Bank wages are hourly unless the release flags them annual."""
    w = load_jobbank_wages(2025)
    return pd.DataFrame({"noc_code": w.noc_code.values, "emp_unit": "vacancies", "wage_unit": np.where(w.annual.values, "year", "hour")})


# ----------------------------------------------------------------------------- firm AI use and the industry mix
def load_csbc_ai_use() -> pd.DataFrame:
    """Share of businesses that used AI in producing goods or delivering services over the last 12 months,
    second quarters of 2024, 2025 and 2026, nationally and by NAICS sector and size (Canadian Survey on
    Business Conditions). Dated at the end of each second quarter."""
    frames = []
    for pid in (33100825, 33101004, 33101167):
        p = STATCAN / f"{pid}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, dtype=str)
        col = next(c for c in d.columns if c.startswith("Use of artificial"))
        d = d[(d.GEO == "Canada") & d[col].str.startswith("Yes, business used AI")]
        year = int(d.REF_DATE.iloc[0])
        for _, r in d.iterrows():
            bc = str(r["Business characteristics"]); pct = pd.to_numeric(r.VALUE, errors="coerce")
            if bc.startswith("North American Industry Classification System (NAICS), all industries"):
                frames.append({"period": f"{year} Q2", "date": year + 0.5, "group_type": "all", "industry": "ALL", "label": "All businesses", "pct": pct})
            elif _sector(bc):
                frames.append({"period": f"{year} Q2", "date": year + 0.5, "group_type": "industry", "industry": _sector(bc), "label": NAICS_SECTORS[_sector(bc)], "pct": pct})
            elif "employees" in bc and bc[0].isdigit():
                frames.append({"period": f"{year} Q2", "date": year + 0.5, "group_type": "size", "industry": bc, "label": bc, "pct": pct})
    if not frames:
        raise FileNotFoundError("Canadian Survey on Business Conditions AI tables (data/raw/statcan/3310*.csv)")
    return pd.DataFrame(frames)


def load_industry_mix() -> pd.DataFrame:
    """Employed persons by NOC minor group (4-digit) and NAICS sector, for the sector shifts."""
    return load_industry_by_minor()


# ----------------------------------------------------------------------------- OaSIS: the Canadian O*NET
def _oasis(name: str, sep: str = ";") -> pd.DataFrame:
    d = pd.read_csv(ESDC / "oasis" / f"{name}.csv", sep=sep, encoding="utf-8-sig", dtype=str)
    d.columns = [c.strip() for c in d.columns]
    d = d.rename(columns={d.columns[0]: "profile"})
    d["noc_code"] = d.profile.str.strip().str[:5]
    return d


def load_oasis_duties() -> pd.DataFrame:
    """OaSIS main duties: one row per duty per unit group (profiles that split a unit group are pooled)."""
    d = _oasis("main_duties")
    d = d.rename(columns={d.columns[1]: "task"})
    d["task"] = d.task.astype(str).str.strip()
    d = d[d.task.str.len() > 3].drop_duplicates(["noc_code", "task"])
    d["task_seq"] = d.groupby("noc_code").cumcount()
    return d[["noc_code", "task_seq", "task", "profile"]].reset_index(drop=True)


def load_oasis_descriptions() -> pd.DataFrame:
    """Lead statement per unit group (the first profile's), and the TEER wording."""
    d = _oasis("lead_statement", sep=",")
    d = d.rename(columns={d.columns[1]: "description"}).drop_duplicates("noc_code")
    s = load_noc_structure().set_index("noc_code")
    d["skill_text"] = d.noc_code.map(lambda c: f"TEER {s.teer.get(c, '')}: {TEER_NAMES.get(int(s.teer.get(c, 0)), '')}" if c in s.index else "")
    return d[["noc_code", "description", "skill_text"]].reset_index(drop=True)


def load_oasis_titles() -> pd.DataFrame:
    """Example and index job titles per unit group (for the page's search)."""
    d = _oasis("example_titles", sep=",")
    d = d.rename(columns={"Job title text": "title", "Job title type": "title_type"})
    d["title"] = d.title.astype(str).str.strip()
    return d[["noc_code", "title", "title_type"]].drop_duplicates(["noc_code", "title"]).reset_index(drop=True)


def load_oasis_ratings() -> pd.DataFrame:
    """Long table of OaSIS ratings per unit group: skills (0-5 proficiency), work activities (0-5) and
    knowledge (0-3), averaged where a unit group has several profiles. Columns: noc_code, family,
    element_name, level."""
    frames = []
    for name, fam in [("skills", "skill"), ("work_activities", "work_activity"), ("knowledge", "knowledge")]:
        d = _oasis(name)
        val_cols = [c for c in d.columns[2:] if c not in ("noc_code",)]
        long = d.melt(id_vars=["noc_code"], value_vars=val_cols, var_name="element_name", value_name="level")
        long["level"] = pd.to_numeric(long.level.astype(str).str.strip().str.replace("\xa0", ""), errors="coerce")
        long["family"] = fam
        frames.append(long.dropna(subset=["level"]))
    out = pd.concat(frames, ignore_index=True)
    return out.groupby(["noc_code", "family", "element_name"])["level"].mean().reset_index()
