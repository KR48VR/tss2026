"""Germany (full depth): employees and pay by five-digit KldB 2010 occupation, the ICT-use survey's AI readings by
industry, and the official KldB-to-ISCO key.

Files:
  ba/beschaeftigte_berufe_<yyyymm>.xlsx   Federal Employment Agency (Bundesagentur für Arbeit), "Beschäftigte nach
                                          Berufen (KldB 2010)", Germany, reference date 30 June: employees subject to
                                          social insurance by four-digit occupation (sheet "SVB - Tabelle II"), with the
                                          four requirement levels (Helfer, Fachkraft, Spezialist, Experte) as columns.
                                          KldB 2010's fifth digit is that requirement level, so the four columns are the
                                          1,300-odd five-digit occupations (Berufsgattungen). Two editions: June 2025 (the
                                          weights) and June 2024 (the employment change)
  ba/kldb2010_isco08_key.xlsx             the agency's conversion key from the five-digit KldB 2010 (revised 2020) to
                                          ISCO-08 unit groups, with English titles, a uniqueness flag and the main target
                                          ("Schwerpunkt") where a code has alternatives
  destatis/62361-0030_<year>.csv          GENESIS-Online (fetch.fetch_genesis, with a registered account): the earnings
                                          survey's median and average gross monthly and hourly earnings by five-digit
                                          KldB occupation, both sexes, April reference month, 2022 to 2025
  destatis/52911-0001.csv, 52911-0005.csv the ICT-use survey's indicators for enterprises, Germany, 2015 to 2024 and
                                          from 2025 (the questionnaire changed, so GENESIS keeps two tables)
  destatis/52911-0002.csv, 52911-0006.csv the same by WZ 2008 economic section
  destatis/12211-0009.csv                 the Microcensus: persons in employment by two-digit occupation (a check on the
                                          employment statistics, which cover employees only)

What is coarse: employment counts employees subject to social insurance (34.9 million in June 2025), not civil
servants, the self-employed or marginal jobs; pay is the median gross monthly earnings of all employees in the
occupation (full- and part-time, excluding special payments); the occupation-by-industry mix is not published for
KldB groups, so each two-digit occupation group's mix is carried from the EU Labour Force Survey's ISCO major
group by NACE section table for Germany through the ISCO key; no official task list.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import RAW

BA = RAW / "ba"
DESTATIS = RAW / "destatis"
SURVEY_DATE_IN_YEAR = 0.4   # the ICT-use survey is run in spring, as Eurostat's is

MAJOR_GROUP_NAMES = {"0": "Military occupations", "1": "Agriculture, forestry, farming and gardening", "2": "Production and manufacturing",
                     "3": "Construction, architecture, surveying and building services", "4": "Natural sciences, geography and informatics",
                     "5": "Transport, logistics, safety and security", "6": "Commercial services, trading, sales, hotels and tourism",
                     "7": "Business organisation, accounting, law and administration", "8": "Health, social work, teaching and education",
                     "9": "Humanities, social sciences, economics, media, arts and culture"}
SUBMAJOR_NAMES = {"01": "Armed forces", "11": "Agriculture, animal husbandry and forestry", "12": "Horticulture and floristry", "21": "Raw materials extraction, glass and ceramics", "22": "Plastics and wood making and processing",
                  "23": "Paper and printing, technical media design", "24": "Metal-making and -working, metal construction", "25": "Machine and vehicle technology", "26": "Mechatronics, energy and electrical occupations",
                  "27": "Technical research, development, design and production control", "28": "Textile and leather occupations", "29": "Food production and processing", "31": "Construction planning, architecture and surveying",
                  "32": "Building construction and civil engineering", "33": "Interior construction", "34": "Building services engineering", "41": "Mathematics, biology, chemistry and physics", "42": "Geology, geography and environmental protection",
                  "43": "Computer science and other ICT occupations", "51": "Transport and logistics (except driving)", "52": "Drivers and operators of vehicles and transport equipment", "53": "Safety, security and surveillance",
                  "54": "Cleaning occupations", "61": "Purchasing, sales and trading", "62": "Retail sales", "63": "Tourism, hotels and restaurants", "71": "Business management and organisation",
                  "72": "Financial services, accounting and tax consultancy", "73": "Law and public administration", "81": "Medical and health care", "82": "Non-medical health care, body care and medical technology",
                  "83": "Education, social work, housekeeping and theology", "84": "Teaching and training", "91": "Humanities, social sciences and economics", "92": "Advertising, marketing, commercial and editorial media",
                  "93": "Product design, artisan craftwork and fine arts", "94": "Performing arts and entertainment"}
# the fifth digit of a KldB 2010 code is its requirement level (Anforderungsniveau)
SKILL_LEVEL_NAMES = {1: "helper and trainee activities (no or short vocational training)", 2: "skilled activities (two to three years' vocational training)",
                     3: "specialist activities (master craftsman, technician or bachelor level)", 4: "highly complex activities (four or more years of university study)"}
LEVEL_NAMES_DE = {"1": "Helfer", "2": "Fachkraft", "3": "Spezialist", "4": "Experte"}
# the ICT-use survey's economic sections (WZ 2008), keyed to NACE sections for the occupation mix
_WZ_TO_SECTIONS = {"WZ08-C": ["C"], "WZ08-D-05": ["D", "E"], "WZ08-F": ["F"], "WZ08-G": ["G"], "WZ08-H-03": ["H"], "WZ08-I": ["I"], "WZ08-J": ["J"], "WZ08-K": ["K"], "WZ08-L": ["L"], "WZ08-M": ["M"], "WZ08-N": ["N"]}


# ----------------------------------------------------------------------------- employment (Bundesagentur für Arbeit)
def _editions() -> list[str]:
    return sorted(m.group(1) for p in BA.glob("beschaeftigte_berufe_*.xlsx") for m in [re.search(r"_(\d{6})\.xlsx$", p.name)] if m)


EDITIONS = _editions()
EDITION = EDITIONS[-1] if EDITIONS else "202506"


def _num(v) -> float:
    s = str(v).strip() if v is not None else ""
    if s in ("", "nan", "None", "-", "x", ".", "*", "()"):
        return np.nan
    try:
        return float(s.replace(".", "").replace(",", ".")) if re.fullmatch(r"\d{1,3}(\.\d{3})+", s) else float(s.replace(",", "."))
    except ValueError:
        return np.nan


def load_employment(edition: str = EDITION) -> pd.DataFrame:
    """Employees subject to social insurance by five-digit KldB occupation on 30 June of the edition's year:
    kldb_code, kldb4_code, title_de (the four-digit title), level, employees. A cell the agency suppresses ('*')
    takes an equal share of what the four-digit row's total leaves after the published levels."""
    t = pd.read_excel(BA / f"beschaeftigte_berufe_{edition}.xlsx", sheet_name="SVB - Tabelle II", header=None, dtype=str)
    rows = []
    for r in t.itertuples(index=False):
        m = re.match(r"^(\d{4}) (.+)$", str(r[0]))
        if not m:
            continue
        total = _num(r[1])
        levels = {lvl: _num(r[4 + i]) for i, lvl in enumerate("1234")}
        known = sum(v for v in levels.values() if not np.isnan(v))
        missing = [lvl for lvl, v in levels.items() if np.isnan(v) and str(r[4 + "1234".index(lvl)]).strip() == "*"]
        for lvl, v in levels.items():
            if np.isnan(v):
                v = max(total - known, 0.0) / len(missing) if (lvl in missing and not np.isnan(total)) else 0.0
            if v > 0:
                rows.append({"kldb_code": m.group(1) + lvl, "kldb4_code": m.group(1), "title_de": m.group(2).strip(), "level": int(lvl), "employees": float(v)})
    return pd.DataFrame(rows)


def load_group_titles_de(edition: str = EDITION) -> dict:
    """German titles of the two- and three-digit groups from the report's first table."""
    t = pd.read_excel(BA / f"beschaeftigte_berufe_{edition}.xlsx", sheet_name="SVB - Tabelle I", header=None, dtype=str)
    out = {}
    for r in t.itertuples(index=False):
        m = re.match(r"^(\d{2,3}) (.+)$", str(r[0]))
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


# ----------------------------------------------------------------------------- the ISCO key (Bundesagentur für Arbeit)
def load_key() -> pd.DataFrame:
    """The agency's conversion key: kldb_code (five digits), title_de, title_en, isco_code, unique (1/0), main (1 for
    the main target of a non-unique code, else the alternative's number), and a share per KldB code: a unique code
    gives its whole weight to its target; otherwise the main target counts twice and each alternative once."""
    d = pd.read_excel(BA / "kldb2010_isco08_key.xlsx", sheet_name=1, header=None, dtype=str)
    d = d[d[0].astype(str).str.fullmatch(r"\d{5}") & d[3].astype(str).str.fullmatch(r"\d{4}")].copy()
    k = pd.DataFrame({"kldb_code": d[0].str.strip(), "title_de": d[1].str.strip(), "title_en": d[2].str.strip(), "isco_code": d[3].str.strip(), "unique": pd.to_numeric(d[6], errors="coerce").fillna(1).astype(int), "main": pd.to_numeric(d[7], errors="coerce").fillna(1).astype(int)})
    k["w"] = np.where(k.unique == 1, 1.0, np.where(k.main == 1, 2.0, 1.0))
    k["share"] = k.w / k.groupby("kldb_code").w.transform("sum")
    return k.drop(columns="w").drop_duplicates(["kldb_code", "isco_code"]).reset_index(drop=True)


def load_isco_link(unit_codes: set[str] | None = None) -> pd.DataFrame:
    """ISCO-08 unit group -> five-digit KldB codes: each KldB code's share of its ISCO targets (the key) times its
    employees, normalised per ISCO code, so an ISCO group named by several KldB codes is split by their size."""
    k = load_key()
    emp = load_employment().set_index("kldb_code").employees
    k = k[k.kldb_code.isin(unit_codes)] if unit_codes else k
    k["w"] = k.share * k.kldb_code.map(emp).fillna(1.0).clip(lower=1.0)
    k["weight"] = k.w / k.groupby("isco_code").w.transform("sum")
    return k.rename(columns={"kldb_code": "nat_code"})[["isco_code", "nat_code", "weight"]].reset_index(drop=True)


# ----------------------------------------------------------------------------- pay (GENESIS 62361-0030)
def _earnings_years() -> list[int]:
    return sorted(int(m.group(1)) for p in DESTATIS.glob("62361-0030_*.csv") for m in [re.search(r"_(\d{4})\.csv$", p.name)] if m)


def load_earnings(year: int) -> pd.DataFrame:
    """The earnings survey's both-sexes rows for the April reference month of `year`: code (the KB10 code without
    its prefix: five, three or two digits), title_en, median_monthly, mean_monthly, median_hourly (euro; NaN where
    GENESIS suppresses the cell)."""
    d = pd.read_csv(DESTATIS / f"62361-0030_{year}.csv", sep=";", dtype=str)
    d = d[d["2_variable_attribute_label"].str.strip() == "Total"].copy()
    d["code"] = d["3_variable_attribute_code"].str.replace("KB10-", "", regex=False).str.strip()
    d["v"] = pd.to_numeric(d.value.str.strip().str.replace(",", "."), errors="coerce")
    p = d.pivot_table(index=["code", "3_variable_attribute_label"], columns="value_variable_code", values="v", aggfunc="first").reset_index()
    p = p.rename(columns={"3_variable_attribute_label": "title_en", "VST052": "median_monthly", "VST047": "mean_monthly", "VST051": "median_hourly", "VST045": "mean_hourly"})
    for c in ("median_monthly", "mean_monthly", "median_hourly", "mean_hourly"):
        if c not in p.columns:
            p[c] = np.nan
    p["year"] = year
    return p[["code", "title_en", "year", "median_monthly", "mean_monthly", "median_hourly", "mean_hourly"]]


def _fill_pay(df: pd.DataFrame, earn: pd.DataFrame) -> pd.DataFrame:
    """Median monthly earnings per five-digit code, else the three-digit group's published median, else the
    two-digit group's, else the national median over all employees."""
    e = earn.set_index("code").median_monthly
    df = df.copy()
    df["wage_monthly"] = df.kldb_code.map(e)
    df["wage_source"] = np.where(df.wage_monthly.notna(), "earnings_survey_median_monthly_5digit", "")
    for n, label in ((3, "earnings_survey_median_monthly_3digit_group"), (2, "earnings_survey_median_monthly_2digit_group")):
        miss = df.wage_monthly.isna()
        df.loc[miss, "wage_monthly"] = df.loc[miss, "kldb_code"].str[:n].map(e)
        df.loc[miss & df.wage_monthly.notna(), "wage_source"] = label
    miss = df.wage_monthly.isna()
    nat = earn[earn.code.str.len() == 5].median_monthly.median()
    df.loc[miss, "wage_monthly"] = nat
    df.loc[miss, "wage_source"] = "earnings_survey_median_over_occupations"
    return df


def build_weights(edition: str = EDITION, year: int | None = None) -> pd.DataFrame:
    """Per five-digit KldB occupation: employees (thousands, June of the edition), median gross monthly earnings
    (April of the latest survey year), their product as the wage weight, English and German titles, the KldB
    area (first digit) and main group (two digits), and the requirement level (fifth digit) as the skill level."""
    emp = load_employment(edition)
    years = _earnings_years()
    year = year or (years[-1] if years else None)
    if year is None:
        raise FileNotFoundError(DESTATIS / "62361-0030_<year>.csv")
    earn = load_earnings(year)
    key = load_key().drop_duplicates("kldb_code").set_index("kldb_code")
    df = pd.DataFrame({"kldb_code": emp.kldb_code.values, "employment_thousands": emp.employees.values / 1000.0, "title_de4": emp.title_de.values, "level": emp.level.values})
    en = df.kldb_code.map(key.title_en)
    en5 = df.kldb_code.map(earn[earn.code.str.len() == 5].set_index("code").title_en)
    df["kldb_title"] = en.fillna(en5).fillna(df.title_de4 + " - " + df.level.astype(str).map(LEVEL_NAMES_DE))
    df["kldb_title_de"] = df.kldb_code.map(key.title_de).fillna(df.title_de4 + " - " + df.level.astype(str).map(LEVEL_NAMES_DE))
    df = _fill_pay(df, earn)
    df["major_group"] = df.kldb_code.str[0]
    df["submajor_label"] = df.kldb_code.str[:2]
    df["submajor_name"] = df.submajor_label.map(SUBMAJOR_NAMES)
    df["skill_level"] = df.level.astype(float)
    df["wage_value"] = df.employment_thousands * df.wage_monthly
    df["employment_method"] = f"employment_statistics_{edition}"
    df["edition"] = int(edition[:4])
    df["pay_year"] = year
    return df[df.employment_thousands > 0][["kldb_code", "kldb_title", "kldb_title_de", "employment_thousands", "major_group", "submajor_label", "submajor_name", "skill_level", "wage_monthly", "wage_source", "wage_value", "employment_method", "edition", "pay_year"]].reset_index(drop=True)


# ----------------------------------------------------------------------------- the firm survey (GENESIS 52911)
def _ict(name: str) -> pd.DataFrame:
    d = pd.read_csv(DESTATIS / f"{name}.csv", sep=";", dtype=str)
    d["v"] = pd.to_numeric(d.value.str.strip().str.replace(",", "."), errors="coerce")
    d["year"] = pd.to_numeric(d.time.str[:4], errors="coerce").astype("Int64")
    return d


def load_ai_points() -> pd.DataFrame:
    """Share of enterprises (10 or more persons employed) using at least one AI technology, nationally and by
    economic section: 2021, 2023 and 2024 from the tables up to 2024 (indicator IKT-0205) and 2025 from the
    tables of the revised questionnaire (IKT25-0071). A year without the question ('-') is skipped; a survey
    year is dated at its spring."""
    rows = []
    for name, dim in (("52911-0001", 2), ("52911-0005", 2)):
        d = _ict(name)
        a = d[d[f"{dim}_variable_attribute_label"].str.contains(r"^Artificial intelligence \(AI\)$", na=False, regex=True)].dropna(subset=["v"])
        rows += [{"date": int(r.year) + SURVEY_DATE_IN_YEAR, "period": str(int(r.year)), "pct": float(r.v), "group_type": "all", "industry": None, "label": "Enterprises with 10 or more persons employed"} for r in a.itertuples()]
    for name in ("52911-0002", "52911-0006"):
        d = _ict(name)
        a = d[d["3_variable_attribute_label"].str.contains(r"^Artificial intelligence \(AI\)$", na=False, regex=True)].dropna(subset=["v"])
        for wz, yr, v in zip(a["2_variable_attribute_code"], a.year, a.v):
            for sec in _WZ_TO_SECTIONS.get(str(wz).strip(), []):
                rows.append({"date": int(yr) + SURVEY_DATE_IN_YEAR, "period": str(int(yr)), "pct": float(v), "group_type": "industry", "industry": sec, "label": _section_name(sec)})
    out = pd.DataFrame(rows, columns=["date", "period", "pct", "group_type", "industry", "label"]).drop_duplicates(["group_type", "industry", "date"])
    return out.sort_values(["group_type", "industry", "date"], na_position="first").reset_index(drop=True)


def _section_name(sec: str) -> str:
    from . import europe as eu
    return eu.NACE_SECTIONS.get(sec, sec)


def load_industry_mix() -> pd.DataFrame:
    """Employees by two-digit KldB group and NACE section, derived: the group's ISCO-major composition (its codes'
    key targets, weighted by employees) times the EU Labour Force Survey's section distribution of each ISCO major
    group for Germany (Eurostat lfsa_eisn2, 2024). Columns: group2, industry, employed."""
    from . import europe as eu
    w = load_employment()[["kldb_code", "employees"]]
    k = load_key()
    m = w.merge(k[["kldb_code", "isco_code", "share"]], on="kldb_code")
    m["major"] = m.isco_code.str[0]
    m["group2"] = m.kldb_code.str[:2]
    m["e"] = m.employees * m.share
    comp = m.groupby(["group2", "major"]).e.sum().reset_index()
    lfs = eu.load_mix(); lfs = lfs[lfs.geo == "DE"].drop(columns="geo")
    lfs["sh"] = lfs.employed / lfs.groupby("major_group").employed.transform("sum")
    x = comp.merge(lfs.rename(columns={"major_group": "major"})[["major", "industry", "sh"]], on="major", how="inner")
    x["employed"] = x.e * x.sh
    return x.groupby(["group2", "industry"]).employed.sum().reset_index()


# ----------------------------------------------------------------------------- rung five, units, titles
def employment_change() -> pd.DataFrame:
    """Per five-digit code: employees in the latest June against the June a year earlier (two editions of the
    employment statistics) and the occupation's own median monthly earnings in the latest April against the one
    before (group-level stand-ins carry no change of their own)."""
    eds = EDITIONS[-2:] if len(EDITIONS) >= 2 else EDITIONS
    e1 = load_employment(eds[-1]).set_index("kldb_code").employees
    e0 = load_employment(eds[0]).set_index("kldb_code").employees if len(eds) == 2 else pd.Series(dtype=float)
    years = _earnings_years()
    y1 = years[-1]; y0 = years[-2] if len(years) >= 2 else None
    p1 = load_earnings(y1); p1 = p1[p1.code.str.len() == 5].set_index("code").median_monthly
    p0 = load_earnings(y0) if y0 else None
    p0 = p0[p0.code.str.len() == 5].set_index("code").median_monthly if p0 is not None else pd.Series(dtype=float)
    codes = sorted(e1.index)
    df = pd.DataFrame({"kldb_code": codes})
    ey0, ey1 = (eds[0][:4] if len(eds) == 2 else str(int(eds[-1][:4]) - 1)), eds[-1][:4]
    df[f"emp_{ey0}"] = df.kldb_code.map(e0); df[f"emp_{ey1}"] = df.kldb_code.map(e1)
    df["emp_change_pct"] = 100.0 * (df[f"emp_{ey1}"] / df[f"emp_{ey0}"].where(df[f"emp_{ey0}"] > 0) - 1.0)
    df[f"wage_{y0 or y1 - 1}"] = df.kldb_code.map(p0); df[f"wage_{y1}"] = df.kldb_code.map(p1)
    df["wage_change_pct"] = 100.0 * (df[f"wage_{y1}"] / df[f"wage_{y0 or y1 - 1}"].where(df[f"wage_{y0 or y1 - 1}"] > 0) - 1.0)
    df["year_from"], df["year_to"], df["wage_years"] = int(ey0), int(ey1), f"{y0 or y1 - 1}-{y1}"
    return df


def outcome_units() -> pd.DataFrame:
    w = load_employment()
    return pd.DataFrame({"kldb_code": w.kldb_code.values, "emp_unit": "people", "wage_unit": "month"})


def load_titles() -> pd.DataFrame:
    """English titles as principal, German titles (from the key) as alternative titles for the search."""
    w = build_weights()
    return pd.concat([pd.DataFrame({"kldb_code": w.kldb_code, "title": w.kldb_title, "category": "principal"}),
                      pd.DataFrame({"kldb_code": w.kldb_code, "title": w.kldb_title_de, "category": "german"})], ignore_index=True).drop_duplicates(["kldb_code", "title"]).reset_index(drop=True)


def no_tasks() -> pd.DataFrame:
    return pd.DataFrame(columns=["kldb_code", "task_seq", "task"])
