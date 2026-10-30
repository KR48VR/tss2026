"""South Korea (medium depth): employment and pay by three-digit KSCO occupation from KOSIS website downloads
(no API key: the KOSIS sign-up needs a Korean mobile phone), the Survey of Business Activities' question on
fourth-industrial-revolution technologies for firm AI use, and Statistics Korea's official links from the Korean
Standard Classification of Occupations (8th revision, 2024) to ISCO-08 and to the 7th revision.

Files (downloaded by fetch.fetch_kosis through the table pages' own download panel; one folder per table):
  kosis/DT_1ES3B32J/DT_1ES3B32J.csv   Local Area Labour Force Survey: employed persons by three-digit occupation of the 8th
                                       revision (163 codes, thousand persons, both sexes), half-yearly 2024 1/2 to 2026 1/2
  kosis/DT_1ES3B32J/ksco8_titles.json  the table's classification tree with Korean and English titles per code
  kosis/DT_1ES3B33J/DT_1ES3B33J.csv   the same survey: employed by industry section (KSIC) and occupation major group, 2026 1/2
  kosis/DT_118N_PAYM41/DT_118N_PAYM41.csv  Ministry of Employment and Labour, Survey on Labour Conditions by Employment Type:
                                       workers, monthly total wages (thousand won) and hours by three-digit occupation of the
                                       7th revision and sex, 2024 and 2025
  kosis/DT_1EP1237_1/DT_1EP1237_1.csv  Statistics Korea, Survey of Business Activities: enterprises developing or using each
                                       fourth-industrial-revolution technology, by industry section and division, 2017 to 2022
  kosis/DT_1EP1337_1/DT_1EP1337_1.csv  the same for 2023 and 2024 (a new table id after the questionnaire changed)
  kostat/ksco8_isco08_link.xlsx, kostat/ksco8_ksco7_link.xlsx  Statistics Korea's link tables (kssc.kostat.go.kr), from which
  reference/ksco8_isco08_link.csv      8th-revision three-digit code -> ISCO-08 unit groups (the union over its four-digit codes)
  reference/ksco8_ksco7_link.csv       8th-revision three-digit code -> the 7th-revision three-digit codes feeding it, with link counts

What is coarse: occupations are three-digit minor groups (163) rather than unit groups; pay comes from a survey of
employees on the older classification (converted through the official 7th-to-8th link, employment-weighted) while
the labour force survey counts the self-employed too; a code without a converted pay line takes its two-digit
group's employment-weighted mean; the firm survey covers enterprises with 50 or more regular employees and capital
of 300 million won or more, and asks which technologies the enterprise develops or uses, so its reading is read
against US firms of the same sizes and shown as an upper bound; no official task list, so job pages carry O*NET's
tasks through the ISCO link.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW

KOSIS = RAW / "kosis"
REFERENCE = RAW.parent / "reference"
EMPLOYMENT_TBL = "DT_1ES3B32J"
MIX_TBL = "DT_1ES3B33J"
PAY_TBL = "DT_118N_PAYM41"
AI_TBLS = ("DT_1EP1237_1", "DT_1EP1337_1")
LINK = REFERENCE / "ksco8_isco08_link.csv"
LINK_78 = REFERENCE / "ksco8_ksco7_link.csv"
TITLES = KOSIS / EMPLOYMENT_TBL / "ksco8_titles.json"

MAJOR_GROUP_NAMES = {"1": "Managers", "2": "Professionals and related workers", "3": "Clerks", "4": "Service workers", "5": "Sales workers",
                     "6": "Skilled agricultural, forestry and fishery workers", "7": "Craft and related trades workers", "8": "Equipment, machine operating and assembling workers", "9": "Elementary workers"}
# KSIC's sections carry the same letters as ISIC Rev. 4 and NACE Rev. 2, so the record names them with europe.NACE_SECTIONS (set in countries.py)


def _read(tbl: str) -> pd.DataFrame:
    """A KOSIS CSV download with codes: row 0 holds the item names, the column names hold the periods."""
    path = KOSIS / tbl / f"{tbl}.csv"
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            text = raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    from io import StringIO
    return pd.read_csv(StringIO(text), dtype=str)


def _tidy(tbl: str, n_dims: int) -> pd.DataFrame:
    """Long form: one row per (dimension values..., period, item, value)."""
    df = _read(tbl)
    items = df.iloc[0].tolist()[n_dims:]
    cols = list(df.columns)[n_dims:]
    periods = [re.match(r"[A-Z]?(\d+)", c).group(1) for c in cols]
    body = df.iloc[1:].copy()
    dims = [f"d{i}" for i in range(n_dims)]
    body.columns = dims + [f"{p}|{it}" for p, it in zip(periods, items)]
    long = body.melt(id_vars=dims, var_name="pi", value_name="value")
    long[["period", "item"]] = long.pi.str.split("|", n=1, expand=True)
    long = long.drop(columns="pi")
    long["value"] = pd.to_numeric(long.value.str.replace(",", "", regex=False), errors="coerce")
    return long


def _period_year(p: str) -> float:
    """'202601' (first half of 2026) -> 2026.25; '2025' -> 2025.0."""
    if len(p) == 6:
        return int(p[:4]) + (0.25 if p[4:] == "01" else 0.75)
    return float(p)


# ----------------------------------------------------------------------------- employment, pay, weights
def load_employment() -> pd.DataFrame:
    """Employed persons (thousands) per three-digit KSCO code and half-year period, both sexes."""
    d = _tidy(EMPLOYMENT_TBL, 1)
    d = d[d.item.str.startswith("T00 ")]
    d["ksco_code"] = d.d0.str.extract(r"^(\d{3}) ")[0]
    d["ksco_title_ko"] = d.d0.str.replace(r"^\d{3} ", "", regex=True).str.strip()
    d = d.dropna(subset=["ksco_code"])
    return d[["ksco_code", "ksco_title_ko", "period", "value"]].rename(columns={"value": "employment_thousands"}).reset_index(drop=True)


def employment_periods() -> list[str]:
    return sorted(load_employment().period.unique())


def _pay_rows(width: int) -> pd.DataFrame:
    """Workers and monthly total wages (won) per 7th-revision code of the given width (1 = major group, 3 = minor
    group) and year, all employees (both sexes)."""
    d = _tidy(PAY_TBL, 2)
    d = d[d.d1.str.contains("전체")]
    d["ksco7_code"] = d.d0.str.extract(r"KSCO7_(\d+) ")[0]
    d = d.dropna(subset=["ksco7_code"])
    d = d[d.ksco7_code.str.len() == width]
    piv = d.pivot_table(index=["ksco7_code", "period"], columns="item", values="value", aggfunc="first").reset_index()
    wage_col = next(c for c in piv.columns if "월임금총액" in str(c))
    work_col = next(c for c in piv.columns if "근로자수" in str(c))
    out = pd.DataFrame({"ksco7_code": piv.ksco7_code, "year": piv.period.astype(int), "workers": piv[work_col], "wage_monthly": piv[wage_col] * 1000.0})
    return out.dropna(subset=["wage_monthly"]).sort_values(["ksco7_code", "year"]).reset_index(drop=True)


def load_pay_ksco7() -> pd.DataFrame:
    """Three-digit codes of the 7th revision (the survey publishes none for managers, major group 1)."""
    return _pay_rows(3)


def load_pay_major() -> pd.DataFrame:
    """The survey's own major-group figures (one digit; the major groups are the same in the 7th and 8th revisions)."""
    return _pay_rows(1).rename(columns={"ksco7_code": "major_group"})


def load_pay() -> pd.DataFrame:
    """The wage survey's pay carried to the 8th revision: each 8th-revision code takes the workers-weighted mean
    monthly wage of the 7th-revision codes that feed it (Statistics Korea's 7th-to-8th link at four digits,
    rolled up to three), per year."""
    p7 = load_pay_ksco7()
    link = pd.read_csv(LINK_78, dtype={"ksco8_code": str, "ksco7_code": str})
    m = link.merge(p7, on="ksco7_code")
    m["w"] = m.n_links * m.workers.fillna(0).clip(lower=1)
    rows = []
    for (k8, y), g in m.groupby(["ksco8_code", "year"]):
        rows.append({"ksco_code": k8, "year": int(y), "wage_monthly": float(np.average(g.wage_monthly, weights=g.w)), "workers": float(g.workers.sum()), "n_sources": int(len(g))})
    return pd.DataFrame(rows).sort_values(["ksco_code", "year"]).reset_index(drop=True)


def pay_years() -> list[int]:
    return sorted(load_pay_ksco7().year.unique())


def load_titles_json() -> pd.DataFrame:
    rows = json.loads(TITLES.read_text(encoding="utf-8"))
    return pd.DataFrame(rows)


def _fill_pay(df: pd.DataFrame, pay: pd.DataFrame, major_pay: pd.DataFrame) -> pd.DataFrame:
    """Pay per code (carried through the 7th-to-8th link), else the two-digit group's employment-weighted mean of
    the codes that have one, else the survey's own major-group figure (managers have no three-digit lines)."""
    p = pay.set_index("ksco_code").wage_monthly
    df = df.copy()
    df["wage_monthly"] = df.ksco_code.map(p)
    df["wage_source"] = np.where(df.wage_monthly.notna(), "labour_conditions_survey_code_via_ksco7_link", "")
    have = df[df.wage_monthly.notna()]
    grp = have.groupby(have.ksco_code.str[:2]).apply(lambda g: float(np.average(g.wage_monthly, weights=np.maximum(g.employment_thousands, 1e-6))), include_groups=False)
    miss = df.wage_monthly.isna()
    df.loc[miss, "wage_monthly"] = df.loc[miss, "ksco_code"].str[:2].map(grp)
    df.loc[miss & df.wage_monthly.notna(), "wage_source"] = "labour_conditions_survey_submajor_mean"
    mp = major_pay.set_index("major_group").wage_monthly
    miss = df.wage_monthly.isna()
    df.loc[miss, "wage_monthly"] = df.loc[miss, "ksco_code"].str[0].map(mp)
    df.loc[miss & df.wage_monthly.notna(), "wage_source"] = "labour_conditions_survey_major_group"
    return df


def build_weights(period: str | None = None, year: int | None = None) -> pd.DataFrame:
    """Employment (latest half-year of the labour force survey) and monthly pay (latest year of the wage survey) per code."""
    emp = load_employment()
    period = period or emp.period.max()
    e = emp[emp.period == period]
    pay = load_pay()
    year = year or int(pay.year.max())
    df = pd.DataFrame({"ksco_code": e.ksco_code.values, "ksco_title": e.ksco_title_ko.values, "employment_thousands": e.employment_thousands.values})
    titles = load_titles_json().set_index("code").en
    df["ksco_title"] = df.ksco_code.map(titles).fillna(df.ksco_title)
    mp = load_pay_major()
    df = _fill_pay(df, pay[pay.year == year], mp[mp.year == year])
    df["major_group"] = df.ksco_code.str[0]
    df["submajor_label"] = df.major_group
    df["submajor_name"] = df.major_group.map(MAJOR_GROUP_NAMES)
    link = load_isco_link(set(df.ksco_code))
    from .no_labour import _skill_level
    lv = link.assign(sl=link.isco_code.map(_skill_level)).groupby("nat_code").sl.mean()
    df["skill_level"] = df.ksco_code.map(lv).fillna(2.0)
    df["wage_value"] = df.employment_thousands * df.wage_monthly
    df["employment_method"] = f"local_area_labour_force_survey_{period}"
    df["edition"] = int(period[:4])
    return df[(df.employment_thousands > 0) & df.wage_monthly.notna()].reset_index(drop=True)


# ----------------------------------------------------------------------------- the link to ISCO-08
def load_isco_link(unit_codes: set[str] | None = None) -> pd.DataFrame:
    """ISCO-08 unit group -> KSCO three-digit code (8th revision). The reference table, built from Statistics Korea's
    official four-digit link, names for each KSCO code the ISCO unit groups it corresponds to; a three-digit entry
    would stand for every unit group under it. An ISCO group named by several KSCO codes is shared equally."""
    key = pd.read_csv(LINK, dtype=str)
    from . import io
    isco = sorted(io.load("xw_onetsoc_isco08").isco_code.astype(str).unique())
    rows = []
    for r in key.itertuples():
        if unit_codes and r.ksco_code not in unit_codes:
            continue
        for tgt in str(r.isco_codes).split(";"):
            tgt = tgt.strip()
            if not tgt:
                continue
            units = [tgt] if len(tgt) == 4 else [i for i in isco if i.startswith(tgt)]
            rows += [{"isco_code": u, "nat_code": r.ksco_code, "weight": 1.0} for u in units]
    k = pd.DataFrame(rows).drop_duplicates(["isco_code", "nat_code"])
    k["weight"] = k.weight / k.groupby("isco_code").weight.transform("sum")
    return k.reset_index(drop=True)


# ----------------------------------------------------------------------------- industries and the firm survey
def load_industry_mix(period: str | None = None) -> pd.DataFrame:
    """Employed (thousands) by occupation major group and KSIC industry section, both sexes, latest half-year."""
    d = _tidy(MIX_TBL, 3)
    d = d[d.item.str.startswith("T1 ") & d.d0.str.startswith("S0 ")]
    period = period or d.period.max()
    d = d[d.period == period]
    d["industry"] = d.d1.str.extract(r"^\d+ ([A-U])\. ")[0]
    d["major_group"] = d.d2.str.extract(r"^(\d) ")[0]
    d = d.dropna(subset=["industry", "major_group"])
    d = d[d.major_group != "0"]
    return d[["major_group", "industry", "value"]].rename(columns={"value": "employed"}).reset_index(drop=True)


def load_ai_points() -> pd.DataFrame:
    """Share of surveyed enterprises developing or using artificial intelligence, nationally and by industry
    section, 2017 to 2024 (Survey of Business Activities: enterprises with 50 or more regular employees and capital
    of 300 million won or more). A year's reading covers that calendar year and is dated at its end."""
    rows = []
    for tbl in AI_TBLS:
        d = _tidy(tbl, 1)
        firms = d[d.item.str.startswith("T01 ")].set_index(["d0", "period"]).value
        ai = d[d.item.str.contains("인공지능")].set_index(["d0", "period"]).value
        share = (ai / firms * 100.0).reset_index().rename(columns={"value": "pct"})
        for r in share.itertuples():
            name = r.d0
            if name.startswith("0 "):
                rows.append({"date": int(r.period) + 1.0, "group_type": "all", "industry": None, "label": str(r.period), "pct": r.pct})
            else:
                m = re.match(r"^([A-U]) ", name)
                if m:
                    rows.append({"date": int(r.period) + 1.0, "group_type": "industry", "industry": m.group(1), "label": str(r.period), "pct": r.pct})
    out = pd.DataFrame(rows).dropna(subset=["pct"]).drop_duplicates(["group_type", "industry", "date"])
    out["period"] = out.label
    return out.sort_values(["group_type", "industry", "date"], na_position="first").reset_index(drop=True)


# ----------------------------------------------------------------------------- rung five, units, titles
def employment_change() -> pd.DataFrame:
    """Per code: employed persons in the latest half-year against the same half a year earlier (labour force
    survey) and monthly pay in the latest wage-survey year against the year before."""
    emp = load_employment()
    p1 = emp.period.max(); p0 = f"{int(p1[:4]) - 1}{p1[4:]}"
    pay = load_pay(); y1 = int(pay.year.max()); y0 = y1 - 1
    e = emp.pivot_table(index="ksco_code", columns="period", values="employment_thousands", aggfunc="first")
    mp = load_pay_major(); base = emp[emp.period == p1][["ksco_code", "employment_thousands"]]
    filled = pd.concat([_fill_pay(base, pay[pay.year == y], mp[mp.year == y]).assign(year=y) for y in (y0, y1) if y in set(pay.year)], ignore_index=True)
    own = filled[filled.wage_source == "labour_conditions_survey_code_via_ksco7_link"]   # a group-level stand-in carries no change of its own
    w = own.pivot_table(index="ksco_code", columns="year", values="wage_monthly", aggfunc="first")
    codes = sorted(emp.ksco_code.unique())
    df = pd.DataFrame({"ksco_code": codes})
    ey0, ey1 = p0[:4], p1[:4]
    df[f"emp_{ey0}"] = df.ksco_code.map(e[p0]) if p0 in e.columns else np.nan
    df[f"emp_{ey1}"] = df.ksco_code.map(e[p1])
    df["emp_change_pct"] = (df[f"emp_{ey1}"] / df[f"emp_{ey0}"] - 1.0) * 100.0
    df[f"wage_{y0}"] = df.ksco_code.map(w[y0]) if y0 in w.columns else np.nan
    df[f"wage_{y1}"] = df.ksco_code.map(w[y1]) if y1 in w.columns else np.nan
    df["wage_change_pct"] = (df[f"wage_{y1}"] / df[f"wage_{y0}"] - 1.0) * 100.0
    df["year_from"] = y0; df["year_to"] = y1; df["wage_years"] = f"{y0}-{y1}"
    return df


def outcome_units() -> pd.DataFrame:
    w = build_weights()
    return pd.DataFrame({"ksco_code": w.ksco_code.values, "emp_unit": "thousands", "wage_unit": "month"})


def load_titles() -> pd.DataFrame:
    """English titles as principal, Korean names as alternative titles for the search."""
    t = load_titles_json()
    t = t[t.lvl == 3]
    return pd.concat([pd.DataFrame({"ksco_code": t.code, "title": t.en, "category": "principal"}),
                      pd.DataFrame({"ksco_code": t.code, "title": t.ko, "category": "korean"})], ignore_index=True)


def no_tasks() -> pd.DataFrame:
    return pd.DataFrame(columns=["ksco_code", "task_seq", "task"])
