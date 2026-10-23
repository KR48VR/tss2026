"""Singapore wages and employment, matched to SSOC codes.

Sources (all public, see docs/DATA_SOURCES.md):
* MOM Occupational Wage Survey, June 2025 (stats.mom.gov.sg, Table 4): median and quartile basic and
  gross monthly wages for 523 occupations **with official SSOC 2024 codes**. Primary wage source.
* MOM Occupational Wage Survey, June 2024 (data.gov.sg): 562 occupations by title only; matched to
  SSOC codes by title (exact, then fuzzy >= 90) and used only for codes the 2025 table lacks.
* MOM Labour Force in Singapore 2025, Section D, Table D8 (stats.mom.gov.sg): employed residents by
  SSOC 2024 sub-major group (2-digit, 43 groups). Primary employment source.
* MOM Labour Force Survey (data.gov.sg / SingStat): employment and median income by major group
  (1-digit); used for checks and as the last-resort wage.

Weights produced per SSOC 5-digit code:
    employment  resident employment allocated from the 2-digit sub-major group total; within a group
                the allocation is uniform across the codes that appear in the wage survey (the
                occupations large enough to be surveyed), with a quarter-weight for the rest, because
                no official count below the 2-digit level is published. `employment_method` says so.
    wage        median gross monthly wage (S$): OWS 2025 by code, else OWS 2024 by title, else the
                sub-major group median, else the major group median income.
    wage_value  employment x wage: the "wage bill" weight Project Iceberg uses.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from . import singapore as sg
from .config import RAW

MOM = RAW / "singapore" / "mom"

# ordered: more specific labels first ("associate professionals" before "professionals")
MAJOR_GROUP_KEYS = [
    ("3", ["associate professionals & technicians", "associate professionals and technicians"]),
    ("1", ["managers & administrators", "managers and administrators", "legislators, senior officials and managers"]),
    ("2", ["professionals"]),
    ("4", ["clerical support workers"]),
    ("5", ["service & sales workers", "service and sales workers"]),
    ("6", ["agricultural & fishery workers", "agricultural and fishery workers"]),
    ("7", ["craftsmen & related trade", "craftsmen and related trades"]),
    ("8", ["plant & machine operators", "plant and machine operators"]),
    ("9", ["cleaners, labourers & related", "cleaners, labourers and related"]),
]


def _major_from_label(label: str) -> str | None:
    s = str(label).strip().lower()
    for code, keys in MAJOR_GROUP_KEYS:
        if any(k in s for k in keys):
            return code
    return None


def _norm_title(s: str) -> str:
    s = str(s).casefold().replace("–", "-").replace("—", "-")
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s*\(\s*", " (", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ----------------------------------------------------------------------------- wages
def load_ows_wages() -> pd.DataFrame:
    df = pd.read_csv(MOM / "ows_2024_wages_by_occupation.csv", dtype=str)
    for c in df.columns:
        if "wage" in c:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={"occ_desc": "ows_occupation", "mthly_gross_wage_50_pctile": "wage_median", "mthly_gross_wage_25_pctile": "wage_p25", "mthly_gross_wage_75_pctile": "wage_p75"})
    return df[["year", "ows_occupation", "wage_median", "wage_p25", "wage_p75"]]


def match_ows_to_ssoc(edition: int = 2024, fuzzy_threshold: int = 90) -> pd.DataFrame:
    """Match OWS occupation titles to SSOC 5-digit codes. Returns one row per OWS title with code,
    match_method and score. Unmatched rows keep code = None."""
    w = load_ows_wages()
    struct = sg.load_ssoc_structure(edition)
    five = struct[struct.level == 5].copy()
    five["key"] = five.ssoc_title.map(_norm_title)
    alpha = sg.load_ssoc_alpha_index(edition)
    alpha = alpha[alpha.title_type.isin(["Principal Title", "Alternative Title"])].copy()
    alpha["key"] = alpha.index_title.map(_norm_title)
    title_map = dict(zip(five.key, five.ssoc_code))
    alpha_map = dict(zip(alpha.key, alpha.ssoc_code))
    out = []
    keys = list(title_map.keys())
    for _, r in w.iterrows():
        k = _norm_title(r.ows_occupation)
        if k in title_map:
            out.append((r.ows_occupation, title_map[k], "exact_title", 100.0))
            continue
        if k in alpha_map:
            out.append((r.ows_occupation, alpha_map[k], "exact_alpha_index", 100.0))
            continue
        # strip trailing bracketed qualifiers like "(e.g. ...)" and retry
        k2 = re.sub(r"\s*\((e\.g\.|eg\.|excluding|including)[^)]*\)", "", k).strip()
        if k2 in title_map:
            out.append((r.ows_occupation, title_map[k2], "exact_title_stripped", 100.0))
            continue
        best = process.extractOne(k, keys, scorer=fuzz.token_set_ratio)
        if best and best[1] >= fuzzy_threshold:
            out.append((r.ows_occupation, title_map[best[0]], "fuzzy_title", float(best[1])))
            continue
        best2 = process.extractOne(k, list(alpha_map.keys()), scorer=fuzz.token_set_ratio)
        if best2 and best2[1] >= fuzzy_threshold:
            out.append((r.ows_occupation, alpha_map[best2[0]], "fuzzy_alpha_index", float(best2[1])))
            continue
        out.append((r.ows_occupation, None, "unmatched", float(best[1]) if best else None))
    m = pd.DataFrame(out, columns=["ows_occupation", "ssoc_code", "match_method", "match_score"])
    m = m.merge(w, on="ows_occupation", how="left")
    m["edition"] = edition
    return m


# ----------------------------------------------------------------------------- employment
def load_lfs_employment_major(year: int | None = None) -> pd.DataFrame:
    """Employed residents by major occupational group (thousands), from the LFS annual series."""
    df = pd.read_csv(MOM / "lfs_employed_by_occupation_age.csv", dtype=str)
    # the file is a stack of blocks: the all-ages block first (header "All Occupation Groups"), then one
    # block per age group (header "Employed Residents Aged ..."). Keep only the first block.
    first_block = []
    started = False
    for _, r in df.iterrows():
        label = str(r.DataSeries)
        if "All Occupation" in label:
            started = True
            continue
        if label.strip().startswith("Employed Residents Aged"):
            break
        if started:
            first_block.append(r)
    d = pd.DataFrame(first_block)
    years = [c for c in d.columns if c.isdigit()]
    y = str(year or max(int(c) for c in years))
    d["major_group"] = d.DataSeries.map(_major_from_label)
    d["employed_thousands"] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=["major_group"])[["major_group", "DataSeries", "employed_thousands"]].rename(columns={"DataSeries": "label"})
    d["label"] = d.label.str.strip()
    d["year"] = int(y)
    return d.reset_index(drop=True)


def load_lfs_median_income_major() -> pd.DataFrame:
    """Median gross monthly income of full-time employed residents by major group (SingStat M920131)."""
    d = json.load(open(MOM / "singstat_M920131.json"))["Data"]
    rows = []
    for r in d["row"]:
        label = r["rowText"]
        mg = _major_from_label(label)
        if mg is None:
            continue
        sex = "male" if label.strip().endswith("- Male") else "female" if label.strip().endswith("- Female") else "all"
        for c in r["columns"]:
            rows.append({"major_group": mg, "sex": sex, "year": int(c["key"]), "median_income": pd.to_numeric(c["value"], errors="coerce")})
    df = pd.DataFrame(rows)
    latest = df.year.max()
    both = df[(df.year == latest)].groupby("major_group").median_income.mean().rename("median_income_major").reset_index()
    both["year"] = latest
    return both


def _xl_rows(path, sheet):
    import openpyxl
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wb = openpyxl.load_workbook(path, read_only=True)
        return list(wb[sheet].iter_rows(values_only=True))


def load_ows2025_wages(sheet: str = "T4") -> pd.DataFrame:
    """OWS June 2025 Table 4 (all industries): one row per SSOC 2024 code with wage percentiles."""
    rows = _xl_rows(MOM / "ows2025" / "mrsd_2025Wages_table4.xlsx", sheet)
    out = []
    for r in rows:
        if r is None or len(r) < 9 or r[1] is None:
            continue
        code = str(r[1]).strip()
        if not re.fullmatch(r"\d{5}", code):
            continue
        vals = [pd.to_numeric(str(v).replace(",", ""), errors="coerce") if v not in (None, "", "s", "-") else None for v in r[3:9]]
        out.append({"ssoc_code": code, "ows_occupation": str(r[2]).strip(), "basic_p25": vals[0], "basic_median": vals[1], "basic_p75": vals[2], "wage_p25": vals[3], "wage_median": vals[4], "wage_p75": vals[5]})
    df = pd.DataFrame(out)
    df["year"] = 2025
    df["source"] = "ows_2025_table4"
    return df


def load_lfs_employment_submajor(edition: int = 2025) -> pd.DataFrame:
    """Table D8 of the Labour Force in Singapore report: employed residents (thousands) by SSOC 2024
    sub-major group (2-digit). Combined rows such as '61-62' and 'X1-X5' are kept with their label."""
    rows = _xl_rows(MOM / f"LFR{edition}_SectionD.xlsx", "D8")
    out = []
    for r in rows:
        if not r or r[0] is None:
            continue
        m = re.match(r"^\s*([0-9X][0-9X]?(?:-[0-9X]{1,2})?)\s+(.*)$", str(r[0]))
        if not m:
            continue
        codes = m.group(1)
        total = pd.to_numeric(r[1], errors="coerce")
        if pd.isna(total):
            continue
        # expand '61-62' -> ['61','62'], 'X1-X5' -> ['X1'..'X5']
        if "-" in codes:
            a, b = codes.split("-")
            if a[0].isdigit():
                members = [str(x) for x in range(int(a), int(b) + 1)]
            else:
                members = [f"{a[0]}{i}" for i in range(int(a[1:]), int(b[1:]) + 1)]
        else:
            members = [codes]
        out.append({"submajor_label": codes, "submajor_members": members, "group_name": m.group(2).strip(), "employed_thousands": float(total), "male_thousands": pd.to_numeric(r[2], errors="coerce"), "female_thousands": pd.to_numeric(r[3], errors="coerce"), "year": edition})
    return pd.DataFrame(out)


def build_ssoc_weights(edition: int = 2024) -> pd.DataFrame:
    """Per SSOC 5-digit code: employment (allocated from the 2-digit group), wage, wage_value, with flags."""
    struct = sg.load_ssoc_structure(edition)
    five = struct[struct.level == 5][["ssoc_code", "ssoc_title"]].copy()
    five["major_group"] = five.ssoc_code.str[0]
    five["submajor"] = five.ssoc_code.str[:2]
    # wages: 2025 coded table first, then 2024 by title, then group medians
    w25 = load_ows2025_wages().dropna(subset=["wage_median"]).groupby("ssoc_code").wage_median.mean()
    w24 = match_ows_to_ssoc(edition).dropna(subset=["ssoc_code"]).groupby("ssoc_code").wage_median.mean()
    five["wage_median"] = five.ssoc_code.map(w25)
    five["wage_source"] = np.where(five.wage_median.notna(), "ows_2025_coded", None)
    m = five.wage_median.isna()
    five.loc[m, "wage_median"] = five.loc[m, "ssoc_code"].map(w24)
    five.loc[m & five.wage_median.notna(), "wage_source"] = "ows_2024_title_match"
    sub_med = five.dropna(subset=["wage_median"]).groupby("submajor").wage_median.median()
    m = five.wage_median.isna()
    five.loc[m, "wage_median"] = five.loc[m, "submajor"].map(sub_med)
    five.loc[m & five.wage_median.notna(), "wage_source"] = "submajor_median"
    inc = load_lfs_median_income_major().set_index("major_group").median_income_major
    m = five.wage_median.isna()
    five.loc[m, "wage_median"] = five.loc[m, "major_group"].map(inc)
    five.loc[m & five.wage_median.notna(), "wage_source"] = "major_group_median_income"
    m = five.wage_median.isna()
    five.loc[m, "wage_median"] = float(five.wage_median.median())
    five.loc[m, "wage_source"] = "overall_median"
    five["in_ows"] = five.wage_source.isin(["ows_2025_coded", "ows_2024_title_match"])
    # employment: 2-digit sub-major totals (Table D8), split within the group
    d8 = load_lfs_employment_submajor(2025)
    sub_total = {}
    for _, r in d8.iterrows():
        for mem in r.submajor_members:
            sub_total[mem] = (r.employed_thousands, r.submajor_label)
    five["submajor_label"] = five.submajor.map(lambda c: sub_total.get(c, (np.nan, None))[1])
    alloc = []
    for lbl, g in five.groupby("submajor_label", dropna=False):
        # members of a combined label such as '61-62' share one published total
        total = float(next((v[0] for k, v in sub_total.items() if v[1] == lbl), 0.0)) if lbl else 0.0
        n_in = int(g.in_ows.sum()); n_out = int((~g.in_ows).sum())
        w_in, w_out = (1.0, 0.25) if n_in else (0.0, 1.0)
        denom = n_in * w_in + n_out * w_out
        for idx, r in g.iterrows():
            alloc.append((idx, total * (w_in if r.in_ows else w_out) / denom if denom else 0.0))
    five["employment_thousands"] = pd.Series(dict(alloc))
    five["employment_method"] = "submajor_group_total_uniform_within_group"
    five["wage_value"] = five.employment_thousands * five.wage_median
    five["edition"] = edition
    return five.drop(columns=["submajor"]).reset_index(drop=True)
