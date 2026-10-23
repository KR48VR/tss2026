"""Singapore wages and employment, matched to SSOC codes.

Sources (all public, see docs/DATA_SOURCES.md):
* MOM Occupational Wage Survey, June 2024 (data.gov.sg): median/quartile gross monthly wages for 562
  occupations, by occupation title only. We match titles to SSOC 2024 5-digit codes through SingStat's
  classification titles and alphabetical index (exact, then fuzzy >= 90).
* MOM Labour Force Survey (data.gov.sg / SingStat Table Builder): employed residents by major
  occupational group (1-digit, SSOC 2024), by year; median income by major group.
* MOM Labour Force in Singapore 2025, Section D (XLSX from stats.mom.gov.sg) when reachable: finer
  occupation breakdowns. The site blocks some automated downloads; drop the file into
  data/raw/singapore/mom/ if the fetcher could not get it.

Weights produced per SSOC 5-digit code:
    employment  resident employment allocated from the major group total; within a group the
                allocation is proportional to the number of wage-survey occupations only if no finer
                table exists (uniform within group), and the method is recorded in `employment_method`.
    wage        median gross monthly wage (S$) from the OWS where matched, else the group median.
    wage_value  employment x wage: the "wage bill" weight Project Iceberg uses.
"""
from __future__ import annotations

import json
import re

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


def build_ssoc_weights(edition: int = 2024) -> pd.DataFrame:
    """Per SSOC 5-digit code: employment (allocated), wage, wage_value, with method flags."""
    struct = sg.load_ssoc_structure(edition)
    five = struct[struct.level == 5][["ssoc_code", "ssoc_title"]].copy()
    five["major_group"] = five.ssoc_code.str[0]
    ows = match_ows_to_ssoc(edition)
    wage = ows.dropna(subset=["ssoc_code"]).groupby("ssoc_code").wage_median.mean()
    five["wage_median"] = five.ssoc_code.map(wage)
    five["wage_source"] = five.wage_median.notna().map({True: "ows_2024", False: "major_group_median"})
    inc = load_lfs_median_income_major().set_index("major_group").median_income_major
    five["wage_median"] = five.wage_median.fillna(five.major_group.map(inc))
    five.loc[five.wage_median.isna(), "wage_source"] = "overall_median"
    five["wage_median"] = five.wage_median.fillna(float(five.wage_median.median()))
    emp = load_lfs_employment_major().set_index("major_group").employed_thousands
    # within a major group we have no official count per 5-digit code: allocate uniformly across the
    # codes that appear in the wage survey (they are the occupations with enough employees to be
    # surveyed), and give the remaining codes a small share.
    five["in_ows"] = five.ssoc_code.isin(set(ows.ssoc_code.dropna()))
    alloc = []
    for mg, g in five.groupby("major_group"):
        total = float(emp.get(mg, 0.0))
        n_in = int(g.in_ows.sum()); n_out = int((~g.in_ows).sum())
        w_in, w_out = (1.0, 0.25) if n_in else (0.0, 1.0)
        denom = n_in * w_in + n_out * w_out
        for _, r in g.iterrows():
            alloc.append(total * (w_in if r.in_ows else w_out) / denom if denom else 0.0)
    five["employment_thousands"] = alloc
    five["employment_method"] = "major_group_total_uniform_within_group"
    five["wage_value"] = five.employment_thousands * five.wage_median
    five["edition"] = edition
    return five.reset_index(drop=True)
