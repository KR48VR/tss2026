"""France (full depth): employees and pay by detailed socio-professional category (PCS-ESE, four characters) from
INSEE's Melodi API, the Eurostat AI readings by industry, INSEE's correspondence matrices to ISCO-08, and the labour
force survey's occupation-by-activity mix.

Files (`data/raw/insee/`):
  melodi_dera_prive.json        Melodi DS_DERA_PRIVE_ANNUAL (fetch.fetch_melodi): private-sector employees in full-time
                                equivalents and mean net monthly salary per FTE by detailed PCS-ESE, 2024, from the
                                Déclaration sociale nominative (the "Base Tous salariés")
  melodi_dera_public.json       DS_DERA_PUBLIC_ANNUAL: the same for the civil service (State, local government and
                                hospitals), 2023, 75 categories
  melodi_eec_pcs_activity.json  DD_EEC_ANNUAL: the labour force survey's employed persons (thousands) by PCS and economic
                                activity, 15 and over, 2025 (the cross with activity is published for the six one-digit
                                groups): the mix that carries industry shifts
  pcs_ese2017_liste.xls         the PCS-ESE 2017 list (codes and French titles at every level)
  pcs_ese2017_liste_n2.xls      its 29 two-digit categories
  pcs2003_pcs2020_matrix.csv    INSEE's probabilistic matrix from PCS 2003 professions (the codes the PCS-ESE keeps)
                                to PCS 2020 professions
  pcs2020_isco08_matrix.csv     INSEE's probabilistic matrix from PCS 2020 professions to ISCO-08 unit groups
Eurostat's enterprise AI survey for France (`eurostat/isoc_eb_ai.json`, `isoc_eb_ain2.json`) is read through europe.py.

What is coarse: the weights count salaried employees in full-time equivalents (private sector 2024 and public sector
2023; the self-employed, about a tenth of employment, are not in the wage tables); pay is the mean net monthly salary
per FTE, not a median; the link to O*NET runs through two probabilistic matrices (PCS 2003 to PCS 2020, PCS 2020 to
ISCO-08); the mix is at the one-digit group; Melodi carries one year of each wage table, so there is no outcome
test yet; no official task list.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .config import RAW

INSEE = RAW / "insee"

MAJOR_GROUP_NAMES = {"1": "Farmers", "2": "Craftspeople, shopkeepers and business owners", "3": "Managers and higher intellectual professions",
                     "4": "Intermediate professions", "5": "Employees", "6": "Manual workers"}
# the 29 two-digit categories (catégories socioprofessionnelles), in English
SUBMAJOR_NAMES = {"10": "Farmers (salaried on their own farm)", "21": "Craftspeople (salaried in their own business)", "22": "Shopkeepers and similar (salaried in their own business)", "23": "Heads of businesses with 10 or more employees",
                  "31": "Liberal professions", "33": "Senior civil service managers", "34": "Teachers and scientific professions", "35": "Information, arts and entertainment professions", "37": "Administrative and commercial managers in companies", "38": "Engineers and technical managers in companies",
                  "42": "Primary school teachers and similar", "43": "Intermediate health and social work professions", "44": "Clergy", "45": "Intermediate administrative professions in the civil service", "46": "Intermediate administrative and commercial professions in companies", "47": "Technicians", "48": "Foremen and supervisors",
                  "52": "Civil service employees", "53": "Police and armed forces", "54": "Administrative employees in companies", "55": "Shop employees", "56": "Personal service workers",
                  "62": "Skilled industrial workers", "63": "Skilled craft workers", "64": "Drivers", "65": "Skilled handling, warehouse and transport workers", "67": "Unskilled industrial workers", "68": "Unskilled craft workers", "69": "Farm workers and similar"}
# the labour force survey's activity codes that make up a NACE section where the section itself is not published
_SECTION_PARTS = {"C": ["C10T12", "C13T15", "C16T18", "C19", "C20", "C21", "C22_23", "C24_25", "C26", "C27", "C28", "C29_30", "C31T33"], "M": ["M69T71", "M72", "M73T75"], "Q": ["Q86", "Q87_88"]}


# ----------------------------------------------------------------------------- Melodi tables
def _melodi(name: str) -> pd.DataFrame:
    d = json.loads((INSEE / f"{name}.json").read_text(encoding="utf-8"))
    rows = [{**o.get("dimensions", {}), "value": (o.get("measures", {}).get("OBS_VALUE_NIVEAU") or {}).get("value")} for o in d["observations"]]
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df.value, errors="coerce")
    return df


def _dera(name: str, totals: dict) -> pd.DataFrame:
    """One wage table's national totals: pcs_code (four characters, upper case), year, fte, net_monthly."""
    d = _melodi(name)
    for k, v in totals.items():
        if k in d.columns:
            d = d[d[k] == v]
    d = d[d.PCS_ESE.str.fullmatch(r"\d{3}[A-Za-z]")].copy()
    d["pcs_code"] = d.PCS_ESE.str.upper()
    p = d.pivot_table(index=["pcs_code", "TIME_PERIOD"], columns="DERA_MEASURE", values="value", aggfunc="first").reset_index()
    return p.rename(columns={"TIME_PERIOD": "year", "EFFECTIFS_EQTP": "fte", "SALAIRE_NET_EQTP_MENSUEL_MOYENNE": "net_monthly"})[["pcs_code", "year", "fte", "net_monthly"]].dropna(subset=["fte"])


def private_wages() -> pd.DataFrame:
    return _dera("melodi_dera_prive", {"ACTIVITY": "_T", "AGE": "_T", "SEX": "_T", "WKTIME": "_T", "NUMBER_EMPL": "_T", "QUANTILE": "_T"})


def public_wages() -> pd.DataFrame:
    return _dera("melodi_dera_public", {"CIVILSERVANT_STATUS": "_T", "PUBLIC_LEGAL_FORM": "_T", "WKTIME": "_T", "SEX": "_T", "AGE": "_T", "QUANTILE": "_T"})


def load_titles_list() -> dict:
    """PCS-ESE 2017 codes (upper case) -> French title, at every level."""
    d = pd.read_excel(INSEE / "pcs_ese2017_liste.xls", header=None, dtype=str)
    out = {}
    for c, t in zip(d[0], d[1]):
        c = str(c).strip()
        if re.fullmatch(r"\d{3}[A-Za-z]|\d{1,2}", c) and isinstance(t, str):
            out[c.upper()] = t.strip()
    return out


def build_weights() -> pd.DataFrame:
    """Per detailed PCS-ESE category: salaried employees in full-time equivalents (thousands; the private sector's
    latest year plus the public sector's latest year), the FTE-weighted mean net monthly salary, their product as
    the wage weight, the French title, the one-digit group and two-digit category, and the skill level as the mean
    ISCO level of the unit groups the category links to."""
    pr = private_wages(); yp = pr.year.max(); pr = pr[pr.year == yp]
    pu = public_wages(); yu = pu.year.max(); pu = pu[pu.year == yu]
    both = pd.concat([pr.assign(sector="private"), pu.assign(sector="public")], ignore_index=True)
    both["fte"] = both.fte.astype(float); both["net_monthly"] = both.net_monthly.astype(float)
    both["w"] = both.fte * both.net_monthly.fillna(0)
    g = both.groupby("pcs_code").agg(fte=("fte", "sum"), w=("w", "sum"), fte_paid=("fte", lambda s: float(s[both.loc[s.index, "net_monthly"].notna()].sum())), sectors=("sector", lambda s: "+".join(sorted(set(s))))).reset_index()
    g["wage_monthly"] = g.w / g.fte_paid.where(g.fte_paid > 0)
    titles = load_titles_list()
    df = pd.DataFrame({"pcs_code": g.pcs_code, "pcs_title": g.pcs_code.map(titles), "employment_thousands": g.fte / 1000.0, "wage_monthly": g.wage_monthly, "sectors": g.sectors})
    df["pcs_title"] = df.pcs_title.fillna(df.pcs_code.str[:2].map(titles).fillna(df.pcs_code))
    df["wage_source"] = np.where(df.wage_monthly.notna(), "dsn_mean_net_monthly_fte_" + df.sectors, "")
    miss = df.wage_monthly.isna()
    cat = df.assign(k=df.pcs_code.str[:2]).dropna(subset=["wage_monthly"]).groupby("k").apply(lambda x: float(np.average(x.wage_monthly, weights=np.maximum(x.employment_thousands, 1e-6))), include_groups=False)
    df.loc[miss, "wage_monthly"] = df.loc[miss, "pcs_code"].str[:2].map(cat); df.loc[miss & df.wage_monthly.notna(), "wage_source"] = "category_mean"
    df["major_group"] = df.pcs_code.str[0]
    df["submajor_label"] = df.pcs_code.str[:2]
    df["submajor_name"] = df.submajor_label.map(SUBMAJOR_NAMES).fillna(df.submajor_label.map(titles))
    link = load_isco_link(set(df.pcs_code))
    from .no_labour import _skill_level
    lv = link.assign(sl=link.isco_code.map(_skill_level)).groupby("nat_code").apply(lambda x: float(np.average(x.sl, weights=np.maximum(x.weight, 1e-9))), include_groups=False)
    df["skill_level"] = df.pcs_code.map(lv)
    df["skill_level"] = df.skill_level.fillna(df.groupby("submajor_label").skill_level.transform("mean")).fillna(2.0)
    df["wage_value"] = df.employment_thousands * df.wage_monthly
    df["employment_method"] = f"dsn_fte_private_{yp}_public_{yu}"
    df["edition"] = int(yp)
    return df[(df.employment_thousands > 0) & df.wage_monthly.notna()].drop(columns="sectors").reset_index(drop=True)


# ----------------------------------------------------------------------------- the link to ISCO-08 through two INSEE matrices
def _matrix(name: str, a: str, b: str, pct: str) -> pd.DataFrame:
    """An INSEE correspondence matrix as (src, dst, p): percentages with a decimal comma; 'ns' (not significant)
    and 'a' (a rounding remainder) are small shares and count as 0.5 percent."""
    m = pd.read_csv(INSEE / name, sep=";", dtype=str)
    m = m.rename(columns={a: "src", b: "dst", pct: "p"})
    m["src"] = m.src.str.strip().str.upper(); m["dst"] = m.dst.str.strip().str.upper()
    m["p"] = pd.to_numeric(m.p.str.replace(",", ".", regex=False), errors="coerce").fillna(0.5)
    m = m[m.p > 0]
    m["p"] = m.p / m.groupby("src").p.transform("sum")
    return m[["src", "dst", "p"]]


def pcs_to_isco() -> pd.DataFrame:
    """PCS-ESE / PCS 2003 profession -> ISCO-08 unit group probabilities: the PCS 2003 to PCS 2020 matrix composed with
    the PCS 2020 to ISCO-08 matrix. A PCS-ESE code the 2003 list does not carry (the 2017 variants, such as 381B and
    381C beside 381A) takes the mean of the PCS 2003 professions with the same three digits."""
    a = _matrix("pcs2003_pcs2020_matrix.csv", "PCS2003", "PCS2020", "pct")
    b = _matrix("pcs2020_isco08_matrix.csv", "PCS4", "ISCO4", "PCT")
    m = a.merge(b, left_on="dst", right_on="src", suffixes=("", "_b"))
    m["p"] = m.p * m.p_b
    out = m.groupby(["src", "dst_b"]).p.sum().reset_index().rename(columns={"src": "pcs_code", "dst_b": "isco_code"})
    out = out[out.isco_code.str.fullmatch(r"\d{4}")]
    return out.reset_index(drop=True)


def load_isco_link(unit_codes: set[str] | None = None) -> pd.DataFrame:
    """ISCO-08 unit group -> PCS-ESE codes: each code's ISCO probabilities times its full-time-equivalent employees,
    normalised per ISCO code."""
    pi = pcs_to_isco()
    pr = private_wages(); pu = public_wages()
    fte = pd.concat([pr[pr.year == pr.year.max()], pu[pu.year == pu.year.max()]]).groupby("pcs_code").fte.sum()
    codes = sorted(unit_codes) if unit_codes else sorted(fte.index)
    by3 = pi.assign(k=pi.pcs_code.str[:3]).groupby(["k", "isco_code"]).p.mean().reset_index()
    by2 = pi.assign(k=pi.pcs_code.str[:2]).groupby(["k", "isco_code"]).p.mean().reset_index()   # the PCS-ESE aggregates 210X and 220X (artisans and shopkeepers salaried in their own business)
    rows = []
    for c in codes:
        g = pi[pi.pcs_code == c]
        if g.empty:
            g = by3[by3.k == c[:3]].rename(columns={"k": "pcs_code"})
        if g.empty:
            g = by2[by2.k == c[:2]].rename(columns={"k": "pcs_code"})
        if g.empty:
            continue
        for r in g.itertuples():
            rows.append({"isco_code": r.isco_code, "nat_code": c, "w": float(r.p) * float(max(fte.get(c, 1.0), 1.0))})
    k = pd.DataFrame(rows)
    k["weight"] = k.w / k.groupby("isco_code").w.transform("sum")
    return k[["isco_code", "nat_code", "weight"]].reset_index(drop=True)


# ----------------------------------------------------------------------------- the firm survey and the mix
def load_ai_points() -> pd.DataFrame:
    """Eurostat's enterprise AI-use survey for France (enterprises with 10 or more persons employed), nationally and
    by NACE activity, 2021 to 2025; INSEE runs the same survey and publishes it on Melodi as DS_TICE."""
    from . import europe as eu
    return eu.ai_points("FR")


def load_industry_mix() -> pd.DataFrame:
    """Employed persons (thousands) by PCS group (one digit: the survey crosses activity with the six groups, not
    the 29 categories) and NACE section, labour force survey 2025, 15 and over. Sections the survey publishes only
    in parts (manufacturing, professional activities, health) are summed from their parts. Columns: major_group,
    industry, employed."""
    d = _melodi("melodi_eec_pcs_activity")
    d = d[d.PCS.str.fullmatch(r"[1-6]0")].copy()
    rows = []
    parts = {p: s for s, ps in _SECTION_PARTS.items() for p in ps}
    for r in d.dropna(subset=["value"]).itertuples():
        act = str(r.ACTIVITY)
        sec = act if re.fullmatch(r"[A-U]", act) else parts.get(act)
        if sec:
            rows.append({"major_group": str(r.PCS)[0], "industry": sec, "employed": float(r.value)})
    return pd.DataFrame(rows, columns=["major_group", "industry", "employed"]).groupby(["major_group", "industry"]).employed.sum().reset_index()


# ----------------------------------------------------------------------------- rung five (not testable yet), units, titles
def employment_change() -> pd.DataFrame:
    """Melodi carries one year of each wage table (private 2024, public 2023), so neither employment nor pay has a
    second reading per category yet: the table is kept in the shared shape with empty changes."""
    w = build_weights(); y = int(w.edition.iloc[0])
    df = pd.DataFrame({"pcs_code": w.pcs_code.values})
    df[f"emp_{y - 1}"] = np.nan; df[f"emp_{y}"] = w.employment_thousands.values * 1000.0; df["emp_change_pct"] = np.nan
    df[f"wage_{y - 1}"] = np.nan; df[f"wage_{y}"] = w.wage_monthly.values; df["wage_change_pct"] = np.nan
    df["year_from"], df["year_to"], df["wage_years"] = y - 1, y, "none"
    return df


def outcome_units() -> pd.DataFrame:
    w = build_weights()
    return pd.DataFrame({"pcs_code": w.pcs_code.values, "emp_unit": "full-time equivalents", "wage_unit": "month"})


def load_titles() -> pd.DataFrame:
    w = build_weights()
    return pd.DataFrame({"pcs_code": w.pcs_code, "title": w.pcs_title, "category": "principal"})


def no_tasks() -> pd.DataFrame:
    return pd.DataFrame(columns=["pcs_code", "task_seq", "task"])
