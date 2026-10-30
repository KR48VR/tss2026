"""New Zealand: the sixth workforce, on ANZSCO (shared with Australia), from the open Stats NZ files and, with a
Stats NZ API key, the Census 2023 tables on the Aotearoa Data Explorer.

Inputs (`data/raw/statsnz/`), the open files being Stats NZ data republished under CC BY 4.0 by Figure.NZ, which
keeps the original Stats NZ file beside its tidy CSV:
  census2023_occupation_sex_region.xlsx   2023 Census: employed people by ANZSCO unit group (level 4), sex
                                          and region (Stats NZ's own file; the "Total New Zealand" rows are used)
  census_occupation_2013_2018_2023.csv    2013, 2018 and 2023 Censuses: employed people by 6-digit occupation
                                          (rolled up to the unit group for the employment change)
  census_industry_2013_2018_2023.csv      the same by ANZSIC06 industry (employment shares by division)
  hlfs_earnings_by_occupation.csv         Household Labour Force Survey income supplement: median and average
                                          weekly and hourly earnings by ANZSCO major group, June years 2009-2026
  ade/CEN23_WRK_011.csv (with the key)    Aotearoa Data Explorer (`fetch.fetch_statsnz_ade`, STATSNZ_API_KEY):
                                          2023 Census total personal income by occupation sub-major group (two
                                          digits), with a median row per group; the pay weight of every unit group
                                          in the sub-major group (the Explorer publishes no unit-group income table)
  ade/CEN23_INC_007.csv (with the key)    the same by major group (a check, and the fallback for a missing group)
  ade/CEN23_WRK_008.csv (with the key)    2023 Census employed people by occupation major group and ANZSIC06
                                          industry division: the mix that carries industry shifts and the sector lens

What New Zealand does not have (October 2026): an official reading of the share of firms using AI (Stats NZ's
first Survey of Business Operations with an AI question reports later in 2026), so its lead on the adoption
curve is borrowed from Australia's survey and marked as such, and the industry shifts are Australia's readings
carried through New Zealand's own occupation-by-industry mix.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import au_labour as au
from .config import RAW

STATSNZ = RAW / "statsnz"
ADE = STATSNZ / "ade"


def load_census_employment() -> pd.DataFrame:
    """Employed people (count) by ANZSCO unit group, 2023 Census, all of New Zealand, both sexes."""
    df = pd.read_excel(STATSNZ / "census2023_occupation_sex_region.xlsx", sheet_name="RC", header=0, dtype=str)
    df = df[(df.geographyVars_label == "Total") & (df.i_sex_at_birth_code == "Total")]
    df = df[df.occupationVars.astype(str).str.fullmatch(r"\d{4}")].copy()
    out = pd.DataFrame({"anzsco_code": df.occupationVars.astype(str), "anzsco_title": df.occupationVars_label.astype(str).str.strip(), "employed": pd.to_numeric(df.Count, errors="coerce")})
    return out.dropna(subset=["employed"]).reset_index(drop=True)


def load_census_occupation_years() -> pd.DataFrame:
    """Employed people by unit group for the 2013, 2018 and 2023 Censuses (6-digit occupations summed up):
    columns anzsco_code, emp_2013, emp_2018, emp_2023."""
    c = pd.read_csv(STATSNZ / "census_occupation_2013_2018_2023.csv", dtype=str)
    c = c[c.Unit == "Count"].copy()
    c["code"] = c["Occupation Code"].astype(str)
    c = c[c.code.str.fullmatch(r"\d{6}")]
    c["anzsco_code"] = c.code.str[:4]
    c["value"] = pd.to_numeric(c.Value, errors="coerce")
    c["year"] = pd.to_numeric(c["Census Year"], errors="coerce").astype(int)
    piv = c.groupby(["anzsco_code", "year"]).value.sum(min_count=1).unstack()
    piv.columns = [f"emp_{int(y)}" for y in piv.columns]
    return piv.reset_index()


def load_census_industry() -> pd.DataFrame:
    """Employed people by ANZSIC06 division (one letter), 2023 Census: division, employed, share."""
    c = pd.read_csv(STATSNZ / "census_industry_2013_2018_2023.csv", dtype=str)
    c = c[(c.Unit == "Count") & (c["Census Year"] == "2023")].copy()
    c["code"] = c["Industry of employment Code"].astype(str)
    c = c[c.code.str.fullmatch(r"[A-S]\d{4}")]
    c["division"] = c.code.str[0]
    c["value"] = pd.to_numeric(c.Value, errors="coerce")
    g = c.groupby("division").value.sum().reset_index().rename(columns={"value": "employed"})
    g["share"] = g.employed / g.employed.sum()
    g["label"] = g.division.map(au.DIVISIONS)
    return g


def load_hlfs_earnings() -> pd.DataFrame:
    """Median and average weekly and hourly earnings from the main wage and salary job by ANZSCO major group,
    all people, year to June: major_group (digit), year, median_weekly, median_hourly, average_weekly."""
    h = pd.read_csv(STATSNZ / "hlfs_earnings_by_occupation.csv", dtype=str)
    h = h[(h.Sex == "Total") & (h["Age group"].str.lower() == "total") & (h["Ethnic group"] == "Total")].copy()
    h["major_group"] = h["Occupation Code"].astype(str)
    h = h[h.major_group.str.fullmatch(r"\d")]
    h["year"] = pd.to_numeric(h["Year as at June"], errors="coerce").astype(int)
    h["value"] = pd.to_numeric(h.Value, errors="coerce")
    piv = h.pivot_table(index=["major_group", "year"], columns="Measure", values="value", aggfunc="first").reset_index()
    piv = piv.rename(columns={"Median Weekly Earnings": "median_weekly", "Median Hourly Earnings": "median_hourly", "Average Weekly Earnings": "average_weekly", "Average Hourly Earnings": "average_hourly"})
    return piv[["major_group", "year", "median_weekly", "median_hourly", "average_weekly", "average_hourly"]].sort_values(["major_group", "year"]).reset_index(drop=True)


_BAND = re.compile(r"\$?([\d,]+)\s*(?:-|to|–)\s*\$?([\d,]+)|\$?([\d,]+)\s*(?:or more|and over|\+)|(loss|zero|nil)", re.I)


def _band_edges(label: str) -> tuple[float, float] | None:
    """The dollar range of a Census income band label ('$30,001-$35,000', '$150,001 or more', 'Zero income')."""
    m = _BAND.search(str(label))
    if not m:
        return None
    if m.group(4):
        return (0.0, 0.0)
    if m.group(3):
        lo = float(m.group(3).replace(",", ""))
        return (lo, lo * 1.5)
    return (float(m.group(1).replace(",", "")), float(m.group(2).replace(",", "")))


def has_ade() -> bool:
    """Whether the Data Explorer extracts (pay by sub-major group and the occupation-by-industry mix) are on disk."""
    return (ADE / "CEN23_WRK_011.csv").exists() and (ADE / "CEN23_WRK_008.csv").exists()


def _ade_frame(name: str) -> pd.DataFrame | None:
    """An Aotearoa Data Explorer extract (CSV with labels) as written by fetch.fetch_statsnz_ade; None when absent."""
    f = ADE / f"{name}.csv"
    if not f.exists():
        return None
    return pd.read_csv(f, dtype=str)


def _ade_totals(d: pd.DataFrame) -> pd.DataFrame:
    """The all-ages, all-genders rows of a Census extract (code 99 on the age and gender dimensions)."""
    age = next(c for c in d.columns if c.startswith("CEN23_AGE"))
    gen = next(c for c in d.columns if c.startswith("CEN23_GEN"))
    return d[(d[age] == "99") & (d[gen] == "99")]


def census_income_medians() -> dict | None:
    """Median total personal income (NZ$ a year, 2023 Census) per occupation group from the Data Explorer: the
    Explorer carries a 'Median ($)' row per group in its income tables, so no interpolation of bands is needed.
    Returns {'submajor': Series by two-digit code, 'major': Series by one-digit code}, or None without the extracts."""
    sub = _ade_frame("CEN23_WRK_011")
    maj = _ade_frame("CEN23_INC_007")
    if sub is None and maj is None:
        return None
    out = {}
    for key, d, occ, width in (("submajor", sub, "CEN23_OCC_003", 2), ("major", maj, "CEN23_OCC_002", 1)):
        if d is None:
            continue
        t = _ade_totals(d)
        t = t[t["Total personal income"].str.startswith("Median") & t[occ].str.fullmatch(r"\d{%d}" % width)]
        out[key] = pd.to_numeric(t.set_index(occ).OBS_VALUE, errors="coerce").dropna()
    return out


def census_income_median(ade_file=ADE / "CEN23_INC_007.csv") -> pd.DataFrame | None:
    """Kept for the band-interpolation route (a unit-group income table, should Stats NZ publish one on the
    Explorer): median total personal income per unit group from income bands, else None."""
    if not ade_file.exists():
        return None
    d = pd.read_csv(ade_file, dtype=str)
    occ_col = next((c for c in d.columns if "occupation" in c.lower() and "code" in c.lower()), None) or next(c for c in d.columns if "OCC" in c.upper())
    inc_col = next(c for c in d.columns if "income" in c.lower() and "code" not in c.lower())
    val_col = next(c for c in d.columns if c.upper() in ("OBS_VALUE", "VALUE"))
    d["code"] = d[occ_col].astype(str).str.strip()
    d = d[d.code.str.fullmatch(r"\d{4}")].copy()
    if d.empty:
        return None
    d["value"] = pd.to_numeric(d[val_col], errors="coerce")
    d["edges"] = d[inc_col].map(_band_edges)
    d = d.dropna(subset=["edges", "value"])
    rows = []
    for code, g in d.groupby("code"):
        g = g.assign(lo=g.edges.str[0], hi=g.edges.str[1]).sort_values("lo")
        tot = g.value.sum()
        if tot <= 0:
            continue
        cum = 0.0
        for r in g.itertuples():
            if cum + r.value >= tot / 2:
                frac = (tot / 2 - cum) / r.value if r.value else 0.0
                rows.append({"anzsco_code": code, "income_median": r.lo + frac * (r.hi - r.lo), "n": tot})
                break
            cum += r.value
    return pd.DataFrame(rows)


def build_anzsco_weights() -> pd.DataFrame:
    """Per unit group: 2023 Census employment (thousands), weekly pay (the 2023 Census median total personal
    income of the unit group's sub-major group divided by 52 where the Data Explorer extract exists, 43 figures;
    else the HLFS median weekly earnings of the major group, eight figures), their product as the wage weight,
    skill level and group names from the ANZSCO structure."""
    emp = load_census_employment()
    # titles, group names and skill levels as the Australian build carries them for the same 358 codes
    # (the two countries' Census and labour force codes coincide), else from the ANZSCO structure
    from . import io
    ref = io.load("au_weights").drop_duplicates("anzsco_code").set_index("anzsco_code") if io.exists("au_weights") else None
    st = au.load_anzsco_structure().drop_duplicates("anzsco_code").set_index("anzsco_code")
    df = emp.copy()
    df["employment_thousands"] = df.employed / 1000.0
    df["major_group"] = df.anzsco_code.str[0]
    df["submajor_label"] = df.anzsco_code.str[:2]
    sub_names = st.reset_index().drop_duplicates("submajor_label").set_index("submajor_label").submajor_name if "submajor_label" in st.columns else st.reset_index().assign(submajor_label=lambda d: d.anzsco_code.str[:2]).drop_duplicates("submajor_label").set_index("submajor_label").get("submajor_name", pd.Series(dtype=str))
    df["submajor_name"] = df.submajor_label.map(sub_names)
    df["skill_level"] = df.anzsco_code.map(st.skill_level)
    title = df.anzsco_code.map(st.anzsco_title)
    if ref is not None:
        df["submajor_name"] = df.submajor_name.where(df.submajor_name.notna(), df.anzsco_code.map(ref.submajor_name))
        df["skill_level"] = df.skill_level.where(df.skill_level.notna(), df.anzsco_code.map(ref.skill_level))
        title = title.where(title.notna(), df.anzsco_code.map(ref.anzsco_title))
    df["anzsco_title"] = title.where(title.notna(), df.anzsco_title)
    h = load_hlfs_earnings()
    latest = int(h.year.max())
    mg = h[h.year == latest].set_index("major_group").median_weekly
    df["wage_weekly"] = df.major_group.map(mg)
    df["wage_source"] = f"hlfs_{latest}_major_group_median_weekly"
    meds = census_income_medians()
    if meds:
        for key, width, label in (("major", 1, "census_2023_major_group_median_income"), ("submajor", 2, "census_2023_submajor_group_median_income")):
            if key not in meds:
                continue
            m = meds[key] / 52.0
            has = df.anzsco_code.str[:width].isin(m.index)
            df.loc[has, "wage_weekly"] = df.loc[has, "anzsco_code"].str[:width].map(m)
            df.loc[has, "wage_source"] = label
    med = census_income_median()   # a unit-group income table, if Stats NZ ever publishes one on the Explorer
    if med is not None and len(med):
        m = med.set_index("anzsco_code").income_median / 52.0
        has = df.anzsco_code.isin(m.index)
        df.loc[has, "wage_weekly"] = df.loc[has, "anzsco_code"].map(m)
        df.loc[has, "wage_source"] = "census_2023_unit_group_median_income"
    df["wage_value"] = df.employment_thousands * df.wage_weekly
    df["employment_method"] = "census_2023"
    df["edition"] = 2023
    return df[["anzsco_code", "employment_thousands", "anzsco_title", "major_group", "submajor_label", "submajor_name", "skill_level", "wage_weekly", "wage_source", "wage_value", "employment_method", "edition"]].reset_index(drop=True)


def employment_change() -> pd.DataFrame:
    """Rung five for New Zealand: employment change between the 2018 and 2023 Censuses per unit group (Stats NZ
    changed its occupation coding in 2023, so small changes are not meaningful), and the change in the major
    group's median weekly earnings over the latest two June years (HLFS)."""
    y = load_census_occupation_years()
    df = y[["anzsco_code", "emp_2018", "emp_2023"]].copy()
    df["emp_change_pct"] = 100.0 * (df.emp_2023 / df.emp_2018.where(df.emp_2018 > 0) - 1.0)
    h = load_hlfs_earnings()
    years = sorted(h.year.unique())[-2:]
    w = h[h.year.isin(years)].pivot(index="major_group", columns="year", values="median_weekly")
    df["major_group"] = df.anzsco_code.str[0]
    df[f"wage_{years[0]}"] = df.major_group.map(w[years[0]])
    df[f"wage_{years[1]}"] = df.major_group.map(w[years[1]])
    df["wage_change_pct"] = 100.0 * (df[f"wage_{years[1]}"] / df[f"wage_{years[0]}"].where(df[f"wage_{years[0]}"] > 0) - 1.0)
    df["year_from"], df["year_to"] = 2018, 2023
    df["wage_years"] = f"{years[0]}-{years[1]}"
    return df.drop(columns=["major_group"]).reset_index(drop=True)


def outcome_units() -> pd.DataFrame:
    """The outcome cells' units: Census employment is a count of people, not thousands; earnings are weekly."""
    e = load_census_employment()
    return pd.DataFrame({"anzsco_code": e.anzsco_code.values, "emp_unit": "people", "wage_unit": "week"})


def load_descriptions() -> pd.DataFrame:
    """The ABS's description and skill-level text per ANZSCO unit group (the classification New Zealand shares)."""
    return au.load_anzsco_pages().drop_duplicates("anzsco_code")[["anzsco_code", "description", "skill_text"]].reset_index(drop=True)


def load_titles() -> pd.DataFrame:
    """Alternative titles and specialisations per unit group from the ABS title index, for the page's search."""
    t = au.load_title_index()
    return t.rename(columns={"unit": "anzsco_code"})[["anzsco_code", "title", "category"]].drop_duplicates(["anzsco_code", "title"]).reset_index(drop=True)


def load_ai_points() -> pd.DataFrame:
    """No New Zealand firm survey has asked about AI use yet (Stats NZ's Survey of Business Operations reports
    later in 2026): an empty table, so the lead is borrowed from the proxy country named on the record."""
    return pd.DataFrame(columns=["date", "period", "pct", "group_type", "industry", "label"])


def load_industry_mix() -> pd.DataFrame:
    """Employed people by occupation major group and ANZSIC06 industry division (2023 Census, all ages and
    genders) when the Data Explorer extract is present; otherwise an empty mix, so no industry shifts are applied."""
    d = _ade_frame("CEN23_WRK_008")
    if d is None:
        return pd.DataFrame(columns=["major_group", "industry", "employed"])
    t = _ade_totals(d).copy()
    t["major_group"] = t.CEN23_OCC_002.astype(str)
    t["industry"] = t.CEN23_IND_003.astype(str)
    t["employed"] = pd.to_numeric(t.OBS_VALUE, errors="coerce")
    t = t[t.major_group.str.fullmatch(r"[1-8]") & t.industry.str.fullmatch(r"[A-S]")]
    return t.groupby(["major_group", "industry"]).employed.sum().reset_index()
