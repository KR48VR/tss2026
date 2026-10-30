"""Sector-specific adoption: a desk job in a bank and one in a restaurant chain should not share one
adoption date.

The US Census BTOS publishes the 'used AI in the last two weeks' share by NAICS sector. Holding the
national growth rate, each sector sits a measured number of years ahead of or behind the national curve
(`ai_signals.fit_sector_levels`). Those shifts are carried to Singapore industries (SSIC sections, as
MOM's Labour Force Survey groups them), then to occupations through their industry mix:

  SSOC occupation  =  0.5 x the industry mix of its major group (LFS Table D2, employed residents by
                      occupation group and industry) + 0.5 x the industries of the Skills Framework
                      roles tagged to that code (when any), so the Singapore-specific tagging refines
                      the official but coarse group mix
  O*NET occupation =  weight_rev-weighted mean over the SSOC codes it maps to
  Skills Framework role = the industry of its framework sector (cross-sector frameworks such as Human
                      Resource stay at the national level)

Shifts are in years at the central growth rate; the fast and slow cases divide them by their growth
multipliers. IMDA's 2023-2024 sector adoption rates are used only to check the ordering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW
from . import io

MOM = RAW / "singapore" / "mom"

# NAICS 2-digit sectors in the Census file -> MOM/SSIC industry groups used in LFS Table D2
NAICS_TO_INDUSTRY = {
    "11": "ABDE", "21": "ABDE", "22": "ABDE", "23": "F", "31": "C", "42": "G", "44": "G", "48": "H", "51": "J", "52": "K",
    "53": "L", "54": "M", "55": "M", "56": "N", "61": "OP", "62": "Q", "71": "R", "72": "I", "81": "SU",
}
INDUSTRY_LABELS = {
    "C": "Manufacturing", "F": "Construction", "G": "Wholesale & Retail Trade", "H": "Transportation & Storage",
    "I": "Accommodation & Food Services", "J": "Information & Communications", "K": "Financial & Insurance Services",
    "L": "Real Estate Services", "M": "Professional Services", "N": "Administrative & Support Services",
    "OP": "Public Administration & Education", "Q": "Health & Social Services", "R": "Arts, Entertainment & Recreation",
    "SU": "Other Community, Social & Personal Services", "ABDE": "Agriculture, Mining, Utilities",
}
# Skills Framework sector -> industry group (None = cross-sector framework, national level)
SFW_SECTOR_TO_INDUSTRY = {
    "Accountancy": "M", "Aerospace": "C", "Agrifood": "ABDE", "Air Transport": "H", "Arts": "R", "BioPharmaceuticals Manufacturing": "C",
    "Built Environment": "F", "Carbon Services and Trading": "M", "Design": "M", "Early Childhood": "OP", "Electronics": "C",
    "Energy and Chemicals": "C", "Energy and Power": "ABDE", "Engineering Services": "M", "Environmental Services": "N",
    "Financial Services": "K", "Food Manufacturing": "C", "Food Services": "I", "Healthcare": "Q", "Hotel and Accommodation Services": "I",
    "Human Resource": None, "Infocomm Technology": "J", "Intellectual Property": "M", "Landscape": "N", "Legal Services": "M",
    "Logistics": "H", "Marine and Offshore": "C", "Media": "J", "Precision Engineering": "C", "Public Transport": "H", "Retail": "G",
    "Sea Transport": "H", "Security": "N", "Social Service": "Q", "Tourism": "R", "Trade Associations and Chambers": "SU",
    "Training and Adult Education": "OP", "Wholesale Trade": "G", "Workplace Safety and Health": None,
}
# LFS Table D2 column order -> SSOC major group (first digit); 'Others' holds agriculture and armed forces
D2_MAJOR_COLUMNS = ["1", "2", "3", "4", "5", "7", "8", "9", "X"]
D2_INDUSTRY_ROWS = {
    "C Manufacturing": "C", "F Construction": "F", "G Wholesale & Retail Trade": "G", "H Transportation & Storage": "H",
    "I Accommodation & Food Services": "I", "J Information & Communications": "J", "K Financial & Insurance Services": "K",
    "L Real Estate Services": "L", "M Professional Services": "M", "N Administrative & Support Service": "N",
    "O,P Public Administration & Educat": "OP", "Q Health & Social Services": "Q", "R Arts, Entertainment & Recreation": "R",
    "S-U Other Community, Social & Pers": "SU", "A, B, D, E Others": "ABDE",
}


def load_lfs_industry_by_major(edition: int = 2025) -> pd.DataFrame:
    """Employed residents (thousands) by industry group and SSOC major group, LFS Table D2, both sexes."""
    import openpyxl

    wb = openpyxl.load_workbook(MOM / f"LFR{edition}_SectionD.xlsx", read_only=True, data_only=True)
    rows = list(wb["D2"].iter_rows(values_only=True))
    out = []
    in_total = False
    for r in rows:
        a, b = (str(r[0]).strip() if r[0] is not None else ""), (str(r[1]).strip() if r[1] is not None else "")
        if a == "Total" and b == "Total":
            in_total = True
            continue
        if in_total and a in ("Male", "Female"):
            break
        if not in_total:
            continue
        key = next((k for k in D2_INDUSTRY_ROWS if b.startswith(k[:22])), None)
        if key is None:
            continue
        for j, mg in enumerate(D2_MAJOR_COLUMNS):
            v = r[3 + j]
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = 0.0
            out.append({"industry": D2_INDUSTRY_ROWS[key], "major_group": mg, "employed_thousands": v})
    df = pd.DataFrame(out)
    df["edition"] = edition
    return df


def industry_shifts(sector_levels: pd.DataFrame) -> pd.DataFrame:
    """Years each Singapore industry group sits ahead (negative) or behind the national adoption curve:
    the mean of its NAICS sectors' shifts. Industries with no surveyed sector get 0."""
    s = sector_levels[sector_levels.sector.isin(NAICS_TO_INDUSTRY)].copy()
    s["industry"] = s.sector.map(NAICS_TO_INDUSTRY)
    g = s.groupby("industry").agg(anchor_shift_years=("anchor_shift_years", "mean"), naics=("sector", lambda x: ",".join(sorted(x))), latest_pct=("latest_pct", "mean")).reset_index()
    all_ind = pd.DataFrame({"industry": list(INDUSTRY_LABELS)})
    g = all_ind.merge(g, on="industry", how="left")
    g["anchor_shift_years"] = g.anchor_shift_years.fillna(0.0)
    g["label"] = g.industry.map(INDUSTRY_LABELS)
    return g.sort_values("anchor_shift_years").reset_index(drop=True)


def ssoc_industry_mix(d2: pd.DataFrame, roles: pd.DataFrame, role_links: pd.DataFrame, ssoc_codes: pd.Series, blend: float = 0.5) -> pd.DataFrame:
    """Long table (ssoc_code, industry, share): the major-group mix from Table D2 blended with the
    industries of the Skills Framework roles tagged to the code."""
    mg = d2.groupby(["major_group", "industry"]).employed_thousands.sum().reset_index()
    mg["share"] = mg.employed_thousands / mg.groupby("major_group").employed_thousands.transform("sum")
    mg = mg[mg.share.notna()]
    rl = role_links.merge(roles[["role_id", "sector"]], on="role_id", how="left")
    rl["industry"] = rl.sector.map(SFW_SECTOR_TO_INDUSTRY)
    rl = rl.dropna(subset=["industry"])
    sf = rl.groupby(["ssoc_code", "industry"]).weight.sum().reset_index()
    sf["share_sfw"] = sf.weight / sf.groupby("ssoc_code").weight.transform("sum")
    out = []
    codes = pd.Series(ssoc_codes).astype(str).unique()
    sf_by = {k: g for k, g in sf.groupby("ssoc_code")}
    mg_by = {k: g for k, g in mg.groupby("major_group")}
    for code in codes:
        base = mg_by.get(code[0], mg_by.get("X"))
        parts = {}
        if base is not None:
            for r in base.itertuples():
                parts[r.industry] = parts.get(r.industry, 0.0) + (1.0 - blend if code in sf_by else 1.0) * r.share
        if code in sf_by:
            for r in sf_by[code].itertuples():
                parts[r.industry] = parts.get(r.industry, 0.0) + blend * r.share_sfw
        tot = sum(parts.values()) or 1.0
        for ind, sh in parts.items():
            out.append({"ssoc_code": code, "industry": ind, "share": sh / tot, "has_sfw_tags": code in sf_by})
    return pd.DataFrame(out)


def occupation_shifts(mix: pd.DataFrame, ind_shift: pd.DataFrame, o2s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ssoc_shift, onet_shift): years ahead/behind the national curve per SSOC code and per O*NET occupation."""
    m = mix.merge(ind_shift[["industry", "anchor_shift_years"]], on="industry", how="left")
    m["anchor_shift_years"] = m.anchor_shift_years.fillna(0.0)
    ssoc = m.assign(_x=m.share * m.anchor_shift_years).groupby("ssoc_code").agg(adoption_shift_years=("_x", "sum"), has_sfw_tags=("has_sfw_tags", "first")).reset_index()
    o = o2s[["onet_soc_code", "ssoc_code", "weight_rev"]].merge(ssoc[["ssoc_code", "adoption_shift_years"]], on="ssoc_code", how="left")
    o["adoption_shift_years"] = o.adoption_shift_years.fillna(0.0)
    o["w"] = o.weight_rev.fillna(0.0)
    onet = o.assign(_x=o.w * o.adoption_shift_years).groupby("onet_soc_code").apply(lambda g: pd.Series({"adoption_shift_years": (g._x.sum() / g.w.sum()) if g.w.sum() > 0 else 0.0}), include_groups=False).reset_index()
    return ssoc, onet


def role_shifts(roles: pd.DataFrame, ind_shift: pd.DataFrame) -> pd.DataFrame:
    r = roles[["role_id", "sector"]].copy()
    r["industry"] = r.sector.map(SFW_SECTOR_TO_INDUSTRY)
    r = r.merge(ind_shift[["industry", "anchor_shift_years"]], on="industry", how="left")
    r["adoption_shift_years"] = r.anchor_shift_years.fillna(0.0)
    return r[["role_id", "sector", "industry", "adoption_shift_years"]]


def build_shifts(sector_levels: pd.DataFrame) -> dict:
    """All shift tables from the processed tables; returns dict of DataFrames (industry, ssoc, onet, role, mix)."""
    roles = io.load("sfw_roles")
    links = io.load("xw_sfw_role_ssoc2024")
    o2s = io.load("xw_onetsoc_ssoc2024")
    weights = io.load("sg_ssoc2024_weights")
    ind = industry_shifts(sector_levels)
    if (MOM / "LFR2025_SectionD.xlsx").exists() or not io.exists("sg_ssoc2024_industry_mix"):
        d2 = load_lfs_industry_by_major(2025)
        codes = pd.concat([weights.ssoc_code.astype(str), o2s.ssoc_code.astype(str), links.ssoc_code.astype(str)]).unique()
        mix = ssoc_industry_mix(d2, roles, links, pd.Series(codes))
    else:   # MOM's server refuses the download from this address: the mix built from Table D2 in an earlier run is kept
        d2 = None
        mix = io.load("sg_ssoc2024_industry_mix")
    ssoc, onet = occupation_shifts(mix, ind, o2s)
    role = role_shifts(roles, ind)
    return {"industry": ind, "ssoc": ssoc, "onet": onet, "role": role, "mix": mix, "d2": d2}


def check_against_imda(ind_shift: pd.DataFrame, sg: pd.DataFrame) -> dict:
    """Rank agreement between the US-derived industry ordering and IMDA's 2023 sector adoption rates."""
    imda_map = {"sector:information_communications": "J", "sector:finance_insurance": "K", "sector:professional_services": "M",
                "sector:real_estate_admin_support": "L", "sector:retail_trade": "G", "sector:construction": "F", "sector:wholesale_trade": "G",
                "sector:manufacturing": "C", "sector:transport_storage": "H", "sector:accommodation_food": "I"}
    s = sg[(sg.year == 2023) & sg.group.isin(imda_map)].copy()
    s["industry"] = s.group.map(imda_map)
    s = s.groupby("industry").pct.mean().reset_index().merge(ind_shift[["industry", "anchor_shift_years"]], on="industry")
    if len(s) < 3:
        return {}
    rho = float(s[["pct", "anchor_shift_years"]].corr(method="spearman").iloc[0, 1])
    return {"n_industries": int(len(s)), "spearman_imda2023_vs_us_shift": rho, "note": "negative is agreement: higher IMDA adoption should pair with an earlier (negative) shift"}
