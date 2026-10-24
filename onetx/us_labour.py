"""United States labour tables: employment and wages by occupation from the BLS Occupational Employment
and Wage Statistics (OEWS), the occupation-by-sector mix for sector-specific adoption, and the year-on-year
employment change used as the US rung-five outcome.

OEWS codes occupations in SOC 2018, which O*NET-SOC 2019 is built on: an O*NET occupation's first seven
characters are its SOC code. Where several O*NET occupations share one SOC code (the .01-.09
specialisations), the code's employment is split evenly between them and each carries the code's wage,
the same convention the Singapore side uses for codes that only carry a group total.

Wages are annual medians in US dollars; OEWS prints "#" for medians at or above $239,200 (kept as that
floor) and "*" where none is published (left missing; the SOC minor group's median fills it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW
from . import io

BLS = RAW / "bls"
WAGE_TOP_CODE = 239200.0
# OEWS combined sectors -> the Census BTOS 2-digit sector codes used in ai_btos_sector_levels
OEWS_SECTOR_TO_BTOS = {"31-33": "31", "44-45": "44", "48-49": "48"}
SECTOR_LABELS = {
    "11": "Agriculture", "21": "Mining & oil and gas", "22": "Utilities", "23": "Construction", "31": "Manufacturing",
    "42": "Wholesale trade", "44": "Retail trade", "48": "Transportation & warehousing", "51": "Information",
    "52": "Finance & insurance", "53": "Real estate", "54": "Professional & technical services",
    "55": "Management of companies", "56": "Administrative & support services", "61": "Education",
    "62": "Health care & social assistance", "71": "Arts, entertainment & recreation", "72": "Accommodation & food services",
    "81": "Other services", "99": "Government",
}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def _wage(s: pd.Series) -> pd.Series:
    v = _num(s)
    return v.where(s.astype(str) != "#", WAGE_TOP_CODE)


def _oews_rows(year: int) -> pd.DataFrame:
    yy = str(year)[-2:]
    path = BLS / f"oesm{yy}nat" / f"national_M{year}_dl.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"{path} (run `python -m onetx fetch`)")
    d = pd.read_excel(path, dtype=str)
    return pd.DataFrame({"soc_2018": d.OCC_CODE.str.strip(), "soc_title": d.OCC_TITLE.str.strip(), "o_group": d.O_GROUP.str.strip(), "employment": _num(d.TOT_EMP), "wage_median_usd": _wage(d.A_MEDIAN), "wage_mean_usd": _wage(d.A_MEAN), "wage_top_coded": d.A_MEDIAN.astype(str) == "#", "year": year})


def load_oews_national(year: int = 2024) -> pd.DataFrame:
    """Detailed occupations (SOC 2018): employment and annual median / mean wages, May of `year`.

    OEWS publishes some SOC detailed codes only combined (a broad code such as 13-1020 carried as a
    detailed row, or an OEWS-specific code such as 21-1018). Codes O*NET uses that OEWS does not publish
    get the remainder of their broad group (broad total minus its published detailed codes), split evenly,
    with the broad group's median wage; when the remainder is nil the work already sits under a published
    combined code and the unpublished code stays at zero."""
    rows = _oews_rows(year)
    det = rows[rows.o_group == "detailed"].drop(columns=["o_group"]).drop_duplicates("soc_2018").copy()
    det["wage_source"] = np.where(det.wage_median_usd.notna(), "oews_median", "minor_group_median")
    det["minor"] = det.soc_2018.str[:4]
    med = det.groupby("minor").wage_median_usd.median()
    det["wage_median_usd"] = det.wage_median_usd.fillna(det.minor.map(med))
    det = det.drop(columns=["minor"])
    try:
        onet_socs = io.load("xw_onetsoc_soc2018").soc_2018.drop_duplicates()
    except Exception:
        onet_socs = pd.Series(dtype=str)
    missing = onet_socs[~onet_socs.isin(det.soc_2018)]
    extra = []
    if len(missing):
        broad = rows[rows.o_group == "broad"].drop_duplicates("soc_2018").set_index("soc_2018")
        pub = det.set_index("soc_2018")
        for bcode, grp in missing.groupby(missing.str[:6] + "0"):
            if bcode not in broad.index:
                continue
            kids = pub[pub.index.str.startswith(bcode[:6]) & (pub.index != bcode)]
            remainder = float(broad.loc[bcode, "employment"] or 0) - float(kids.employment.sum())
            share = max(remainder, 0.0) / len(grp)
            for code in grp:
                extra.append({"soc_2018": code, "soc_title": f"{broad.loc[bcode, 'soc_title']} (unpublished detail, broad remainder)", "employment": share, "wage_median_usd": broad.loc[bcode, "wage_median_usd"], "wage_mean_usd": broad.loc[bcode, "wage_mean_usd"], "wage_top_coded": bool(broad.loc[bcode, "wage_top_coded"]), "year": year, "wage_source": "broad_group_median"})
    out = pd.concat([det, pd.DataFrame(extra)], ignore_index=True) if extra else det
    return out.reset_index(drop=True)


def load_oews_sector(year: int = 2024) -> pd.DataFrame:
    """Employment by detailed occupation and NAICS sector (BTOS 2-digit codes), May of `year`."""
    yy = str(year)[-2:]
    path = BLS / f"oesm{yy}in4" / f"natsector_M{year}_dl.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"{path} (run `python -m onetx fetch`)")
    d = pd.read_excel(path, dtype=str, usecols=["NAICS", "NAICS_TITLE", "OCC_CODE", "O_GROUP", "TOT_EMP"])
    d = d[d.O_GROUP == "detailed"].copy()
    out = pd.DataFrame({"soc_2018": d.OCC_CODE.str.strip(), "sector": d.NAICS.str.strip().map(lambda x: OEWS_SECTOR_TO_BTOS.get(x, x)), "sector_title": d.NAICS_TITLE.str.strip(), "employment": _num(d.TOT_EMP)})
    out = out.dropna(subset=["employment"])
    out["year"] = year
    return out.reset_index(drop=True)


def build_onet_weights(year: int = 2024) -> pd.DataFrame:
    """One row per O*NET occupation: US employment (thousands), annual median wage, wage value, and the
    ISCO-08 major group it mostly maps to (so the two countries can be compared on the same groups)."""
    nat = load_oews_national(year)
    o2soc = io.load("xw_onetsoc_soc2018")[["onet_soc_code", "onet_title", "soc_2018"]].copy()
    n_per_soc = o2soc.groupby("soc_2018").onet_soc_code.transform("size")
    o2soc["n_onet_in_soc"] = n_per_soc
    t = o2soc.merge(nat, on="soc_2018", how="left")
    t["employment_thousands"] = t.employment / 1000.0 / t.n_onet_in_soc
    t["employment_method"] = np.where(t.employment.notna(), np.where(t.n_onet_in_soc > 1, "soc_total_split_evenly", "soc_total"), "not_in_oews")
    t["wage_median"] = t.wage_median_usd
    t["wage_value"] = t.employment_thousands * t.wage_median
    o2i = io.load("xw_onetsoc_isco08")[["onet_soc_code", "isco_code", "weight"]].copy()
    o2i["major_group"] = o2i.isco_code.astype(str).str[0]
    mg = o2i.groupby(["onet_soc_code", "major_group"]).weight.sum().reset_index().sort_values(["onet_soc_code", "weight"], ascending=[True, False]).drop_duplicates("onet_soc_code")
    t = t.merge(mg[["onet_soc_code", "major_group"]], on="onet_soc_code", how="left")
    # military and a few residual codes have no ISCO route; keep them out of group tables but in the total
    t["major_group"] = t.major_group.fillna("X")
    # SOC minor groups (the US counterpart of Singapore's sub-major groups), named from the OEWS file
    minor = load_oews_minor_groups(year)
    t["submajor_label"] = t.soc_2018.str[:4] + "000"   # SOC minor group: major digits plus the first digit after the hyphen
    t["submajor_name"] = t.submajor_label.map(minor.set_index("soc_minor").title)
    t["edition"] = year
    cols = ["onet_soc_code", "onet_title", "soc_2018", "soc_title", "major_group", "submajor_label", "submajor_name", "employment_thousands", "employment_method", "wage_median", "wage_source", "wage_top_coded", "wage_value", "edition"]
    return t[cols].reset_index(drop=True)


def load_oews_minor_groups(year: int = 2024) -> pd.DataFrame:
    """SOC minor groups (codes like 11-1000) with titles and employment, from the OEWS national file."""
    rows = _oews_rows(year)
    m = rows[rows.o_group == "minor"].drop_duplicates("soc_2018")
    return pd.DataFrame({"soc_minor": m.soc_2018.values, "title": m.soc_title.values, "employment": m.employment.values, "year": year})


# ---- career clusters and the DOL industry competency models (the US stand-ins for Skills Framework sectors)
REF = RAW.parent / "reference"


def load_career_clusters() -> pd.DataFrame:
    """O*NET OnLine's 14 career clusters (National Career Clusters Framework) to O*NET occupations, from
    data/reference/onet_career_clusters_2026.csv (collected from the website; not in the database download)."""
    path = REF / "onet_career_clusters_2026.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    d = pd.read_csv(path, dtype=str)
    return d[["cluster_code", "cluster", "onet_soc_code", "onet_title"]].drop_duplicates(["cluster_code", "onet_soc_code"]).reset_index(drop=True)


def load_cmc_cluster_crosswalk() -> pd.DataFrame:
    path = REF / "cmc_models_to_career_clusters.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def _sql_rows(path) -> pd.DataFrame:
    """Rows from an Oracle-style script of 'Insert into "t" ("a", "b") Values (1, 'x');' statements,
    using each statement's own column list (NULL columns are sometimes left out)."""
    import re

    s = open(path, encoding="utf-8", errors="ignore").read()
    out = []
    for m in re.finditer(r"Insert\s+into\s+\"\w+\"\s*\(([^)]*)\)\s*Values\s*\((.*?)\);", s, flags=re.S | re.I):
        cols = re.findall(r'"(\w+)"', m.group(1))
        vals = re.findall(r"'((?:[^']|'')*)'|(-?\d+(?:\.\d+)?)|(NULL)", m.group(2))
        vals = [v[0].replace("''", "'") if v[0] else (v[1] if v[1] else None) for v in vals]
        out.append(dict(zip(cols, vals)))
    return pd.DataFrame(out)


def load_competency_models() -> dict:
    """The Competency Model Clearinghouse database (DOL ETA, Sept 2025) from its SQL scripts: models,
    tiers, competency blocks, key behaviours and the model-to-block links."""
    base = RAW / "dol"
    folders = [p for p in base.glob("CMC Data Downloads*") if p.is_dir()] if base.exists() else []
    if not folders:
        raise FileNotFoundError(f"{base}/CMC Data Downloads* (run `python -m onetx fetch`)")
    d = folders[0]
    models = _sql_rows(d / "1-cmc_models.sql").reindex(columns=["model_id", "model_name"])
    tiers = _sql_rows(d / "4-tiers.sql").reindex(columns=["tier_id", "tier_name", "tier_desc", "tier_type_id"])
    blocks = _sql_rows(d / "5-blocks.sql").reindex(columns=["tier_id", "block_id", "block_name", "block_desc"])
    kb = _sql_rows(d / "6-keybehaviors.sql").reindex(columns=["block_id", "kb_level_id", "kb_id", "kb_name", "kb_desc", "kb_sort_id"])
    mb = _sql_rows(d / "7-cmc_model_blocks.sql").reindex(columns=["model_id", "tier_id", "block_id"])
    for df, cols in [(models, ["model_id"]), (tiers, ["tier_id", "tier_type_id"]), (blocks, ["tier_id", "block_id"]), (kb, ["block_id", "kb_level_id", "kb_id", "kb_sort_id"]), (mb, ["model_id", "tier_id", "block_id"])]:
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    tiers["tier_type"] = tiers.tier_type_id.map({1: "foundation", 2: "industry", 3: "occupation"})
    import re as _re
    kb["kb_desc"] = kb.kb_desc.fillna("").map(lambda x: _re.sub(r"<[^>]+>", " ", x).strip())
    blocks["block_desc"] = blocks.block_desc.fillna("").map(lambda x: _re.sub(r"<[^>]+>", " ", x).strip())
    return {"models": models, "tiers": tiers, "blocks": blocks, "key_behaviors": kb, "model_blocks": mb}


def onet_sector_shifts(sector_levels: pd.DataFrame, year: int = 2024) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(onet_shift, mix): years each O*NET occupation's industries run ahead (negative) or behind the US
    national adoption curve, the employment-weighted mean of its OEWS sectors' Census shifts; sectors the
    Census does not survey (government) sit at the national level (0)."""
    sec = load_oews_sector(year)
    lv = sector_levels[["sector", "anchor_shift_years"]].copy()
    lv["sector"] = lv.sector.astype(str)
    sec = sec.merge(lv, on="sector", how="left")
    sec["anchor_shift_years"] = sec.anchor_shift_years.fillna(0.0)
    tot = sec.groupby("soc_2018").employment.transform("sum")
    sec["share"] = sec.employment / tot.where(tot > 0, np.nan)
    sec = sec.dropna(subset=["share"])
    mix = sec[["soc_2018", "sector", "share"]].copy()
    by_soc = sec.assign(_x=sec.share * sec.anchor_shift_years).groupby("soc_2018")._x.sum().rename("adoption_shift_years").reset_index()
    o2soc = io.load("xw_onetsoc_soc2018")[["onet_soc_code", "soc_2018"]]
    out = o2soc.merge(by_soc, on="soc_2018", how="left")
    out["adoption_shift_years"] = out.adoption_shift_years.fillna(0.0)
    return out[["onet_soc_code", "soc_2018", "adoption_shift_years"]], mix


def employment_change(year_from: int = 2023, year_to: int = 2024) -> pd.DataFrame:
    """Per SOC code: employment and median wage in both years and the percentage changes (OEWS May)."""
    a = load_oews_national(year_from)[["soc_2018", "employment", "wage_median_usd"]].rename(columns={"employment": f"emp_{year_from}", "wage_median_usd": f"wage_{year_from}"})
    b = load_oews_national(year_to)[["soc_2018", "soc_title", "employment", "wage_median_usd"]].rename(columns={"employment": f"emp_{year_to}", "wage_median_usd": f"wage_{year_to}"})
    d = b.merge(a, on="soc_2018", how="inner")
    d["emp_change_pct"] = 100.0 * (d[f"emp_{year_to}"] / d[f"emp_{year_from}"].where(d[f"emp_{year_from}"] > 0) - 1.0)
    d["wage_change_pct"] = 100.0 * (d[f"wage_{year_to}"] / d[f"wage_{year_from}"].where(d[f"wage_{year_from}"] > 0) - 1.0)
    d["year_from"] = year_from
    d["year_to"] = year_to
    return d
