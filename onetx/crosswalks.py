"""Occupation crosswalks.

The chain we build:

    O*NET-SOC 2019  ->  SOC 2018            (official O*NET file)
    O*NET-SOC 2019  ->  ISCO-08             (two routes, combined: the O*NET/ESCO crosswalk, whose
                                             ESCO codes start with the ISCO-08 unit group; and the
                                             BLS SOC-2010 -> ISCO-08 table reached via O*NET's
                                             2010 -> 2019 crosswalk)
    ISCO-08         ->  SSOC 2024 / 2020    (SingStat correspondence tables)
    O*NET-SOC 2019  ->  SSOC 2024           (composition of the two above, with path weights)
    SFw job role    ->  SSOC 2024           (SingStat's own tagging of Skills Framework roles in the
                                             SSOC alphabetical index, joined to the SFw dataset)

All many-to-many links carry a `weight` that sums to 1 across the targets of each source code, so
that any score can be pushed through a crosswalk as a weighted average without double counting.
"""
from __future__ import annotations

import re

import pandas as pd
from rapidfuzz import fuzz, process

from . import singapore as sg
from .config import RAW

XW = RAW / "crosswalks"


def _norm_weights(df: pd.DataFrame, by: str, col: str = "weight") -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col] / df.groupby(by)[col].transform("sum")
    return df


# ----------------------------------------------------------------------------- O*NET-SOC -> SOC 2018
def onetsoc_to_soc2018() -> pd.DataFrame:
    df = pd.read_csv(XW / "onetsoc2019_to_soc2018.csv", dtype=str)
    df.columns = ["onet_soc_code", "onet_title", "soc_2018", "soc_2018_title"]
    return df


# ----------------------------------------------------------------------------- O*NET-SOC -> ISCO-08
def onetsoc_to_isco08() -> pd.DataFrame:
    """Combined two-route crosswalk. Columns: onet_soc_code, isco_code, weight, via_esco, via_soc2010, n_routes."""
    # Route A: ESCO crosswalk published by O*NET. ESCO occupation codes look like "2512.1" or "0110.10";
    # the digits before the dot are the ISCO-08 unit group.
    esco = pd.read_excel(XW / "esco_to_onetsoc2019.xlsx", header=3, dtype=str)
    esco.columns = ["esco_code", "esco_title", "onet_soc_code", "onet_title"]
    esco = esco.dropna(subset=["esco_code", "onet_soc_code"])
    esco["isco_code"] = esco["esco_code"].str.extract(r"^(\d{4})")[0]
    esco = esco.dropna(subset=["isco_code"]).drop_duplicates(["onet_soc_code", "isco_code"])
    esco["weight"] = 1.0
    a = _norm_weights(esco[["onet_soc_code", "isco_code", "weight"]], "onet_soc_code").rename(columns={"weight": "w_esco"})

    # Route B: O*NET-SOC 2019 -> O*NET-SOC 2010 -> SOC 2010 -> ISCO-08 (BLS table).
    w = pd.read_csv(XW / "onetsoc2010_to_onetsoc2019.csv", dtype=str)
    w.columns = ["onet_2010", "t2010", "onet_soc_code", "t2019"]
    w["soc_2010"] = w["onet_2010"].str[:7]
    s2i = pd.read_csv(XW / "soc2010_to_isco08.csv", dtype=str)
    s2i.columns = ["soc_2010", "soc_2010_title", "isco_code", "isco_title"]
    s2i["isco_code"] = s2i["isco_code"].str.zfill(4)
    b = w.merge(s2i, on="soc_2010", how="inner")[["onet_soc_code", "isco_code"]].drop_duplicates()
    b["weight"] = 1.0
    b = _norm_weights(b, "onet_soc_code").rename(columns={"weight": "w_soc2010"})

    out = a.merge(b, on=["onet_soc_code", "isco_code"], how="outer")
    out["via_esco"] = out["w_esco"].notna()
    out["via_soc2010"] = out["w_soc2010"].notna()
    out["n_routes"] = out["via_esco"].astype(int) + out["via_soc2010"].astype(int)
    out["weight"] = out[["w_esco", "w_soc2010"]].fillna(0).mean(axis=1)
    out = _norm_weights(out, "onet_soc_code")
    isco_titles = s2i.drop_duplicates("isco_code").set_index("isco_code").isco_title
    out["isco_title"] = out["isco_code"].map(isco_titles)
    return out[["onet_soc_code", "isco_code", "isco_title", "weight", "via_esco", "via_soc2010", "n_routes"]].sort_values(["onet_soc_code", "weight"], ascending=[True, False]).reset_index(drop=True)


# ----------------------------------------------------------------------------- ISCO-08 -> SSOC
def isco08_to_ssoc(edition: int = 2024) -> pd.DataFrame:
    """Union of SingStat's two correspondence directions. Weight = 1/n SSOC codes per ISCO unit group."""
    t = sg.load_ssoc_isco(edition)
    t = t[t.ssoc_code.str.fullmatch(r"\d{5}") & t.isco_code.str.fullmatch(r"\d{4}")]
    pairs = t.groupby(["isco_code", "ssoc_code"]).agg(part=("part", "max"), ssoc_title=("ssoc_title", "first"), isco_title=("isco_title", "first")).reset_index()
    pairs["weight"] = 1.0
    pairs = _norm_weights(pairs, "isco_code")
    pairs["edition"] = edition
    return pairs


# ----------------------------------------------------------------------------- O*NET-SOC -> SSOC (composed)
def onetsoc_to_ssoc(edition: int = 2024) -> pd.DataFrame:
    """Compose O*NET->ISCO and ISCO->SSOC. `weight` sums to 1 per O*NET occupation (push O*NET scores
    to Singapore); `weight_rev` sums to 1 per SSOC code (pull O*NET scores into each SSOC code)."""
    a = onetsoc_to_isco08()[["onet_soc_code", "isco_code", "weight"]].rename(columns={"weight": "w1"})
    b = isco08_to_ssoc(edition)[["isco_code", "ssoc_code", "ssoc_title", "weight"]].rename(columns={"weight": "w2"})
    m = a.merge(b, on="isco_code", how="inner")
    m["path_w"] = m["w1"] * m["w2"]
    g = m.groupby(["onet_soc_code", "ssoc_code"]).agg(ssoc_title=("ssoc_title", "first"), path_w=("path_w", "sum"), n_isco_paths=("isco_code", "nunique")).reset_index()
    g["weight"] = g["path_w"] / g.groupby("onet_soc_code").path_w.transform("sum")
    g["weight_rev"] = g["path_w"] / g.groupby("ssoc_code").path_w.transform("sum")
    g["edition"] = edition
    return g.drop(columns="path_w").sort_values(["onet_soc_code", "weight"], ascending=[True, False]).reset_index(drop=True)


# ----------------------------------------------------------------------------- Skills Framework role -> SSOC
def _norm_role(s: str) -> str:
    s = s.casefold().replace("–", "-").replace("—", "-")
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sfw_role_to_ssoc(edition: int = 2024, fuzzy_threshold: int = 92) -> pd.DataFrame:
    """Join SingStat's '(SFw-Sector)' index entries to Skills Framework roles.

    match_method: exact (sector+title), exact_with_track (title plus track in brackets), fuzzy
    (rapidfuzz token_set_ratio >= threshold within the sector). Roles the index does not list stay
    unmatched here and are picked up by the text matcher in onetx.match.
    """
    idx = sg.load_ssoc_alpha_index(edition)
    idx = idx[idx.sfw_sector.notna()].copy()
    idx["sector_key"] = idx["sfw_sector"].str.casefold().str.strip()
    idx["title_key"] = idx["sfw_role_title"].map(_norm_role)
    roles = sg.load_sfw_roles().copy()
    roles["sector_key"] = roles["sector"].str.casefold().str.strip()
    roles["title_key"] = roles["job_role"].map(_norm_role)
    roles["title_track_key"] = (roles["job_role"] + " (" + roles["track"].fillna("") + ")").map(_norm_role)

    out = []
    by_sector = {k: g for k, g in roles.groupby("sector_key")}
    for _, r in idx.iterrows():
        cands = by_sector.get(r.sector_key)
        if cands is None:
            out.append({**r.to_dict(), "role_id": None, "match_method": "sector_not_in_dataset", "match_score": None})
            continue
        hit = cands[cands.title_key == r.title_key]
        method = "exact"
        if hit.empty:
            hit = cands[cands.title_track_key == r.title_key]
            method = "exact_with_track"
        if hit.empty:
            best = process.extractOne(r.title_key, cands.title_key.tolist(), scorer=fuzz.token_set_ratio)
            if best and best[1] >= fuzzy_threshold:
                hit = cands[cands.title_key == best[0]]
                method = "fuzzy"
        if hit.empty:
            out.append({**r.to_dict(), "role_id": None, "match_method": "unmatched", "match_score": None})
        else:
            for _, h in hit.iterrows():
                out.append({**r.to_dict(), "role_id": h.role_id, "match_method": method, "match_score": 100 if method.startswith("exact") else None})
    df = pd.DataFrame(out)
    df = df[["role_id", "sfw_sector", "sfw_role_title", "ssoc_code", "match_method", "match_score", "edition"]]
    m = df[df.role_id.notna()].drop_duplicates(["role_id", "ssoc_code"]).copy()
    m["weight"] = 1.0
    m = _norm_weights(m, "role_id")
    return m.reset_index(drop=True), df[df.role_id.isna()].reset_index(drop=True)
