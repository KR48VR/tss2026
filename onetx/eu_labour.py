"""Sweden, Finland, Denmark, the Netherlands and Ireland from their statistics offices' open APIs
(`fetch.fetch_further_offices`), each with the same four ingredients where the office publishes them:
employment by detailed occupation, pay at the same detail, a firm survey of AI use by industry, and the
occupation-by-industry mix (the EU Labour Force Survey's, from `europe.load_mix`).

Depth, as marked on the page:
  Sweden    full    register employees and median monthly salary by four-digit SSYK 2012 (430 occupations);
                    AI use since 2021 by industry; the link to O*NET runs through SCB's own key, which is
                    written at the ISCO minor-group (three-digit) level
  Finland   full    register employed and median hourly earnings by four-digit occupation (ISCO-08 codes);
                    AI use 2021-2025 by industry
  Denmark   medium  hourly earnings by four-digit DISCO-08 and AI use 2023-2026 by industry, but employment by
                    detailed occupation is not in the open API: each two-digit group's EU-LFS employment is
                    split across its four-digit occupations the way Norway's register splits the same codes
  Netherlands medium employed and median hourly wage for the 114 BRC 2014 occupation groups (between two
                    and three ISCO digits), AI use 2023-2025 by industry; ROA and CBS publish the ISCO
                    composition of every group, which gives the link to O*NET
  Ireland   medium  Census 2022 labour force by 329 detailed occupations (UK SOC 2010 codes, linked through
                    the ONS mapping), pay only by major group (Eurostat SES), AI use 2023-2025 nationally
                    with no industry breakdown, so no industry shifts
"""
from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import europe as eu
from .config import RAW
from .no_labour import MAJOR_GROUP_NAMES, SKILL_LEVEL_NAMES, _skill_level, jsonstat_frame

SURVEY_DATE_IN_YEAR = 0.4
_DIV_TO_SECTION = [(1, 3, "A"), (5, 9, "B"), (10, 33, "C"), (35, 35, "D"), (36, 39, "E"), (41, 43, "F"), (45, 47, "G"), (49, 53, "H"), (55, 56, "I"), (58, 63, "J"), (64, 66, "K"), (68, 68, "L"), (69, 75, "M"), (77, 82, "N"), (84, 84, "O"), (85, 85, "P"), (86, 88, "Q"), (90, 93, "R"), (94, 96, "S")]


def sections_from_label(label: str) -> list[str]:
    """NACE sections named by an industry label: letters ('C-E', 'GHI', 'L, N') or two-digit division
    ranges ('10-33', '41-43', '95.1'). Used to key each office's survey industries to the EU-LFS mix."""
    s = str(label)
    out = []
    for a, b in re.findall(r"\b(\d{2})\s*(?:[-–]\s*(\d{2}))?\b", s):
        lo, hi = int(a), int(b) if b else int(a)
        for d0, d1, sec in _DIV_TO_SECTION:
            if lo <= d1 and hi >= d0 and sec not in out:
                out.append(sec)
    if not out:
        head = re.split(r"\s(?=[a-z])", s.strip(), maxsplit=1)[0]   # the code part before the first lower-case word
        for a, b in re.findall(r"([A-U])\s*[-–]\s*([A-U])", head):
            out += [chr(c) for c in range(ord(a), ord(b) + 1) if chr(c) not in out]
        if not out and re.fullmatch(r"[A-U]{2,4}", head):   # 'GHI': a run of sections
            out = list(head)
        if not out:
            out = [c for c in re.findall(r"\b([A-U])\b", re.sub(r"[A-U]\d+", lambda m: m.group(0)[0], head)) if c not in out]
    return out


def _jsonstat1_frame(path: Path) -> pd.DataFrame:
    """Statistics Denmark's JSON-stat 1.x ({"dataset": {...}}) as a long table."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))["dataset"]
    dims = d["dimension"]["id"]
    cats = []
    for k in dims:
        c = d["dimension"][k]["category"]
        order = sorted(c["index"], key=lambda x: c["index"][x]) if isinstance(c["index"], dict) else list(c["index"])
        cats.append([(code, c.get("label", {}).get(code, code)) for code in order])
    rows = list(itertools.product(*cats))
    vals = d["value"]
    v = [vals.get(str(i)) for i in range(len(rows))] if isinstance(vals, dict) else vals
    out = pd.DataFrame({k: [r[i][0] for r in rows] for i, k in enumerate(dims)})
    for i, k in enumerate(dims):
        out[f"label_{k}"] = [r[i][1] for r in rows]
    out["value"] = pd.to_numeric(pd.Series(v), errors="coerce")
    return out


def _jsonrpc_frame(path: Path) -> pd.DataFrame:
    """CSO PxStat JSON-RPC answers wrap a JSON-stat 2.0 dataset in "result"."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    tmp = Path(path).with_suffix(".inner.json")
    tmp.write_text(json.dumps(d.get("result", d)))
    try:
        return jsonstat_frame(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def _submajor_names() -> dict:
    """ISCO-08 two-digit titles, from Eurostat's employment table labels."""
    e = eu.load_employment_2d().drop_duplicates("isco2_code")
    return dict(zip(e.isco2_code, e.isco2_title))


def _points(nat: list[tuple[float, str, float]], ind: list[tuple[float, str, float, str]], national_label: str) -> pd.DataFrame:
    rows = [{"date": d, "period": p, "pct": v, "group_type": "all", "industry": None, "label": national_label} for d, p, v in nat]
    rows += [{"date": d, "period": p, "pct": v, "group_type": "industry", "industry": sec, "label": eu.NACE_SECTIONS[sec]} for d, p, v, sec in ind]
    df = pd.DataFrame(rows, columns=["date", "period", "pct", "group_type", "industry", "label"])
    # a section quoted by two survey industries in one year keeps their mean
    g = df[df.group_type == "industry"].groupby(["date", "period", "group_type", "industry", "label"], dropna=False).pct.mean().reset_index()
    return pd.concat([df[df.group_type == "all"], g], ignore_index=True).sort_values(["group_type", "industry", "date"]).reset_index(drop=True)


def _mix(geo: str) -> pd.DataFrame:
    m = eu.load_mix()
    return m[m.geo == geo].drop(columns="geo").reset_index(drop=True)


# =============================================================================== Sweden
SCB = RAW / "scb"


def se_employment() -> pd.DataFrame:
    d = jsonstat_frame(SCB / "yreg50bas_employees_by_ssyk.json")
    d = d[d.Yrke2012.str.fullmatch(r"\d{4}") & ~d.Yrke2012.isin(["0001", "0002"]) & (d.ArbetsSektor == "010")]   # all sectors (the other codes nest inside it)
    g = d.groupby(["Yrke2012", "label_Yrke2012", "Tid"]).value.sum(min_count=1).reset_index()
    return g.rename(columns={"Yrke2012": "ssyk_code", "label_Yrke2012": "ssyk_title", "Tid": "year", "value": "employed"})


def se_salary() -> pd.DataFrame:
    d = jsonstat_frame(SCB / "lone_salary_by_ssyk.json")
    d = d[d.Yrke2012.str.fullmatch(r"\d{4}") & (d.ContentsCode == "000007CE")]
    return d.rename(columns={"Yrke2012": "ssyk_code", "Tid": "year", "value": "median_monthly"})[["ssyk_code", "year", "median_monthly"]].dropna()


def se_weights() -> pd.DataFrame:
    emp = se_employment(); y = emp.year.max(); e = emp[emp.year == y]
    sal = se_salary(); ys = sal.year.max(); s = sal[sal.year == ys].set_index("ssyk_code").median_monthly
    df = pd.DataFrame({"ssyk_code": e.ssyk_code.values, "ssyk_title": e.ssyk_title.values, "employment_thousands": e.employed.values / 1000.0})
    df["wage_monthly"] = df.ssyk_code.map(s); df["wage_source"] = f"scb_salary_{ys}_median_monthly"
    for n, label in ((3, "minor_group_median"), (2, "submajor_group_median")):
        grp = df.assign(k=df.ssyk_code.str[:n]).dropna(subset=["wage_monthly"]).groupby("k").wage_monthly.median()
        miss = df.wage_monthly.isna(); df.loc[miss, "wage_monthly"] = df.loc[miss, "ssyk_code"].str[:n].map(grp); df.loc[miss & df.wage_monthly.notna(), "wage_source"] = label
    df["major_group"] = df.ssyk_code.str[0]; df["submajor_label"] = df.ssyk_code.str[:2]
    names = _submajor_names(); key = pd.read_csv(SCB / "ssyk2012_isco08_key.csv", dtype=str)
    via_key = key.assign(s2=key.ssyk_code.str[:2], i2=key.isco_minor.str[:2]).groupby("s2").i2.agg(lambda x: x.mode().iloc[0]).map(names)
    df["submajor_name"] = df.submajor_label.map(names).fillna(df.submajor_label.map(via_key))   # an SSYK-only group takes the name of the ISCO group its classes come from
    df["skill_level"] = df.ssyk_code.map(_skill_level)
    df["wage_value"] = df.employment_thousands * df.wage_monthly
    df["employment_method"] = f"register_{y}"; df["edition"] = int(y)
    return df[df.employment_thousands > 0].dropna(subset=["wage_monthly"]).reset_index(drop=True)


def se_isco_link(unit_codes: set[str] | None = None) -> pd.DataFrame:
    """ISCO-08 unit group -> SSYK 2012 occupations through SCB's translation key, which pairs each ISCO-08
    minor group (three digits) with the SSYK classes it maps to; a four-digit SSYK code that equals the ISCO
    code takes the whole weight, else the minor group's SSYK occupations share it equally."""
    key = pd.read_csv(SCB / "ssyk2012_isco08_key.csv", dtype=str)
    codes = sorted(unit_codes) if unit_codes else sorted(se_employment().ssyk_code.unique())
    by3 = {}
    for c in codes:
        by3.setdefault(c[:3], []).append(c)
    minor_to_ssyk = {}
    for r in key.itertuples():
        targets = [r.ssyk_code] if len(r.ssyk_code) == 4 else by3.get(r.ssyk_code, [])
        minor_to_ssyk.setdefault(r.isco_minor, set()).update(t for t in targets if t in codes)
    from . import io
    isco = sorted(io.load("xw_onetsoc_isco08").isco_code.astype(str).unique())
    rows = []
    for i in isco:
        targets = sorted(minor_to_ssyk.get(i[:3], set()))
        if not targets:
            targets = by3.get(i[:3], [])   # no key line: the same three digits
        if i in targets:
            rows.append({"isco_code": i, "nat_code": i, "weight": 1.0})
        else:
            rows += [{"isco_code": i, "nat_code": t, "weight": 1.0 / len(targets)} for t in targets]
    return pd.DataFrame(rows, columns=["isco_code", "nat_code", "weight"])


def se_ai_points() -> pd.DataFrame:
    d = jsonstat_frame(SCB / "ai_use_by_domain.json").dropna(subset=["value"])
    nat = [(int(r.Tid) + SURVEY_DATE_IN_YEAR, r.Tid, float(r.value)) for r in d[d.Redovisningsgrupp == "Tot250"].itertuples()]
    ind = []
    for r in d.itertuples():
        if re.search(r"SNI", str(r.label_Redovisningsgrupp)) and not str(r.label_Redovisningsgrupp).startswith("total"):
            for sec in sections_from_label(r.label_Redovisningsgrupp):
                ind.append((int(r.Tid) + SURVEY_DATE_IN_YEAR, r.Tid, float(r.value), sec))
    return _points(nat, ind, "Enterprises with 10 or more employees")


def se_outcomes() -> pd.DataFrame:
    emp = se_employment(); ys = sorted(emp.year.unique())[-2:]
    w = emp.pivot(index="ssyk_code", columns="year", values="employed")
    sal = se_salary(); ysal = sorted(sal.year.unique())[-2:]
    sw = sal.pivot(index="ssyk_code", columns="year", values="median_monthly")
    df = pd.DataFrame({"ssyk_code": w.index, f"emp_{ys[0]}": w[ys[0]].values, f"emp_{ys[1]}": w[ys[1]].values})
    df["emp_change_pct"] = 100.0 * (df[f"emp_{ys[1]}"] / df[f"emp_{ys[0]}"].where(df[f"emp_{ys[0]}"] > 0) - 1.0)
    df[f"wage_{ysal[0]}"] = df.ssyk_code.map(sw[ysal[0]]); df[f"wage_{ysal[1]}"] = df.ssyk_code.map(sw[ysal[1]])
    df["wage_change_pct"] = 100.0 * (df[f"wage_{ysal[1]}"] / df[f"wage_{ysal[0]}"].where(df[f"wage_{ysal[0]}"] > 0) - 1.0)
    df["year_from"], df["year_to"], df["wage_years"] = int(ys[0]), int(ys[1]), f"{ysal[0]}-{ysal[1]}"
    return df.reset_index(drop=True)


def se_units() -> pd.DataFrame:
    w = se_weights(); return pd.DataFrame({"ssyk_code": w.ssyk_code.values, "emp_unit": "people", "wage_unit": "month"})


def se_titles() -> pd.DataFrame:
    w = se_weights(); return pd.DataFrame({"ssyk_code": w.ssyk_code, "title": w.ssyk_title, "category": "principal"})


# =============================================================================== Finland
STATFIN = RAW / "statfin"


def fi_employment() -> pd.DataFrame:
    d = jsonstat_frame(STATFIN / "tyokay_115q_employed_by_occupation.json")
    col = "ammatti_104_20161021"
    d = d[d[col].str.fullmatch(r"\d{4}")]
    d["title"] = d[f"label_{col}"].str.replace(r"^\d+\s+", "", regex=True).str.replace(r"\s*\(Level \d\)\s*$", "", regex=True)
    return d.rename(columns={col: "tk_code", "timeperiod_y": "year", "value": "employed"})[["tk_code", "title", "year", "employed"]].dropna()


def fi_earnings() -> pd.DataFrame:
    d = jsonstat_frame(STATFIN / "pra_15ay_earnings_by_occupation.json")
    col = "ammatti_19_20180101"
    d = d[d[col].str.fullmatch(r"\d{4}")]
    w = d.pivot_table(index=[col, "timeperiod_y"], columns="contentscode", values="value", aggfunc="first").reset_index()
    return w.rename(columns={col: "tk_code", "timeperiod_y": "year", "psaaja_tunti_kans_med": "median_hourly", "psaaja_lkm": "n_earners"})[["tk_code", "year", "median_hourly", "n_earners"]]


def fi_weights() -> pd.DataFrame:
    emp = fi_employment(); y = emp.year.max(); e = emp[emp.year == y]
    earn = fi_earnings(); ye = earn.year.max(); s = earn[earn.year == ye].set_index("tk_code").median_hourly
    df = pd.DataFrame({"tk_code": e.tk_code.values, "tk_title": e.title.values, "employment_thousands": e.employed.values / 1000.0})
    df["wage_hourly"] = df.tk_code.map(s); df["wage_source"] = f"statfin_earnings_{ye}_median_hourly"
    for n, label in ((3, "minor_group_median"), (2, "submajor_group_median")):
        grp = df.assign(k=df.tk_code.str[:n]).dropna(subset=["wage_hourly"]).groupby("k").wage_hourly.median()
        miss = df.wage_hourly.isna(); df.loc[miss, "wage_hourly"] = df.loc[miss, "tk_code"].str[:n].map(grp); df.loc[miss & df.wage_hourly.notna(), "wage_source"] = label
    df["major_group"] = df.tk_code.str[0]; df["submajor_label"] = df.tk_code.str[:2]; df["submajor_name"] = df.submajor_label.map(_submajor_names())
    df["skill_level"] = df.tk_code.map(_skill_level)
    df["wage_value"] = df.employment_thousands * df.wage_hourly
    df["employment_method"] = f"register_{y}"; df["edition"] = int(y)
    return df[df.employment_thousands > 0].dropna(subset=["wage_hourly"]).reset_index(drop=True)


def fi_isco_link(unit_codes: set[str] | None = None) -> pd.DataFrame:
    codes = sorted(unit_codes) if unit_codes else sorted(fi_employment().tk_code.unique())
    return pd.DataFrame({"isco_code": codes, "nat_code": codes, "weight": 1.0})


def fi_ai_points() -> pd.DataFrame:
    d = jsonstat_frame(STATFIN / "icte_14yc_ict_in_enterprises.json").dropna(subset=["value"])
    col = "toimiala_79_20180101"
    nat = [(int(r.timeperiod_y) + SURVEY_DATE_IN_YEAR, r.timeperiod_y, float(r.value)) for r in d[d[col] == "SSS"].itertuples()]
    ind = []
    for r in d[d[col] != "SSS"].itertuples():
        for sec in sections_from_label(getattr(r, f"label_{col}")):
            ind.append((int(r.timeperiod_y) + SURVEY_DATE_IN_YEAR, r.timeperiod_y, float(r.value), sec))
    return _points(nat, ind, "Enterprises with 10 or more persons employed")


def fi_outcomes() -> pd.DataFrame:
    emp = fi_employment(); ys = sorted(emp.year.unique())[-2:]
    w = emp.pivot(index="tk_code", columns="year", values="employed")
    df = pd.DataFrame({"tk_code": w.index, f"emp_{ys[0]}": w[ys[0]].values, f"emp_{ys[1]}": w[ys[1]].values})
    df["emp_change_pct"] = 100.0 * (df[f"emp_{ys[1]}"] / df[f"emp_{ys[0]}"].where(df[f"emp_{ys[0]}"] > 0) - 1.0)
    earn = fi_earnings(); ye = sorted(earn.year.unique())
    if len(ye) >= 2:
        ew = earn.pivot(index="tk_code", columns="year", values="median_hourly")
        df[f"wage_{ye[-2]}"] = df.tk_code.map(ew[ye[-2]]); df[f"wage_{ye[-1]}"] = df.tk_code.map(ew[ye[-1]])
        df["wage_change_pct"] = 100.0 * (df[f"wage_{ye[-1]}"] / df[f"wage_{ye[-2]}"].where(df[f"wage_{ye[-2]}"] > 0) - 1.0)
        df["wage_years"] = f"{ye[-2]}-{ye[-1]}"
    else:   # one year of the earnings structure published so far: no pay change to test
        df[f"wage_{ye[-1]}"] = df.tk_code.map(earn[earn.year == ye[-1]].set_index("tk_code").median_hourly); df["wage_change_pct"] = np.nan; df["wage_years"] = str(ye[-1])
    df["year_from"], df["year_to"] = int(ys[0]), int(ys[1])
    return df.reset_index(drop=True)


def fi_units() -> pd.DataFrame:
    w = fi_weights(); return pd.DataFrame({"tk_code": w.tk_code.values, "emp_unit": "people", "wage_unit": "hour"})


def fi_titles() -> pd.DataFrame:
    w = fi_weights(); return pd.DataFrame({"tk_code": w.tk_code, "title": w.tk_title, "category": "principal"})


# =============================================================================== Denmark
DST = RAW / "dst"


def dk_earnings() -> pd.DataFrame:
    d = _jsonstat1_frame(DST / "lons20_earnings_by_occupation.json")
    d = d[d.ARBF.str.fullmatch(r"\d{4}")]
    return d.rename(columns={"ARBF": "disco_code", "label_ARBF": "disco_title", "Tid": "year", "value": "hourly_dkk"})[["disco_code", "disco_title", "year", "hourly_dkk"]]


def dk_employment_split() -> pd.DataFrame:
    """Employment per four-digit DISCO code: the two-digit group's EU-LFS employment (Denmark, 2025) split
    across the group's four-digit codes in the proportions of Norway's register, the nearest country that
    publishes the same ISCO codes at that detail; equal shares where Norway has no code in the group."""
    from . import no_labour as no
    e2 = eu.load_employment_2d(); e2 = e2[e2.geo == "DK"].set_index("isco2_code").employment_thousands
    nor = no.build_styrk_weights()[["styrk_code", "employment_thousands"]].set_index("styrk_code").employment_thousands
    earn = dk_earnings(); codes = sorted(earn.disco_code.unique())
    rows = []
    for g2, grp in pd.Series(codes).groupby(pd.Series(codes).str[:2]):
        total = float(e2.get(g2, np.nan))
        if np.isnan(total):
            continue
        w = np.array([float(nor.get(c, np.nan)) for c in grp])
        w = np.where(np.isnan(w), 0.0, w)
        shares = w / w.sum() if w.sum() > 0 else np.full(len(grp), 1.0 / len(grp))
        rows += [{"disco_code": c, "employment_thousands": total * s, "split": "norway_register" if w.sum() > 0 else "equal"} for c, s in zip(grp, shares)]
    return pd.DataFrame(rows)


def dk_weights() -> pd.DataFrame:
    earn = dk_earnings(); ye = earn.year.max(); e = earn[earn.year == ye]
    emp = dk_employment_split().set_index("disco_code")
    df = pd.DataFrame({"disco_code": e.disco_code.values, "disco_title": e.disco_title.values, "wage_hourly": e.hourly_dkk.values})
    df["employment_thousands"] = df.disco_code.map(emp.employment_thousands)
    df["employment_method"] = "eu_lfs_2d_split_" + df.disco_code.map(emp.split).fillna("none")
    df = df.dropna(subset=["employment_thousands"])
    df["wage_source"] = f"dst_lons20_{ye}_hourly_earnings"
    for n, label in ((3, "minor_group_median"), (2, "submajor_group_median")):
        grp = df.assign(k=df.disco_code.str[:n]).dropna(subset=["wage_hourly"]).groupby("k").wage_hourly.median()
        miss = df.wage_hourly.isna(); df.loc[miss, "wage_hourly"] = df.loc[miss, "disco_code"].str[:n].map(grp); df.loc[miss & df.wage_hourly.notna(), "wage_source"] = label
    df["major_group"] = df.disco_code.str[0]; df["submajor_label"] = df.disco_code.str[:2]; df["submajor_name"] = df.submajor_label.map(_submajor_names())
    df["skill_level"] = df.disco_code.map(_skill_level)
    df["wage_value"] = df.employment_thousands * df.wage_hourly
    df["edition"] = int(ye)
    return df[df.employment_thousands > 0].dropna(subset=["wage_hourly"]).reset_index(drop=True)


def dk_isco_link(unit_codes: set[str] | None = None) -> pd.DataFrame:
    codes = sorted(unit_codes) if unit_codes else sorted(dk_earnings().disco_code.unique())
    return pd.DataFrame({"isco_code": codes, "nat_code": codes, "weight": 1.0})


def dk_ai_points() -> pd.DataFrame:
    d = _jsonstat1_frame(DST / "itav28_ai_use_by_industry.json").dropna(subset=["value"])
    d = d[d.VIRKSTRRDA == "1010"]
    nat = [(int(r.Tid) + SURVEY_DATE_IN_YEAR, r.Tid, float(r.value)) for r in d[d.BRANCHEDB25UDVALG.str.startswith("C-O")].itertuples()]
    ind = []
    for r in d[~d.BRANCHEDB25UDVALG.str.startswith("C-O")].itertuples():
        for sec in sections_from_label(r.BRANCHEDB25UDVALG.split("_")[0]):
            ind.append((int(r.Tid) + SURVEY_DATE_IN_YEAR, r.Tid, float(r.value), sec))
    return _points(nat, ind, "Private non-agricultural enterprises with 10 or more employees")


def dk_outcomes() -> pd.DataFrame:
    earn = dk_earnings(); ys = sorted(earn.year.unique())[-2:]
    w = earn.pivot(index="disco_code", columns="year", values="hourly_dkk")
    df = pd.DataFrame({"disco_code": w.index, f"emp_{ys[0]}": np.nan, f"emp_{ys[1]}": np.nan, "emp_change_pct": np.nan, f"wage_{ys[0]}": w[ys[0]].values, f"wage_{ys[1]}": w[ys[1]].values})
    df["wage_change_pct"] = 100.0 * (df[f"wage_{ys[1]}"] / df[f"wage_{ys[0]}"].where(df[f"wage_{ys[0]}"] > 0) - 1.0)
    df["year_from"], df["year_to"], df["wage_years"] = int(ys[0]), int(ys[1]), f"{ys[0]}-{ys[1]}"
    return df.reset_index(drop=True)


def dk_units() -> pd.DataFrame:
    w = dk_weights(); return pd.DataFrame({"disco_code": w.disco_code.values, "emp_unit": "people", "wage_unit": "hour"})


def dk_titles() -> pd.DataFrame:
    w = dk_weights(); return pd.DataFrame({"disco_code": w.disco_code, "title": w.disco_title, "category": "principal"})


# =============================================================================== Netherlands
CBS = RAW / "cbs"


def _cbs_rows(name: str) -> list[dict]:
    return json.loads((CBS / name).read_text(encoding="utf-8"))["value"]


def nl_groups() -> pd.DataFrame:
    """The 114 BRC 2014 occupation groups: key, code, Dutch title."""
    rows = []
    for r in _cbs_rows("86320NED_Beroep.json"):
        m = re.match(r"^(\d{4})\s+(.+)$", r["Title"].strip())
        if m:
            rows.append({"key": r["Key"], "brc_code": m.group(1), "brc_title": m.group(2)})
    return pd.DataFrame(rows)


def nl_employment() -> pd.DataFrame:
    g = nl_groups().set_index("key")
    rows = [{"key": r["Beroep"], "year": r["Perioden"][:4], "employed_thousands": r.get("WerkzameBeroepsbevolking_1")} for r in _cbs_rows("86320NED_employed_by_occupation.json") if r["Perioden"].endswith("JJ00")]
    d = pd.DataFrame(rows); d = d[d.key.isin(g.index)].copy()
    d["brc_code"] = d.key.map(g.brc_code); d["brc_title"] = d.key.map(g.brc_title)
    return d.dropna(subset=["employed_thousands"]).reset_index(drop=True)


def nl_wages() -> pd.DataFrame:
    g = {r["Key"]: re.match(r"^(\d{4})\s", r["Title"].strip()).group(1) for r in _cbs_rows("86355NED_Beroep.json") if re.match(r"^\d{4}\s", r["Title"].strip())}
    rows = [{"brc_code": g[r["Beroep"]], "year": r["Perioden"][:4], "median_hourly": r.get("k_50ePercentielMediaan_3"), "employees_thousands": r.get("Werknemer_1")} for r in _cbs_rows("86355NED_hourly_wage_by_occupation.json") if r["Beroep"] in g and r["Perioden"].endswith("JJ00")]
    return pd.DataFrame(rows).dropna(subset=["median_hourly"]).reset_index(drop=True)


def nl_key() -> pd.DataFrame:
    """ROA and CBS: the ISCO 2008 unit groups inside every BRC 2014 group, with 2025 workers and shares."""
    k = pd.read_csv(CBS / "brc2014_isco08_key.csv", dtype={"brc_code": str, "isco_code": str})
    k["workers_thousands_2025"] = pd.to_numeric(k.workers_thousands_2025, errors="coerce").fillna(0.5).clip(lower=0.5)   # a suppressed or rounded-to-zero cell is small, not absent
    return k


def nl_weights() -> pd.DataFrame:
    emp = nl_employment(); y = emp.year.max(); e = emp[emp.year == y]
    w = nl_wages(); yw = w.year.max()
    latest = w[w.year >= "2022"].sort_values("year").drop_duplicates("brc_code", keep="last").set_index("brc_code")   # each group's latest median since 2022 (a small group can be suppressed in the latest year)
    df = pd.DataFrame({"brc_code": e.brc_code.values, "brc_title": e.brc_title.values, "employment_thousands": e.employed_thousands.values.astype(float)})
    df["wage_hourly"] = df.brc_code.map(latest.median_hourly); df["wage_source"] = "cbs_hourly_wage_" + df.brc_code.map(latest.year).fillna("none") + "_median"
    seg = df.assign(k=df.brc_code.str[:3]).dropna(subset=["wage_hourly"]).groupby("k").wage_hourly.median()
    miss = df.wage_hourly.isna(); df.loc[miss, "wage_hourly"] = df.loc[miss, "brc_code"].str[:3].map(seg); df.loc[miss & df.wage_hourly.notna(), "wage_source"] = "segment_median"
    miss = df.wage_hourly.isna(); df.loc[miss, "wage_hourly"] = df.wage_hourly.median(); df.loc[miss, "wage_source"] = "all_groups_median"
    key = nl_key()
    top = key.sort_values("workers_thousands_2025", ascending=False).drop_duplicates("brc_code").set_index("brc_code").isco_code
    df["major_group"] = df.brc_code.map(top).str[0].fillna("0"); df["submajor_label"] = df.brc_code.str[:2]
    cls = {r["Key"]: r["Title"].strip() for r in _cbs_rows("86320NED_Beroep.json")}
    seg_names = {re.match(r"^(\d{2})\s+(.+)$", t).group(1): re.match(r"^(\d{2})\s+(.+)$", t).group(2) for t in cls.values() if re.match(r"^\d{2}\s", t)}
    df["submajor_name"] = df.submajor_label.map(seg_names)
    df["skill_level"] = df.brc_code.map(top).map(lambda c: _skill_level(c) if isinstance(c, str) else np.nan)
    df["skill_level"] = df.skill_level.fillna(df.groupby(df.brc_code.str[:2]).skill_level.transform("median")).fillna(2.0)
    df["wage_value"] = df.employment_thousands * df.wage_hourly
    df["employment_method"] = f"lfs_{y}"; df["edition"] = int(y)
    return df[df.employment_thousands > 0].reset_index(drop=True)


def nl_isco_link(unit_codes: set[str] | None = None) -> pd.DataFrame:
    k = nl_key()
    k = k[k.brc_code.isin(unit_codes)] if unit_codes else k
    k["weight"] = k.workers_thousands_2025 / k.groupby("isco_code").workers_thousands_2025.transform("sum")
    return k.rename(columns={"brc_code": "nat_code"})[["isco_code", "nat_code", "weight"]].reset_index(drop=True)


def nl_ai_points() -> pd.DataFrame:
    nat, ind = [], []
    for year, data, inds in (("2023", "85735NED_ict_by_industry_2023.json", "85735NED_industries.json"), ("2024", "86075NED_ict_by_industry_2024.json", "86075NED_industries.json"), ("2025", "86120NED_ict_by_industry_2025.json", "86120NED_industries.json")):
        names = {r["Key"]: r["Title"].strip() for r in _cbs_rows(inds)}
        size_total = None
        if (CBS / "85735NED_sizes.json").exists() and year == "2023":   # the 2023 table is also by size class: keep all sizes together
            size_total = next((r["Key"] for r in _cbs_rows("85735NED_sizes.json") if re.search(r"totaal|10 of meer|10 en meer", r["Title"], re.I)), None)
        for r in _cbs_rows(data):
            if size_total is not None and r.get("Bedrijfsgrootte") != size_total:
                continue
            fld = next((f for f in r if f.startswith("AIGebruikt")), None)
            if fld is None or r.get(fld) is None:
                continue
            lab = names.get(r["BedrijfstakkenBranchesSBI2008"], "")
            if lab.lower().startswith("totaal"):
                nat.append((int(year) + SURVEY_DATE_IN_YEAR, year, float(r[fld])))
            else:
                for sec in sections_from_label(lab):
                    ind.append((int(year) + SURVEY_DATE_IN_YEAR, year, float(r[fld]), sec))
    return _points(nat, ind, "Enterprises with 10 or more employed persons (sections C to N and Q)")


def nl_outcomes() -> pd.DataFrame:
    emp = nl_employment(); ys = sorted(emp.year.unique())[-2:]
    w = emp.pivot_table(index="brc_code", columns="year", values="employed_thousands", aggfunc="first")
    df = pd.DataFrame({"brc_code": w.index, f"emp_{ys[0]}": w[ys[0]].values, f"emp_{ys[1]}": w[ys[1]].values})
    df["emp_change_pct"] = 100.0 * (df[f"emp_{ys[1]}"] / df[f"emp_{ys[0]}"].where(df[f"emp_{ys[0]}"] > 0) - 1.0)
    wg = nl_wages(); yw = sorted(wg.year.unique())[-2:]
    ww = wg.pivot_table(index="brc_code", columns="year", values="median_hourly", aggfunc="first")
    df[f"wage_{yw[0]}"] = df.brc_code.map(ww[yw[0]]); df[f"wage_{yw[1]}"] = df.brc_code.map(ww[yw[1]])
    df["wage_change_pct"] = 100.0 * (df[f"wage_{yw[1]}"] / df[f"wage_{yw[0]}"].where(df[f"wage_{yw[0]}"] > 0) - 1.0)
    df["year_from"], df["year_to"], df["wage_years"] = int(ys[0]), int(ys[1]), f"{yw[0]}-{yw[1]}"
    return df.reset_index(drop=True)


def nl_units() -> pd.DataFrame:
    w = nl_weights(); return pd.DataFrame({"brc_code": w.brc_code.values, "emp_unit": "thousands", "wage_unit": "hour"})


def nl_titles() -> pd.DataFrame:
    k = nl_key()
    rows = [{"brc_code": r.brc_code, "title": r.isco_name_nl, "category": "isco_component"} for r in k.itertuples()]
    w = nl_weights(); rows += [{"brc_code": c, "title": t, "category": "principal"} for c, t in zip(w.brc_code, w.brc_title)]
    return pd.DataFrame(rows).drop_duplicates(["brc_code", "title"]).reset_index(drop=True)


# =============================================================================== Ireland
CSO = RAW / "cso"


def ie_census() -> pd.DataFrame:
    """Labour force (persons at work or looking for work) by detailed occupational group, all of the State, 2011, 2016, 2022."""
    d = _jsonrpc_frame(CSO / "F7042_labour_force_by_detailed_occupation.json")
    occ, cty = "C02843V04551", "C04104V04868"
    state = d[d[f"label_{cty}"].str.contains(r"^(?:State|Ireland)", regex=True, case=False)]
    if state.empty:
        state = d[d[cty] == "-"]
    state = state.rename(columns={occ: "soc10_code", f"label_{occ}": "soc10_title", "TLIST(A1)": "year", "value": "labour_force"})
    return state[["soc10_code", "soc10_title", "year", "labour_force"]].dropna().reset_index(drop=True)


def ie_weights() -> pd.DataFrame:
    c = ie_census(); y = c.year.max(); e = c[(c.year == y) & c.soc10_code.str.fullmatch(r"\d{4}")]
    earn = eu.load_earnings_1d(); earn = earn[earn.geo == "IE"].set_index("major_group").mean_hourly_eur
    soc_to_isco_major = {"1": "1", "2": "2", "3": "3", "4": "4", "5": "7", "6": "5", "7": "5", "8": "8", "9": "9"}
    df = pd.DataFrame({"soc10_code": e.soc10_code.values, "soc10_title": e.soc10_title.values, "employment_thousands": e.labour_force.values / 1000.0})
    df["major_group"] = df.soc10_code.str[0]
    df["wage_hourly"] = df.major_group.map(soc_to_isco_major).map(earn); df["wage_source"] = "ses_major_group_mean_hourly_eur"
    df["wage_hourly"] = df.wage_hourly.fillna(df.wage_hourly.mean())
    df["submajor_label"] = df.soc10_code.str[:2]; df["submajor_name"] = df.submajor_label.map(_ie_submajor_names())
    df["skill_level"] = df.major_group.map(soc_to_isco_major).map(lambda m: _skill_level(m + "000"))
    df["wage_value"] = df.employment_thousands * df.wage_hourly
    df["employment_method"] = f"census_{y}_labour_force"; df["edition"] = int(y)
    return df[df.employment_thousands > 0].reset_index(drop=True)


def _ie_submajor_names() -> dict:
    k = pd.ExcelFile(CSO / "soc2010_isco08.xls").parse("1to1s", header=None, dtype=str)
    names = {}
    for r in k.itertuples(index=False):
        c, t = str(r[0]).strip(), str(r[1]).strip()
        if re.fullmatch(r"\d{4}", c):
            names.setdefault(c[:2], t)
    return names


def ie_isco_link(unit_codes: set[str] | None = None) -> pd.DataFrame:
    """SOC 2010 unit group -> ISCO-08 unit groups from the ONS mapping (one-to-one, and one-to-two with 50:50
    or 40:60 splits), inverted to ISCO -> SOC weights."""
    x = pd.ExcelFile(CSO / "soc2010_isco08.xls")
    rows = []
    for sheet, shares in (("1to1s", [1.0]), ("1to2 50-50", [0.5, 0.5]), ("1to2 40-60", [0.4, 0.6])):
        df = x.parse(sheet, header=None, dtype=str)
        for r in df.itertuples(index=False):
            cells = [str(v).strip() for v in r]
            if not re.fullmatch(r"\d{4}", cells[0]):
                continue
            iscos = [c for c in cells[1:] if re.fullmatch(r"\d{4}", c)]
            for isco, w in zip(iscos, shares):
                rows.append({"soc10_code": cells[0], "isco_code": isco, "w": w})
    k = pd.DataFrame(rows)
    k = k[k.soc10_code.isin(unit_codes)] if unit_codes else k
    k["weight"] = k.w / k.groupby("isco_code").w.transform("sum")
    return k.rename(columns={"soc10_code": "nat_code"})[["isco_code", "nat_code", "weight"]].reset_index(drop=True)


def ie_ai_points() -> pd.DataFrame:
    d = _jsonrpc_frame(CSO / "ICA237_enterprise_ai_use.json").dropna(subset=["value"])
    first_stat = d.STATISTIC.iloc[0]
    d = d[(d.STATISTIC == first_stat) & (d["C03066V03701"] == "-")]
    nat = [(int(r["TLIST(A1)"]) + SURVEY_DATE_IN_YEAR, str(r["TLIST(A1)"]), float(r.value)) for _, r in d.iterrows()]
    return _points(nat, [], "Enterprises with 10 or more persons employed")


def ie_outcomes() -> pd.DataFrame:
    c = ie_census(); ys = sorted(c.year.unique())[-2:]
    w = c[c.soc10_code.str.fullmatch(r"\d{4}")].pivot_table(index="soc10_code", columns="year", values="labour_force", aggfunc="first")
    df = pd.DataFrame({"soc10_code": w.index, f"emp_{ys[0]}": w[ys[0]].values, f"emp_{ys[1]}": w[ys[1]].values})
    df["emp_change_pct"] = 100.0 * (df[f"emp_{ys[1]}"] / df[f"emp_{ys[0]}"].where(df[f"emp_{ys[0]}"] > 0) - 1.0)
    df[f"wage_{ys[0]}"], df[f"wage_{ys[1]}"], df["wage_change_pct"] = np.nan, np.nan, np.nan
    df["year_from"], df["year_to"], df["wage_years"] = int(ys[0]), int(ys[1]), "none"
    return df.reset_index(drop=True)


def ie_units() -> pd.DataFrame:
    w = ie_weights(); return pd.DataFrame({"soc10_code": w.soc10_code.values, "emp_unit": "people", "wage_unit": "hour"})


def ie_titles() -> pd.DataFrame:
    w = ie_weights(); return pd.DataFrame({"soc10_code": w.soc10_code, "title": w.soc10_title, "category": "principal"})


def empty_tasks(code_col: str):
    return lambda: pd.DataFrame(columns=[code_col, "task_seq", "task"])
