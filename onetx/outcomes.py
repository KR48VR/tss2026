"""Rung five of the ladder: what has actually happened to jobs so far.

MOM publishes, by occupation, the things a disruption would eventually show up in:
  * job vacancies by detailed occupation (SSOC 5-digit) as at September 2023, 2024 and 2025
    (2023 and 2024 as the share of vacancies within each occupational group, 2025 as numbers)
  * employed residents by occupation sub-major group (2-digit), every June from 2011 to 2025
  * retrenchment incidence, job vacancy rate and resignation rate by occupational group
    (PMET / clerical, sales and service / production, transport, cleaners and labourers), yearly
  * median gross monthly wages by occupation, June 2024 and June 2025

This module loads them, builds one outcome row per SSOC 2024 code and per sub-major group, and
tests whether the map's exposure signals relate to them yet (rank correlations with plain readings).
No causal claim: two years of outcomes against a forecast that mostly starts in 2029.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .config import RAW
from . import io

MOM = RAW / "singapore" / "mom"
GROUP_OF_MAJOR = {"1": "pmet", "2": "pmet", "3": "pmet", "4": "clerical_sales_service", "5": "clerical_sales_service", "6": "production_transport_cleaners_labourers", "7": "production_transport_cleaners_labourers", "8": "production_transport_cleaners_labourers", "9": "production_transport_cleaners_labourers"}
GROUP_LABELS = {"pmet": "professionals, managers, executives and technicians", "clerical_sales_service": "clerical, sales and service workers", "production_transport_cleaners_labourers": "production and transport operators, cleaners and labourers"}


def _num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return np.nan


# ----------------------------------------------------------------------------- vacancies by detailed occupation
def load_jv_by_occupation(year: int) -> pd.DataFrame:
    """Rows: year, occ_group, ssoc_version, code, occupation, value, kind ('number' or 'share_within_group')."""
    import openpyxl

    wb = openpyxl.load_workbook(MOM / "jv" / f"mrsd_{year}_jv_by_ssoc_ind_occ_edu.xlsx", read_only=True, data_only=True)
    rows = list(wb["Occupational Group"].iter_rows(values_only=True))
    kind = "number" if "NUMBER" in str(rows[0][0]).upper() else "share_within_group"
    version = "2024" if "2024" in str(rows[4][1]) else "2020"
    out, group = [], None
    for r in rows[5:]:
        if r[0] not in (None, ""):
            if str(r[0]).startswith("Notes") or re.match(r"^\d\)", str(r[0])):
                break
            group = str(r[0]).strip()
        if r[1] in (None, "") or group is None:
            continue
        out.append({"year": year, "occ_group": group, "ssoc_version": version, "code": str(r[1]).strip(), "occupation": str(r[2]).strip(), "value": _num(r[3]), "kind": kind})
    return pd.DataFrame(out)


def jv_shares_by_minor_group() -> pd.DataFrame:
    """Vacancy share within occupational group at the SSOC minor-group level (first four digits), for 2023,
    2024 and 2025 on a common basis: 2025 numbers are converted to shares among the listed occupations;
    2023 and 2024 shares (SSOC 2020 codes) are carried to SSOC 2024 minor groups through the official
    correspondence table, with four-digit 2020 codes mapped as minor groups directly."""
    xw = io.load("xw_ssoc2024_ssoc2020")
    xw["minor24"] = xw.ssoc2024_code.astype(str).str[:4]
    xw["minor20"] = xw.ssoc2020_code.astype(str).str[:4]
    minor_map = xw.drop_duplicates("minor20").set_index("minor20").minor24.to_dict()  # most minor groups unchanged between editions
    frames = []
    for y in (2023, 2024, 2025):
        try:
            d = load_jv_by_occupation(y)
        except FileNotFoundError:
            continue
        if d.empty:
            continue
        if d.kind.iloc[0] == "number":
            d["share"] = d.value / d.groupby("occ_group").value.transform("sum") * 100.0
        else:
            d["share"] = d.value
        d["minor"] = d.code.str[:4]
        if d.ssoc_version.iloc[0] == "2020":
            d["minor"] = d.minor.map(lambda m: minor_map.get(m, m))
        g = d.groupby(["year", "occ_group", "minor"], as_index=False).share.sum()
        frames.append(g)
    if not frames:
        return pd.DataFrame(columns=["occ_group", "minor", "share_2023", "share_2024", "share_2025"])
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index=["occ_group", "minor"], columns="year", values="share", aggfunc="sum").reset_index()
    wide.columns = [c if isinstance(c, str) else f"share_{c}" for c in wide.columns]
    return wide


# ----------------------------------------------------------------------------- employment by sub-major group, 2011-2025
def load_employment_submajor_series() -> pd.DataFrame:
    """Employed residents (thousands) by SSOC sub-major group (2-digit), June of each year 2011-2025, both sexes."""
    import openpyxl

    wb = openpyxl.load_workbook(MOM / "mrsd_69_Emp_Res_DetailedOcc_Sex.xlsx", read_only=True, data_only=True)
    rows = list(wb["Sheet1"].iter_rows(values_only=True))
    hdr = rows[3]
    years = [int(c) for c in hdr if isinstance(c, (int, float)) or (isinstance(c, str) and c.strip().isdigit())]
    first_year_col = next(i for i, c in enumerate(hdr) if c not in (None, ""))
    out = []
    for r in rows[4:]:
        if r[0] in ("Male", "Female"):
            break
        label = str(r[1]).strip() if r[1] is not None else ""
        if not label or label == "Total":
            if label == "Total":
                out.append({"submajor_label": "TOTAL", "group_name": "Total", **{f"emp_{y}": _num(r[first_year_col + i]) for i, y in enumerate(years)}})
            continue
        m = re.match(r"^([0-9X]{2})(?:\s*-\s*([0-9X]{2}))?\s+(.*)$", label)
        if not m:
            continue
        code = m.group(1) if not m.group(2) else f"{m.group(1)}-{m.group(2)}"
        out.append({"submajor_label": code, "group_name": m.group(3).strip(), **{f"emp_{y}": _num(r[first_year_col + i]) for i, y in enumerate(years)}})
    df = pd.DataFrame(out)
    for a, b in [(2019, 2025), (2023, 2025), (2024, 2025), (2011, 2025)]:
        if f"emp_{a}" in df and f"emp_{b}" in df:
            df[f"emp_change_{a}_{b}_pct"] = (df[f"emp_{b}"] / df[f"emp_{a}"] - 1.0) * 100.0
    return df


# ----------------------------------------------------------------------------- yearly series by occupational group (3 groups)
def _group_series(path, sheet_prefix: str, value_name: str, annual_only: bool = True) -> pd.DataFrame:
    """Parse MOM's time-series layout: blocks per OCCUPATIONAL GROUP, rows per INDUSTRY, columns per year
    (with quarters in between for the quarterly tables). Returns the INDUSTRY == TOTAL row of each block."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = [s for s in wb.sheetnames if s.startswith(sheet_prefix)][-1]
    rows = list(wb[sheet].iter_rows(values_only=True))
    hi = next(i for i, r in enumerate(rows) if r and str(r[0]).strip().upper().startswith("OCCUPATIONAL GROUP"))
    hdr = rows[hi]
    year_cols = [(i, int(str(c))) for i, c in enumerate(hdr) if i >= 3 and c is not None and str(c).strip().isdigit()]
    out, group = [], None
    for r in rows[hi + 1:]:
        if r[0] not in (None, ""):
            group = str(r[0]).strip()
        if group is None or r[2] is None:
            continue
        if str(r[2]).strip().upper() == "TOTAL":
            for i, y in year_cols:
                out.append({"occ_group_label": group, "year": y, value_name: _num(r[i])})
    df = pd.DataFrame(out)

    def norm(s):
        return re.sub(r"\s+", " ", str(s).upper().replace("AND", "&").replace("SERVICES", "SERVICE")).strip()
    key = {"TOTAL": "total", "PROFESSIONALS, MANAGERS, EXECUTIVES & TECHNICIANS": "pmet", "CLERICAL, SALES & SERVICE WORKERS": "clerical_sales_service", "PRODUCTION & TRANSPORT OPERATORS, CLEANERS & LABOURERS": "production_transport_cleaners_labourers"}
    df["occ_group"] = df.occ_group_label.map(lambda s: key.get(norm(s)) or next((v for k, v in key.items() if k in norm(s)), None))
    df["occ_group"] = df.occ_group.fillna(df.occ_group_label.str.lower())
    return df.dropna(subset=[value_name])


def load_retrenchment_incidence() -> pd.DataFrame:
    return _group_series(MOM / "mrsd_33_incidence_of_retrenchment_by_ind_occ_grp.xlsx", "2006-", "retrenchment_per_1000")


def load_vacancy_rate_by_group() -> pd.DataFrame:
    return _group_series(MOM / "mrsd_24_Qtly_and_annl_tsd_on_job_vac_rate_by_ind_and_occ_grp.xlsx", "2006-", "job_vacancy_rate_pct")


def load_resignation_rate_by_group() -> pd.DataFrame:
    return _group_series(MOM / "mrsd_26_Qtly_and_annl_tsd_on_resgn_by_ind_and_occ_grp.xlsx", "2006-", "resignation_rate_pct")


def load_retrenchment_by_occupation_annual() -> pd.DataFrame:
    """data.gov.sg: number of retrenched employees by occupational group, yearly (the same three groups)."""
    p = MOM / "retrench_by_occupation_annual.csv"
    if not p.exists():
        return pd.DataFrame(columns=["year", "occupation", "retrench"])
    d = pd.read_csv(p, dtype=str)
    d["year"] = pd.to_numeric(d.year, errors="coerce"); d["retrench"] = pd.to_numeric(d.retrench, errors="coerce")
    return d.dropna(subset=["year"])


# ----------------------------------------------------------------------------- wages, two years
def wage_change_by_ssoc() -> pd.DataFrame:
    """Median gross monthly wage by SSOC 2024 code, June 2024 (title-matched survey, exact matches only) and June 2025 (coded)."""
    w25 = io.load("sg_ows2025_wages")[["ssoc_code", "wage_median"]].rename(columns={"wage_median": "wage_2025"})
    w24 = io.load("sg_ows_wages_ssoc2024")
    w24 = w24[(w24.match_score >= 95) | (w24.match_method.astype(str).str.contains("exact"))][["ssoc_code", "wage_median"]].rename(columns={"wage_median": "wage_2024"})
    w25["ssoc_code"] = w25.ssoc_code.astype(str); w24["ssoc_code"] = w24.ssoc_code.astype(str)
    d = w25.merge(w24, on="ssoc_code", how="left").drop_duplicates("ssoc_code")
    d["wage_change_pct"] = (d.wage_2025 / d.wage_2024 - 1.0) * 100.0
    return d


# ----------------------------------------------------------------------------- outcome tables
def build_outcomes() -> dict:
    """Per SSOC 2024 code and per sub-major group: what has happened so far."""
    jv25 = load_jv_by_occupation(2025)
    jv25 = jv25[jv25.kind == "number"][["code", "value"]].rename(columns={"code": "ssoc_code", "value": "vacancies_2025"})
    shares = jv_shares_by_minor_group()
    emp = load_employment_submajor_series()
    wages = wage_change_by_ssoc()
    ret = load_retrenchment_incidence(); vr = load_vacancy_rate_by_group(); rs = load_resignation_rate_by_group()
    weights = io.load("sg_ssoc2024_weights")[["ssoc_code", "ssoc_title", "major_group", "submajor_label"]]
    weights["ssoc_code"] = weights.ssoc_code.astype(str)
    t = weights.merge(jv25, on="ssoc_code", how="left").merge(wages, on="ssoc_code", how="left")
    t["minor"] = t.ssoc_code.str[:4]
    sh = shares.groupby("minor", as_index=False)[[c for c in shares.columns if c.startswith("share_")]].sum(min_count=1)
    t = t.merge(sh, on="minor", how="left")
    if "share_2023" in t and "share_2025" in t:
        t["vac_share_change_2023_2025_pp"] = t.share_2025 - t.share_2023
    # sub-major employment series joined by label (D8 labels like '11', '61-62', 'X1-X5')
    emp["key"] = emp.submajor_label.str.replace(" ", "")
    t["key"] = t.submajor_label.astype(str).str.replace(" ", "")
    emp_cols = [c for c in emp.columns if c.startswith("emp_")]
    t = t.merge(emp[["key"] + emp_cols], on="key", how="left").drop(columns=["key"])
    # occupational-group series (latest year and change since 2019)
    t["occ_group"] = t.major_group.astype(str).map(GROUP_OF_MAJOR)
    def latest_and_change(df, col):
        out = {}
        for g, d in df.groupby("occ_group"):
            d = d.sort_values("year")
            last = d.iloc[-1]; base = d[d.year == 2019]
            out[g] = {f"{col}_latest": float(last[col]), f"{col}_latest_year": int(last.year), f"{col}_change_since_2019": float(last[col] - base[col].iloc[0]) if len(base) else np.nan}
        return out
    for df, col in [(ret, "retrenchment_per_1000"), (vr, "job_vacancy_rate_pct"), (rs, "resignation_rate_pct")]:
        if df.empty:
            continue
        m = latest_and_change(df, col)
        for k in [f"{col}_latest", f"{col}_latest_year", f"{col}_change_since_2019"]:
            t[k] = t.occ_group.map(lambda g: (m.get(g) or {}).get(k, np.nan))
    submajor = emp.copy()
    return {"ssoc": t, "submajor": submajor, "jv_shares_minor": shares, "retrenchment": ret, "vacancy_rate": vr, "resignation_rate": rs, "wages": wages}


# ----------------------------------------------------------------------------- is it showing up yet?
def _spearman(a: pd.Series, b: pd.Series) -> tuple[float | None, int]:
    m = a.notna() & b.notna()
    if m.sum() < 8:
        return None, int(m.sum())
    return float(pd.concat([a[m], b[m]], axis=1).corr(method="spearman").iloc[0, 1]), int(m.sum())


def _reading(rho: float | None, n: int, expected_sign: int) -> str:
    """expected_sign = -1 when disruption should lower the outcome (employment, wages, vacancies)."""
    if rho is None or n < 8:
        return "too few groups to say"
    if abs(rho) < 0.15:
        return "no clear sign yet"
    if np.sign(rho) == expected_sign:
        return "an early sign in the direction disruption would show" if abs(rho) < 0.35 else "a clear sign in the direction disruption would show"
    return "the opposite of what disruption would show: more exposed jobs are doing better so far"


def outcome_tests(outcomes: dict, forecast_ssoc: pd.DataFrame, frontier_ssoc: pd.DataFrame, econ: pd.DataFrame) -> dict:
    """Rank correlations between the map's exposure signals and the observed outcomes."""
    t = outcomes["ssoc"].copy()
    f = forecast_ssoc[["ssoc_code", "done_by_2030", "done_by_2035", "quarter_done_year_p50", "share_by_2030"]].copy(); f["ssoc_code"] = f.ssoc_code.astype(str)
    fr = frontier_ssoc[["ssoc_code", "observed_automation", "share_weight_observed"]].copy(); fr["ssoc_code"] = fr.ssoc_code.astype(str)
    t = t.merge(f, on="ssoc_code", how="left").merge(fr, on="ssoc_code", how="left")
    out = {"tests": [], "n_ssoc_with_wages_both_years": int(t.wage_change_pct.notna().sum()), "n_ssoc_with_vacancies_2025": int(t.vacancies_2025.notna().sum()), "n_minor_groups_with_vacancy_shares": int(t.dropna(subset=["vac_share_change_2023_2025_pp"]).minor.nunique()) if "vac_share_change_2023_2025_pp" in t else 0}
    # detailed occupations: wages 2024->2025 and vacancy-share change 2023->2025 against exposure
    for ycol, ylabel, sign in [("wage_change_pct", "median wage growth, June 2024 to June 2025", -1), ("vac_share_change_2023_2025_pp", "vacancy share within occupational group, 2023 to 2025 (percentage points)", -1)]:
        if ycol not in t:
            continue
        for xcol, xlabel in [("observed_automation", "AI doing it today"), ("done_by_2030", "AI doing by 2030 (central)"), ("quarter_done_year_p50", "year AI could be doing a quarter (central)")]:
            rho, n = _spearman(t[xcol], t[ycol])
            s = sign if xcol != "quarter_done_year_p50" else -sign  # a later year means less exposed
            out["tests"].append({"level": "occupation", "outcome": ylabel, "signal": xlabel, "n": n, "spearman": rho, "reading": _reading(rho, n, s)})
    # sub-major groups: employment change against employment-weighted exposure
    e = econ[(econ.level == "submajor_group") & (econ.weighting == "employment")].copy()
    e["key"] = e.scope.str.replace("submajor_", "", regex=False).str.replace(" ", "")
    sm = outcomes["submajor"].copy(); sm["key"] = sm.submajor_label.str.replace(" ", "")
    e = e.merge(sm, on="key", how="inner")
    for ycol, ylabel in [("emp_change_2023_2025_pct", "resident employment change, 2023 to 2025"), ("emp_change_2019_2025_pct", "resident employment change, 2019 to 2025")]:
        if ycol not in e:
            continue
        for xcol, xlabel in [("observed_automation", "AI doing it today"), ("done_by_2030", "AI doing by 2030 (central)"), ("quarter_done_year_central_from_curve", "year AI could be doing a quarter (central)")]:
            if xcol not in e:
                continue
            rho, n = _spearman(e[xcol], e[ycol])
            s = -1 if xcol != "quarter_done_year_central_from_curve" else 1
            out["tests"].append({"level": "sub-major group", "outcome": ylabel, "signal": xlabel, "n": n, "spearman": rho, "reading": _reading(rho, n, s)})
    # occupational-group series: the three groups, latest year vs 2019
    for name, df, col in [("retrenchment", outcomes["retrenchment"], "retrenchment_per_1000"), ("vacancy_rate", outcomes["vacancy_rate"], "job_vacancy_rate_pct"), ("resignation_rate", outcomes["resignation_rate"], "resignation_rate_pct")]:
        if df.empty:
            continue
        rows = {}
        for g, d in df[df.occ_group.isin(["total", "pmet", "clerical_sales_service", "production_transport_cleaners_labourers"])].groupby("occ_group"):
            d = d.sort_values("year"); rows[g] = {int(r.year): float(getattr(r, col)) for r in d.itertuples() if r.year >= 2019}
        out[name] = rows
    # one headline reading: the median absolute correlation across the occupation-level tests, with direction
    occ_tests = [x for x in out["tests"] if x["level"] == "occupation" and x["spearman"] is not None]
    sm_tests = [x for x in out["tests"] if x["level"] == "sub-major group" and x["spearman"] is not None]
    def summary(tests):
        if not tests:
            return None
        signed = [x["spearman"] * (-1 if "year" in x["signal"] else 1) for x in tests]  # make 'more exposed' positive
        return {"median_signed_rho": float(np.median(signed)), "n_tests": len(tests), "readings": sorted({x["reading"] for x in tests})}
    out["headline"] = {"occupation": summary(occ_tests), "submajor": summary(sm_tests)}
    return out
