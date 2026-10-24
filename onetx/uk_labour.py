"""The United Kingdom: the fourth workforce. Employee jobs and pay by SOC 2020 unit group, the ISCO-08
link to O*NET from the ONS coding index, firm AI use from the fortnightly business survey, the Census
occupation-by-industry mix for sector shifts, the ONS's own task list per unit group, and the
apprenticeship standards (Skills England, formerly IfATE) as the Skills Framework stand-in.

Sources (data/raw/ons/ and data/raw/ifate/, see docs/DATA_SOURCES.md):
  ashe/.../Table 14.7a Annual pay - Gross <year>.xlsx  Annual Survey of Hours and Earnings table 14:
                                number of employee jobs (thousands), median and mean gross annual pay by
                                SOC 2020 unit group; 2025 provisional and 2024 revised
  soc2020_coding_index.xlsx     32,670 job titles with their SOC 2020 code and the ISCO-08 code the ONS
                                assigns to each ("ISCO-08 code based on SOC2020")
  soc2020_structure.xlsx        SOC 2020 volume 1: group titles, descriptions, entry routes and the
                                official task list per unit group
  bics/bicswave165.xlsx         Business Insights and Conditions Survey: 'which AI technologies does your
                                business currently use', every six waves since September 2023, by industry
                                and size band (the time-series sheet in the latest wave carries all waves)
  census2021_occupation_industry.json  Census 2021 (England and Wales): usual residents in employment by
                                occupation minor group (3-digit) and industry division (2-digit), from the
                                ONS flexible table API
  ../ifate/apprenticeship_standards.json  every apprenticeship standard with its route, level, typical job
                                titles, duties and knowledge/skills/behaviours
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW

ONS = RAW / "ons"
IFATE = RAW / "ifate"

MAJOR_GROUP_NAMES = {"1": "Managers, directors and senior officials", "2": "Professional occupations", "3": "Associate professional occupations",
                     "4": "Administrative and secretarial occupations", "5": "Skilled trades occupations", "6": "Caring, leisure and other service occupations",
                     "7": "Sales and customer service occupations", "8": "Process, plant and machine operatives", "9": "Elementary occupations"}
# SOC 2020 volume 1, table 1: skill level by sub-major group (sub-major 12 is level 3; the rest follow the major group)
SKILL_LEVEL_BY_MAJOR = {"1": 4, "2": 4, "3": 3, "4": 2, "5": 3, "6": 2, "7": 2, "8": 2, "9": 1}
SKILL_LEVEL_NAMES = {1: "competence from compulsory education", 2: "longer work-related training or GCSE-level qualifications", 3: "post-16 training, a body of knowledge or apprenticeship", 4: "degree-level knowledge, often with professional training"}
SECTIONS = {"A": "Agriculture, forestry and fishing", "B": "Mining and quarrying", "C": "Manufacturing", "D": "Electricity and gas", "E": "Water supply and waste",
            "F": "Construction", "G": "Wholesale and retail trade", "H": "Transportation and storage", "I": "Accommodation and food services", "J": "Information and communication",
            "K": "Financial and insurance activities", "L": "Real estate activities", "M": "Professional, scientific and technical activities", "N": "Administrative and support services",
            "O": "Public administration and defence", "P": "Education", "Q": "Human health and social work", "R": "Arts, entertainment and recreation", "S": "Other service activities",
            "T": "Households as employers", "U": "Extraterritorial organisations"}
_SECTION_RANGES = [("A", 1, 3), ("B", 5, 9), ("C", 10, 33), ("D", 35, 35), ("E", 36, 39), ("F", 41, 43), ("G", 45, 47), ("H", 49, 53), ("I", 55, 56), ("J", 58, 63), ("K", 64, 66), ("L", 68, 68),
                   ("M", 69, 75), ("N", 77, 82), ("O", 84, 84), ("P", 85, 85), ("Q", 86, 88), ("R", 90, 93), ("S", 94, 96), ("T", 97, 98), ("U", 99, 99)]
BICS_INDUSTRY_TO_SECTION = {"manufacturing": "C", "water supply": "E", "construction": "F", "wholesale and retail": "G", "transportation and storage": "H", "accommodation and food": "I",
                            "information and communication": "J", "real estate": "L", "professional, scientific": "M", "administrative and support": "N", "education": "P",
                            "human health": "Q", "arts, entertainment": "R", "other service": "S"}


def sic_section(division: str) -> str | None:
    try:
        d = int(str(division)[:2])
    except ValueError:
        return None
    for s, lo, hi in _SECTION_RANGES:
        if lo <= d <= hi:
            return s
    return None


def _rows(path: Path, sheet: str, min_row: int = 1) -> list[tuple]:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    return list(wb[sheet].iter_rows(min_row=min_row, values_only=True))


def _num(v):
    try:
        x = float(str(v).replace(",", "").strip())
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ----------------------------------------------------------------------------- classification
def load_soc_structure() -> pd.DataFrame:
    """One row per SOC 2020 unit group with its group names, description, entry routes and task text."""
    rows = _rows(ONS / "soc2020_structure.xlsx", "SOC2020 descriptions", 1)
    names, out = {}, []
    clean = lambda v: None if v is None or str(v).strip() in ("<blank>", "") else str(v).strip()
    for r in rows[1:]:
        maj, sub, minr, unit, title = (clean(r[i]) for i in range(1, 6))
        for code in (maj, sub, minr):
            if code and title:
                names[code] = title
        if unit:
            out.append({"soc_code": unit, "soc_title": title, "major_group": unit[0], "submajor": unit[:2], "minor": unit[:3], "description": clean(r[7]), "entry_routes": clean(r[8]), "tasks_text": clean(r[9]), "related_titles": clean(r[10]) if len(r) > 10 else None})
    df = pd.DataFrame(out)
    df["submajor_name"] = df.submajor.map(names)
    df["minor_name"] = df.minor.map(names)
    df["major_group_name"] = df.major_group.map(MAJOR_GROUP_NAMES)
    df["skill_level"] = np.where(df.submajor == "12", 3, df.major_group.map(SKILL_LEVEL_BY_MAJOR)).astype(float)
    return df


def load_soc_tasks() -> pd.DataFrame:
    """One row per official task item ('~ ...' lines in the structure file)."""
    s = load_soc_structure()
    out = []
    for _, r in s.dropna(subset=["tasks_text"]).iterrows():
        for i, line in enumerate(str(r.tasks_text).split("\n")):
            line = re.sub(r"^[\s~\-•*]+", "", line).strip().rstrip(";.")
            if len(line) > 3:
                out.append({"soc_code": r.soc_code, "task_seq": i, "task": line})
    return pd.DataFrame(out)


def load_soc_to_isco() -> pd.DataFrame:
    """SOC 2020 unit group -> ISCO-08 unit group, weighted by the number of coding-index job titles that
    carry each pair. `weight` sums to 1 per SOC code, `weight_rev` to 1 per ISCO code."""
    rows = _rows(ONS / "soc2020_coding_index.xlsx", "SOC2020 coding index", 1)
    hdr = [str(c).strip() if c is not None else "" for c in rows[0]]
    i_soc, i_isco = hdr.index("SOC_2020"), next(i for i, c in enumerate(hdr) if c.lower().startswith("isco-08 code"))
    pairs = []
    for r in rows[1:]:
        soc, isco = r[i_soc], r[i_isco]
        if soc is None or isco is None:
            continue
        soc, isco = str(soc).strip(), str(isco).strip()
        if re.fullmatch(r"\d{4}", soc) and re.fullmatch(r"\d{4}", isco):
            pairs.append((soc, isco))
    df = pd.DataFrame(pairs, columns=["soc_code", "isco_code"]).value_counts().reset_index(name="n_titles")
    df["weight"] = df.n_titles / df.groupby("soc_code").n_titles.transform("sum")
    df["weight_rev"] = df.n_titles / df.groupby("isco_code").n_titles.transform("sum")
    return df.sort_values(["soc_code", "weight"], ascending=[True, False]).reset_index(drop=True)


# ----------------------------------------------------------------------------- jobs and pay
def _ashe_file(year: int) -> Path:
    hits = sorted((ONS / "ashe").rglob(f"*Table 14.7a*Annual pay - Gross {year}.xlsx"))
    if not hits:
        raise FileNotFoundError(f"ASHE table 14.7a for {year} under {ONS / 'ashe'}")
    return hits[0]


def load_ashe_occupation(year: int = 2025) -> pd.DataFrame:
    """Employee jobs (thousands), median and mean gross annual pay (GBP) by SOC 2020 unit group, all
    employees. 'x' (suppressed) becomes missing."""
    rows = _rows(_ashe_file(year), "All", 6)
    out = []
    for r in rows:
        code = str(r[1]).strip() if r[1] is not None else ""
        if not re.fullmatch(r"\d{4}", code):
            continue
        out.append({"soc_code": code, "ashe_title": str(r[0]).strip(), "jobs_thousands": _num(r[2]), "median_annual": _num(r[3]), "mean_annual": _num(r[5]), "year": year})
    return pd.DataFrame(out)


def build_soc_weights(year: int = 2025) -> pd.DataFrame:
    """Per SOC 2020 unit group: employee jobs (ASHE, April of the year), median gross annual pay (ASHE;
    the mean, then the minor, sub-major or major group median where suppressed), their product as the
    wage weight, skill level and group names."""
    a = load_ashe_occupation(year)
    s = load_soc_structure().set_index("soc_code")
    df = a[["soc_code", "jobs_thousands", "median_annual", "mean_annual"]].rename(columns={"jobs_thousands": "employment_thousands"}).copy()
    df["soc_title"] = df.soc_code.map(s.soc_title).fillna(a.set_index("soc_code").ashe_title)
    df["major_group"] = df.soc_code.str[0]
    df["submajor_label"] = df.soc_code.str[:2]
    df["minor"] = df.soc_code.str[:3]
    df["submajor_name"] = df.submajor_label.map(s.drop_duplicates("submajor").set_index("submajor").submajor_name)
    df["skill_level"] = df.soc_code.map(s.skill_level).fillna(df.major_group.map(SKILL_LEVEL_BY_MAJOR).astype(float))
    df["wage_median"] = df.median_annual
    df["wage_source"] = np.where(df.wage_median.notna(), "ashe_median", None)
    m = df.wage_median.isna() & df.mean_annual.notna()
    df.loc[m, "wage_median"] = df.loc[m, "mean_annual"]
    df.loc[m, "wage_source"] = "ashe_mean_no_median"
    pub = df.dropna(subset=["median_annual"])
    for lvl, n in [("minor", 3), ("submajor_label", 2), ("major_group", 1)]:
        grp = pub.groupby(pub.soc_code.str[:n]).median_annual.median()
        m = df.wage_median.isna()
        df.loc[m, "wage_median"] = df.loc[m, lvl].map(grp)
        df.loc[m & df.wage_median.notna(), "wage_source"] = f"{lvl.replace('_label', '')}_group_median"
    df["employment_thousands"] = df.employment_thousands.fillna(0.0)
    df["wage_value"] = df.employment_thousands * df.wage_median
    df["employment_method"] = "ashe_employee_jobs"
    df["edition"] = year
    return df.drop(columns=["minor", "median_annual", "mean_annual"]).reset_index(drop=True)


def employment_change(year_from: int = 2024, year_to: int = 2025) -> pd.DataFrame:
    """Rung five for the UK: change in employee jobs and in median gross annual pay between the two ASHE
    editions (April 2024 revised, April 2025 provisional), per unit group."""
    a0, a1 = load_ashe_occupation(year_from).set_index("soc_code"), load_ashe_occupation(year_to).set_index("soc_code")
    codes = sorted(set(a0.index) | set(a1.index))
    df = pd.DataFrame({"soc_code": codes})
    df[f"emp_{year_from}"] = df.soc_code.map(a0.jobs_thousands)
    df[f"emp_{year_to}"] = df.soc_code.map(a1.jobs_thousands)
    df["emp_change_pct"] = 100.0 * (df[f"emp_{year_to}"] / df[f"emp_{year_from}"].where(df[f"emp_{year_from}"] > 0) - 1.0)
    df[f"wage_{year_from}"] = df.soc_code.map(a0.median_annual)
    df[f"wage_{year_to}"] = df.soc_code.map(a1.median_annual)
    df["wage_change_pct"] = 100.0 * (df[f"wage_{year_to}"] / df[f"wage_{year_from}"].where(df[f"wage_{year_from}"] > 0) - 1.0)
    df["year_from"], df["year_to"] = year_from, year_to
    df["wage_years"] = f"{year_from}-{year_to}"
    return df


# ----------------------------------------------------------------------------- industry mix and firm AI use
def load_census_occupation_industry() -> pd.DataFrame:
    """Census 2021, England and Wales: employed usual residents by SOC 2020 minor group (3-digit) and
    SIC 2007 section (2-digit divisions rolled up)."""
    d = json.loads((ONS / "census2021_occupation_industry.json").read_text())
    agg: dict[tuple, float] = {}
    for o in d["observations"]:
        occ = ind = None
        for dim in o["dimensions"]:
            if dim["dimension_id"].startswith("occupation"):
                occ = str(dim["option"])[:3]
            elif dim["dimension_id"].startswith("industry"):
                ind = sic_section(str(dim["option"])[:2])
        if occ is None or not occ.isdigit() or ind is None:
            continue
        agg[(occ, ind)] = agg.get((occ, ind), 0.0) + float(o["observation"])
    return pd.DataFrame([{"minor": k[0], "section": k[1], "section_name": SECTIONS[k[1]], "employed": v} for k, v in agg.items()])


def load_bics_ai_use(wave_file: Path | None = None) -> pd.DataFrame:
    """Share of businesses currently using at least one AI technology, by survey wave: 100 minus the
    'does not currently use' share minus 'not sure' (a business unsure whether it uses AI is not counted
    as a user). Nationally, by industry section and by size band; dated at the end of each wave's
    two-week reference period."""
    path = wave_file or sorted((ONS / "bics").glob("bicswave*.xlsx"))[-1]
    rows = _rows(path, "AI Current Usage TS (WTD)", 6)
    hdr = [str(c).strip().lower() if c is not None else "" for c in rows[0]]
    i_not = next(i for i, c in enumerate(hdr) if c.startswith("business does not currently use"))
    i_ns = next(i for i, c in enumerate(hdr) if c == "not sure")
    out = []
    for r in rows[1:]:
        if not r[1] or not str(r[1]).startswith("Wave") or r[2] is None:
            continue
        label = str(r[2]).strip()
        not_use, not_sure = _num(r[i_not]), _num(r[i_ns])
        if np.isnan(not_use):
            continue
        m = re.search(r"to (\d{1,2} \w+ \d{4})", str(r[0]))
        end = pd.to_datetime(m.group(1), format="%d %B %Y") if m else pd.NaT
        if label == "All businesses":
            gt, sec = "all", "ALL"
        elif re.match(r"^\d", label):
            gt, sec = "size", label
        elif label.startswith("All size bands"):
            continue
        else:
            sec = next((v for k, v in BICS_INDUSTRY_TO_SECTION.items() if label.lower().startswith(k)), None)
            if sec is None:
                continue
            gt = "industry"
        pct = 100.0 * (1.0 - not_use - (0.0 if np.isnan(not_sure) else not_sure))
        out.append({"wave": int(re.sub(r"\D", "", str(r[1]))), "ref_end": end, "date": end.year + (end.dayofyear - 1) / 365.25 if pd.notna(end) else np.nan, "group_type": gt, "industry": sec, "label": SECTIONS.get(sec, label), "pct": pct, "pct_not_sure": 100.0 * not_sure if not np.isnan(not_sure) else np.nan})
    return pd.DataFrame(out).sort_values(["group_type", "industry", "ref_end"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- apprenticeship standards (the Skills Framework stand-in)
def load_ifate_standards(status: str = "Approved for delivery") -> pd.DataFrame:
    """One row per apprenticeship standard currently approved for delivery: route, level, title, typical
    job titles, summary, duties and the knowledge, skills and behaviours statements."""
    d = json.loads((IFATE / "apprenticeship_standards.json").read_text(encoding="utf-8"))
    out = []
    strip = lambda s: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(s or "").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))).strip()
    for s in d:
        if status and s.get("status") != status:
            continue
        titles = s.get("typicalJobTitles") or []
        if isinstance(titles, str):
            titles = [titles]
        duties = [strip(x.get("dutyDetail")) for x in (s.get("duties") or []) if x.get("dutyDetail")]
        core = [bool(x.get("isThisACoreDuty", 1)) for x in (s.get("duties") or []) if x.get("dutyDetail")]
        out.append({"reference": s.get("referenceNumber"), "title": strip(s.get("title")), "level": int(str(s.get("level") or 0)) if str(s.get("level") or "").isdigit() else None, "route": s.get("route"), "status": s.get("status"),
                    "version": s.get("version"), "url": s.get("standardPageUrl") or s.get("url"), "summary": strip(s.get("occupationalSummary"))[:600], "typical_job_titles": [strip(t) for t in titles if t],
                    "duties": duties, "duty_core": core, "knowledges": [strip(x.get("detail")) for x in (s.get("knowledges") or []) if x.get("detail")], "skills": [strip(x.get("detail")) for x in (s.get("skills") or []) if x.get("detail")],
                    "behaviours": [strip(x.get("detail")) for x in (s.get("behaviours") or []) if x.get("detail")], "typical_duration_months": s.get("typicalDuration"), "approved_for_delivery": str(s.get("approvedForDelivery") or "")[:10]})
    return pd.DataFrame(out).sort_values(["route", "title"]).reset_index(drop=True)


def _norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def coding_index_titles() -> pd.DataFrame:
    """Job titles in natural word order with their SOC 2020 code, from the coding index."""
    rows = _rows(ONS / "soc2020_coding_index.xlsx", "SOC2020 coding index", 1)
    hdr = [str(c).strip() if c is not None else "" for c in rows[0]]
    i_soc, i_nat = hdr.index("SOC_2020"), next(i for i, c in enumerate(hdr) if c.startswith("INDEXOCC_-_natural"))
    out = [(str(r[i_soc]).strip(), str(r[i_nat]).strip()) for r in rows[1:] if r[i_soc] is not None and r[i_nat] is not None]
    df = pd.DataFrame(out, columns=["soc_code", "title"])
    df = df[df.soc_code.str.fullmatch(r"\d{4}")]
    df["key"] = df.title.map(_norm_title)
    return df.drop_duplicates(["soc_code", "key"]).reset_index(drop=True)


def match_standards_to_soc(standards: pd.DataFrame, structure: pd.DataFrame | None = None, text_k: int = 1) -> pd.DataFrame:
    """Standard -> SOC 2020 unit group(s). First by exact title: each typical job title (and the standard's
    own title) is looked up in the ONS coding index and the standard is shared equally over the codes
    found. Standards with no title in the index are matched by meaning (title + summary against each
    unit group's title, description and task list) and take the single best code with its score."""
    from .embed import HybridMatcher

    idx = coding_index_titles()
    by_key = idx.groupby("key").soc_code.apply(set).to_dict()
    rows, unmatched = [], []
    for i, s in standards.iterrows():
        codes: dict[str, int] = {}
        for t in list(s.typical_job_titles) + [s.title]:
            for c in by_key.get(_norm_title(t), set()):
                codes[c] = codes.get(c, 0) + 1
        if codes:
            tot = sum(codes.values())
            for c, n in codes.items():
                rows.append({"reference": s.reference, "soc_code": c, "share": n / tot, "method": "title_in_coding_index", "score": None})
        else:
            unmatched.append(i)
    if unmatched:
        st = load_soc_structure() if structure is None else structure
        corpus = (st.soc_title + ". " + st.description.fillna("") + " Tasks: " + st.tasks_text.fillna("").str.replace("~", " ")).tolist()
        m = HybridMatcher(corpus)
        q = [f"{standards.loc[i].title}. {standards.loc[i].summary}" for i in unmatched]
        ix, sc = m.query(q, k=text_k)
        for j, i in enumerate(unmatched):
            rows.append({"reference": standards.loc[i].reference, "soc_code": st.soc_code.iat[int(ix[j, 0])], "share": 1.0, "method": "text_match", "score": float(sc[j, 0])})
    return pd.DataFrame(rows)
