"""Load Singapore sources: SingStat's SSOC classification tables and SkillsFuture's Skills Framework.

SSOC (Singapore Standard Occupational Classification) is Singapore's occupation code list. It follows
the ISCO-08 structure (1 to 5 digits). SingStat publishes correspondence tables to ISCO-08 and an
alphabetical index in which Skills Framework job roles are tagged with "(SFw-<Sector>)".

The Skills Framework (SFw) dataset from the Jobs-Skills Portal lists, for 38+ sectors, every job role
with its Critical Work Functions, Key Tasks, and the Technical Skills & Competencies (TSC) and
Critical Core Skills (CCS) it needs, with proficiency levels.
"""
from __future__ import annotations

import re
import warnings

import pandas as pd

from .config import RAW

SG = RAW / "singapore"
SFW_FILES = {
    "dataset": SG / "jobsandskills-skillsfuture-skills-framework-dataset.xlsx",
    "unique_skills": SG / "jobsandskills-skillsfuture-unique-skills-list.xlsx",
    "tsc_mapping": SG / "jobsandskills-skillsfuture-tsc-to-unique-skills-mapping.xlsx",
}
SSOC_FILES = {
    2024: {
        "structure": SG / "ssoc2024_structure.xlsx",
        "definitions": SG / "ssoc2024_detailed_definitions.xlsx",
        "alpha": SG / "ssoc2024_alpha_index.xlsx",
        "isco": SG / "ssoc2024_to_isco08.xlsx",
        "to_2020": SG / "ssoc2024_to_ssoc2020.xlsx",
    },
    2020: {
        "structure": SG / "ssoc2020_structure.xlsx",
        "definitions": SG / "ssoc2020_detailed_definitions.xlsx",
        "alpha": SG / "ssoc2020_alpha_index.xlsx",
        "isco": SG / "ssoc2020_to_isco08.xlsx",
    },
}


def _xl(path, sheet=0, header=0) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.read_excel(path, sheet_name=sheet, header=header, dtype=str)


def _clean(s):
    if not isinstance(s, str):
        return None
    s = s.replace("\x9f", "").replace("\xa0", " ").strip()
    return None if s in ("", "<Blank>", "-", "nan") else s


# ----------------------------------------------------------------------------- SSOC
def load_ssoc_structure(edition: int = 2024) -> pd.DataFrame:
    df = _xl(SSOC_FILES[edition]["structure"], header=4).iloc[:, :2]
    df.columns = ["ssoc_code", "ssoc_title"]
    df = df.dropna(subset=["ssoc_code"])
    df["ssoc_code"] = df["ssoc_code"].astype(str).str.strip()
    df["ssoc_title"] = df["ssoc_title"].map(_clean)
    df["level"] = df["ssoc_code"].str.len()
    df["parent_code"] = df["ssoc_code"].str[:-1].where(df["level"] > 1, None)
    df["edition"] = edition
    return df.reset_index(drop=True)


def load_ssoc_definitions(edition: int = 2024) -> pd.DataFrame:
    """Definitions at every level. SingStat gives the bullet list of tasks at the 4-digit unit-group
    level and prose definitions at both 4- and 5-digit levels."""
    df = _xl(SSOC_FILES[edition]["definitions"], header=4)
    df.columns = ["ssoc_code", "ssoc_title", "groups_under", "definition", "tasks_text", "notes", "examples_included", "examples_excluded"][: len(df.columns)]
    df = df.dropna(subset=["ssoc_code"])
    for c in df.columns[1:]:
        df[c] = df[c].map(_clean)
    df["ssoc_code"] = df["ssoc_code"].astype(str).str.strip()
    df["level"] = df["ssoc_code"].str.len()
    df["edition"] = edition
    return df.reset_index(drop=True)


def load_ssoc_tasks(edition: int = 2024) -> pd.DataFrame:
    """One row per SSOC unit-group (4-digit) task bullet, with the unit-group code."""
    d = load_ssoc_definitions(edition)
    d = d[d.tasks_text.notna()]
    rows = []
    for _, r in d.iterrows():
        for i, line in enumerate(str(r.tasks_text).split("\n")):
            line = re.sub(r"^[\s\-•·*]+", "", line).strip()
            if len(line) < 4:
                continue
            rows.append({"ssoc_code": r.ssoc_code, "ssoc_title": r.ssoc_title, "task_seq": i, "ssoc_task": line, "edition": edition})
    out = pd.DataFrame(rows)
    out["ssoc_task_id"] = out["ssoc_code"] + "-" + out["task_seq"].astype(str)
    return out


_SFW_RE = re.compile(r"^(?P<title>.*?)\s*\(SFw-(?P<sector>[^)]+)\)\s*$")


def load_ssoc_alpha_index(edition: int = 2024) -> pd.DataFrame:
    """Alphabetical index: principal titles, alternative titles and Skills Framework job-role titles,
    each with the SSOC code SingStat assigned."""
    df = _xl(SSOC_FILES[edition]["alpha"], sheet=0, header=7)
    df = df.iloc[:, :3]
    df.columns = ["ssoc_code", "index_title", "title_type"]
    df = df.dropna(subset=["ssoc_code", "index_title"])
    df["ssoc_code"] = df["ssoc_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    df = df[df.ssoc_code.str.fullmatch(r"\d{5}")]
    df["title_type"] = df["title_type"].str.strip()
    m = df["index_title"].str.extract(_SFW_RE)
    df["sfw_role_title"] = m["title"].str.strip()
    df["sfw_sector"] = m["sector"].str.strip()
    df["edition"] = edition
    return df.reset_index(drop=True)


def load_ssoc_isco(edition: int = 2024) -> pd.DataFrame:
    """SSOC <-> ISCO-08 correspondence (SingStat). `part` flags a partial correspondence."""
    f = SSOC_FILES[edition]["isco"]
    a = _xl(f, sheet=0, header=4)
    a.columns = ["ssoc_title", "ssoc_code", "isco_code", "isco_part", "isco_title"][: len(a.columns)]
    b = _xl(f, sheet=1, header=4)
    b.columns = ["isco_title", "isco_code", "ssoc_code", "ssoc_part", "ssoc_title"][: len(b.columns)]
    a["direction"] = "ssoc_to_isco"
    b["direction"] = "isco_to_ssoc"
    out = pd.concat([a, b], ignore_index=True)
    for c in ["ssoc_code", "isco_code"]:
        out[c] = out[c].map(_clean)
    out = out.dropna(subset=["ssoc_code", "isco_code"])
    out["ssoc_code"] = out["ssoc_code"].str.replace(r"\.0$", "", regex=True).str.strip()
    out["isco_code"] = out["isco_code"].str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(4)
    out["part"] = out["isco_part"].map(_clean).notna() | out["ssoc_part"].map(_clean).notna()
    out["edition"] = edition
    return out[["ssoc_code", "ssoc_title", "isco_code", "isco_title", "part", "direction", "edition"]].reset_index(drop=True)


def load_ssoc_2024_2020() -> pd.DataFrame:
    a = _xl(SSOC_FILES[2024]["to_2020"], sheet=1, header=4).iloc[:, :5]
    a.columns = ["ssoc2024_title", "ssoc2024_code", "ssoc2020_code", "part", "ssoc2020_title"]
    a = a.dropna(subset=["ssoc2024_code", "ssoc2020_code"])
    for c in ["ssoc2024_code", "ssoc2020_code"]:
        a[c] = a[c].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    a["part"] = a["part"].map(_clean).notna()
    return a.reset_index(drop=True)


# ----------------------------------------------------------------------------- Skills Framework
def _sfw_sheet(sheet: str) -> pd.DataFrame:
    df = _xl(SFW_FILES["dataset"], sheet=sheet)
    df.columns = [re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower() for c in df.columns]
    return df


def load_sfw_roles() -> pd.DataFrame:
    df = _sfw_sheet("Job Role_Description")
    df["role_id"] = _role_id(df)
    return df[["role_id", "sector", "track", "job_role", "job_role_description", "performance_expectation"]]


def _role_id(df: pd.DataFrame) -> pd.Series:
    return (df["sector"].str.strip() + " | " + df["track"].fillna("").str.strip() + " | " + df["job_role"].str.strip())


def load_sfw_key_tasks() -> pd.DataFrame:
    df = _sfw_sheet("Job Role_CWF_KT")
    df["role_id"] = _role_id(df)
    df = df.rename(columns={"critical_work_function": "cwf", "key_tasks": "key_task"})
    df = df.dropna(subset=["key_task"]).reset_index(drop=True)
    df["key_task_id"] = range(len(df))
    return df[["key_task_id", "role_id", "sector", "track", "job_role", "cwf", "key_task"]]


def load_sfw_role_tscs() -> pd.DataFrame:
    df = _sfw_sheet("Job Role_TSC_CCS")
    df["role_id"] = _role_id(df)
    df = df.rename(columns={"tsc_ccs_title": "tsc_title", "tsc_ccs_type": "tsc_type", "tsc_ccs_code": "tsc_code"})
    df["proficiency_level"] = pd.to_numeric(df["proficiency_level"], errors="coerce")
    return df[["role_id", "sector", "track", "job_role", "tsc_code", "tsc_title", "tsc_type", "proficiency_level"]]


def load_sfw_tsc_key() -> pd.DataFrame:
    df = _sfw_sheet("TSC_CCS_Key")
    df = df.rename(columns={"tsc_ccs_category": "tsc_category", "tsc_ccs_title": "tsc_title", "tsc_ccs_description": "tsc_description", "tsc_ccs_type": "tsc_type"})
    return df[["tsc_code", "sector", "tsc_category", "tsc_title", "tsc_description", "tsc_type", "latest_update_date"]]


def load_sfw_tsc_ka() -> pd.DataFrame:
    df = _sfw_sheet("TSC_CCS_K&A")
    df = df.rename(columns={"tsc_ccs_type": "tsc_type", "tsc_ccs_code": "tsc_code", "tsc_ccs_category": "tsc_category", "tsc_ccs_title": "tsc_title", "tsc_ccs_description": "tsc_description", "knowledge_ability_items": "ka_item", "knowledge_ability_classification": "ka_type"})
    df["proficiency_level"] = pd.to_numeric(df["proficiency_level"], errors="coerce")
    return df[["tsc_code", "tsc_type", "sector", "tsc_category", "tsc_title", "tsc_description", "proficiency_level", "proficiency_description", "ka_item", "ka_type"]]


def load_unique_skills() -> pd.DataFrame:
    df = _xl(SFW_FILES["unique_skills"], sheet="Unique Skills List")
    df.columns = [re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower() for c in df.columns]
    df = df.rename(columns={"emerging_skills": "emerging", "casl_skills": "casl"})
    for c in ["emerging", "casl"]:
        df[c] = df[c].astype(str).str.lower().eq("true")
    return df


def load_tsc_to_unique_skill() -> pd.DataFrame:
    df = _xl(SFW_FILES["tsc_mapping"], sheet="data")
    df.columns = [re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower() for c in df.columns]
    df = df.rename(columns={"skills_framework_skill_code": "tsc_code", "unique_skill_updated_skill_title": "unique_skill_title", "unique_skill_updated_skill_sfs_status": "emerging", "unique_skill_updated_casl_status": "casl"})
    for c in ["emerging", "casl"]:
        df[c] = df[c].astype(str).str.lower().eq("true")
    return df[["tsc_code", "skills_framework_skill_title", "unique_skill_title", "emerging", "casl"]]
