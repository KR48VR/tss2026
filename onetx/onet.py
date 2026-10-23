"""Load O*NET text releases into tidy tables.

Every loader returns a pandas DataFrame with snake_case columns and an `onet_version` column.
Nothing here is Singapore-specific; this is the "O*NET core" the rest of the project hangs off.

Key identifiers
---------------
onet_soc_code   O*NET-SOC 2019 code, e.g. "15-1252.00"
task_id         integer task id (stable across releases)
dwa_id/iwa_id   detailed / intermediate work activity ids in the *classic* format used up to 30.1
                (e.g. 4.A.1.a.1.I01.D01). O*NET 31.0 renumbered these; we map 31.0 ids back to the
                classic ids by name so that every external dataset lines up.
gwa_id          generalized work activity element id (4.A.x.y.z), unchanged across versions.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ONET_BASE_VERSION, ONET_DIRS

# Expected frequency, in occurrences per year, for each O*NET frequency category (scale FT).
# Used only to turn the frequency distribution into a single "how often" number.
FREQ_PER_YEAR = {1: 0.5, 2: 6, 3: 24, 4: 100, 5: 250, 6: 750, 7: 2000}


def _dir(version: str) -> Path:
    d = ONET_DIRS[version]
    if not d.exists():
        raise FileNotFoundError(f"O*NET {version} not found at {d}; run `onetx fetch` first")
    return d


def _read(version: str, name: str) -> pd.DataFrame:
    return pd.read_csv(_dir(version) / f"{name}.txt", sep="\t", dtype=str, keep_default_na=False, na_values=["n/a", ""])


def _snake(c: str) -> str:
    c = c.replace("O*NET-SOC", "onet_soc").replace("DWA ID", "dwa_id").replace("IWA ID", "iwa_id")
    c = re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower()
    return c


# ----------------------------------------------------------------------------- occupations
def load_occupations(version: str = ONET_BASE_VERSION) -> pd.DataFrame:
    df = _read(version, "Occupation Data")
    df.columns = [_snake(c) for c in df.columns]
    df = df.rename(columns={"onet_soc_code": "onet_soc_code"})
    df["soc_2018"] = df["onet_soc_code"].str[:7]
    df["onet_version"] = version
    return df[["onet_soc_code", "soc_2018", "title", "description", "onet_version"]]


def load_job_zones(version: str = ONET_BASE_VERSION) -> pd.DataFrame:
    df = _read(version, "Job Zones")
    df.columns = [_snake(c) for c in df.columns]
    df["job_zone"] = df["job_zone"].astype(int)
    return df[["onet_soc_code", "job_zone"]]


def load_alternate_titles(version: str = ONET_BASE_VERSION) -> pd.DataFrame:
    name = "Alternate Titles" if (ONET_DIRS[version] / "Alternate Titles.txt").exists() else "Job Titles"
    df = _read(version, name)
    df.columns = [_snake(c) for c in df.columns]
    col = "alternate_title" if "alternate_title" in df.columns else "job_title"
    out = df[["onet_soc_code", col]].rename(columns={col: "alt_title"})
    out["onet_version"] = version
    return out


def load_related_occupations(version: str = ONET_BASE_VERSION) -> pd.DataFrame:
    df = _read(version, "Related Occupations")
    df.columns = [_snake(c) for c in df.columns]
    return df.rename(columns={"related_onet_soc_code": "related_onet_soc_code"})


# ----------------------------------------------------------------------------- tasks
def load_tasks(version: str = ONET_BASE_VERSION) -> pd.DataFrame:
    """Task statements plus a few summary ratings per task.

    importance   mean importance rating (1-5)
    relevance    % of incumbents saying the task is relevant
    freq_per_year expected frequency from the FT category distribution
    task_weight  importance share within the occupation (sums to 1 per occupation) - the default
                 weight used when rolling tasks up to occupations.
    """
    ts = _read(version, "Task Statements")
    ts.columns = [_snake(c) for c in ts.columns]
    ts = ts.rename(columns={"task": "task_statement"})
    ts["task_id"] = ts["task_id"].astype(int)
    tr = _read(version, "Task Ratings")
    tr.columns = [_snake(c) for c in tr.columns]
    tr["task_id"] = tr["task_id"].astype(int)
    tr["data_value"] = pd.to_numeric(tr["data_value"], errors="coerce")
    im = tr[tr.scale_id == "IM"].groupby("task_id").data_value.mean().rename("importance")
    rt = tr[tr.scale_id == "RT"].groupby("task_id").data_value.mean().rename("relevance")
    ft = tr[tr.scale_id == "FT"].copy()
    ft["category"] = pd.to_numeric(ft["category"], errors="coerce")
    ft["w"] = ft["category"].map(FREQ_PER_YEAR) * ft["data_value"] / 100.0
    fq = ft.groupby("task_id").w.sum().rename("freq_per_year")
    out = ts.merge(im, on="task_id", how="left").merge(rt, on="task_id", how="left").merge(fq, on="task_id", how="left")
    # default roll-up weight: importance share within occupation (missing importance -> mean of occupation)
    out["importance_filled"] = out.groupby("onet_soc_code").importance.transform(lambda s: s.fillna(s.mean() if s.notna().any() else 3.0))
    out["task_weight"] = out["importance_filled"] / out.groupby("onet_soc_code").importance_filled.transform("sum")
    out["task_norm"] = out["task_statement"].map(normalize_text)
    out["onet_version"] = version
    cols = ["task_id", "onet_soc_code", "task_statement", "task_norm", "task_type", "importance", "relevance", "freq_per_year", "task_weight", "date", "domain_source", "onet_version"]
    return out[[c for c in cols if c in out.columns]]


def load_emerging_tasks(version: str = ONET_BASE_VERSION) -> pd.DataFrame:
    df = _read(version, "Emerging Tasks")
    df.columns = [_snake(c) for c in df.columns]
    df["onet_version"] = version
    return df


# ----------------------------------------------------------------------------- work activities hierarchy
def load_dwa_hierarchy(version: str = ONET_BASE_VERSION) -> pd.DataFrame:
    """One row per DWA with its IWA and GWA. Always expressed in *classic* ids (30.1 format).

    For 31.0 the file "GWAs to IWAs to DWAs" carries new ids; we map them to classic ids by DWA name
    (names are identical across the two releases) and keep the new id in `dwa_id_v31`.
    """
    gwa_names = load_content_model(version).set_index("element_id").element_name
    if (ONET_DIRS[version] / "DWA Reference.txt").exists():
        d = _read(version, "DWA Reference")
        d.columns = [_snake(c) for c in d.columns]
        d = d.rename(columns={"element_id": "gwa_id", "dwa_title": "dwa_name"})
        i = _read(version, "IWA Reference")
        i.columns = [_snake(c) for c in i.columns]
        i = i.rename(columns={"element_id": "gwa_id", "iwa_title": "iwa_name"})
        d = d.merge(i[["iwa_id", "iwa_name"]], on="iwa_id", how="left")
        d["dwa_id_v31"] = np.nan
        d["iwa_id_v31"] = np.nan
    else:
        d = _read(version, "GWAs to IWAs to DWAs")
        d.columns = [_snake(c) for c in d.columns]
        d = d.rename(columns={"gwa_element_id": "gwa_id", "iwa_element_id": "iwa_id_v31", "dwa_element_id": "dwa_id_v31", "dwa_element_name": "dwa_name"})
        i = _read(version, "GWAs to IWAs")
        i.columns = [_snake(c) for c in i.columns]
        i = i.rename(columns={"iwa_element_id": "iwa_id_v31", "iwa_element_name": "iwa_name"})
        d = d.merge(i[["iwa_id_v31", "iwa_name"]], on="iwa_id_v31", how="left")
        base = load_dwa_hierarchy(ONET_BASE_VERSION)[["dwa_id", "iwa_id", "dwa_name"]]
        d = d.merge(base, on="dwa_name", how="left")
    d["gwa_name"] = d["gwa_id"].map(gwa_names)
    d["onet_version"] = version
    return d[["dwa_id", "dwa_name", "iwa_id", "iwa_name", "gwa_id", "gwa_name", "dwa_id_v31", "iwa_id_v31", "onet_version"]]


def load_task_dwa(version: str = ONET_BASE_VERSION) -> pd.DataFrame:
    """Task -> DWA links in classic DWA ids."""
    df = _read(version, "Tasks to DWAs")
    df.columns = [_snake(c) for c in df.columns]
    df["task_id"] = df["task_id"].astype(int)
    if "dwa_element_id" in df.columns:  # 31.0 new ids -> classic via hierarchy
        h = load_dwa_hierarchy(version)[["dwa_id_v31", "dwa_id"]]
        df = df.rename(columns={"dwa_element_id": "dwa_id_v31"}).merge(h, on="dwa_id_v31", how="left")
    df["onet_version"] = version
    return df[["onet_soc_code", "task_id", "dwa_id", "onet_version"]]


# ----------------------------------------------------------------------------- descriptors (skills, abilities, knowledge, work activities)
DESCRIPTOR_TABLES = {
    "30.1": {"skill": "Skills", "ability": "Abilities", "knowledge": "Knowledge", "work_activity": "Work Activities", "work_style": "Work Styles"},
    "31.0": {"essential_skill": "Essential Skills", "transferable_skill": "Transferable Skills", "ability": "Abilities", "knowledge": "Knowledge", "work_activity": "Work Activities", "work_style": "Work Styles"},
}


def load_descriptor_ratings(version: str = ONET_BASE_VERSION) -> pd.DataFrame:
    """Long table: occupation x element x scale (IM importance, LV level) for skills/abilities/knowledge/work activities."""
    frames = []
    for family, fname in DESCRIPTOR_TABLES[version].items():
        df = _read(version, fname)
        df.columns = [_snake(c) for c in df.columns]
        df = df[df.scale_id.isin(["IM", "LV"])][["onet_soc_code", "element_id", "element_name", "scale_id", "data_value"]].copy()
        df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")
        df["family"] = family
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.pivot_table(index=["onet_soc_code", "element_id", "element_name", "family"], columns="scale_id", values="data_value").reset_index()
    out.columns.name = None
    out = out.rename(columns={"IM": "importance", "LV": "level"})
    out["onet_version"] = version
    return out


def load_descriptor_to_work_activity(version: str = ONET_BASE_VERSION) -> pd.DataFrame:
    """O*NET's own links from skills/abilities/work styles to the GWAs they are used in."""
    names = {
        "30.1": {"skill": "Skills to Work Activities", "ability": "Abilities to Work Activities", "work_style": "Work Styles to Work Activities"},
        "31.0": {"essential_skill": "Essential Skills to Work Activities", "transferable_skill": "Transferable Skills to Work Activities", "ability": "Abilities to Work Activities", "work_style": "Work Styles to Work Activities"},
    }[version]
    frames = []
    for family, fname in names.items():
        if not (ONET_DIRS[version] / f"{fname}.txt").exists():
            continue
        df = _read(version, fname)
        cols = list(df.columns)
        df = df.rename(columns={cols[0]: "element_id", cols[1]: "element_name", cols[2]: "gwa_id", cols[3]: "gwa_name"})
        df["family"] = family
        frames.append(df[["family", "element_id", "element_name", "gwa_id", "gwa_name"]])
    out = pd.concat(frames, ignore_index=True)
    out["onet_version"] = version
    return out


def load_technology_skills(version: str = ONET_BASE_VERSION) -> pd.DataFrame:
    name = "Technology Skills" if (ONET_DIRS[version] / "Technology Skills.txt").exists() else "Software Skills"
    df = _read(version, name)
    df.columns = [_snake(c) for c in df.columns]
    df = df.rename(columns={"example": "workplace_example", "commodity_code": "element_id", "commodity_title": "element_name"})
    df["onet_version"] = version
    keep = [c for c in ["onet_soc_code", "workplace_example", "element_id", "element_name", "hot_technology", "in_demand", "onet_version"] if c in df.columns]
    return df[keep]


def load_content_model(version: str = ONET_BASE_VERSION) -> pd.DataFrame:
    df = _read(version, "Content Model Reference")
    df.columns = [_snake(c) for c in df.columns]
    df["onet_version"] = version
    return df


# ----------------------------------------------------------------------------- text normalisation shared with other modules
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Lower-case, unify quotes/dashes, collapse whitespace, strip trailing period. Used for joining
    datasets that carry task statements as text rather than ids."""
    if not isinstance(s, str):
        return ""
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace("\xa0", " ")
    s = _WS.sub(" ", s).strip().lower()
    return s.rstrip(".").strip()
