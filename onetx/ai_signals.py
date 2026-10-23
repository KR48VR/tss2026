"""Frontier-AI usage and capability signals, loaded into a few tidy tables.

Sources
-------
Anthropic Economic Index (AEI)   share of Claude conversations per O*NET task, per vintage; plus per-task
                                 collaboration pattern (automation vs augmentation), AI autonomy, the time
                                 a human would need without AI, task-success rate. Six vintages, Dec 2024
                                 to May 2026, for Claude.ai and (from Aug 2025) the first-party API.
AEI labour-market impacts        per-task "penetration" and per-SOC "observed exposure".
AEI robot exposure               per-task physical flag and robot-exposure tier (E0-E3).
Microsoft "Working with AI"      Bing Copilot usage per O*NET intermediate work activity (IWA): activity
                                 share, completion rate, scope of impact; AI applicability per SOC.
OpenAI GDPval                    220 expert-written tasks in 44 occupations (which occupations the labs
                                 are benchmarking).
METR time horizons               per-model 50% task-length horizon, giving the capability frontier over
                                 time and its doubling time.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import onet
from .config import AEI_VINTAGES, RAW

AEI = RAW / "aei"
COLLAB_PATTERNS = ["directive", "feedback_loop", "validation", "task_iteration", "learning", "none", "not_classified"]
AUTOMATION_PATTERNS = ["directive", "feedback_loop"]
AUGMENTATION_PATTERNS = ["validation", "task_iteration", "learning"]


# ----------------------------------------------------------------------------- task-text -> task_id
def task_text_index(tasks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Map normalised task text to every O*NET task id carrying that text (identical statements can be
    shared by several occupations)."""
    t = onet.load_tasks() if tasks is None else tasks
    return t[["task_norm", "task_id", "onet_soc_code"]]


def _attach_task_ids(df: pd.DataFrame, text_col: str, tasks: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df.copy()
    df["task_norm"] = df[text_col].map(onet.normalize_text)
    idx = task_text_index(tasks)
    out = df.merge(idx, on="task_norm", how="left")
    return out


# ----------------------------------------------------------------------------- AEI vintages
def _pattern(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def _load_task_pct_schema(release: str) -> pd.DataFrame:
    base = AEI / release
    pct_file = base / ("task_pct_v2.csv" if (base / "task_pct_v2.csv").exists() else "onet_task_mappings.csv")
    pct = pd.read_csv(pct_file).rename(columns={"task_name": "task_text", "pct": "value"})
    pct["metric"] = "usage_pct"
    frames = [pct]
    aa = base / "automation_vs_augmentation_by_task.csv"
    if aa.exists():
        a = pd.read_csv(aa).rename(columns={"task_name": "task_text"})
        a = a.melt(id_vars="task_text", var_name="metric", value_name="value")
        a["metric"] = "collab_" + a["metric"].map(_pattern).replace({"filtered": "not_classified"})
        a["value"] = a["value"] * 100.0  # ratios -> percent of the task's conversations
        frames.append(a)
    out = pd.concat(frames, ignore_index=True)
    out["source"] = "claude_ai"
    return out


def _load_raw_weekly_schema(release: str) -> pd.DataFrame:
    frames = []
    for f in sorted((AEI / release).rglob("aei_raw_*.csv")):
        source = "1p_api" if "1p_api" in f.name else "claude_ai"
        df = pd.read_csv(f, dtype=str, usecols=["geo_id", "facet", "variable", "cluster_name", "value"])
        df = df[(df.geo_id == "GLOBAL") & df.facet.str.startswith("onet_task")].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        rows = []
        # plain usage share
        u = df[(df.facet == "onet_task") & (df.variable == "onet_task_pct")]
        rows.append(pd.DataFrame({"task_text": u.cluster_name, "metric": "usage_pct", "value": u.value}))
        # collaboration split: cluster_name = "<task>::<pattern>", value = pct of the task's conversations
        c = df[(df.facet == "onet_task::collaboration") & (df.variable == "onet_task_collaboration_pct")]
        if len(c):
            parts = c.cluster_name.str.rsplit("::", n=1, expand=True)
            rows.append(pd.DataFrame({"task_text": parts[0], "metric": "collab_" + parts[1].map(_pattern), "value": c.value}))
        # mean-valued facets
        for facet, name in [("onet_task::human_only_time", "human_only_time_mean"), ("onet_task::human_with_ai_time", "human_with_ai_time_mean"), ("onet_task::ai_autonomy", "ai_autonomy_mean")]:
            m = df[(df.facet == facet) & (df.variable == f"onet_task_{facet.split('::')[1]}_mean")]
            if len(m) == 0:
                continue
            parts = m.cluster_name.str.rsplit("::", n=1, expand=True)
            rows.append(pd.DataFrame({"task_text": parts[0], "metric": name, "value": m.value}))
        # yes/no facets -> percent yes
        for facet, name in [("onet_task::task_success", "task_success_pct"), ("onet_task::human_only_ability", "human_only_ability_pct"), ("onet_task::use_case", "use_case_work_pct")]:
            m = df[(df.facet == facet) & df.variable.str.endswith("_pct")]
            if len(m) == 0:
                continue
            parts = m.cluster_name.str.rsplit("::", n=1, expand=True)
            want = "work" if facet.endswith("use_case") else "yes"
            keep = parts[1].str.strip().str.lower() == want
            rows.append(pd.DataFrame({"task_text": parts[0][keep], "metric": name, "value": m.value[keep]}))
        out = pd.concat(rows, ignore_index=True)
        out["source"] = source
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def _load_monthly_schema(release: str, month_start: str) -> pd.DataFrame:
    frames = []
    for f in sorted((AEI / release / "data").glob("aei_*.csv")):
        source = "1p_api" if "1p_api" in f.name else "claude_ai"
        df = pd.read_csv(f, dtype=str, usecols=["date_start", "geo_id", "category_name", "hierarchy_level", "metric_id", "value", "node_name", "node_external_id"])
        df = df[(df.geo_id == "GLOBAL") & (df.category_name == "onet") & (df.hierarchy_level == "0") & (df.date_start == month_start)].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        ren = {"pct": "usage_pct", "ai_autonomy_mean": "ai_autonomy_mean", "human_only_time_mean": "human_only_time_mean", "human_with_ai_time_mean": "human_with_ai_time_mean", "human_only_ability_pct": "human_only_ability_pct", "use_case_work_pct": "use_case_work_pct"}
        for p in COLLAB_PATTERNS:
            ren[f"collaboration_{p}_pct"] = f"collab_{p}"
        df = df[df.metric_id.isin(ren)]
        out = pd.DataFrame({"task_id": pd.to_numeric(df.node_external_id, errors="coerce"), "task_text": df.node_name, "metric": df.metric_id.map(ren), "value": df.value})
        out["source"] = source
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def load_aei_task_metrics(tasks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long table of task-level AEI metrics across vintages and sources.

    Columns: vintage, mid_date, source, task_id, metric, value, plus derived `automation_share`.
    Usage shares are expressed in percent of all classified conversations in that vintage/source.
    """
    tasks = onet.load_tasks() if tasks is None else tasks
    frames = []
    for v in AEI_VINTAGES:
        if v["schema"] == "task_pct":
            df = _load_task_pct_schema(v["release"])
        elif v["schema"] == "raw_weekly":
            df = _load_raw_weekly_schema(v["release"])
        else:
            df = _load_monthly_schema(v["release"], v["mid_date"][:8] + "01")
        if "task_id" not in df.columns or df["task_id"].isna().all():
            df = _attach_task_ids(df, "task_text", tasks).drop(columns=["onet_soc_code"])
        df["vintage"] = v["vintage"]
        df["mid_date"] = pd.Timestamp(v["mid_date"])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["task_id", "value"])
    out["task_id"] = out["task_id"].astype(int)
    # put usage shares on one scale: percent of conversations. Early files are fractions.
    tot = out[out.metric == "usage_pct"].groupby(["vintage", "source"]).value.sum()
    frac = tot[tot < 2.0].index
    mask = (out.metric == "usage_pct") & out.set_index(["vintage", "source"]).index.isin(frac)
    out.loc[mask, "value"] = out.loc[mask, "value"] * 100.0
    out = out.groupby(["vintage", "mid_date", "source", "task_id", "metric"], as_index=False).value.mean()
    return out


def aei_task_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per vintage x source x task with the main metrics as columns."""
    w = long.pivot_table(index=["vintage", "mid_date", "source", "task_id"], columns="metric", values="value").reset_index()
    w.columns.name = None
    for p in COLLAB_PATTERNS:
        if f"collab_{p}" not in w.columns:
            w[f"collab_{p}"] = np.nan
    classified = w[[f"collab_{p}" for p in AUTOMATION_PATTERNS + AUGMENTATION_PATTERNS]].sum(axis=1, min_count=1)
    w["automation_share"] = w[[f"collab_{p}" for p in AUTOMATION_PATTERNS]].sum(axis=1, min_count=1) / classified.replace(0, np.nan)
    return w


# ----------------------------------------------------------------------------- AEI labour-market impacts & robots
def load_aei_task_penetration(tasks: pd.DataFrame | None = None) -> pd.DataFrame:
    df = pd.read_csv(AEI / "labor_market_impacts" / "task_penetration.csv").rename(columns={"task": "task_text"})
    df = _attach_task_ids(df, "task_text", tasks).dropna(subset=["task_id"])
    df["task_id"] = df["task_id"].astype(int)
    return df[["task_id", "onet_soc_code", "penetration"]].drop_duplicates(["task_id", "onet_soc_code"])


def load_aei_job_exposure() -> pd.DataFrame:
    df = pd.read_csv(AEI / "labor_market_impacts" / "job_exposure.csv").rename(columns={"occ_code": "soc_2018"})
    return df


def load_robot_exposure_tasks() -> pd.DataFrame:
    df = pd.read_csv(AEI / "robot_exposure" / "robot_exposure_tasks.csv", usecols=["onetsoc_code", "task_id", "time_share", "physical", "exposure_tier"], dtype={"task_id": "Int64"})
    df = df.rename(columns={"onetsoc_code": "onet_soc_code", "time_share": "time_share_claude"})
    df["physical"] = df["physical"].eq("Yes")
    df["robot_tier"] = df["exposure_tier"].map({"E0": 0, "E1": 1, "E2": 2, "E3": 3})
    return df[["onet_soc_code", "task_id", "time_share_claude", "physical", "exposure_tier", "robot_tier"]]


def load_robot_exposure_occupations() -> pd.DataFrame:
    return pd.read_csv(AEI / "robot_exposure" / "robot_exposure_occupations.csv").rename(columns={"onetsoc_code": "onet_soc_code"})


# ----------------------------------------------------------------------------- Microsoft
def load_microsoft_iwa() -> pd.DataFrame:
    df = pd.read_csv(RAW / "microsoft" / "iwa_metrics.csv").rename(columns={"IWA": "iwa_id", "title": "iwa_name"})
    return df


def load_microsoft_soc() -> pd.DataFrame:
    a = pd.read_csv(RAW / "microsoft" / "soc_metrics.csv").rename(columns={"SOC Code": "soc_2018"})
    b = pd.read_csv(RAW / "microsoft" / "ai_applicability_scores.csv").rename(columns={"SOC Code": "soc_2018"})[["soc_2018", "ai_applicability_score"]]
    return a.merge(b, on="soc_2018", how="left")


def load_microsoft_physical_tasks() -> pd.DataFrame:
    df = pd.read_csv(RAW / "microsoft" / "physical_tasks.csv").rename(columns={"Task ID": "task_id", "Physical": "physical_ms"})
    df["physical_ms"] = df["physical_ms"].astype(str).str.lower().eq("true")
    return df


# ----------------------------------------------------------------------------- OpenAI GDPval
def load_gdpval() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "openai" / "gdpval.parquet", columns=["task_id", "sector", "occupation", "prompt"])
    df = df.rename(columns={"task_id": "gdpval_task_id", "occupation": "soc_title"})
    occ = onet.load_occupations()
    occ["title_key"] = occ.title.str.casefold().str.strip()
    df["title_key"] = df.soc_title.str.casefold().str.strip()
    m = df.merge(occ[["title_key", "onet_soc_code", "soc_2018"]], on="title_key", how="left")
    return m[["gdpval_task_id", "sector", "soc_title", "onet_soc_code", "soc_2018", "prompt"]]


# ----------------------------------------------------------------------------- METR time horizons
def load_metr_runs() -> pd.DataFrame:
    rows = []
    with open(RAW / "metr" / "runs_th11.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            rows.append({k: r.get(k) for k in ["task_id", "task_family", "alias", "model", "score_binarized", "human_minutes", "equal_task_weight", "invsqrt_task_weight"]})
    return pd.DataFrame(rows)


def load_metr_release_dates() -> pd.DataFrame:
    import yaml

    y = yaml.safe_load(open(RAW / "metr" / "release_dates.yaml"))
    d = y.get("date", y)
    return pd.DataFrame({"alias": list(d.keys()), "release_date": pd.to_datetime(list(d.values()))})


def compute_metr_horizons(min_runs: int = 100) -> pd.DataFrame:
    """Fit, per model, P(success) = sigmoid(a + b * log2(human_minutes)) and report the 50% and 80%
    horizons in minutes. Mirrors METR's method in spirit (weighted logistic fit); not their exact
    bootstrap pipeline, so treat as approximate.
    """
    from sklearn.linear_model import LogisticRegression

    runs = load_metr_runs().dropna(subset=["human_minutes", "score_binarized"])
    runs = runs[runs.human_minutes > 0]
    dates = load_metr_release_dates().set_index("alias").release_date
    out = []
    for alias, g in runs.groupby("alias"):
        if len(g) < min_runs or g.score_binarized.nunique() < 2:
            continue
        X = np.log2(g.human_minutes.values).reshape(-1, 1)
        y = g.score_binarized.astype(int).values
        w = g.equal_task_weight.fillna(1.0).values
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(X, y, sample_weight=w)
        a, b = float(lr.intercept_[0]), float(lr.coef_[0][0])
        if b >= 0:  # success should fall with task length; skip degenerate fits
            continue
        h50 = 2 ** (-a / b)
        h80 = 2 ** ((np.log(0.2 / 0.8) - a) / b)
        out.append({"alias": alias, "release_date": dates.get(alias), "n_runs": len(g), "horizon50_min": h50, "horizon80_min": h80, "slope": b})
    df = pd.DataFrame(out).dropna(subset=["release_date"]).sort_values("release_date").reset_index(drop=True)
    return df


def metr_frontier_trend(h: pd.DataFrame, since: str = "2024-01-01") -> dict:
    """Exponential fit of the running-max 50% horizon against time -> doubling time in days."""
    d = h[h.release_date >= pd.Timestamp(since)].copy().sort_values("release_date")
    d["frontier"] = d["horizon50_min"].cummax()
    d = d[d["horizon50_min"] >= d["frontier"]]  # frontier-setting models only
    x = (d.release_date - pd.Timestamp("2024-01-01")).dt.days.values.astype(float)
    y = np.log2(d.horizon50_min.values)
    b, a = np.polyfit(x, y, 1)
    return {"since": since, "n_models": int(len(d)), "doubling_days": float(1 / b), "log2_horizon_at_2024_01_01": float(a), "models": d.alias.tolist()}


# ----------------------------------------------------------------------------- OpenAI (GPTs are GPTs, 2023) occupation exposure
def load_openai_gpts_exposure() -> pd.DataFrame:
    """Eloundou et al. (2023) occupation-level exposure. beta = E1 + 0.5*E2 ("exposure with some tooling");
    dv = GPT-4 ratings, human = annotators."""
    df = pd.read_csv(RAW / "openai" / "gpts_are_gpts_occ_level.csv")
    df = df.rename(columns={"O*NET-SOC Code": "onet_soc_code", "Title": "title"})
    return df[["onet_soc_code", "dv_rating_alpha", "dv_rating_beta", "dv_rating_gamma", "human_rating_alpha", "human_rating_beta", "human_rating_gamma"]]


# ----------------------------------------------------------------------------- hand-transcribed reference tables
REFERENCE = RAW.parent / "reference"


def load_openai_gwa_shares() -> pd.DataFrame:
    df = pd.read_csv(REFERENCE / "openai_chatgpt_gwa_shares_2025.csv")
    cm = onet.load_content_model()
    gwa = cm[cm.element_id.str.match(r"^4\.A\.\d\.[a-z]\.\d$")][["element_id", "element_name"]]
    return df.merge(gwa, left_on="gwa_name", right_on="element_name", how="left").drop(columns="element_name").rename(columns={"element_id": "gwa_id"})


def load_google_atlas_occupations() -> pd.DataFrame:
    df = pd.read_csv(REFERENCE / "google_atlas_v1_occupations.csv")
    occ = onet.load_occupations()
    key = lambda s: s.str.casefold().str.replace(r"[^a-z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    occ["k"] = key(occ.title)
    df["k"] = key(df.occupation_title)
    m = df.merge(occ[["k", "onet_soc_code"]], on="k", how="left").drop(columns="k")
    return m


# ----------------------------------------------------------------------------- adoption series (for the timing model)
def load_microsoft_diffusion_countries() -> pd.DataFrame:
    """Share of people aged 15-64 who used a generative-AI product, by economy (Microsoft AI Economy
    Institute, Global AI Diffusion Q2 2026)."""
    return pd.read_csv(RAW / "microsoft" / "ai_diffusion_2026_q2_countries.csv")


def load_btos_ai_adoption() -> pd.DataFrame:
    """US Census Business Trends and Outlook Survey: share of firms using AI in producing goods or
    services in the last two weeks (biweekly, Sept 2023 onwards), as shipped in AEI release 2025-09-15."""
    import openpyxl

    wb = openpyxl.load_workbook(AEI / "release_2025_09_15" / "data" / "input" / "BTOS_National.xlsx", read_only=True)
    dates = {}
    for r in wb["Collection and Reference Dates"].iter_rows(values_only=True):
        if r[0] and str(r[0]).strip().isdigit():
            dates[str(r[0]).strip()] = pd.to_datetime(str(r[4]), errors="coerce")
    ws = wb["Response Estimates"]
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(h) for h in rows[0]]
    out = []
    for r in rows[1:]:
        if str(r[0]) in ("7", "24") and r[3] == "Yes":
            q = "current_use" if str(r[0]) == "7" else "expected_use_6m"
            for i in range(4, len(r)):
                if r[i] in (None, ""):
                    continue
                out.append({"question": q, "sample_period": hdr[i], "ref_end": dates.get(hdr[i]), "pct": float(str(r[i]).rstrip("%"))})
    df = pd.DataFrame(out).dropna(subset=["ref_end"]).sort_values(["question", "ref_end"]).reset_index(drop=True)
    return df


def fit_adoption_logistic(series: pd.DataFrame, saturation: float = 0.9) -> dict:
    """Fit log(p/(K-p)) = a + b*t on the BTOS 'current use' series (K = saturation share) and return the
    growth rate and the dates at which the fitted curve crosses 25%, 50% and 75% of firms."""
    d = series[series.question == "current_use"].copy()
    d["t"] = (d.ref_end - pd.Timestamp("2024-01-01")).dt.days / 365.25
    p = d.pct.values / 100.0
    y = np.log(p / (saturation - p))
    b, a = np.polyfit(d.t.values, y, 1)
    def cross(level):
        return 2024.0 + (np.log(level / (saturation - level)) - a) / b
    return {"saturation": saturation, "growth_per_year": float(b), "intercept": float(a), "n_points": int(len(d)), "last_ref_end": str(d.ref_end.max().date()), "last_pct": float(d.pct.iloc[-1]), "year_cross_25pct": float(cross(0.25)), "year_cross_50pct": float(cross(0.50)), "year_cross_75pct": float(cross(0.75))}
