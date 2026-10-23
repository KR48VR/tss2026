"""Per-task frontier scores and their propagation through the crosswalk graph.

The unit of analysis is the O*NET task (18,796 of them). For each task we assemble:

usage_latest          share (%) of Claude.ai conversations mapped to the task, mean of the two latest
                      monthly vintages. 0 = not observed in the published data.
usage_index           percentile rank of usage_latest among tasks with any usage (0 for none).
usage_trend           yearly change in log usage share, fitted across vintages (>= 3 points).
automation_share      among classified conversations, the share with an automation-style pattern
                      (directive + feedback loop). Imputed from the task's DWA / IWA / global mean when
                      the task has no published value (flag automation_imputed).
ai_autonomy           1-5 mean autonomy rating in the latest vintage.
human_only_minutes    time a human would need without AI (AEI estimate), minutes.
penetration           Anthropic's labour-market "task penetration".
ms_share/ms_completion  Microsoft Copilot activity share and completion rate for the task's IWAs.
physical/robot_tier   from Anthropic's robot-exposure release.
observed_automation   = 1{usage_latest > 0} x automation_share : how much of the observed use looks like
                      the AI doing the task rather than helping with it (Project Iceberg's signal).
status                plain-language bucket, see STATUS_RULES below.

Roll-ups push these through: task -> DWA -> IWA -> GWA; task -> occupation -> SOC/ISCO/SSOC;
occupation -> skills/knowledge/abilities; SSOC -> Skills Framework roles; key tasks and TSCs via the
text crosswalks. All roll-ups are weighted means where weights are documented in each function.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ai_signals as ai
from . import onet

STATUS_RULES = """
physical_robotics        task is physical (robot-exposure release); chat-AI usage is not the right signal
observed_automation      observed usage and automation_share >= 0.5 (AI mostly does the task when used)
observed_augmentation    observed usage and automation_share < 0.5 (AI mostly helps a person do the task)
not_observed             no published usage in any vintage
"""


def _weighted_mean(df: pd.DataFrame, by, col: str, w: str) -> pd.Series:
    d = df[[*(by if isinstance(by, list) else [by]), col, w]].dropna(subset=[col])
    d = d.assign(_wx=d[col] * d[w])
    g = d.groupby(by)
    return g._wx.sum() / g[w].sum()


def _usage_trend(wide: pd.DataFrame, source: str = "claude_ai", min_points: int = 3) -> pd.Series:
    """Per task: slope of log(usage) vs years, over vintages with usage > 0."""
    d = wide[(wide.source == source) & (wide.usage_pct > 0)].copy()
    d["t"] = (d.mid_date - pd.Timestamp("2025-01-01")).dt.days / 365.25
    d["ly"] = np.log(d.usage_pct)
    out = {}
    for tid, g in d.groupby("task_id"):
        if len(g) < min_points:
            continue
        b = np.polyfit(g.t.values, g.ly.values, 1)[0]
        out[tid] = b
    return pd.Series(out, name="usage_trend")


def build_task_frontier(latest_vintages=("2026-04", "2026-05")) -> pd.DataFrame:
    tasks = onet.load_tasks()
    td = onet.load_task_dwa()
    h = onet.load_dwa_hierarchy()
    long = ai.load_aei_task_metrics(tasks)
    wide = ai.aei_task_wide(long)

    latest = wide[(wide.source == "claude_ai") & wide.vintage.isin(latest_vintages)].groupby("task_id").agg(
        usage_latest=("usage_pct", "mean"), automation_share=("automation_share", "mean"), ai_autonomy=("ai_autonomy_mean", "mean"),
        human_only_hours=("human_only_time_mean", "mean"), human_with_ai_minutes=("human_with_ai_time_mean", "mean"),
        task_success_pct=("task_success_pct", "mean"), human_only_ability_pct=("human_only_ability_pct", "mean"), use_case_work_pct=("use_case_work_pct", "mean"))
    latest_api = wide[(wide.source == "1p_api") & wide.vintage.isin(latest_vintages)].groupby("task_id").agg(usage_latest_api=("usage_pct", "mean"), automation_share_api=("automation_share", "mean"))
    first = wide[(wide.source == "claude_ai") & (wide.vintage == "2024-12")].groupby("task_id").usage_pct.mean().rename("usage_first")
    nv = wide[(wide.source == "claude_ai") & (wide.usage_pct > 0)].groupby("task_id").vintage.nunique().rename("usage_n_vintages")
    # fall back to earlier vintages for the time estimate when the latest months do not carry it
    any_time = wide[(wide.source == "claude_ai")].dropna(subset=["human_only_time_mean"]).sort_values("mid_date").groupby("task_id").human_only_time_mean.last().rename("human_only_hours_any")
    any_auto = wide[(wide.source == "claude_ai")].dropna(subset=["automation_share"]).sort_values("mid_date").groupby("task_id").automation_share.last().rename("automation_share_any")

    tf = tasks.merge(latest, on="task_id", how="left").merge(latest_api, on="task_id", how="left").merge(first, on="task_id", how="left").merge(nv, on="task_id", how="left")
    tf = tf.merge(any_time, on="task_id", how="left").merge(any_auto, on="task_id", how="left")
    tf["usage_latest"] = tf["usage_latest"].fillna(0.0)
    tf["usage_n_vintages"] = tf["usage_n_vintages"].fillna(0).astype(int)
    tf["automation_share"] = tf["automation_share"].fillna(tf["automation_share_any"])
    tf["human_only_hours"] = tf["human_only_hours"].fillna(tf["human_only_hours_any"])
    tf = tf.drop(columns=["automation_share_any", "human_only_hours_any"])
    tf = tf.merge(_usage_trend(wide), left_on="task_id", right_index=True, how="left")

    pos = tf.usage_latest > 0
    tf["usage_index"] = 0.0
    tf.loc[pos, "usage_index"] = tf.loc[pos, "usage_latest"].rank(pct=True)

    # impute automation share: DWA mean -> IWA mean -> global mean (observed tasks only)
    tf["automation_imputed"] = tf["automation_share"].isna()
    obs = tf[tf.automation_share.notna()][["task_id", "automation_share"]].merge(td[["task_id", "dwa_id"]], on="task_id").merge(h[["dwa_id", "iwa_id"]], on="dwa_id")
    dwa_mean = obs.groupby("dwa_id").automation_share.mean()
    iwa_mean = obs.groupby("iwa_id").automation_share.mean()
    glob = float(tf.automation_share.mean())
    t2d = td.merge(h[["dwa_id", "iwa_id"]], on="dwa_id")
    t2d["dwa_m"] = t2d.dwa_id.map(dwa_mean)
    t2d["iwa_m"] = t2d.iwa_id.map(iwa_mean)
    imp = t2d.groupby("task_id").agg(dwa_m=("dwa_m", "mean"), iwa_m=("iwa_m", "mean"))
    tf = tf.merge(imp, on="task_id", how="left")
    tf["automation_share"] = tf["automation_share"].fillna(tf["dwa_m"]).fillna(tf["iwa_m"]).fillna(glob)
    tf = tf.drop(columns=["dwa_m", "iwa_m"])

    # penetration, robots, Microsoft
    pen = ai.load_aei_task_penetration(tasks).groupby("task_id").penetration.mean()
    tf["penetration"] = tf.task_id.map(pen)
    rb = ai.load_robot_exposure_tasks().drop_duplicates("task_id").set_index("task_id")
    tf["physical"] = tf.task_id.map(rb.physical).fillna(False).astype(bool)
    tf["robot_tier"] = tf.task_id.map(rb.robot_tier)
    tf["time_share_claude"] = tf.task_id.map(rb.time_share_claude)
    ms = ai.load_microsoft_iwa().set_index("iwa_id")
    t2i = t2d.groupby("task_id").iwa_id.apply(list)
    tf["ms_share"] = tf.task_id.map(t2i.apply(lambda L: float(np.nanmean([ms.share_user.get(i, np.nan) for i in L])) if L else np.nan))
    tf["ms_completion"] = tf.task_id.map(t2i.apply(lambda L: float(np.nanmean([ms.completion_user.get(i, np.nan) for i in L])) if L else np.nan))
    tf["ms_scope"] = tf.task_id.map(t2i.apply(lambda L: float(np.nanmean([ms.impact_scope_user.get(i, np.nan) for i in L])) if L else np.nan))

    tf["observed_automation"] = np.where(tf.usage_latest > 0, tf.automation_share, 0.0)
    tf["status"] = np.select(
        [tf.physical, (tf.usage_latest > 0) & (tf.automation_share >= 0.5), tf.usage_latest > 0],
        ["physical_robotics", "observed_automation", "observed_augmentation"], default="not_observed")
    return tf


# ----------------------------------------------------------------------------- roll-ups
ROLL_COLS = ["usage_index", "automation_share", "observed_automation", "ai_autonomy", "penetration", "ms_share", "ms_completion"]


def rollup_activities(tf: pd.DataFrame) -> pd.DataFrame:
    """DWA / IWA / GWA level means (each task counted once per activity it maps to, weighted by task_weight)."""
    td = onet.load_task_dwa()[["task_id", "dwa_id"]]
    h = onet.load_dwa_hierarchy()
    d = tf.merge(td, on="task_id").merge(h, on="dwa_id")
    frames = []
    for level, key, name in [("dwa", "dwa_id", "dwa_name"), ("iwa", "iwa_id", "iwa_name"), ("gwa", "gwa_id", "gwa_name")]:
        g = d.groupby(key)
        out = pd.DataFrame({"name": g[name].first(), "n_tasks": g.task_id.nunique(), "share_tasks_observed": g.apply(lambda x: (x.usage_latest > 0).mean()), "share_physical": g.physical.mean()})
        for c in ROLL_COLS:
            out[c] = _weighted_mean(d, key, c, "task_weight")
        out["level"] = level
        out = out.reset_index().rename(columns={key: "activity_id"})
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def rollup_occupations(tf: pd.DataFrame) -> pd.DataFrame:
    g = tf.groupby("onet_soc_code")
    out = pd.DataFrame({"n_tasks": g.task_id.nunique(), "share_weight_observed": g.apply(lambda x: x.task_weight[x.usage_latest > 0].sum()), "share_weight_physical": g.apply(lambda x: x.task_weight[x.physical].sum()), "share_weight_observed_automation": g.apply(lambda x: x.task_weight[x.status == "observed_automation"].sum())})
    for c in ROLL_COLS:
        out[c] = _weighted_mean(tf, "onet_soc_code", c, "task_weight")
    out = out.reset_index()
    occ = onet.load_occupations()[["onet_soc_code", "soc_2018", "title"]]
    out = occ.merge(out, on="onet_soc_code", how="left")
    out = out.merge(onet.load_job_zones(), on="onet_soc_code", how="left")
    out = out.merge(ai.load_aei_job_exposure()[["soc_2018", "observed_exposure"]].rename(columns={"observed_exposure": "aei_job_exposure"}), on="soc_2018", how="left")
    out = out.merge(ai.load_microsoft_soc()[["soc_2018", "ai_applicability_score", "coverage_user", "completion_user"]].rename(columns={"coverage_user": "ms_coverage", "completion_user": "ms_completion_occ"}), on="soc_2018", how="left")
    out = out.merge(ai.load_robot_exposure_occupations()[["onet_soc_code", "robot_exposure_index"]], on="onet_soc_code", how="left")
    gd = ai.load_gdpval().groupby("onet_soc_code").gdpval_task_id.nunique().rename("gdpval_n_tasks")
    out = out.merge(gd, on="onet_soc_code", how="left")
    out["gdpval_covered"] = out.gdpval_n_tasks.fillna(0) > 0
    return out


def rollup_skills(occ: pd.DataFrame, act: pd.DataFrame, version: str = "30.1") -> pd.DataFrame:
    """Two views per descriptor element (skill / knowledge / ability / work activity):

    via_occupations  importance-weighted mean of occupation-level observed_automation across occupations
                     where the element is at least moderately important (importance >= 3)
    via_activities   mean observed_automation of the GWAs O*NET links the element to (skills/abilities)
    """
    r = onet.load_descriptor_ratings(version)
    r = r[r.importance >= 3.0].merge(occ[["onet_soc_code", "observed_automation", "usage_index", "automation_share"]], on="onet_soc_code")
    g = r.groupby(["family", "element_id", "element_name"])
    out = pd.DataFrame({"n_occupations": g.onet_soc_code.nunique(), "via_occupations_observed_automation": g.apply(lambda x: np.average(x.observed_automation.fillna(0), weights=x.importance)), "via_occupations_usage_index": g.apply(lambda x: np.average(x.usage_index.fillna(0), weights=x.importance))}).reset_index()
    links = onet.load_descriptor_to_work_activity(version)
    gwa = act[act.level == "gwa"].set_index("activity_id")
    links["gwa_obs"] = links.gwa_id.map(gwa.observed_automation)
    links["gwa_usage"] = links.gwa_id.map(gwa.usage_index)
    la = links.groupby("element_id").agg(via_activities_observed_automation=("gwa_obs", "mean"), via_activities_usage_index=("gwa_usage", "mean"), n_linked_gwas=("gwa_id", "nunique"))
    out = out.merge(la, on="element_id", how="left")
    out["onet_version"] = version
    return out


def push_to_ssoc(occ: pd.DataFrame, o2s: pd.DataFrame, edition: int = 2024) -> pd.DataFrame:
    """SSOC 5-digit scores = weight_rev-weighted means of the O*NET occupations that map in."""
    cols = ["observed_automation", "usage_index", "automation_share", "share_weight_observed", "share_weight_physical", "aei_job_exposure", "ai_applicability_score", "robot_exposure_index"]
    m = o2s.merge(occ[["onet_soc_code", *cols]], on="onet_soc_code", how="inner")
    out = pd.DataFrame({"ssoc_title": m.groupby("ssoc_code").ssoc_title.first(), "n_onet_occupations": m.groupby("ssoc_code").onet_soc_code.nunique()})
    for c in cols:
        out[c] = _weighted_mean(m, "ssoc_code", c, "weight_rev")
    out = out.reset_index()
    out["ssoc_unit"] = out.ssoc_code.str[:4]
    out["edition"] = edition
    return out


def push_to_sfw(role_ssoc: pd.DataFrame, ssoc_scores: pd.DataFrame, keytask_links: pd.DataFrame, tf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Skills Framework roles scored two ways: through their SSOC code(s) and through their key tasks."""
    from . import singapore as sg

    roles = sg.load_sfw_roles()
    cols = ["observed_automation", "usage_index", "automation_share", "share_weight_observed", "share_weight_physical", "aei_job_exposure", "ai_applicability_score"]
    m = role_ssoc.merge(ssoc_scores[["ssoc_code", *cols]], on="ssoc_code", how="left")
    via_ssoc = pd.DataFrame({f"{c}_via_ssoc": _weighted_mean(m, "role_id", c, "weight") for c in cols})
    # key-task route: score each key task by its top-3 matched O*NET tasks (score-weighted), then average per role
    kt = sg.load_sfw_key_tasks()[["key_task_id", "role_id"]]
    k = keytask_links.merge(tf[["task_id", "observed_automation", "usage_index", "automation_share", "usage_latest", "physical", "human_only_hours"]], on="task_id", how="left")
    k["w"] = k.score.clip(lower=0.01)
    per_kt = pd.DataFrame({c: _weighted_mean(k, "key_task_id", c, "w") for c in ["observed_automation", "usage_index", "automation_share", "human_only_hours"]})
    per_kt["share_physical"] = k.groupby("key_task_id").physical.mean()
    per_kt["top_score"] = k[k["rank"] == 1].set_index("key_task_id").score
    per_kt["top_task_id"] = k[k["rank"] == 1].set_index("key_task_id").task_id
    per_kt["any_observed"] = k.groupby("key_task_id").usage_latest.max() > 0
    per_kt = per_kt.reset_index().merge(kt, on="key_task_id")
    g = per_kt.groupby("role_id")
    via_kt = pd.DataFrame({"n_key_tasks": g.key_task_id.nunique(), "observed_automation_via_tasks": g.observed_automation.mean(), "usage_index_via_tasks": g.usage_index.mean(), "automation_share_via_tasks": g.automation_share.mean(), "share_key_tasks_observed": g.any_observed.mean(), "share_key_tasks_physical": g.share_physical.mean(), "mean_match_score": g.top_score.mean()})
    out = roles.merge(via_ssoc, left_on="role_id", right_index=True, how="left").merge(via_kt, left_on="role_id", right_index=True, how="left")
    return out, per_kt


# ----------------------------------------------------------------------------- barriers: why a task may stay human even when AI could do it
INTERPERSONAL_GWAS = {
    "4.A.4.a.4": "Establishing and Maintaining Interpersonal Relationships",
    "4.A.4.a.5": "Assisting and Caring for Others",
    "4.A.4.a.6": "Selling or Influencing Others",
    "4.A.4.a.7": "Resolving Conflicts and Negotiating with Others",
    "4.A.4.a.8": "Performing for or Working Directly with the Public",
    "4.A.4.b.1": "Coordinating the Work and Activities of Others",
    "4.A.4.b.2": "Developing and Building Teams",
    "4.A.4.b.3": "Training and Teaching Others",
    "4.A.4.b.4": "Guiding, Directing, and Motivating Subordinates",
    "4.A.4.b.5": "Coaching and Developing Others",
    "4.A.4.c.2": "Staffing Organizational Units",
}
BARRIER_WEIGHTS = {"interpersonal": 0.35, "in_person": 0.25, "accountability": 0.25, "handover": 0.15}


def _work_context_rank(names: list[str], version: str = "30.1") -> pd.Series:
    """Percentile rank (0-1) per occupation of the mean of the named work-context elements (CX scale)."""
    wc = onet._read(version, "Work Context")
    wc.columns = [onet._snake(c) for c in wc.columns]
    wc = wc[(wc.scale_id == "CX") & wc.element_name.isin(names)].copy()
    wc["data_value"] = pd.to_numeric(wc.data_value, errors="coerce")
    return wc.groupby("onet_soc_code").data_value.mean().rank(pct=True)


def task_barriers(tf: pd.DataFrame) -> pd.DataFrame:
    """Per-task barrier components (0-1) and a combined index.

    b_interpersonal   share of the task's detailed-work-activity links that fall in interpersonal GWAs
                      (caring, selling, negotiating, teaching, supervising, working with the public)
    b_in_person       occupation percentile of physical proximity, face-to-face discussion, contact with
                      others, dealing with external customers (O*NET work context)
    b_accountability  occupation percentile of consequence of error, impact and frequency of decisions
    b_handover        from usage data: how often people kept the task even with AI in the loop
                      (human_only_ability_pct), rescaled 50..100 -> 0..1; occupation mean where missing
    barrier_index     weighted sum (weights in BARRIER_WEIGHTS); physical tasks are handled separately
    barrier_type      the dominant component, or "none" when the index is below 0.3
    """
    td = onet.load_task_dwa()[["task_id", "dwa_id"]]
    h = onet.load_dwa_hierarchy()[["dwa_id", "gwa_id"]]
    m = td.merge(h, on="dwa_id")
    m["inter"] = m.gwa_id.isin(INTERPERSONAL_GWAS)
    b = tf[["task_id", "onet_soc_code", "physical", "human_only_ability_pct"]].copy()
    b["b_interpersonal"] = b.task_id.map(m.groupby("task_id").inter.mean()).fillna(0.0)
    inp = _work_context_rank(["Physical Proximity", "Face-to-Face Discussions with Individuals and Within Teams", "Contact With Others", "Deal With External Customers or the Public in General"])
    acc = _work_context_rank(["Consequence of Error", "Impact of Decisions on Co-workers or Company Results", "Frequency of Decision Making"])
    b["b_in_person"] = b.onet_soc_code.map(inp).fillna(float(inp.median()))
    b["b_accountability"] = b.onet_soc_code.map(acc).fillna(float(acc.median()))
    ho = ((b.human_only_ability_pct - 50.0) / 50.0).clip(0, 1)
    occ_mean = ho.groupby(b.onet_soc_code).transform("mean")
    b["b_handover"] = ho.fillna(occ_mean).fillna(float(ho.mean()))
    b["b_handover_imputed"] = b.human_only_ability_pct.isna()
    comps = ["b_interpersonal", "b_in_person", "b_accountability", "b_handover"]
    b["barrier_index"] = sum(BARRIER_WEIGHTS[c[2:]] * b[c] for c in comps)
    dom = b[comps].idxmax(axis=1).str[2:]
    b["barrier_type"] = np.where(b.physical, "physical", np.where(b.barrier_index < 0.3, "none", dom))
    return b.drop(columns=["onet_soc_code", "physical", "human_only_ability_pct"])


# ----------------------------------------------------------------------------- provider consensus at occupation level
def occupation_consensus(occ: pd.DataFrame) -> pd.DataFrame:
    """Add other providers' occupation-level signals and a simple consensus count.

    anthropic_flag   share_weight_observed >= 0.25 (a quarter of task weight shows up in Claude usage)
    microsoft_flag   ai_applicability_score in the top third of occupations
    openai_flag      Eloundou et al. (2023) beta exposure (GPT-4 rating) >= 0.5
    google_flag      listed by Google ATLAS as over-represented or most task-saturated (True), or
                     under-represented / least saturated / not observed (False); NaN if not listed
    n_providers_positive  count of the first three flags (Google only where listed)
    """
    o = occ.copy()
    g = ai.load_openai_gpts_exposure()
    o = o.merge(g[["onet_soc_code", "dv_rating_beta", "human_rating_beta"]].rename(columns={"dv_rating_beta": "openai_exposure_beta_gpt4", "human_rating_beta": "openai_exposure_beta_human"}), on="onet_soc_code", how="left")
    atlas = ai.load_google_atlas_occupations().dropna(subset=["onet_soc_code"])
    pos = set(atlas[atlas.atlas_flag.isin(["over_represented", "most_task_saturated"])].onet_soc_code)
    neg = set(atlas[atlas.atlas_flag.isin(["under_represented", "least_task_saturated", "not_observed"])].onet_soc_code)
    o["google_atlas_flag"] = np.where(o.onet_soc_code.isin(pos), "positive", np.where(o.onet_soc_code.isin(neg), "negative", None))
    o["anthropic_flag"] = o.share_weight_observed.fillna(0) >= 0.25
    thr = o.ai_applicability_score.quantile(2 / 3)
    o["microsoft_flag"] = o.ai_applicability_score >= thr
    o["openai_flag"] = o.openai_exposure_beta_gpt4 >= 0.5
    o["n_providers_positive"] = o[["anthropic_flag", "microsoft_flag", "openai_flag"]].fillna(False).sum(axis=1) + (o.google_atlas_flag == "positive").astype(int)
    o["n_providers_with_data"] = 3 - o[["ai_applicability_score", "openai_exposure_beta_gpt4"]].isna().sum(axis=1).clip(upper=2) + o.google_atlas_flag.notna().astype(int)
    return o
