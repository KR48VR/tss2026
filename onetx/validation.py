"""Track record of the capability ratings against what was later observed, and the 'ladder' conversion
rates used by the page's claim translator.

Rungs of the ladder (every claim about AI and work sits on one):
  1 rated     someone rated the task doable
  2 seen      people have been seen using AI on it (Economic Index, any release)
  3 doing     when they do, AI mostly completes it (automation share >= 0.5)
  4 adopted   typical employers use AI (Census / IMDA firm surveys)
  5 changed   the job itself changed (vacancies, wages) - not in the map yet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EXP_SCORE = {"E0": 0, "E2": 1, "E1": 2}
AUT_SCORE = {"A0": 0, "A1": 1, "A2": 2, "A3": 3}


def _auc(score: pd.Series, y: pd.Series) -> float | None:
    from sklearn.metrics import roc_auc_score

    m = score.notna() & y.notna()
    if m.sum() < 50 or y[m].nunique() < 2:
        return None
    return float(roc_auc_score(y[m].astype(int), score[m].astype(float)))


def _kappa(a: pd.Series, b: pd.Series) -> float | None:
    m = a.notna() & b.notna()
    if m.sum() < 50:
        return None
    cats = sorted(set(a[m]) | set(b[m]))
    ct = pd.crosstab(a[m], b[m]).reindex(index=cats, columns=cats, fill_value=0).values
    n = ct.sum()
    po = np.trace(ct) / n
    pe = float(ct.sum(axis=1) @ ct.sum(axis=0)) / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else None


def rating_validation(tf: pd.DataFrame, labels: pd.DataFrame | None, ratings: pd.DataFrame | None, repeat: pd.DataFrame | None = None) -> dict:
    d = tf[["task_id", "physical", "usage_n_vintages", "automation_share", "ms_share"]].copy()
    d["task_id"] = d.task_id.astype(str)
    if labels is not None and len(labels):
        lab = labels.copy(); lab["task_id"] = lab.task_id.astype(str)
        d = d.merge(lab[["task_id", "exposure_human_2023", "exposure_gpt4_2023", "beta_2023"]], on="task_id", how="left")
    else:
        for c in ["exposure_human_2023", "exposure_gpt4_2023", "beta_2023"]:
            d[c] = np.nan
    if ratings is not None and len(ratings):
        r = ratings.copy(); r["task_id"] = r.task_id.astype(str)
        d = d.merge(r[["task_id", "exposure", "autonomy", "barrier"]].rename(columns={"exposure": "exposure_2026", "autonomy": "autonomy_2026", "barrier": "barrier_2026"}), on="task_id", how="left")
    else:
        for c in ["exposure_2026", "autonomy_2026", "barrier_2026"]:
            d[c] = np.nan
    desk = ~d.physical.astype(bool)
    seen = (d.usage_n_vintages.fillna(0) >= 1)
    doing = seen & (d.automation_share >= 0.5)
    out: dict = {"n_tasks": int(len(d)), "n_desk": int(desk.sum()), "n_seen_desk": int((seen & desk).sum())}

    # --- rung 1 -> rung 2: does a rating predict being seen in use? (desk tasks)
    h = d.exposure_human_2023.map(EXP_SCORE); g = d.exposure_gpt4_2023.map(EXP_SCORE); e26 = d.exposure_2026.map(EXP_SCORE); a26 = d.autonomy_2026.map(AUT_SCORE)
    out["auc_seen"] = {"human_2023": _auc(h[desk], seen[desk]), "gpt4_2023": _auc(g[desk], seen[desk]), "beta_2023": _auc(d.beta_2023[desk], seen[desk]), "exposure_2026": _auc(e26[desk], seen[desk]), "autonomy_2026": _auc(a26[desk], seen[desk])}
    out["share_seen_by_label"] = {}
    for name, col in [("human_2023", "exposure_human_2023"), ("gpt4_2023", "exposure_gpt4_2023"), ("exposure_2026", "exposure_2026"), ("autonomy_2026", "autonomy_2026")]:
        if d[col].notna().any():
            out["share_seen_by_label"][name] = {k: round(float(v), 4) for k, v in seen[desk].groupby(d[col][desk]).mean().items()}
    # --- rung 2 -> rung 3: among seen desk tasks, does a rating predict AI doing most of it?
    sd = seen & desk
    out["auc_doing_given_seen"] = {"human_2023": _auc(h[sd], doing[sd]), "gpt4_2023": _auc(g[sd], doing[sd]), "exposure_2026": _auc(e26[sd], doing[sd]), "autonomy_2026": _auc(a26[sd], doing[sd])}
    if d.autonomy_2026.notna().any():
        out["automation_share_by_autonomy_2026"] = {k: round(float(v), 4) for k, v in d.automation_share[sd].groupby(d.autonomy_2026[sd]).mean().items()}
        out["corr_autonomy_automation_share"] = float(np.corrcoef(a26[sd & a26.notna() & d.automation_share.notna()], d.automation_share[sd & a26.notna() & d.automation_share.notna()])[0, 1])
    # --- agreement between raters (yes = E1/E2)
    yh, yg, y26 = d.exposure_human_2023.isin(["E1", "E2"]), d.exposure_gpt4_2023.isin(["E1", "E2"]), d.exposure_2026.isin(["E1", "E2"])
    has23 = d.exposure_human_2023.notna() & d.exposure_gpt4_2023.notna(); has26 = d.exposure_2026.notna()
    def agree(a, b, m):
        return {"agree": float((a[m] == b[m]).mean()), "kappa": _kappa(a[m].astype(str), b[m].astype(str)), "n": int(m.sum())} if m.sum() else None
    out["agreement"] = {"human_vs_gpt4_2023": agree(yh, yg, has23), "claude2026_vs_human_2023": agree(y26, yh, has23 & has26), "claude2026_vs_gpt4_2023": agree(y26, yg, has23 & has26)}
    # --- how the 2023 'no' aged
    no23 = has23 & ~yh & ~yg & desk
    out["aging_of_2023_no"] = {"n": int(no23.sum()), "share_seen_by_2026": float(seen[no23].mean()) if no23.sum() else None, "share_rated_yes_2026": float(y26[no23 & has26].mean()) if (no23 & has26).sum() else None}
    yes23 = has23 & (yh | yg) & desk
    out["aging_of_2023_yes"] = {"n": int(yes23.sum()), "share_seen_by_2026": float(seen[yes23].mean()) if yes23.sum() else None, "share_rated_no_2026": float((~y26)[yes23 & has26].mean()) if (yes23 & has26).sum() else None}
    # --- physical consistency of the 2026 rating
    if d.barrier_2026.notna().any():
        out["physical_consistency_2026"] = {"robot_flagged_rated_physical": float((d.barrier_2026[d.physical.astype(bool)] == "physical").mean()), "desk_rated_physical": float((d.barrier_2026[desk] == "physical").mean())}
    # --- repeat reliability
    if repeat is not None and len(repeat):
        rp = repeat
        out["repeat_2026"] = {"n": int(len(rp)), "exposure_same": float((rp.exposure_1 == rp.exposure_2).mean()), "exposure_yes_no_same": float((rp.exposure_1.isin(["E1", "E2"]) == rp.exposure_2.isin(["E1", "E2"])).mean()),
                              "autonomy_same": float((rp.autonomy_1 == rp.autonomy_2).mean()), "autonomy_within_one": float((abs(rp.autonomy_1.str[1].astype(int) - rp.autonomy_2.str[1].astype(int)) <= 1).mean()), "barrier_same": float((rp.barrier_1 == rp.barrier_2).mean())}
    # --- ladder conversions (desk tasks), the numbers the claim translator uses
    rated_yes = has23 & (yh | yg)
    ladder = {
        "rated_to_seen": float(seen[desk & rated_yes].mean()) if (desk & rated_yes).sum() else None,          # of tasks rated doable (2023, either rater), share seen in use by 2026
        "rated_no_to_seen": float(seen[desk & has23 & ~(yh | yg)].mean()) if (desk & has23).sum() else None,  # of tasks rated not doable, share seen anyway
        "seen_to_doing": float(doing[sd].mean()) if sd.sum() else None,                                        # of seen desk tasks, share where AI mostly does it
        "seen_mean_automation_share": float(d.automation_share[sd].mean()) if sd.sum() else None,
        "all_desk_seen": float(seen[desk].mean()),
        "all_desk_doing": float(doing[desk].mean()),
        "years_between": "2023 ratings to May 2026 usage",
    }
    if has26.any():
        ladder["rated_2026_to_seen"] = float(seen[desk & y26].mean())
        ladder["autonomy_full_to_doing"] = float(doing[sd & (d.autonomy_2026 == "A3")].mean()) if (sd & (d.autonomy_2026 == "A3")).sum() else None
    out["ladder"] = ladder
    return out
