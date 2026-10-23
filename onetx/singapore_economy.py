"""Singapore economy-level views: employment- and wage-weighted exposure and timing.

Takes the SSOC-level frontier scores and timing curves, weights them by MOM employment and wages
(see singapore_labour), and reports the share of Singapore's resident employment and wage bill that
is exposed / disruptable, overall and by major occupational group and Skills Framework sector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import singapore as sg
from . import singapore_labour as sl
from .forecast import YEARS

MAJOR_GROUP_NAMES = {"1": "Managers & administrators", "2": "Professionals", "3": "Associate professionals & technicians", "4": "Clerical support workers", "5": "Service & sales workers", "6": "Agricultural & fishery workers", "7": "Craftsmen & related trades", "8": "Plant & machine operators", "9": "Cleaners, labourers & related"}


def ssoc_weighted_table(frontier_ssoc: pd.DataFrame, forecast_ssoc: pd.DataFrame, weights: pd.DataFrame | None = None) -> pd.DataFrame:
    w = sl.build_ssoc_weights() if weights is None else weights
    share_cols = [f"share_by_{y}" for y in YEARS] + [f"share_by_{y}_{q}" for q in ("p10", "p90") for y in YEARS if f"share_by_{y}_{q}" in forecast_ssoc.columns] + ["share_beyond_horizon", "share_already_observed", "half_year_p10", "half_year_p50", "half_year_p90"]
    t = w.merge(frontier_ssoc[["ssoc_code", "observed_automation", "usage_index", "automation_share", "share_weight_observed", "share_weight_physical", "ai_applicability_score"]], on="ssoc_code", how="left")
    t = t.merge(forecast_ssoc[["ssoc_code", *share_cols]], on="ssoc_code", how="left")
    t["major_group_name"] = t.major_group.map(MAJOR_GROUP_NAMES)
    if "submajor_label" in t.columns:
        d8 = sl.load_lfs_employment_submajor(2025).set_index("submajor_label").group_name
        t["submajor_name"] = t.submajor_label.map(d8)
    t["has_scores"] = t.observed_automation.notna()
    return t


def _wavg(d: pd.DataFrame, col: str, w: str) -> float:
    x = d[[col, w]].dropna()
    return float(np.average(x[col], weights=x[w])) if len(x) and x[w].sum() > 0 else np.nan


def economy_exposure(t: pd.DataFrame) -> pd.DataFrame:
    """One row per (scope, weighting). Scopes: whole economy and each major group."""
    rows = []
    scopes = [("economy", t)] + [(f"major_{mg}", g) for mg, g in t.groupby("major_group") if g.employment_thousands.sum() > 0]
    if "submajor_label" in t.columns:
        scopes += [(f"submajor_{lbl}", g) for lbl, g in t.groupby("submajor_label") if g.employment_thousands.sum() > 0]
    for scope, d in scopes:
        if not d.has_scores.any():
            continue  # e.g. the X1-X5 'not elsewhere classified' codes carry employment but no O*NET link
        for wname in ["employment_thousands", "wage_value"]:
            if scope == "economy":
                scope_name, level = "Whole economy (residents)", "economy"
            elif scope.startswith("major_"):
                scope_name, level = MAJOR_GROUP_NAMES.get(scope[-1], scope), "major_group"
            else:
                scope_name, level = f"{scope.split('_', 1)[1]} {d.submajor_name.iloc[0] if 'submajor_name' in d else ''}".strip(), "submajor_group"
            row = {"scope": scope, "level": level, "scope_name": scope_name, "weighting": "employment" if wname.startswith("emp") else "wage_value", "n_ssoc": int(len(d)), "n_ssoc_scored": int(d.has_scores.sum()), "weight_total": float(d[wname].sum()), "weight_scored_share": float(d.loc[d.has_scores, wname].sum() / d[wname].sum()) if d[wname].sum() else np.nan}
            for c in ["observed_automation", "usage_index", "automation_share", "share_weight_observed", "share_weight_physical", "ai_applicability_score"]:
                row[c] = _wavg(d, c, wname)
            for y in YEARS:
                row[f"share_by_{y}"] = _wavg(d, f"share_by_{y}", wname)
                for q in ("p10", "p90"):
                    if f"share_by_{y}_{q}" in d.columns:
                        row[f"share_by_{y}_{q}"] = _wavg(d, f"share_by_{y}_{q}", wname)
            row["share_beyond_horizon"] = _wavg(d, "share_beyond_horizon", wname)
            for q in ["p10", "p50", "p90"]:
                row[f"half_year_{q}"] = _wavg(d, f"half_year_{q}", wname)
            rows.append(row)
    out = pd.DataFrame(rows)
    # exact half-way year from the weighted share-by-year curve (central case)
    def first_year(r, suffix=""):
        prev_y, prev_v = None, None
        for y in YEARS:
            v = r.get(f"share_by_{y}{suffix}")
            if pd.notna(v) and v >= 0.5:
                # linear interpolation between the previous year and this one
                if prev_v is not None and v > prev_v:
                    return round(prev_y + (0.5 - prev_v) / (v - prev_v), 1)
                return float(y)
            prev_y, prev_v = y, v if pd.notna(v) else prev_v
        return np.nan
    out["half_year_central_from_curve"] = out.apply(first_year, axis=1)
    if any(c.endswith("_p10") for c in out.columns):
        out["half_year_fast_from_curve"] = out.apply(lambda r: first_year(r, "_p10"), axis=1)
        out["half_year_slow_from_curve"] = out.apply(lambda r: first_year(r, "_p90"), axis=1)
    return out


def sector_view(role_scores: pd.DataFrame, role_curves: pd.DataFrame) -> pd.DataFrame:
    """Skills Framework sectors: unweighted mean over roles (no employment by SFw sector is published)."""
    r = role_scores.merge(role_curves[["role_id", "share_by_2028", "share_by_2030", "share_by_2035", "half_year_p10", "half_year_p50", "half_year_p90"]], on="role_id", how="left")
    g = r.groupby("sector")
    out = pd.DataFrame({"n_roles": g.role_id.nunique(), "observed_automation_via_tasks": g.observed_automation_via_tasks.mean(), "observed_automation_via_ssoc": g.observed_automation_via_ssoc.mean(), "share_key_tasks_observed": g.share_key_tasks_observed.mean(), "share_key_tasks_physical": g.share_key_tasks_physical.mean(), "share_by_2028": g.share_by_2028.mean(), "share_by_2030": g.share_by_2030.mean(), "share_by_2035": g.share_by_2035.mean(), "half_year_p50_median_role": g.half_year_p50.median(), "half_year_p90_median_role": g.half_year_p90.median()})
    return out.reset_index().sort_values("half_year_p50_median_role")
