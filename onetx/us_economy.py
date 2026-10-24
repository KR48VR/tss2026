"""United States economy-level views, and the Singapore-US comparison.

The same tasks, the same AI evidence and the same timing model are weighted by a second workforce:
BLS OEWS employment and wages by occupation (us_labour.py). Timing differs from Singapore's in two
measured ways: the US sits on its own adoption curve (no lead), and each occupation's industries carry
the Census sector shifts through the US occupation-by-sector mix rather than Singapore's.

The gap between the two countries' curves is split into two parts by running each country's weights
on the other's timing:
  job-mix effect   = Singapore's weights on US timing  -  US weights on US timing
  timing effect    = Singapore's weights on SG timing  -  Singapore's weights on US timing
and the two add up to the full gap, scenario by scenario.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import singapore_economy as se
from .forecast import YEARS
from .outcomes import _reading, _spearman

SCORE_COLS = ["observed_automation", "usage_index", "automation_share", "share_weight_observed", "share_weight_physical", "ai_applicability_score"]


def onet_weighted_table(frontier_occ: pd.DataFrame, forecast_occ: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """One row per O*NET occupation with US weights, frontier scores and timing curves, shaped like the
    Singapore SSOC table so `singapore_economy.economy_exposure` can aggregate it."""
    share_cols = [c for c in forecast_occ.columns if c.startswith("share_by_") or c.startswith("done_by_") or c.startswith("half_year_") or c.startswith("quarter_done_year_") or c.startswith("half_done_year_") or c in ("share_beyond_horizon", "share_already_observed", "ceiling_ai_share")]
    t = weights.merge(frontier_occ[["onet_soc_code", *SCORE_COLS]], on="onet_soc_code", how="left")
    t = t.merge(forecast_occ[["onet_soc_code", *share_cols]], on="onet_soc_code", how="left")
    t = t[t.employment_thousands.notna() & (t.employment_thousands > 0)].copy()
    t["major_group_name"] = t.major_group.map(se.MAJOR_GROUP_NAMES)
    t["has_scores"] = t.observed_automation.notna()
    return t


def economy_exposure(t: pd.DataFrame, country: str = "USA") -> pd.DataFrame:
    out = se.economy_exposure(t)
    out["country"] = country
    out.loc[out.scope == "economy", "scope_name"] = "Whole economy (US)" if country == "USA" else out.loc[out.scope == "economy", "scope_name"]
    return out


def decomposition(e_sg: pd.DataFrame, e_us: pd.DataFrame, e_sgw_ust: pd.DataFrame, e_usw_sgt: pd.DataFrame, weighting: str = "wage_value") -> dict:
    """Split the Singapore-minus-US gap in 'AI doing by year' into a job-mix part and a timing part,
    for the central, fast and slow cases, plus the same for the quarter-done year."""
    def row(e):
        return e[(e.scope == "economy") & (e.weighting == weighting)].iloc[0]
    sg, us, x, y = row(e_sg), row(e_us), row(e_sgw_ust), row(e_usw_sgt)
    out = {"weighting": weighting, "by_year": {}}
    for q, suf in [("central", ""), ("fast", "_p10"), ("slow", "_p90")]:
        for yv in (2030, 2035, 2040):
            c = f"done_by_{yv}{suf}"
            gap = float(sg[c] - us[c]); jobmix = float(x[c] - us[c]); timing = float(sg[c] - x[c])
            # the alternative ordering (timing first, then job mix) gives the same total; report its mean with the first
            jobmix2 = float(sg[c] - y[c]); timing2 = float(y[c] - us[c])
            out["by_year"][f"{q}_{yv}"] = {"singapore": float(sg[c]), "us": float(us[c]), "gap": gap, "job_mix": 0.5 * (jobmix + jobmix2), "timing": 0.5 * (timing + timing2), "sg_weights_us_timing": float(x[c]), "us_weights_sg_timing": float(y[c])}
    for q in ("central", "fast", "slow"):
        c = f"quarter_done_year_{q}_from_curve"
        if c in sg and c in us:
            out[f"quarter_done_year_{q}"] = {"singapore": float(sg[c]) if pd.notna(sg[c]) else None, "us": float(us[c]) if pd.notna(us[c]) else None, "sg_weights_us_timing": float(x[c]) if pd.notna(x[c]) else None, "us_weights_sg_timing": float(y[c]) if pd.notna(y[c]) else None}
    return out


def outcome_tests(change: pd.DataFrame, weights: pd.DataFrame, forecast_occ: pd.DataFrame, frontier_occ: pd.DataFrame) -> dict:
    """US rung five: does OEWS employment or wage change between the two Mays relate to exposure, across
    detailed SOC codes? Exposure is the employment-weighted mean over the code's O*NET occupations."""
    w = weights[["onet_soc_code", "soc_2018", "employment_thousands"]].merge(forecast_occ[["onet_soc_code", "done_by_2035", "share_by_2035", "quarter_done_year_p50"]], on="onet_soc_code", how="left").merge(frontier_occ[["onet_soc_code", "share_weight_observed", "observed_automation"]], on="onet_soc_code", how="left")
    w["w"] = w.employment_thousands.fillna(0)
    def wmean(col):
        d = w.dropna(subset=[col])
        return d.assign(_x=d[col] * d.w).groupby("soc_2018").apply(lambda g: g._x.sum() / g.w.sum() if g.w.sum() > 0 else np.nan, include_groups=False)
    soc = pd.DataFrame({"done_by_2035": wmean("done_by_2035"), "share_by_2035": wmean("share_by_2035"), "share_weight_observed": wmean("share_weight_observed"), "observed_automation": wmean("observed_automation")}).reset_index()
    d = change.merge(soc, on="soc_2018", how="inner")
    yf, yt = int(change.year_from.iloc[0]), int(change.year_to.iloc[0])
    tests = []
    for outcome, label in [("emp_change_pct", f"employment change {yf} to {yt}"), ("wage_change_pct", f"median wage change {yf} to {yt}")]:
        for sig, sig_label in [("done_by_2035", "AI doing by 2035 (central)"), ("share_weight_observed", "share of work seen in AI use"), ("observed_automation", "AI already doing the work")]:
            rho, n = _spearman(d[outcome], d[sig])
            tests.append({"level": "detailed occupation (SOC)", "outcome": label, "signal": sig_label, "rho": rho, "n": n, "reading": _reading(rho, n, -1)})
    big = d[d[f"emp_{yf}"] >= 100000]
    for sig, sig_label in [("done_by_2035", "AI doing by 2035 (central)")]:
        rho, n = _spearman(big.emp_change_pct, big[sig])
        tests.append({"level": "occupations with 100,000+ employed", "outcome": f"employment change {yf} to {yt}", "signal": sig_label, "rho": rho, "n": n, "reading": _reading(rho, n, -1)})
    occ_rhos = [t["rho"] for t in tests if t["level"].startswith("detailed") and t["rho"] is not None]
    headline = {"occupation": {"median_signed_rho": float(np.median(occ_rhos)) if occ_rhos else None, "n_tests": len(occ_rhos)}}
    tot_from, tot_to = float(d[f"emp_{yf}"].sum()), float(d[f"emp_{yt}"].sum())
    # employment change by exposure tercile, employment-weighted
    d["tercile"] = pd.qcut(d.done_by_2035.rank(method="first"), 3, labels=["least exposed third", "middle third", "most exposed third"])
    terc = d.groupby("tercile", observed=True).apply(lambda g: 100.0 * (g[f"emp_{yt}"].sum() / g[f"emp_{yf}"].sum() - 1.0), include_groups=False)
    return {"tests": tests, "headline": headline, "n_soc": int(len(d)), "years": [yf, yt], "total_employment_change_pct": 100.0 * (tot_to / tot_from - 1.0), "emp_change_by_exposure_tercile": {str(k): float(v) for k, v in terc.items()}}
