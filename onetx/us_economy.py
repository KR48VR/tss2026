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


# ---- career clusters as the US sector lens, and skill levels against AI
def cluster_view(weights: pd.DataFrame, frontier_occ: pd.DataFrame, forecast_occ: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    """One row per career cluster, employment-weighted over its occupations (an occupation in two clusters
    counts in both), with the same curve columns as the economy table: shares, AI doing by year, ceiling,
    and the quarter-done years read off the cluster's own curve."""
    t = onet_weighted_table(frontier_occ, forecast_occ, weights)
    t = t.drop(columns=[c for c in ["major_group", "major_group_name", "submajor_label", "submajor_name"] if c in t.columns])
    t = t.merge(clusters[["cluster_code", "cluster", "onet_soc_code"]], on="onet_soc_code", how="inner")
    t["major_group"] = t.cluster_code
    e = se.economy_exposure(t)
    e = e[(e.level == "major_group") & (e.weighting == "employment")].copy()
    e["cluster_code"] = e.scope.str.replace("major_", "", regex=False)
    names = clusters.drop_duplicates("cluster_code").set_index("cluster_code").cluster
    e["cluster"] = e.cluster_code.map(names)
    n = t.groupby("cluster_code").agg(n_occupations=("onet_soc_code", "nunique"), employment_thousands=("employment_thousands", "sum"), wage_bill=("wage_value", "sum")).reset_index()
    e = e.merge(n, on="cluster_code", how="left")
    e["scope_name"] = e.cluster
    return e.sort_values("quarter_done_year_central_from_curve").reset_index(drop=True)


LEVEL_BANDS = [("L", "low or not needed (level below 2.5)", -1.0, 2.5), ("M", "middle (2.5 to 4.5)", 2.5, 4.5), ("H", "high (4.5 and above)", 4.5, 99.0)]


def skill_level_exposure(weights: pd.DataFrame, ratings: pd.DataFrame, frontier_occ: pd.DataFrame, forecast_occ: pd.DataFrame, families: tuple = ("essential_skill", "transferable_skill")) -> dict:
    """Does AI reach work that needs a low or a high level of a skill first? For each skill, occupations are
    banded by the level the job needs (O*NET's 0-7 scale) and the bands are compared, employment-weighted:
    share of work seen in AI use, AI doing by 2035, the quarter-done year. Also a rank correlation of level
    with 'AI doing by 2035' across occupations, and the same for the average level over the essential skills."""
    from .outcomes import _spearman

    base = onet_weighted_table(frontier_occ, forecast_occ, weights)
    base = base.drop(columns=[c for c in ["major_group", "major_group_name", "submajor_label", "submajor_name"] if c in base.columns])
    r = ratings[ratings.family.isin(families)][["onet_soc_code", "element_id", "element_name", "family", "importance", "level"]].copy()
    r = r.dropna(subset=["level"])
    frames, corr = [], []

    def bands_for(label, fam, lv: pd.Series):
        d = base.merge(lv.rename("lv"), left_on="onet_soc_code", right_index=True, how="inner")
        d["major_group"] = pd.cut(d.lv, bins=[b[2] for b in LEVEL_BANDS] + [LEVEL_BANDS[-1][3]], labels=[b[0] for b in LEVEL_BANDS], right=False).astype(str)
        e = se.economy_exposure(d)
        e = e[(e.level == "major_group") & (e.weighting == "employment")].copy()
        e["band"] = e.scope.str.replace("major_", "", regex=False)
        e["band_name"] = e.band.map({b[0]: b[1] for b in LEVEL_BANDS})
        e["element_name"] = label
        e["family"] = fam
        e["n_occupations"] = e.band.map(d.groupby("major_group").onet_soc_code.nunique())
        rho, n = _spearman(d.lv, d.done_by_2035)
        rho_seen, _ = _spearman(d.lv, d.share_weight_observed)
        corr.append({"element_name": label, "family": fam, "n_occupations": n, "rho_level_vs_done_2035": rho, "rho_level_vs_seen": rho_seen, "mean_level": float(d.lv.mean())})
        return e[["element_name", "family", "band", "band_name", "n_occupations", "weight_total", "share_weight_observed", "observed_automation", "share_weight_physical", "share_by_2035", "done_by_2030", "done_by_2035", "done_by_2035_p10", "done_by_2035_p90", "ceiling_ai_share", "quarter_done_year_central_from_curve", "quarter_done_year_fast_from_curve", "quarter_done_year_slow_from_curve"]]

    for (eid, name, fam), g in r.groupby(["element_id", "element_name", "family"]):
        frames.append(bands_for(name, fam, g.set_index("onet_soc_code")["level"]))
    ess = r[r.family == "essential_skill"].groupby("onet_soc_code")["level"].mean()
    frames.append(bands_for("All essential skills (average level)", "index", ess))
    out = pd.concat(frames, ignore_index=True)
    return {"bands": out, "correlations": pd.DataFrame(corr)}
