"""The second workforce: US weights, sector shifts and the Singapore-US decomposition."""
from __future__ import annotations

import numpy as np
import pandas as pd

from onetx import us_economy as ue
from onetx.forecast import YEARS


def _econ(doing_by_year: dict, quarter: float, weighting: str = "wage_value") -> pd.DataFrame:
    row = {"scope": "economy", "weighting": weighting, "quarter_done_year_central_from_curve": quarter}
    for y in YEARS:
        v = doing_by_year.get(y, 0.0)
        row[f"done_by_{y}"] = v
        row[f"done_by_{y}_p10"] = min(1.0, v * 1.2)
        row[f"done_by_{y}_p90"] = v * 0.8
    return pd.DataFrame([row])


def test_decomposition_adds_up_in_every_case():
    sg = _econ({2030: 0.20, 2035: 0.42, 2040: 0.46}, 2030.6)
    us = _econ({2030: 0.15, 2035: 0.36, 2040: 0.43}, 2031.4)
    x = _econ({2030: 0.17, 2035: 0.39, 2040: 0.45}, 2031.0)   # Singapore weights on US timing
    y = _econ({2030: 0.18, 2035: 0.40, 2040: 0.44}, 2030.9)   # US weights on Singapore timing
    d = ue.decomposition(sg, us, x, y)
    for key, part in d["by_year"].items():
        assert abs(part["gap"] - (part["job_mix"] + part["timing"])) < 1e-9, key
    c35 = d["by_year"]["central_2035"]
    assert abs(c35["gap"] - 0.06) < 1e-9
    assert c35["job_mix"] > 0 and c35["timing"] > 0
    assert d["quarter_done_year_central"]["singapore"] == 2030.6


def test_onet_weighted_table_keeps_only_weighted_rows_and_names_groups():
    weights = pd.DataFrame({"onet_soc_code": ["a", "b", "c"], "major_group": ["2", "4", "X"], "employment_thousands": [10.0, 0.0, 5.0], "wage_value": [1000.0, 0.0, 200.0]})
    frontier = pd.DataFrame({"onet_soc_code": ["a", "b", "c"], **{c: [0.1, 0.2, np.nan] for c in ue.SCORE_COLS}})
    fc = pd.DataFrame({"onet_soc_code": ["a", "b", "c"], **{f"share_by_{y}": [0.5, 0.5, 0.5] for y in YEARS}, **{f"done_by_{y}": [0.2, 0.2, 0.2] for y in YEARS}, "ceiling_ai_share": [0.4, 0.4, 0.4], "half_year_p10": [2030.0] * 3, "half_year_p50": [2032.0] * 3, "half_year_p90": [2035.0] * 3})
    t = ue.onet_weighted_table(frontier, fc, weights)
    assert list(t.onet_soc_code) == ["a", "c"]           # zero-employment row dropped
    assert t.major_group_name.iloc[0] == "Professionals"
    assert bool(t.has_scores.iloc[0]) and not bool(t.has_scores.iloc[1])


def test_sector_shift_is_employment_weighted(monkeypatch):
    from onetx import us_labour as ul

    sec = pd.DataFrame({"soc_2018": ["11-1011"] * 2 + ["11-1021"], "sector": ["51", "72", "99"], "sector_title": ["Information", "Food", "Government"], "employment": [300.0, 100.0, 50.0], "year": 2024})
    monkeypatch.setattr(ul, "load_oews_sector", lambda year=2024: sec)
    monkeypatch.setattr(ul.io, "load", lambda name: pd.DataFrame({"onet_soc_code": ["11-1011.00", "11-1011.03", "11-1021.00"], "soc_2018": ["11-1011", "11-1011", "11-1021"]}))
    levels = pd.DataFrame({"sector": ["51", "72"], "anchor_shift_years": [-2.0, 2.0]})
    sh, mix = ul.onet_sector_shifts(levels)
    s = sh.set_index("onet_soc_code").adoption_shift_years
    assert abs(s["11-1011.00"] - (-2.0 * 0.75 + 2.0 * 0.25)) < 1e-9   # 300 vs 100 employed
    assert s["11-1011.03"] == s["11-1011.00"]                          # specialisations share the SOC mix
    assert s["11-1021.00"] == 0.0                                      # government: not surveyed, national level
