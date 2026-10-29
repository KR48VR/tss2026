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


def _fc_table(codes, done35):
    fc = pd.DataFrame({"onet_soc_code": codes, "ceiling_ai_share": [0.5] * len(codes), "half_year_p10": [2030.0] * len(codes), "half_year_p50": [2032.0] * len(codes), "half_year_p90": [2035.0] * len(codes), "share_beyond_horizon": [0.0] * len(codes), "share_already_observed": [0.1] * len(codes)})
    for y in YEARS:
        frac = min(1.0, max(0.0, (y - 2027) / 8.0))
        fc[f"share_by_{y}"] = [min(1.0, 2 * d * frac) for d in done35]
        fc[f"done_by_{y}"] = [d * frac for d in done35]
        fc[f"done_by_{y}_p10"] = [min(1.0, d * frac * 1.2) for d in done35]
        fc[f"done_by_{y}_p90"] = [d * frac * 0.8 for d in done35]
    return fc


def test_cluster_view_weights_by_employment_and_counts_shared_occupations():
    codes = ["a", "b", "c"]
    weights = pd.DataFrame({"onet_soc_code": codes, "employment_thousands": [100.0, 300.0, 50.0], "wage_value": [1e4, 3e4, 5e3], "major_group": ["2", "4", "9"]})
    frontier = pd.DataFrame({"onet_soc_code": codes, **{c: [0.3, 0.6, 0.1] for c in ue.SCORE_COLS}})
    fc = _fc_table(codes, [0.3, 0.6, 0.1])
    clusters = pd.DataFrame({"cluster_code": ["X", "X", "Y"], "cluster": ["Office", "Office", "Field"], "onet_soc_code": ["a", "b", "b"]})
    v = ue.cluster_view(weights, frontier, fc, clusters).set_index("cluster_code")
    assert v.loc["X", "n_occupations"] == 2 and v.loc["Y", "n_occupations"] == 1
    assert abs(v.loc["X", "done_by_2035"] - (100 * 0.3 + 300 * 0.6) / 400) < 1e-9   # employment-weighted
    assert abs(v.loc["Y", "employment_thousands"] - 300.0) < 1e-9                   # b counts in both clusters
    assert v.loc["Y", "quarter_done_year_central_from_curve"] < v.loc["X", "quarter_done_year_central_from_curve"] or v.loc["Y", "done_by_2035"] > v.loc["X", "done_by_2035"]


def test_skill_level_bands_and_correlation_direction():
    codes = [f"o{i}" for i in range(30)]
    levels = np.linspace(0.5, 6.5, 30)
    done = levels / 7.0                                   # higher level, more AI doing
    weights = pd.DataFrame({"onet_soc_code": codes, "employment_thousands": [10.0] * 30, "wage_value": [100.0] * 30, "major_group": ["2"] * 30})
    frontier = pd.DataFrame({"onet_soc_code": codes, **{c: done for c in ue.SCORE_COLS}})
    fc = _fc_table(codes, done)
    ratings = pd.DataFrame({"onet_soc_code": codes * 2, "element_id": ["2.A.1.a"] * 30 + ["2.A.1.e"] * 30, "element_name": ["Reading Comprehension"] * 30 + ["Mathematics"] * 30, "family": ["essential_skill"] * 60, "importance": [4.0] * 60, "level": list(levels) + list(levels[::-1])})
    out = ue.skill_level_exposure(weights, ratings, frontier, fc)
    b = out["bands"]; c = out["correlations"].set_index("element_name")
    assert set(b.family) == {"essential_skill", "index"}
    rc = b[b.element_name == "Reading Comprehension"].set_index("band")
    assert rc.loc["H", "done_by_2035"] > rc.loc["M", "done_by_2035"] > rc.loc["L", "done_by_2035"]
    assert c.loc["Reading Comprehension", "rho_level_vs_done_2035"] > 0.9
    assert c.loc["Mathematics", "rho_level_vs_done_2035"] < -0.9                      # reversed levels, opposite sign
    assert "All essential skills (average level)" in c.index


# ---------------------------------------------------------------------------- further countries (Australia, the United Kingdom)
from onetx import au_labour as au
from onetx import uk_labour as uk
from onetx import countries as cn


def test_nfd_redistribution_keeps_totals_and_removes_nfd_codes():
    emp = pd.Series({"1000": 10.0, "1100": 4.0, "1110": 2.0, "1111": 30.0, "1112": 10.0, "1211": 60.0, "2000": 5.0, "2111": 95.0})
    out = au.redistribute_nfd(emp)
    assert abs(out.sum() - emp.sum()) < 1e-9
    assert not any(c.endswith("0") for c in out.index)
    # 1110 goes to 1111/1112 in proportion (3:1), 1100 likewise (only 111x codes under 11), 1000 over every 1xxx code
    assert out["1111"] > out["1112"] and out["1211"] > 60.0


def test_isco_link_weights_sum_to_one_per_isco_code():
    d = pd.DataFrame({"isco_code": ["1120", "1120", "2512"], "nat_code": ["1111", "1112", "2136"], "weight": [0.5, 0.5, 1.0]})
    assert np.allclose(d.groupby("isco_code").weight.sum(), 1.0)


def test_bics_use_definition_excludes_not_sure():
    # 64.1% do not use, 6.2% not sure -> 29.7% use
    not_use, not_sure = 0.641, 0.062
    assert abs(100.0 * (1.0 - not_use - not_sure) - 29.7) < 1e-9


def test_sic_section_ranges():
    assert uk.sic_section("01") == "A" and uk.sic_section("33") == "C" and uk.sic_section("62") == "J" and uk.sic_section("86") == "Q" and uk.sic_section("99") == "U"


def test_firm_lead_reads_the_us_curve_at_matched_shares():
    fit = {"saturation": 0.9, "intercept": -2.0, "growth_per_year": 0.5}
    # a point sitting exactly on the US curve has a lead of zero; a higher share at the same date leads
    K, a, b = fit["saturation"], fit["intercept"], fit["growth_per_year"]
    t = 1.5
    p_on = K / (1 + np.exp(-(a + b * t)))
    pts = pd.DataFrame({"group_type": ["all", "all"], "date": [2024.0 + t, 2024.0 + t], "pct": [100 * p_on, 100 * min(K - 0.01, p_on * 1.5)], "period": ["x", "y"]})
    out = cn.firm_lead(fit, pts, min_year=2024.0)
    assert abs(out["points"][0]["lead_years"]) < 1e-6
    assert out["points"][1]["lead_years"] > 0
    assert out["n_points_used"] == 2


def test_industry_offsets_sign_and_threshold():
    fit = {"saturation": 0.9, "intercept": -2.0, "growth_per_year": 0.5}
    pts = pd.DataFrame({"group_type": ["all", "industry", "industry", "all", "industry"], "date": [2025.5, 2025.5, 2025.5, 2022.5, 2022.5], "industry": ["ALL", "J", "I", "ALL", "J"], "label": ["all", "Info", "Food", "all", "Info"], "pct": [12.0, 38.0, 5.0, 1.4, 7.7]})
    out = cn.industry_offsets(pts, fit, min_national_pct=3.0).set_index("industry")
    assert out.loc["J", "anchor_shift_years"] < 0 < out.loc["I", "anchor_shift_years"]   # earlier for the high-adoption industry
    assert out.loc["J", "n_points"] == 1                                                # the 1.4% national reading is below the threshold


def test_occupation_shifts_inherit_group_mix():
    spec = cn.SPECS["AUS"]
    mix = pd.DataFrame({"major_group": ["1", "1", "2"], "industry": ["J", "I", "J"], "employed": [300.0, 100.0, 50.0]})
    ind = pd.DataFrame({"industry": ["J", "I"], "anchor_shift_years": [-2.0, 2.0]})
    natw = pd.DataFrame({"anzsco_code": ["1111", "2611"], "employment_thousands": [1.0, 1.0]})
    xw = pd.DataFrame({"onet_soc_code": ["a", "a"], "nat_code": ["1111", "2611"], "weight_rev": [1.0, 1.0]})
    grp, nat, onet = cn.occupation_shifts(mix, ind, natw, xw, spec)
    s = nat.set_index("nat_code").adoption_shift_years
    assert abs(s["1111"] - (-2.0 * 0.75 + 2.0 * 0.25)) < 1e-9 and abs(s["2611"] - (-2.0)) < 1e-9
    assert abs(onet.adoption_shift_years.iloc[0] - 0.5 * (s["1111"] + s["2611"])) < 1e-9


def test_pair_decomposition_adds_up_and_is_symmetric_in_names():
    a = _econ({2030: 0.20, 2035: 0.42, 2040: 0.46}, 2030.6)
    b = _econ({2030: 0.15, 2035: 0.36, 2040: 0.43}, 2031.4)
    x = _econ({2030: 0.17, 2035: 0.39, 2040: 0.45}, 2031.0)
    y = _econ({2030: 0.18, 2035: 0.40, 2040: 0.44}, 2030.9)
    d = cn.pair_decomposition(a, b, x, y, "wage_value", "AUS", "GBR")
    for key, part in d["by_year"].items():
        assert abs(part["gap"] - (part["job_mix"] + part["timing"])) < 1e-9, key
        assert "AUS" in part and "GBR" in part
    assert abs(d["by_year"]["central_2035"]["gap"] - 0.06) < 1e-9


# ---------------------------------------------------------------------------- Canada: a direct O*NET link and the national ratings test
def test_direct_onet_link_is_used_when_the_spec_has_one(monkeypatch):
    spec = cn.SPECS["CAN"]
    link = pd.DataFrame({"onet_soc_code": ["a", "a", "b"], "nat_code": ["11100", "11101", "11100"], "weight": [0.5, 0.5, 1.0]})
    spec2 = cn.CountrySpec(**{**spec.__dict__, "onet_link": lambda codes: link})
    monkeypatch.setattr(cn.io, "load", lambda name: (_ for _ in ()).throw(AssertionError("ISCO table must not be read")))
    xw = cn.national_crosswalk(spec2, {"11100", "11101"})
    assert np.allclose(xw.groupby("onet_soc_code").weight.sum(), 1.0) and np.allclose(xw.groupby("nat_code").weight_rev.sum(), 1.0)
    assert xw[(xw.onet_soc_code == "a") & (xw.nat_code == "11100")].weight_rev.iloc[0] == 1 / 3


def test_rating_level_exposure_bands_and_direction():
    spec = cn.SPECS["CAN"]
    codes = [f"{i:05d}" for i in range(30)]
    levels = np.linspace(0, 5, 30)
    t = pd.DataFrame({"nat_code": codes, "employment_thousands": 10.0, "share_weight_observed": levels / 5, "observed_automation": levels / 10, "done_by_2035": levels / 6, "done_by_2035_p10": levels / 5, "done_by_2035_p90": levels / 7, "quarter_done_year_p10": 2030.0, "quarter_done_year_p50": 2032.0, "quarter_done_year_p90": 2035.0})
    ratings = pd.DataFrame({"noc_code": codes * 2, "family": ["skill"] * 60, "element_name": ["Numeracy"] * 30 + ["Writing"] * 30, "level": list(levels) + list(levels[::-1])})
    out = cn.rating_level_exposure(t, ratings, spec)
    b = out["bands"]; c = out["correlations"].set_index("element_name")
    assert set(b.band) == {"L", "M", "H"} and "index" in set(b.family)
    num = b[b.element_name == "Numeracy"].set_index("band")
    assert num.loc["H", "done_by_2035"] > num.loc["M", "done_by_2035"] > num.loc["L", "done_by_2035"]
    assert c.loc["Numeracy", "rho_level_vs_done_2035"] > 0.9 and c.loc["Writing", "rho_level_vs_done_2035"] < -0.9


def test_group_len_drives_the_mix_key():
    spec = cn.SPECS["CAN"]
    mix = pd.DataFrame({"minor": ["1110", "1110", "2110"], "industry": ["52", "54", "54"], "employed": [300.0, 100.0, 50.0]})
    ind = pd.DataFrame({"industry": ["52", "54"], "anchor_shift_years": [-2.0, 2.0]})
    natw = pd.DataFrame({"noc_code": ["11100", "21101"], "employment_thousands": [1.0, 1.0]})
    xw = pd.DataFrame({"onet_soc_code": ["a", "a"], "nat_code": ["11100", "21101"], "weight_rev": [1.0, 1.0]})
    grp, nat, onet = cn.occupation_shifts(mix, ind, natw, xw, spec)
    s = nat.set_index("nat_code").adoption_shift_years
    assert abs(s["11100"] - (-2.0 * 0.75 + 2.0 * 0.25)) < 1e-9 and abs(s["21101"] - 2.0) < 1e-9


def test_adoption_now_text_reads_a_quarter_or_a_financial_year():
    from onetx.site import _adoption_now_text
    assert _adoption_now_text("CAN", "2026 Q2") == "the second quarter of 2026"
    assert _adoption_now_text("AUS", "2024-25") == "2024-25 financial year"
    assert _adoption_now_text("GBR", "2026 Q2") is None
    assert _adoption_now_text("CAN", None) is None
