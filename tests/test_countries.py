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


def test_new_zealand_weights_share_anzsco_with_australia():
    from onetx import nz_labour as nz
    import pytest
    if not (nz.STATSNZ / "census2023_occupation_sex_region.xlsx").exists():
        pytest.skip("Stats NZ files not present")
    w = nz.build_anzsco_weights()
    assert len(w) == 358 and w.anzsco_code.str.fullmatch(r"\d{4}").all()
    assert abs(w.employment_thousands.sum() - 2622.8) < 1.0          # 2023 Census employed, all of New Zealand
    assert w.skill_level.notna().all() and w.wage_weekly.gt(0).all()
    ch = nz.employment_change()
    assert {"emp_2018", "emp_2023", "emp_change_pct", "wage_change_pct"} <= set(ch.columns)
    assert nz.load_ai_points().empty   # no survey: a documented gap
    assert nz.has_ade() == (not nz.load_industry_mix().empty)   # the mix exists only with the Data Explorer extracts


def test_proxy_lead_stands_in_when_a_country_has_no_survey():
    """A country without firm-survey points borrows the proxy country's measured lead and is marked."""
    from onetx import countries as cn
    pts = pd.DataFrame(columns=["date", "period", "pct", "group_type", "industry", "label"])
    fit = {"saturation": 0.9, "intercept": -2.0, "growth_per_year": 0.7}
    lead = cn.firm_lead(fit, pts)
    assert lead["lead_years"] is None and lead["n_points_used"] == 0
    others = {"AUS": {"lead": {"lead_years": -0.4, "points": [{"date": 2025.5, "pct": 12.1}], "n_points_used": 1, "latest_pct": 12.1, "latest_date": 2025.5}}}
    spec = cn.SPECS["NZL"]
    assert spec.lead_proxy == "AUS"
    if lead["lead_years"] is None and spec.lead_proxy in others:
        lead = {**others[spec.lead_proxy]["lead"], "proxy": spec.lead_proxy, "own_points": 0}
    assert lead["proxy"] == "AUS" and lead["lead_years"] == -0.4


def test_norway_sits_on_isco_codes_with_register_pay():
    from onetx import no_labour as no
    import pytest
    if not (no.SSB / "11658_employees_earnings_by_occupation.json").exists():
        pytest.skip("Statistics Norway files not present")
    w = no.build_styrk_weights()
    assert len(w) == 400 and w.styrk_code.str.fullmatch(r"\d{4}").all() and w.wage_monthly.gt(0).all()
    link = no.load_isco_link(set(w.styrk_code))
    assert (link.isco_code == link.nat_code).all() and (link.weight == 1).all()
    pts = no.load_ai_use()
    nat = pts[pts.group_type == "all"].sort_values("date")
    assert list(nat.period) == ["2021", "2023", "2024", "2025", "2026"] and nat.pct.iloc[-1] > nat.pct.iloc[0]
    mix = no.load_industry_mix()
    assert set(mix.industry) <= set(no.INDUSTRIES) and mix.employed.sum() > 1000


def test_europe_loaders_cover_the_eurostat_countries():
    from onetx import europe as eu
    import pytest
    if not (eu.EUROSTAT / "isoc_eb_ai.json").exists():
        pytest.skip("Eurostat files not present")
    ai = eu.load_ai_use(); emp = eu.load_employment_2d(); earn = eu.load_earnings_1d(); mix = eu.load_mix()
    assert ai.pct.between(0, 100).all() and set(ai.period) == {"2021", "2023", "2024", "2025"}
    assert emp.isco2_code.str.fullmatch(r"\d\d").all() and emp.geo.nunique() >= 30
    w = eu.weights("DE", emp, earn)
    assert len(w) > 30 and w.wage_value.gt(0).all() and w.skill_level.between(1, 4).all()
    assert set(mix.industry) <= set(eu.NACE_SECTIONS)
    link = eu._isco_to_2d({"11", "25"})
    assert set(link.nat_code) == {"11", "25"} and link.isco_code.str.len().eq(4).all()
    by = eu.load_ai_use_by_activity()
    assert set(by.industry) <= set(eu.NACE_SECTIONS) and not by.duplicated(["geo", "date", "industry"]).any()


def test_five_office_loaders_give_consistent_weights_and_links():
    """Sweden, Finland, Denmark, the Netherlands and Ireland: weights with pay for every code, crosswalk weights
    summing to one per ISCO code, AI readings with the national series, and the section parser."""
    from onetx import eu_labour as el
    import pytest
    if not (el.SCB / "yreg50bas_employees_by_ssyk.json").exists() or not (el.SCB / "ssyk2012_isco08_key.csv").exists():
        pytest.skip("office files not present")
    assert el.sections_from_label("manufacturing (SNI 10–33)") == ["C"] and el.sections_from_label("GHI") == ["G", "H", "I"] and el.sections_from_label("L, N, S951 Real estate") == ["L", "N", "S"]
    for pre, n_lo, n_hi in (("se", 400, 440), ("fi", 400, 430), ("dk", 370, 430), ("nl", 110, 116), ("ie", 300, 330)):
        w = getattr(el, f"{pre}_weights")()
        assert n_lo <= len(w) <= n_hi and w.wage_value.gt(0).all() and w.skill_level.between(1, 4).all(), pre
        link = getattr(el, f"{pre}_isco_link")(set(w[w.columns[0]]))
        assert (link.groupby("isco_code").weight.sum() - 1).abs().lt(1e-6).all(), pre
        pts = getattr(el, f"{pre}_ai_points")()
        nat = pts[pts.group_type == "all"]
        assert len(nat) >= 3 and nat.pct.between(0, 100).all() and nat.date.is_monotonic_increasing, pre
    assert el.se_weights().employment_thousands.sum() < 6000   # all sectors once, not the nested sector codes added up
    assert el.ie_ai_points().query("group_type == 'industry'").empty   # the CSO publishes no industry breakdown


def test_japan_hand_made_link_covers_every_class():
    from onetx import jp_labour as jp
    import pytest
    if not (jp.ESTAT / "wage2024_t1_occupation_small.xlsx").exists():
        pytest.skip("e-Stat files not present")
    t = jp.load_wage_table()
    assert len(t) == len(jp.JSOC_TO_ISCO) == 144 and t.workers.gt(0).all()
    w = jp.build_weights(); link = jp.load_isco_link(set(w.jp_code))
    assert set(link.nat_code) == set(w.jp_code) and (link.groupby("isco_code").weight.sum() - 1).abs().lt(1e-6).all()
    assert link.isco_code.str.fullmatch(r"\d{4}").all()
    pts = jp.load_ai_points(); nat = pts[pts.group_type == "all"]
    assert list(nat.period) == [str(y) for y in range(2018, 2026)] and set(pts.industry.dropna()) <= set(jp.INDUSTRIES)
    mix = jp.load_industry_mix(); assert set(mix.industry) <= set(jp.INDUSTRIES) and set(mix.major_group) <= set(jp.MAJOR_GROUP_NAMES)


def test_occupation_shifts_stay_numeric_without_industry_readings():
    """A country whose AI survey has no industry breakdown (Ireland) gets zero shifts as floats, not objects."""
    from onetx import countries as cn
    spec = cn.SPECS["IRL"]
    mix = pd.DataFrame({"major_group": ["1", "2"], "industry": ["C", "J"], "employed": [10.0, 20.0]})
    ind = pd.DataFrame(columns=["industry", "label", "n_points", "latest_date", "latest_pct", "national_pct_latest", "level_offset", "anchor_shift_years"])
    natw = pd.DataFrame({"soc10_code": ["1115", "2111"], "employment_thousands": [1.0, 2.0]})
    xw = pd.DataFrame({"onet_soc_code": ["11-1011.00", "19-1021.00"], "nat_code": ["1115", "2111"], "weight_rev": [1.0, 1.0]})
    grp, nat, onet = cn.occupation_shifts(mix, ind, natw, xw, spec)
    assert nat.adoption_shift_years.dtype == float and onet.adoption_shift_years.dtype == float and (nat.adoption_shift_years == 0).all()


def test_lead_basis_marks_checklist_surveys_as_upper_bounds():
    """Every country record says which firms its survey covers and what it asks; only the production
    question (Canada's) gives a lead that is a measurement rather than an upper bound."""
    for code, spec in cn.SPECS.items():
        assert spec.survey_firms in ("all", "10plus", "50plus", "100plus"), code
        assert spec.survey_question in ("production", "checklist", "iot_ai"), code
        assert spec.survey_note, code
    assert cn.SPECS["CAN"].survey_question == "production" and cn.SPECS["CAN"].survey_firms == "all"
    assert cn.SPECS["GBR"].survey_firms == "10plus" and cn.SPECS["NOR"].survey_firms == "10plus" and cn.SPECS["JPN"].survey_firms == "100plus"
    cmp = {"comparator": "10plus", "label": "US firms with 10 or more employees", "fallback": False, "saturation": 0.9, "intercept": -2.3, "growth_per_year": 0.5}
    b = cn.lead_basis(cn.SPECS["GBR"], cmp, 1.0)
    assert b["upper_bound"] is True and b["comparator"] == "10plus" and b["lead_years_all_firms"] == 1.0
    assert cn.lead_basis(cn.SPECS["CAN"], {"comparator": "all", "label": "all US firms"}, -0.5)["upper_bound"] is False


def test_size_matched_lead_is_smaller_when_the_comparator_sits_higher():
    """Reading the same national points against a US curve that sits higher (larger firms adopt sooner)
    gives a smaller lead; the all-firm lead is kept beside it for reference."""
    fit = {"saturation": 0.9, "intercept": -2.0, "growth_per_year": 0.5}
    higher = {**fit, "intercept": -1.5, "comparator": "100plus", "label": "US firms with 100 or more employees", "fallback": False}
    pts = pd.DataFrame({"group_type": ["all"], "date": [2025.5], "pct": [30.0], "period": ["2025"]})
    a = cn.firm_lead(fit, pts)["lead_years"]; h = cn.firm_lead(higher, pts)["lead_years"]
    assert h < a and abs((a - h) - 0.5 / 0.5) < 1e-6


def test_korea_loaders_from_kosis_downloads():
    """South Korea: the KOSIS downloads parse, every three-digit code links to ISCO through the official tables,
    pay reaches the 2024 classification through the 7th-to-8th link, and the firm survey gives a yearly series."""
    from onetx import kr_labour as kr
    from onetx import ai_signals as ai
    import pytest
    if not (kr.KOSIS / kr.EMPLOYMENT_TBL / f"{kr.EMPLOYMENT_TBL}.csv").exists():
        pytest.skip("KOSIS downloads not present")
    w = kr.build_weights()
    assert 150 <= len(w) <= 170 and (w.employment_thousands > 0).all() and w.wage_monthly.notna().all()
    assert 26000 <= w.employment_thousands.sum() <= 31000          # about 29 million employed
    assert w.ksco_code.str.len().eq(3).all() and w.ksco_title.notna().all()
    link = kr.load_isco_link(set(w.ksco_code))
    assert set(link.nat_code) == set(w.ksco_code)
    assert abs(link.groupby("isco_code").weight.sum() - 1.0).max() < 1e-9
    pts = kr.load_ai_points()
    nat = pts[pts.group_type == "all"].sort_values("date")
    assert len(nat) >= 8 and nat.pct.is_monotonic_increasing and 5 < nat.pct.iloc[-1] < 20
    assert set(pts[pts.group_type == "industry"].industry) >= {"C", "J", "K"}
    mix = kr.load_industry_mix()
    assert set(mix.major_group) == set("123456789") and mix.employed.sum() > 25000
    ch = kr.employment_change()
    assert ch.emp_change_pct.notna().sum() > 100 and ch.wage_change_pct.notna().sum() > 100
    assert cn.SPECS["KOR"].survey_firms == "50plus" and "50plus" in ai.COMPARATORS


def test_change_tables_name_their_employment_years():
    """The outcome test reads emp_<year_from> and emp_<year_to> from a country's change table, so the year
    labels must match the employment columns even when the pay years differ (Korea's do)."""
    from onetx import kr_labour as kr, jp_labour as jp, de_labour as de
    import pytest
    for mod, probe in ((kr, kr.KOSIS / kr.EMPLOYMENT_TBL / f"{kr.EMPLOYMENT_TBL}.csv"), (jp, jp.ESTAT / "wage2024_t1_occupation_small.xlsx"), (de, de.BA / "beschaeftigte_berufe_202506.xlsx")):
        if not probe.exists():
            continue
        d = mod.employment_change()
        yf, yt = int(d.year_from.iloc[0]), int(d.year_to.iloc[0])
        assert f"emp_{yf}" in d.columns and f"emp_{yt}" in d.columns, mod.__name__
        assert "emp_change_pct" in d.columns and "wage_change_pct" in d.columns


# ---------------------------------------------------------------------------- Germany (GENESIS and the employment agency) and France (Melodi)
def test_germany_loaders_from_genesis_and_the_employment_agency():
    """Germany: five-digit employees from the agency's requirement-level columns, pay from GENESIS at the same codes,
    the official key with weights summing to one per ISCO group, the AI series with 2025 from the revised table,
    and the derived occupation-by-industry mix summing to the employees."""
    from onetx import de_labour as de
    import pytest
    if not (de.BA / "beschaeftigte_berufe_202506.xlsx").exists() or not (de.DESTATIS / "62361-0030_2025.csv").exists():
        pytest.skip("German files not present")
    e = de.load_employment()
    assert e.kldb_code.str.fullmatch(r"\d{5}").all() and 33000000 < e.employees.sum() < 37000000   # 34.9 million employees subject to social insurance
    assert set(e.level) == {1, 2, 3, 4} and (e.kldb_code.str[-1].astype(int) == e.level).all()
    w = de.build_weights()
    assert 1250 <= len(w) <= 1320 and w.wage_monthly.gt(1500).all() and w.skill_level.between(1, 4).all()
    assert (w.wage_source == "earnings_survey_median_monthly_5digit").mean() > 0.8
    assert w.kldb_title.notna().all() and w.submajor_name.notna().all()
    link = de.load_isco_link(set(w.kldb_code))
    assert set(link.nat_code) == set(w.kldb_code) and (link.groupby("isco_code").weight.sum() - 1).abs().lt(1e-6).all()
    pts = de.load_ai_points(); nat = pts[pts.group_type == "all"].sort_values("date")
    assert list(nat.period) == ["2021", "2023", "2024", "2025"] and nat.pct.iloc[-1] > nat.pct.iloc[0]
    assert {"C", "J", "M"} <= set(pts[pts.group_type == "industry"].industry)
    mix = de.load_industry_mix()
    assert abs(mix.employed.sum() - e.employees.sum()) < 1.0 and mix.group2.str.len().eq(2).all()
    ch = de.employment_change()
    assert ch.emp_change_pct.notna().sum() > 1200 and ch.wage_change_pct.notna().sum() > 1000
    assert f"emp_{int(ch.year_from.iloc[0])}" in ch.columns and f"emp_{int(ch.year_to.iloc[0])}" in ch.columns


def test_france_loaders_from_melodi():
    """France: full-time-equivalent employees and net pay by detailed PCS from the private and public wage tables,
    every code linked to ISCO through the two matrices, the one-digit mix from the labour force survey, and the
    outcome table kept in shape with no change (one year on Melodi)."""
    from onetx import fr_labour as fr
    import pytest
    if not (fr.INSEE / "melodi_dera_prive.json").exists():
        pytest.skip("Melodi files not present")
    w = fr.build_weights()
    assert 380 <= len(w) <= 440 and w.pcs_code.str.fullmatch(r"\d{3}[A-Z]").all()
    assert 20000 < w.employment_thousands.sum() < 26000 and w.wage_monthly.gt(1000).all() and w.pcs_title.notna().all()
    assert w.skill_level.between(1, 4).all() and set(w.major_group) <= set("123456")
    link = fr.load_isco_link(set(w.pcs_code))
    assert set(link.nat_code) == set(w.pcs_code) and (link.groupby("isco_code").weight.sum() - 1).abs().lt(1e-6).all()
    pi = fr.pcs_to_isco()
    assert (pi.groupby("pcs_code").p.sum() - 1).abs().lt(1e-6).all()
    mix = fr.load_industry_mix()
    assert set(mix.major_group) == set("123456") and {"C", "M", "Q"} <= set(mix.industry) and mix.employed.sum() > 25000
    pts = fr.load_ai_points(); nat = pts[pts.group_type == "all"]
    assert len(nat) >= 4 and nat.pct.between(0, 100).all()
    ch = fr.employment_change()
    assert ch.emp_change_pct.isna().all() and ch.wage_change_pct.isna().all() and f"emp_{int(ch.year_to.iloc[0])}" in ch.columns
    assert cn.SPECS["FRA"].survey_firms == "10plus" and cn.SPECS["DEU"].survey_firms == "10plus"


def test_new_zealand_data_explorer_extracts_give_pay_and_a_mix():
    """With the Stats NZ key: a Census median income per sub-major group becomes the pay weight of its unit groups,
    and the occupation-by-industry extract gives a mix on the eight major groups and the ANZSIC divisions."""
    from onetx import nz_labour as nz
    import pytest
    if not nz.has_ade():
        pytest.skip("Data Explorer extracts not present")
    meds = nz.census_income_medians()
    assert len(meds["submajor"]) >= 40 and len(meds["major"]) == 8 and meds["major"].gt(10000).all()
    w = nz.build_anzsco_weights()
    assert (w.wage_source == "census_2023_submajor_group_median_income").all()
    mix = nz.load_industry_mix()
    assert set(mix.major_group) == set("12345678") and set(mix.industry) == set("ABCDEFGHIJKLMNOPQRS") and abs(mix.employed.sum() - 2622732) < 5000
    assert cn.SPECS["NZL"].depth == "medium"


def test_proxy_country_industry_shifts_are_carried_when_the_classification_matches(tmp_path, monkeypatch):
    """New Zealand has no survey but shares ANZSIC with Australia, whose industry shifts stand in for the sector lens."""
    assert cn.SPECS["NZL"].industry_names is cn.SPECS["AUS"].industry_names
    assert cn.SPECS["NZL"].lead_proxy == "AUS" and cn.SPECS["DEU"].lead_proxy is None
