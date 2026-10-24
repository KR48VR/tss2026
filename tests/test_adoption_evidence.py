"""Unit tests for the adoption curve with a wording step, sector levels, the Singapore lead, evidence
tiers and the rating parser. Synthetic data only; no downloads needed."""
import numpy as np
import pandas as pd

from onetx import ai_signals as ai
from onetx import forecast as fcm
from onetx import rate
from onetx import sector_adoption as sa


def _logistic_series(K, a, b, step, t0, t1, n, wording_switch):
    t = np.linspace(t0, t1, n)
    rows = []
    for ti in t:
        new = ti >= wording_switch
        p = K / (1 + np.exp(-(a + b * ti + (step if new else 0.0))))
        rows.append({"question": "current_use", "sample_period": "x", "ref_end": pd.Timestamp("2024-01-01") + pd.Timedelta(days=ti * 365.25), "pct": 100 * p, "wording": "si_2025" if new else "ai_2023"})
    return pd.DataFrame(rows)


def test_adoption_fit_recovers_growth_and_step():
    s = _logistic_series(0.9, -3.0, 0.5, 0.6, -0.3, 2.8, 70, 1.9)
    fit = ai.fit_adoption_logistic(s, saturation=0.9)
    assert abs(fit["growth_per_year"] - 0.5) < 0.02
    assert abs(fit["wording_step"] - 0.6) < 0.05
    assert abs(fit["intercept"] - (-2.4)) < 0.05
    assert fit["year_cross_50pct"] < fit["year_cross_50pct_old_wording"]
    assert fit["year_cross_25pct"] < fit["year_cross_50pct"] < fit["year_cross_75pct"]


def test_adoption_fit_without_step_matches_plain_logistic():
    s = _logistic_series(0.9, -3.0, 0.5, 0.0, 0.0, 2.0, 40, 99.0)  # never switches wording
    fit = ai.fit_adoption_logistic(s, saturation=0.9)
    assert abs(fit["growth_per_year"] - 0.5) < 0.02 and abs(fit["wording_step"]) < 1e-9


def test_sector_levels_shift_sign():
    fit = {"saturation": 0.9, "intercept": -2.4, "growth_per_year": 0.5, "year_cross_50pct": 2029.2}
    rows = []
    for sector, off in [("51", 1.0), ("72", -1.0)]:
        for ti in np.linspace(2.0, 2.8, 10):
            p = 0.9 / (1 + np.exp(-(-2.4 + 0.5 * ti + off)))
            rows.append({"sector": sector, "question": "current_use", "sample_period": "x", "ref_end": pd.Timestamp("2024-01-01") + pd.Timedelta(days=ti * 365.25), "pct": 100 * p, "wording": "si_2025"})
    lv = ai.fit_sector_levels(pd.DataFrame(rows), fit).set_index("sector")
    assert lv.loc["51", "anchor_shift_years"] < 0 < lv.loc["72", "anchor_shift_years"]
    assert abs(lv.loc["51", "anchor_shift_years"] + 2.0) < 0.05  # offset 1.0 at growth 0.5 = 2 years earlier


def test_singapore_lead_reads_matched_levels():
    fit = {"saturation": 0.9, "intercept": -2.4, "growth_per_year": 0.5}
    # a Singapore point exactly on the US curve one year earlier than its year-end date gives a lead of 1
    us_year_for_20pct = 2024.0 + (np.log(0.2 / 0.7) - (-2.4)) / 0.5
    sg = pd.DataFrame([{"year": int(np.floor(us_year_for_20pct - 1.0)) - 1, "group": "all", "pct": 20.0, "source": "test"}])
    sg["year"] = int(round(us_year_for_20pct - 1.0 - 1.0))
    out = ai.singapore_firm_lead(fit, sg, min_year=2000)
    assert abs(out["lead_years"] - (us_year_for_20pct - (sg.year.iloc[0] + 1.0))) < 1e-9


def test_evidence_tiers_logic():
    tf = pd.DataFrame({"task_id": ["1", "2", "3", "4", "5", "6"], "physical": [False, False, False, False, False, True], "usage_n_vintages": [2, 0, 0, 0, 0, 0]})
    labels = pd.DataFrame({"task_id": ["1", "2", "3", "4", "5", "6"], "exposure_human_2023": ["E0", "E1", "E0", "E0", "E1", "E0"], "exposure_gpt4_2023": ["E0", "E2", "E1", "E0", "E0", "E0"], "automation_gpt4_2023": ["T0"] * 6})
    t = fcm.evidence_tiers(tf, labels).set_index("task_id").evidence_tier
    assert t["1"] == "seen" and t["2"] == "rated" and t["3"] == "mixed" and t["4"] == "unproven" and t["6"] == "physical"
    ratings = pd.DataFrame({"task_id": ["2", "3", "4", "5"], "exposure": ["E1", "E0", "E1", "E0"], "autonomy": ["A2"] * 4, "barrier": ["none"] * 4})
    t2 = fcm.evidence_tiers(tf, labels, ratings).set_index("task_id").evidence_tier
    assert t2["2"] == "rated" and t2["3"] == "mixed" and t2["4"] == "newly_rated" and t2["5"] == "mixed"


def test_evidence_lag_delays_unproven_desk_tasks():
    p = fcm.ForecastParams()
    p.evidence_lag_years = {"mixed": (0.5, 1.0, 2.0), "unproven": (1.0, 2.0, 4.0), "newly_rated": (0.0, 0.0, 0.0)}
    tf = pd.DataFrame({"task_id": [1, 2], "onet_soc_code": ["11-1011.00"] * 2, "task_weight": [1.0, 1.0], "physical": [False, False], "robot_tier": [np.nan, np.nan], "status": ["not_observed"] * 2,
                       "usage_latest": [0.0, 0.0], "automation_share": [0.4, 0.4], "observed_automation": [0.0, 0.0], "human_only_ability_pct": [np.nan, np.nan], "usage_index": [0.0, 0.0], "usage_trend": [np.nan, np.nan], "human_only_hours": [2.0, 2.0]})
    barriers = pd.DataFrame({"task_id": [1, 2], "barrier_index": [0.2, 0.2], "barrier_type": ["none"] * 2, "b_interpersonal": [0, 0], "b_in_person": [0, 0], "b_accountability": [0, 0], "b_handover": [0, 0]})
    tiers = pd.DataFrame({"task_id": ["1", "2"], "evidence_tier": ["rated", "unproven"], "exposure_2026": [None, None], "autonomy_2026": [None, None], "barrier_2026": [None, None]})
    p = fcm.calibrate(p, tf.assign(usage_latest=[0.1, 0.1], automation_share=[0.5, 0.6]))
    fc = fcm.forecast_tasks(tf, barriers, p, now="2026-10-01", tiers=tiers).set_index("task_id")
    assert abs((fc.loc[2, "disrupt_year_p50"] - fc.loc[1, "disrupt_year_p50"]) - 2.0) < 1e-6


def test_industry_mix_and_shifts():
    d2 = pd.DataFrame([{"industry": "J", "major_group": "2", "employed_thousands": 50}, {"industry": "I", "major_group": "2", "employed_thousands": 50}, {"industry": "I", "major_group": "5", "employed_thousands": 100}])
    roles = pd.DataFrame({"role_id": ["r1"], "sector": ["Infocomm Technology"]})
    links = pd.DataFrame({"role_id": ["r1"], "ssoc_code": ["25100"], "weight": [1.0]})
    mix = sa.ssoc_industry_mix(d2, roles, links, pd.Series(["25100", "51000"]))
    m25 = mix[mix.ssoc_code == "25100"].set_index("industry").share
    assert abs(m25["J"] - 0.75) < 1e-9 and abs(m25["I"] - 0.25) < 1e-9  # half D2 mix, half the role's industry
    assert abs(mix[mix.ssoc_code == "51000"].set_index("industry").share["I"] - 1.0) < 1e-9
    lv = pd.DataFrame({"sector": ["51", "72"], "anchor_shift_years": [-2.0, 2.0], "latest_pct": [40, 10]})
    ind = sa.industry_shifts(lv).set_index("industry").anchor_shift_years
    assert ind["J"] == -2.0 and ind["I"] == 2.0 and ind["C"] == 0.0


def test_rating_parser_is_strict():
    txt = '```json\n[{"n":1,"exposure":"E1","autonomy":"A2","barrier":"judgement"},{"n":2,"exposure":"E9","autonomy":"A0","barrier":"none"},{"n":3,"exposure":"e0","autonomy":"a1","barrier":"in_person"}]\n```'
    out = rate._parse(txt, 3)
    assert out[0] == {"exposure": "E1", "autonomy": "A2", "barrier": "judgement"}
    assert out[1] is None
    assert out[2] == {"exposure": "E0", "autonomy": "A1", "barrier": "in_person"}


def test_validation_ladder_and_agreement():
    from onetx import validation as val

    n = 400
    rng = np.random.default_rng(0)
    seen_n = rng.integers(0, 3, n)
    tf = pd.DataFrame({"task_id": [str(i) for i in range(n)], "physical": [False] * n, "usage_n_vintages": seen_n, "automation_share": rng.uniform(0, 1, n), "ms_share": np.nan})
    # a rating that is informative: 'E1' more often when seen
    human = np.where(rng.uniform(0, 1, n) < np.where(seen_n > 0, 0.8, 0.3), "E1", "E0")
    labels = pd.DataFrame({"task_id": tf.task_id, "exposure_human_2023": human, "exposure_gpt4_2023": human, "automation_gpt4_2023": "T1", "beta_2023": np.where(human == "E1", 1.0, 0.0)})
    ratings = pd.DataFrame({"task_id": tf.task_id, "exposure": human, "autonomy": np.where(tf.automation_share > 0.5, "A3", "A1"), "barrier": "none"})
    out = val.rating_validation(tf, labels, ratings)
    assert out["auc_seen"]["human_2023"] > 0.6
    assert out["agreement"]["human_vs_gpt4_2023"]["agree"] == 1.0 and abs(out["agreement"]["human_vs_gpt4_2023"]["kappa"] - 1.0) < 1e-9
    assert out["auc_doing_given_seen"]["autonomy_2026"] > 0.9
    lad = out["ladder"]
    assert 0 < lad["rated_to_seen"] <= 1 and 0 <= lad["rated_no_to_seen"] < lad["rated_to_seen"]
    assert 0 <= lad["seen_to_doing"] <= 1
