"""Integrity checks on the processed tables (run after `python -m onetx build`)."""
import pytest

from onetx.io import exists, load

pytestmark = pytest.mark.skipif(not exists("frontier_tasks"), reason="run `python -m onetx build` first")


def test_weights_sum_to_one():
    for name, key in [("xw_onetsoc_isco08", "onet_soc_code"), ("xw_onetsoc_ssoc2024", "onet_soc_code"), ("xw_isco08_ssoc2024", "isco_code"), ("xw_sfw_role_ssoc2024", "role_id")]:
        df = load(name)
        s = df.groupby(key).weight.sum()
        assert (s.sub(1).abs() < 1e-6).all(), name
    rev = load("xw_onetsoc_ssoc2024").groupby("ssoc_code").weight_rev.sum()
    assert (rev.sub(1).abs() < 1e-6).all()


def test_task_weights_sum_to_one_per_occupation():
    t = load("onet_tasks")
    s = t.groupby("onet_soc_code").task_weight.sum()
    assert (s.sub(1).abs() < 1e-6).all()


def test_every_task_has_a_status_and_forecast():
    tf = load("frontier_tasks")
    fc = load("forecast_tasks")
    assert set(tf.status.unique()) <= {"physical_robotics", "observed_automation", "observed_augmentation", "not_observed"}
    assert len(fc) == len(tf)
    ok = fc.dropna(subset=["disrupt_year_p50"])
    assert (ok.disrupt_year_p10 <= ok.disrupt_year_p50 + 1e-9).all()
    assert (ok.disrupt_year_p50 <= ok.disrupt_year_p90.fillna(9999) + 1e-9).all()


def test_dwa_ids_align_across_versions():
    a = load("onet_activities")
    b = load("onet_activities_v31")
    assert a.dwa_id.isna().sum() == 0 and b.dwa_id.isna().sum() == 0
    assert set(a.dwa_id) == set(b.dwa_id)


def test_singapore_links_cover_most_roles():
    r = load("sfw_roles")
    x = load("xw_sfw_role_ssoc2024")
    assert x.role_id.nunique() >= 0.95 * r.role_id.nunique()
    assert (x[x.source == "singstat_index"].role_id.nunique()) >= 1500


def test_singapore_weights_match_lfs_totals():
    from onetx import singapore_labour as sl

    w = load("sg_ssoc2024_weights")
    d8 = sl.load_lfs_employment_submajor(2025)
    covered = d8[d8.submajor_members.apply(lambda m: any(x in set(w.ssoc_code.str[:2]) for x in m))]
    assert abs(w.employment_thousands.sum() - covered.employed_thousands.sum()) < 1e-6
    assert (w.wage_median > 0).all()
    assert (w.wage_source == "ows_2025_coded").sum() >= 500


def test_barriers_and_forecast_components():
    b = load("frontier_task_barriers")
    for c in ["b_interpersonal", "b_in_person", "b_accountability", "b_handover", "barrier_index"]:
        assert b[c].between(0, 1).all(), c
    fc = load("forecast_tasks")
    assert (fc.barrier_years_central >= 0).all()
    assert fc.adoption_anchor_year_central.iloc[0] > 2026
    occ = load("frontier_occupations")
    assert "n_providers_positive" in occ.columns and occ.n_providers_positive.max() >= 3


def test_economy_exposure_shape():
    e = load("sg_economy_exposure")
    econ = e[e.scope == "economy"]
    assert set(econ.weighting) == {"employment", "wage_value"}
    assert econ.weight_scored_share.min() > 0.95
    assert econ.share_by_2035.between(0, 1).all()
