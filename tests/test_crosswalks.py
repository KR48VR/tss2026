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
