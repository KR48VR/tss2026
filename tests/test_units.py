"""Pure unit tests that do not need the data."""
import numpy as np

from onetx.forecast import ForecastParams, _frac_year
from onetx.onet import normalize_text


def test_normalize_text():
    assert normalize_text("  Write   new programs.  ") == "write new programs"
    assert normalize_text("Analyze clients’ data – fast.") == "analyze clients' data - fast"
    assert normalize_text(None) == ""


def test_frac_year():
    import pandas as pd

    assert abs(_frac_year(pd.Timestamp("2026-01-01")) - 2026.0) < 1e-9
    assert 2026.49 < _frac_year(pd.Timestamp("2026-07-01")) < 2026.51


def test_params_defaults_are_ordered():
    p = ForecastParams()
    assert p.doubling_days_fast < p.doubling_days_central < p.doubling_days_slow
    assert p.max_barrier_years[0] < p.max_barrier_years[1] < p.max_barrier_years[2]
    assert p.adoption_growth_mult[0] > p.adoption_growth_mult[1] > p.adoption_growth_mult[2]
    for tier, (f, c, s) in p.robot_tier_years.items():
        if tier:
            assert f <= c <= s


def test_adoption_curve_monotone():
    from onetx.forecast import ForecastParams, adoption_cross_year, geo_lead_years

    p = ForecastParams()
    y25, y50, y75 = (adoption_cross_year(p, l) for l in (0.25, 0.5, 0.75))
    assert y25 < y50 < y75
    assert adoption_cross_year(p, 0.5, 1.5) < y50 < adoption_cross_year(p, 0.5, 0.6)
    assert geo_lead_years(p) > 0
    p.geo = "USA"
    assert geo_lead_years(p) == 0.0
