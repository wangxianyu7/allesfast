"""Tests for the midtimes linear-ephemeris prior feature."""
import os, textwrap
import numpy as np
import pytest


_MIDTIMES_CSV = """\
midtimes,err,pl_letter
2455870.4505400,0.0010417,b
2456271.6588800,0.0009722,b
2456524.6261700,0.0012500,b
"""


def _make_basement(tmp_path, settings_extra='', params_extra='',
                   midtimes_csv=None):
    """Helper: build a minimal BASEMENT in a temp directory."""
    from allesfast.basement import Basement

    settings = textwrap.dedent(f"""\
        companions_rv,b
        companions_phot,
        inst_rv,HARPS
        inst_phot,
        multiprocess,False
        baseline_rv_HARPS,hybrid_offset
        {settings_extra}
    """)
    params = textwrap.dedent(f"""\
        #name,value,fit,bounds,label,unit,coupled_with,coupled_tolerance
        b_period,1.360029,0,none,$P$,days,,
        b_epoch,2455870.45054,0,none,$T_0$,BJD,,
        b_rr,0.1,0,none,$R_p/R_s$,,,
        b_rsuma,0.15,0,none,$(R_p+R_s)/a$,,,
        b_cosi,0.0,0,none,$\\cos i$,,,
        b_ecc,0.0,0,none,$e$,,,
        b_f_c,0.0,0,none,$f_c$,,,
        b_f_s,0.0,0,none,$f_s$,,,
        {params_extra}
    """)
    rv_data = "2455870.0,0.0,0.001\n2455871.0,0.1,0.001\n2455872.0,-0.1,0.001\n"

    (tmp_path / 'settings.csv').write_text(settings)
    (tmp_path / 'params.csv').write_text(params)
    (tmp_path / 'HARPS.csv').write_text(rv_data)

    if midtimes_csv is not None:
        (tmp_path / 'midtimes.csv').write_text(midtimes_csv)

    return Basement(str(tmp_path), quiet=True)


def test_midtimes_file_setting_parsed(tmp_path):
    """midtimes_file,midtimes.csv is parsed correctly."""
    b = _make_basement(tmp_path,
                       settings_extra='midtimes_file,midtimes.csv',
                       midtimes_csv=_MIDTIMES_CSV)
    assert b.settings['midtimes_file'] == 'midtimes.csv'


def test_midtimes_file_default_none(tmp_path):
    """midtimes_file defaults to None when absent from settings.csv."""
    b = _make_basement(tmp_path)
    assert b.settings['midtimes_file'] is None


def test_midtimes_data_loaded(tmp_path):
    """midtimes.csv is loaded into BASEMENT.midtimes_data."""
    b = _make_basement(tmp_path,
                       settings_extra='midtimes_file,midtimes.csv',
                       midtimes_csv=_MIDTIMES_CSV)
    assert 'b' in b.midtimes_data
    T_obs, sigma = b.midtimes_data['b']
    assert len(T_obs) == 3
    np.testing.assert_allclose(T_obs[0], 2455870.4505400)
    np.testing.assert_allclose(sigma[0], 0.0010417)


def test_midtimes_data_empty_when_no_setting(tmp_path):
    """midtimes_data is empty when midtimes_file is not set."""
    b = _make_basement(tmp_path)
    assert b.midtimes_data == {}


def test_midtimes_file_missing_raises(tmp_path):
    """FileNotFoundError when midtimes_file is set but file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        _make_basement(tmp_path,
                       settings_extra='midtimes_file,midtimes.csv')


def _params_for_b(tc, period):
    """Return a minimal params dict for companion b."""
    return {
        'b_epoch':  tc,
        'b_period': period,
        'b_rr':     0.1,
        'b_rsuma':  0.15,
        'b_cosi':   0.0,
        'b_ecc':    0.0,
        'b_f_c':    0.0,
        'b_f_s':    0.0,
    }


def _full_params(basement, tc, period):
    """Build a complete params dict from basement defaults, overriding epoch/period."""
    p = dict(basement.params)
    p.update(_params_for_b(tc, period))
    rr = p.get('b_rr', 0.1)
    rsuma = p.get('b_rsuma', 0.15)
    p['b_radius_1'] = rsuma / (1.0 + rr)
    p['b_radius_2'] = p['b_radius_1'] * rr
    p['b_incl'] = 90.0
    return p


def test_midtimes_penalty_zero_on_perfect_ephemeris(tmp_path):
    """chi2 is zero when params exactly reproduce all observed midtimes."""
    from allesfast import config
    from allesfast.computer import calculate_external_priors

    tc = 2455870.4505400
    P  = 1.360029

    Ns = np.array([0, 295, 481])
    T_obs  = tc + Ns * P
    sigma  = np.full(3, 0.001)

    b = _make_basement(tmp_path,
                       settings_extra='midtimes_file,midtimes.csv',
                       midtimes_csv=_MIDTIMES_CSV)
    b.midtimes_data = {'b': (T_obs, sigma)}
    config.BASEMENT = b

    lnp = calculate_external_priors(_full_params(b, tc, P))
    assert np.isfinite(lnp)
    assert abs(lnp) < 1e-10


def test_midtimes_penalty_nonzero_when_period_offset(tmp_path):
    """chi2 grows when period is wrong."""
    from allesfast import config
    from allesfast.computer import calculate_external_priors

    tc_true = 2455870.4505400
    P_true  = 1.360029
    Ns = np.array([0, 295, 481])
    T_obs  = tc_true + Ns * P_true
    sigma  = np.full(3, 0.001)

    b = _make_basement(tmp_path,
                       settings_extra='midtimes_file,midtimes.csv',
                       midtimes_csv=_MIDTIMES_CSV)
    b.midtimes_data = {'b': (T_obs, sigma)}
    config.BASEMENT = b

    lnp_good = calculate_external_priors(_full_params(b, tc_true, P_true))
    lnp_bad  = calculate_external_priors(_full_params(b, tc_true, P_true + 0.01))
    assert lnp_good > lnp_bad


def test_midtimes_penalty_skipped_when_no_setting(tmp_path):
    """No penalty when midtimes_file is not set."""
    from allesfast import config
    from allesfast.computer import calculate_external_priors

    b = _make_basement(tmp_path)
    config.BASEMENT = b

    lnp = calculate_external_priors(_full_params(b, 2455870.4505400, 99.0))
    assert np.isfinite(lnp)
