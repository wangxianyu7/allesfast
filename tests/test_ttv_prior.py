"""Tests for the TTV linear-ephemeris prior feature."""
import os, textwrap
import numpy as np
import pytest


def _make_basement(tmp_path, settings_extra='', params_extra='', ttv_files=None):
    """Helper: build a minimal BASEMENT in a temp directory."""
    from allesfast.basement import Basement

    settings = textwrap.dedent(f"""\
        companions_rv,b
        companions_phot,
        inst_rv,HARPS
        inst_phot,
        multiprocess,False
        use_ttv_prior,True
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

    if ttv_files:
        for fname, content in ttv_files.items():
            (tmp_path / fname).write_text(content)

    return Basement(str(tmp_path), quiet=True)


def test_use_ttv_prior_setting_parsed_true(tmp_path):
    """use_ttv_prior,True is parsed as boolean True."""
    b = _make_basement(tmp_path)
    assert b.settings['use_ttv_prior'] is True


def test_use_ttv_prior_setting_default_false(tmp_path):
    """use_ttv_prior defaults to False when absent from settings.csv."""
    from allesfast.basement import Basement
    settings = textwrap.dedent("""\
        companions_rv,b
        companions_phot,
        inst_rv,HARPS
        inst_phot,
        multiprocess,False
    """)
    params = textwrap.dedent("""\
        #name,value,fit,bounds,label,unit,coupled_with,coupled_tolerance
        b_period,1.360029,0,none,$P$,days,,
        b_epoch,2455870.45054,0,none,$T_0$,BJD,,
        b_rr,0.1,0,none,$R_p/R_s$,,,
        b_rsuma,0.15,0,none,$(R_p+R_s)/a$,,,
        b_cosi,0.0,0,none,$\\cos i$,,,
        b_ecc,0.0,0,none,$e$,,,
        b_f_c,0.0,0,none,$f_c$,,,
        b_f_s,0.0,0,none,$f_s$,,,
    """)
    (tmp_path / 'settings.csv').write_text(settings)
    (tmp_path / 'params.csv').write_text(params)
    (tmp_path / 'HARPS.csv').write_text("2455870.0,0.0,0.001\n2455871.0,0.1,0.001\n2455872.0,-0.1,0.001\n")
    b = Basement(str(tmp_path), quiet=True)
    assert b.settings['use_ttv_prior'] is False


_TTV_CONTENT = """\
2455870.4505400 0.0010417
2456271.6588800 0.0009722
2456524.6261700 0.0012500
"""


def test_ttv_data_loaded_for_companion(tmp_path):
    """*.b.ttv file is loaded into BASEMENT.ttv_data['b']."""
    b = _make_basement(tmp_path, ttv_files={'obs.b.ttv': _TTV_CONTENT})
    assert 'b' in b.ttv_data
    T_obs, sigma = b.ttv_data['b']
    assert len(T_obs) == 3
    np.testing.assert_allclose(T_obs[0], 2455870.4505400)
    np.testing.assert_allclose(sigma[0], 0.0010417)


def test_ttv_data_empty_when_no_files(tmp_path):
    """ttv_data is an empty dict when no .ttv files exist."""
    b = _make_basement(tmp_path)
    assert b.ttv_data == {}


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
    # Compute derived radius_1 so eccentricity guard doesn't crash
    rr = p.get('b_rr', 0.1)
    rsuma = p.get('b_rsuma', 0.15)
    p['b_radius_1'] = rsuma / (1.0 + rr)
    p['b_radius_2'] = p['b_radius_1'] * rr
    p['b_incl'] = 90.0
    return p


def test_ttv_penalty_zero_on_perfect_ephemeris(tmp_path):
    """chi2 is zero when params exactly reproduce all observed midtimes."""
    from allesfast import config
    from allesfast.computer import calculate_external_priors

    tc = 2455870.4505400
    P  = 1.360029

    Ns = np.array([0, 295, 481])
    T_obs  = tc + Ns * P
    sigma  = np.full(3, 0.001)

    b = _make_basement(tmp_path)
    b.ttv_data = {'b': (T_obs, sigma)}
    config.BASEMENT = b

    lnp = calculate_external_priors(_full_params(b, tc, P))
    assert np.isfinite(lnp)
    assert abs(lnp) < 1e-10


def test_ttv_penalty_nonzero_when_period_offset(tmp_path):
    """chi2 grows when period is wrong."""
    from allesfast import config
    from allesfast.computer import calculate_external_priors

    tc_true = 2455870.4505400
    P_true  = 1.360029
    Ns = np.array([0, 295, 481])
    T_obs  = tc_true + Ns * P_true
    sigma  = np.full(3, 0.001)

    b = _make_basement(tmp_path)
    b.ttv_data = {'b': (T_obs, sigma)}
    config.BASEMENT = b

    lnp_good = calculate_external_priors(_full_params(b, tc_true, P_true))
    lnp_bad  = calculate_external_priors(_full_params(b, tc_true, P_true + 0.01))
    assert lnp_good > lnp_bad


def test_ttv_penalty_skipped_when_setting_false(tmp_path):
    """No penalty when use_ttv_prior is False, even if ttv_data is loaded."""
    import textwrap
    from allesfast.basement import Basement
    from allesfast import config
    from allesfast.computer import calculate_external_priors

    settings = textwrap.dedent("""\
        companions_rv,b
        companions_phot,
        inst_rv,HARPS
        inst_phot,
        use_ttv_prior,False
        multiprocess,False
    """)
    params_txt = textwrap.dedent("""\
        #name,value,fit,bounds,label,unit,coupled_with,coupled_tolerance
        b_period,1.360029,0,none,$P$,days,,
        b_epoch,2455870.45054,0,none,$T_0$,BJD,,
        b_rr,0.1,0,none,$R_p/R_s$,,,
        b_rsuma,0.15,0,none,$(R_p+R_s)/a$,,,
        b_cosi,0.0,0,none,$\\cos i$,,,
        b_ecc,0.0,0,none,$e$,,,
        b_f_c,0.0,0,none,$f_c$,,,
        b_f_s,0.0,0,none,$f_s$,,,
    """)
    rv_data = "2455870.0,0.0,0.001\n2455871.0,0.0,0.001\n2455872.0,0.0,0.001\n"
    (tmp_path / 'settings.csv').write_text(settings)
    (tmp_path / 'params.csv').write_text(params_txt)
    (tmp_path / 'HARPS.csv').write_text(rv_data)

    b = Basement(str(tmp_path), quiet=True)
    b.ttv_data = {'b': (np.array([2455870.4505400, 2456271.6588800]),
                        np.array([0.001, 0.001]))}
    config.BASEMENT = b

    # Wildly wrong period — but use_ttv_prior=False so lnp must be finite
    lnp = calculate_external_priors(_full_params(b, 2455870.4505400, 99.0))
    assert np.isfinite(lnp)
