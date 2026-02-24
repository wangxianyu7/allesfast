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
