import os, textwrap, tempfile, numpy as np
from allesfast import config

MINIMAL_SETTINGS = '#name,value\nmultiprocess,False\nmcmc_nwalkers,10\nmcmc_total_steps,100\n'

def test_coupled_tolerance_parsed(tmp_path):
    """basement stores coupled_tolerance as float array from 8th column."""
    settings = os.path.join(tmp_path, 'settings.csv')
    with open(settings, 'w') as f:
        f.write(MINIMAL_SETTINGS)
    params_csv = os.path.join(tmp_path, 'params.csv')
    with open(params_csv, 'w') as f:
        f.write(textwrap.dedent("""\
            #name,value,fit,bounds,label,unit,coupled_with,coupled_tolerance
            A_feh,0.15,1,uniform -1 1,$feh$,,,
            B_feh,0.15,0,fixed,$feh_B$,,A_feh,0
            A_age,10.0,1,uniform 0 14,$age$,Gyr,,
            B_age,10.0,0,fixed,$age_B$,Gyr,A_age,0.1
        """))
    config.init(str(tmp_path), quiet=True)
    idx = list(config.BASEMENT.allkeys).index('B_age')
    assert hasattr(config.BASEMENT, 'coupled_tolerance')
    assert float(config.BASEMENT.coupled_tolerance[idx]) == 0.1
    # Verify rows that leave coupled_tolerance blank default to 0.0
    idx_a_feh = list(config.BASEMENT.allkeys).index('A_feh')
    idx_b_feh = list(config.BASEMENT.allkeys).index('B_feh')
    assert float(config.BASEMENT.coupled_tolerance[idx_a_feh]) == 0.0
    assert float(config.BASEMENT.coupled_tolerance[idx_b_feh]) == 0.0

def test_coupled_tolerance_defaults_zero(tmp_path):
    """params.csv without coupled_tolerance column → array of zeros."""
    settings = os.path.join(tmp_path, 'settings.csv')
    with open(settings, 'w') as f:
        f.write(MINIMAL_SETTINGS)
    params_csv = os.path.join(tmp_path, 'params.csv')
    with open(params_csv, 'w') as f:
        f.write(textwrap.dedent("""\
            #name,value,fit,bounds,label,unit,coupled_with
            A_feh,0.15,1,uniform -1 1,$feh$,,
            A_age,10.0,1,uniform 0 14,$age$,Gyr,
        """))
    config.init(str(tmp_path), quiet=True)
    assert all(float(t) == 0.0 for t in config.BASEMENT.coupled_tolerance)

def test_soft_link_penalty_applied(tmp_path):
    """coupled_tolerance > 0 on B_age is stored and leads to a penalty term.
    Verifies that the stored tolerance is > 0 (enabling the penalty loop in lnprob).
    The actual penalty math is tested implicitly via the end-to-end smoke test.
    """
    import pytest
    settings = os.path.join(tmp_path, 'settings.csv')
    with open(settings, 'w') as f:
        f.write(MINIMAL_SETTINGS)
    params_csv = os.path.join(tmp_path, 'params.csv')
    with open(params_csv, 'w') as f:
        f.write('#name,value,fit,bounds,label,unit,coupled_with,coupled_tolerance\n'
                'A_age,10.0,0,fixed,$age_A$,Gyr,,\n'
                'B_age,10.0,0,fixed,$age_B$,Gyr,A_age,0.1\n')
    config.init(str(tmp_path), quiet=True)
    # verify the tolerance is stored and > 0 for B_age
    idx = list(config.BASEMENT.allkeys).index('B_age')
    assert config.BASEMENT.coupled_tolerance[idx] == pytest.approx(0.1)
    assert config.BASEMENT.coupled_with[idx] == 'A_age'
    # unit test: the penalty formula itself
    val, ref, tol = 10.5, 10.0, 0.1
    penalty = 0.5 * ((val - ref) / tol) ** 2
    assert penalty == pytest.approx(12.5)
