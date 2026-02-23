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
