"""Tests for age as a derived parameter when MIST is enabled."""
import os, textwrap, numpy as np
import pytest
from allesfast import config


MIST_SETTINGS = (
    '#name,value\n'
    'multiprocess,False\n'
    'mcmc_nwalkers,10\n'
    'mcmc_total_steps,100\n'
    'use_mist_prior,True\n'
)


def test_age_forced_nonfree_with_mist(tmp_path):
    """When use_mist_prior=True, age with fit=1 is removed from fitkeys."""
    settings = os.path.join(tmp_path, 'settings.csv')
    with open(settings, 'w') as f:
        f.write(MIST_SETTINGS)
    params_csv = os.path.join(tmp_path, 'params.csv')
    with open(params_csv, 'w') as f:
        f.write(textwrap.dedent("""\
            #name,value,fit,bounds,label,unit
            A_rstar,1.0,1,uniform 0.1 10,$R_A$,Rsun
            A_teff,5800,1,uniform 4000 7000,$T_A$,K
            A_feh,0.0,1,uniform -1 1,$feh_A$,
            A_logmstar,0.0,1,uniform -1 1,$logM_A$,
            A_eep,350,1,uniform 1 808,$EEP_A$,
            A_age,5.0,1,uniform 0 14,$age_A$,Gyr
        """))
    config.init(str(tmp_path), quiet=True)
    assert 'A_age' not in config.BASEMENT.fitkeys
    assert 'A_age' in config.BASEMENT.derived_keys


def test_age_with_prior_stored(tmp_path):
    """fit=1 + normal bounds on age → stored in derived_priors, removed from fitkeys."""
    settings = os.path.join(tmp_path, 'settings.csv')
    with open(settings, 'w') as f:
        f.write(MIST_SETTINGS)
    params_csv = os.path.join(tmp_path, 'params.csv')
    with open(params_csv, 'w') as f:
        f.write(textwrap.dedent("""\
            #name,value,fit,bounds,label,unit
            A_rstar,1.0,1,uniform 0.1 10,$R_A$,Rsun
            A_teff,5800,1,uniform 4000 7000,$T_A$,K
            A_feh,0.0,1,uniform -1 1,$feh_A$,
            A_logmstar,0.0,1,uniform -1 1,$logM_A$,
            A_eep,350,1,uniform 1 808,$EEP_A$,
            A_age,5.0,1,normal 5.0 1.0,$age_A$,Gyr
        """))
    config.init(str(tmp_path), quiet=True)
    assert 'A_age' not in config.BASEMENT.fitkeys
    assert 'A_age' in config.BASEMENT.derived_priors
    assert config.BASEMENT.derived_priors['A_age'] == ('normal', 5.0, 1.0)


def test_age_not_derived_without_mist(tmp_path):
    """Without use_mist_prior, age with fit=1 stays in fitkeys normally."""
    settings = os.path.join(tmp_path, 'settings.csv')
    with open(settings, 'w') as f:
        f.write('#name,value\nmultiprocess,False\nmcmc_nwalkers,10\nmcmc_total_steps,100\n')
    params_csv = os.path.join(tmp_path, 'params.csv')
    with open(params_csv, 'w') as f:
        f.write(textwrap.dedent("""\
            #name,value,fit,bounds,label,unit
            A_age,5.0,1,uniform 0 14,$age_A$,Gyr
        """))
    config.init(str(tmp_path), quiet=True)
    assert 'A_age' in config.BASEMENT.fitkeys


def test_age_derived_in_update_params(tmp_path):
    """update_params computes age from eep+mstar+initfeh when MIST is on."""
    settings = os.path.join(tmp_path, 'settings.csv')
    with open(settings, 'w') as f:
        f.write(MIST_SETTINGS)
    params_csv = os.path.join(tmp_path, 'params.csv')
    with open(params_csv, 'w') as f:
        f.write(textwrap.dedent("""\
            #name,value,fit,bounds,label,unit
            A_rstar,1.0,1,uniform 0.1 10,$R_A$,Rsun
            A_teff,5800,1,uniform 4000 7000,$T_A$,K
            A_feh,0.0,1,uniform -1 1,$feh_A$,
            A_logmstar,0.0,1,uniform -1 1,$logM_A$,
            A_eep,350,1,uniform 1 808,$EEP_A$,
        """))
    config.init(str(tmp_path), quiet=True)
    from allesfast.computer import update_params
    from allesfast.star.massradius_mist import get_mistage
    params = update_params(config.BASEMENT.theta_0)
    expected_age = get_mistage(
        float(params['A_eep']), float(params['A_mstar']),
        float(params['A_feh']),
    )
    assert np.isfinite(params['A_age'])
    assert params['A_age'] == pytest.approx(expected_age, rel=1e-10)


def test_age_couple_tolerance_setting(tmp_path):
    """age_couple_tolerance parsed from settings.csv."""
    settings = os.path.join(tmp_path, 'settings.csv')
    with open(settings, 'w') as f:
        f.write(MIST_SETTINGS + 'age_couple_tolerance,0.5\n')
    params_csv = os.path.join(tmp_path, 'params.csv')
    with open(params_csv, 'w') as f:
        f.write(textwrap.dedent("""\
            #name,value,fit,bounds,label,unit
            A_rstar,1.0,1,uniform 0.1 10,$R_A$,Rsun
            A_teff,5800,1,uniform 4000 7000,$T_A$,K
            A_feh,0.0,1,uniform -1 1,$feh_A$,
            A_logmstar,0.0,1,uniform -1 1,$logM_A$,
            A_eep,350,1,uniform 1 808,$EEP_A$,
        """))
    config.init(str(tmp_path), quiet=True)
    assert float(config.BASEMENT.settings['age_couple_tolerance']) == 0.5


def test_vsini_derived_prior_unified(tmp_path):
    """vsini with fit=1 + normal bounds under sv-param → derived_priors, not fitkeys."""
    settings = os.path.join(tmp_path, 'settings.csv')
    with open(settings, 'w') as f:
        f.write('#name,value\nmultiprocess,False\nmcmc_nwalkers,10\nmcmc_total_steps,100\n')
    params_csv = os.path.join(tmp_path, 'params.csv')
    with open(params_csv, 'w') as f:
        f.write(textwrap.dedent("""\
            #name,value,fit,bounds,label,unit
            A_svsinicoslambda,2.0,1,uniform -10 10,$sv_c$,
            A_svsinisinlambda,0.5,1,uniform -10 10,$sv_s$,
            A_vsini,4.5,1,normal 4.5 0.5,$vsini$,km/s
        """))
    config.init(str(tmp_path), quiet=True)
    assert 'A_vsini' not in config.BASEMENT.fitkeys
    assert 'A_vsini' in config.BASEMENT.derived_priors
    assert config.BASEMENT.derived_priors['A_vsini'] == ('normal', 4.5, 0.5)
