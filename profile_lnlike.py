#!/usr/bin/env python3
"""Profile the likelihood computation for K2-140."""
import sys, time
sys.path.insert(0, '/Users/wangxianyu/Program/Github')
import numpy as np

# Initialize
from allesfast import config
from allesfast import computer

path = 'examples/K2-140'
config.init(path, quiet=True)
computer.setup_transit_models()

print('BASEMENT loaded.')
print('N phot K2:', len(config.BASEMENT.data['K2']['time']))
print('N RV CORALIE:', len(config.BASEMENT.data['CORALIE']['rv']))
print('N RV Keck:', len(config.BASEMENT.data['Keck']['rv']))
print('N fitkeys:', len(config.BASEMENT.fitkeys))
print()

# Build a nominal theta from the params
theta = np.array([config.BASEMENT.params[k] for k in config.BASEMENT.fitkeys], dtype=float)

# Warmup
from allesfast.computer import (
    calculate_lnlike_total, update_params, calculate_external_priors,
    calculate_model, calculate_yerr_w, calculate_baseline, calculate_stellar_var
)
params = update_params(theta)
_ = calculate_lnlike_total(params)

# ---- fine-grained timing ----
N = 100

# 1. update_params
t0 = time.perf_counter()
for _ in range(N):
    params = update_params(theta)
t_update = (time.perf_counter() - t0) / N * 1e3

# 2. external priors  (MIST + SED inside)
params = update_params(theta)
t0 = time.perf_counter()
for _ in range(N):
    lnp = calculate_external_priors(params)
t_prior = (time.perf_counter() - t0) / N * 1e3

# 3a. transit model (K2 photometry)
t0 = time.perf_counter()
for _ in range(N):
    model_flux = calculate_model(params, 'K2', 'flux')
t_transit = (time.perf_counter() - t0) / N * 1e3

# 3b. RV model CORALIE
t0 = time.perf_counter()
for _ in range(N):
    model_rv = calculate_model(params, 'CORALIE', 'rv')
t_rv_cor = (time.perf_counter() - t0) / N * 1e3

# 3c. RV model Keck
t0 = time.perf_counter()
for _ in range(N):
    model_rv = calculate_model(params, 'Keck', 'rv')
t_rv_keck = (time.perf_counter() - t0) / N * 1e3

# 4. baselines
t0 = time.perf_counter()
for _ in range(N):
    yerr_w = calculate_yerr_w(params, 'K2', 'flux')
    baseline = calculate_baseline(params, 'K2', 'flux',
                                   model=model_flux, yerr_w=yerr_w)
t_baseline = (time.perf_counter() - t0) / N * 1e3

# 5. full likelihood
t0 = time.perf_counter()
for _ in range(N):
    params = update_params(theta)
    lnl = calculate_lnlike_total(params)
t_total = (time.perf_counter() - t0) / N * 1e3

print(f"{'Component':<35} {'ms/call':>10}")
print("-" * 47)
print(f"{'update_params':<35} {t_update:>10.3f}")
print(f"{'external_priors (MIST+SED)':<35} {t_prior:>10.3f}")
print(f"{'transit model K2 (flux)':<35} {t_transit:>10.3f}")
print(f"{'rv model CORALIE':<35} {t_rv_cor:>10.3f}")
print(f"{'rv model Keck':<35} {t_rv_keck:>10.3f}")
print(f"{'yerr_w + baseline K2':<35} {t_baseline:>10.3f}")
print("-" * 47)
print(f"{'TOTAL lnlike_total (incl all)':<35} {t_total:>10.3f}")
print()
print(f"lnlike value: {lnl:.3f}")

# Also break down external_priors into MIST vs SED
from allesfast.star import mist_chi2, sed_chi2, StellarInputs
from astropy import units as u
import os

_distance = params.get('host_distance', None)
if _distance is None:
    _parallax = params.get('host_parallax', None)
    if _parallax is not None and _parallax > 0:
        _distance = 1000.0 / _parallax
star = StellarInputs(
    teff=params.get('host_teff', None),
    logg=params.get('host_logg', None),
    feh=params.get('host_feh', None),
    rstar=params.get('host_rstar', None),
    mstar=params.get('host_mstar', None),
    age=params.get('host_age', None),
    av=params.get('host_av', None),
    distance=_distance,
)

sed_file = config.BASEMENT.settings.get('sed_file', None)
if sed_file and not os.path.isabs(sed_file):
    sed_file = os.path.join(config.BASEMENT.datadir, sed_file)

t0 = time.perf_counter()
for _ in range(N):
    mist_chi2(star, config={'mist_root': None, 'vvcrit': 0.0, 'alpha': 0.0, 'allowold': False})
t_mist = (time.perf_counter() - t0) / N * 1e3

t0 = time.perf_counter()
for _ in range(N):
    sed_chi2(star, sed_file=sed_file, config={'errscale': 1.0})
t_sed = (time.perf_counter() - t0) / N * 1e3

print(f"\nBreakdown of external_priors:")
print(f"{'  mist_chi2':<35} {t_mist:>10.3f} ms")
print(f"{'  sed_chi2':<35} {t_sed:>10.3f} ms")
print(f"  sum                               {t_mist+t_sed:>10.3f} ms  (vs {t_prior:.3f} ms measured)")
