"""
Empirical estimates of microturbulence (vmic / xi) and macroturbulence (vmac / zeta)
from stellar parameters (Teff, logg, [Fe/H]).

Functions are adapted from iSpec (Blanco-Cuaresma et al. 2014):
  https://github.com/marblestation/iSpec

Relations:
  vmic — Bruntt et al. (2010), with GES fallback
  vmac — Doyle et al. (2014), with GES fallback
"""

import numpy as np


# ---------------------------------------------------------------------------
# Macroturbulence (vmac / zeta)
# ---------------------------------------------------------------------------

def _vmac_doyle2014(teff, logg, feh):
    """Doyle et al. (2014); valid for Teff 5200–6400 K, logg 4.0–4.6."""
    t0, g0 = 5777, 4.44
    if teff < 5200 or teff > 6400 or logg < 4.0 or logg > 4.6:
        return np.nan
    return 3.21 + 2.33e-3*(teff - t0) + 2e-6*(teff - t0)**2 - 2*(logg - g0)


def _vmac_ges(teff, logg, feh):
    """Bergemann (Gaia ESO Survey); broader validity."""
    t0, g0 = 5500, 4.0
    if logg >= 3.5:
        if teff >= 5000:
            return 3*(1.15 + 7e-4*(teff-t0) + 1.2e-6*(teff-t0)**2
                      - 0.13*(logg-g0) + 0.13*(logg-g0)**2
                      - 0.37*feh - 0.07*feh**2)
        else:
            return 3*(1.15 + 2e-4*(teff-t0) + 3.95e-7*(teff-t0)**2
                      - 0.13*(logg-g0) + 0.13*(logg-g0)**2)
    else:
        return 3*(1.15 + 2.2e-5*(teff-t0) - 0.5e-7*(teff-t0)**2
                  - 0.1*(logg-g0) + 0.04*(logg-g0)**2
                  - 0.37*feh - 0.07*feh**2)


def estimate_vmac(teff, logg, feh):
    """
    Macroturbulence velocity (km/s).
    Tries Doyle2014 first; falls back to GES if out of range.
    """
    v = _vmac_doyle2014(teff, logg, feh)
    if np.isnan(v):
        v = _vmac_ges(teff, logg, feh)
    return max(float(v), 0.0)


# ---------------------------------------------------------------------------
# Microturbulence (vmic / xi)
# ---------------------------------------------------------------------------

def _vmic_bruntt2010(teff, logg, feh):
    """Bruntt et al. (2010); valid for logg > 4, 5000 < Teff < 6500 K."""
    if logg < 4 or teff < 5000 or teff > 6500:
        return np.nan
    t0 = 5700
    return 1.01 + 4.561e-4*(teff - t0) + 2.75e-7*(teff - t0)**2


def _vmic_ges(teff, logg, feh):
    """Bergemann (Gaia ESO Survey)."""
    t0, g0 = 5500, 4.0
    if logg >= 3.5:
        teff_eff = teff if teff >= 5000 else 5000
        return (1.05 + 2.51e-4*(teff_eff-t0) + 1.5e-7*(teff_eff-t0)**2
                - 0.14*(logg-g0) - 0.005*(logg-g0)**2
                + 0.05*feh + 0.01*feh**2)
    else:
        return (1.25 + 4.01e-4*(teff-t0) + 3.1e-7*(teff-t0)**2
                - 0.14*(logg-g0) - 0.005*(logg-g0)**2
                + 0.05*feh + 0.01*feh**2)


def estimate_vmic(teff, logg, feh):
    """
    Microturbulence velocity (km/s).
    Tries Bruntt2010 first; falls back to GES if out of range.
    """
    v = _vmic_bruntt2010(teff, logg, feh)
    if np.isnan(v):
        v = _vmic_ges(teff, logg, feh)
    return max(float(v), 0.0)
