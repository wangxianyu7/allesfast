"""
MIST + SED interfaces.

These functions are designed as stable call points for integrating EXOZIPPy
components into allesfast.
"""

import os
from typing import Any, Dict, Optional

import numpy as np

from .models import StellarInputs
from .massradius_mist import massradius_mist, get_mistage
from .sed_utils import mistmultised, read_sed_file


def _derive_logg(mstar: float, rstar: float) -> float:
    gravity_sun = 27420.011  # cgs
    g = gravity_sun * mstar / rstar**2
    return float(np.log10(g))


def _derive_lstar(teff: float, rstar: float) -> float:
    # L/Lsun = (R/Rsun)^2 * (Teff/Tsun)^4, Tsun ~ 5772 K
    return float((rstar**2) * (teff / 5772.0) ** 4)


def mist_chi2(star: 'StellarInputs | list[StellarInputs]',
              config: Optional[Dict[str, Any]] = None,
              params: Optional[dict] = None) -> float:
    """
    Compute MIST penalty chi2 for one or more stars.

    EEP (star.eep, 1-808, continuous) is the primary MIST parameter.
    Age is derived from the evolutionary track and is NOT a free parameter.

    Parameters
    ----------
    star : StellarInputs or list[StellarInputs]
        A single star or a list of stars (e.g. a binary system).
    config : dict, optional
        Configuration options (vvcrit, alpha, allowold, etc.).
    params : dict, optional
        If provided, the derived age for each star is stored as
        params[f'{label}_age'] where label is 'A', 'B', 'C', 'D'.
    """
    cfg = config or {}
    stars = [star] if isinstance(star, StellarInputs) else list(star)
    labels = ['A', 'B', 'C', 'D']
    total = 0.0
    for idx, s in enumerate(stars):
        if any(v is None for v in [s.eep, s.mstar, s.feh, s.teff, s.rstar]):
            return np.inf
        try:
            chi2 = massradius_mist(
                eep=float(s.eep),
                mstar=float(s.mstar),
                feh=float(s.feh),
                teff=float(s.teff),
                rstar=float(s.rstar),
                age=float(s.age) if s.age is not None else None,
                vvcrit=cfg.get('vvcrit', None),
                alpha=cfg.get('alpha', None),
                allowold=cfg.get('allowold', False),
            )
            if not np.isfinite(chi2):
                return np.inf
            total += chi2
            # store derived age so coupled_tolerance can act on it
            if params is not None and idx < len(labels):
                age = get_mistage(
                    float(s.eep), float(s.mstar), float(s.feh),
                    vvcrit=cfg.get('vvcrit', None),
                    alpha=cfg.get('alpha', None),
                )
                params[f'{labels[idx]}_age'] = age
        except Exception:
            return np.inf
    return total


def sed_chi2(
    star: 'StellarInputs | list[StellarInputs]',
    sed_file: str,
    sed_data=None,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute SED chi2 for one or more stars.

    Parameters
    ----------
    star : StellarInputs or list[StellarInputs]
        A single star or a list of stars (e.g. a binary system).
    sed_data : pre-loaded SED data dict from read_sed_file (e.g. stored in
               BASEMENT.sed_data at init time).  If None, the file is read here.
    """
    cfg = config or {}
    stars = [star] if isinstance(star, StellarInputs) else list(star)
    nstars = len(stars)

    required = ['teff', 'rstar', 'feh', 'av', 'distance', 'mstar']
    for s in stars:
        if any(getattr(s, k, None) is None for k in required):
            return np.inf
    if not sed_file or not os.path.exists(sed_file):
        return np.inf

    try:
        teff_arr     = np.array([float(s.teff)     for s in stars])
        logg_arr     = np.array([_derive_logg(float(s.mstar), float(s.rstar)) for s in stars])
        feh_arr      = np.array([float(s.feh)      for s in stars])
        av_arr       = np.array([float(s.av)       for s in stars])
        distance_arr = np.array([float(s.distance) for s in stars])
        lstar_arr    = np.array([_derive_lstar(float(s.teff), float(s.rstar)) for s in stars])

        if sed_data is None:
            sed_data = read_sed_file(os.path.abspath(sed_file), nstars=nstars)

        chi2, _, _, _ = mistmultised(
            teff_arr,
            logg_arr,
            feh_arr,
            av_arr,
            distance_arr,
            lstar_arr,
            cfg.get("errscale", 1.0),
            sed_file,
            sed_data=sed_data,
        )
        return float(chi2) if np.isfinite(chi2) else np.inf
    except Exception:
        return np.inf
