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
              config: Optional[Dict[str, Any]] = None):
    """
    Compute MIST penalty chi2 for one or more stars.

    EEP (star.eep, 1-808, continuous) is the primary MIST parameter.
    Age is derived from the evolutionary track in update_params() and is
    NOT a free parameter.  Age priors and multi-star coevality are handled
    in calculate_external_priors().

    Parameters
    ----------
    star : StellarInputs or list[StellarInputs]
        A single star or a list of stars (e.g. a binary system).
    config : dict, optional
        Configuration options (vvcrit, alpha, allowold, etc.).

    Returns
    -------
    chi2 : float
        Total MIST chi2 penalty across all stars.
    ln_ageweight : float
        Sum of ln(ageweight) across all stars, used to correct the
        uniform EEP prior to a uniform Age prior (EXOFASTv2 convention).
    """
    cfg = config or {}
    stars = [star] if isinstance(star, StellarInputs) else list(star)
    labels = ['A', 'B', 'C', 'D']
    total = 0.0
    total_ln_ageweight = 0.0
    for idx, s in enumerate(stars):
        if any(v is None for v in [s.eep, s.mstar, s.feh, s.teff, s.rstar]):
            return np.inf, 0.0
        # Use initfeh (initial birth metallicity) as MIST track input if available;
        # otherwise fall back to feh (spectroscopic). EXOFASTv2 style: initfeh drives
        # track interpolation, feh is the observed surface value compared to mistfeh.
        _initfeh = float(s.initfeh) if s.initfeh is not None else float(s.feh)
        try:
            chi2, ageweight = massradius_mist(
                eep=float(s.eep),
                mstar=float(s.mstar),
                feh=_initfeh,
                teff=float(s.teff),
                rstar=float(s.rstar),
                obsfeh=float(s.feh),
                vvcrit=cfg.get('vvcrit', None),
                alpha=cfg.get('alpha', None),
                allowold=cfg.get('allowold', False),
            )
            if not np.isfinite(chi2):
                return np.inf, 0.0
            total += chi2
            if ageweight > 0:
                total_ln_ageweight += np.log(ageweight)
        except Exception:
            return np.inf, 0.0
    return total, total_ln_ageweight


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
        # EXOFASTv2 style: use teffsed/rstarsed for SED if available,
        # else fall back to teff/rstar.  logg and lstar derived from rstarsed.
        # NOTE: fehsed is permanently disabled; SED always uses feh.
        teff_arr     = np.array([float(s.teffsed  if s.teffsed  is not None else s.teff)  for s in stars])
        rstar_sed    = np.array([float(s.rstarsed if s.rstarsed is not None else s.rstar) for s in stars])
        logg_arr     = np.array([_derive_logg(float(s.mstar), rstar_sed[i]) for i, s in enumerate(stars)])
        feh_arr      = np.array([float(s.feh) for s in stars])
        av_arr       = np.array([float(s.av)       for s in stars])
        distance_arr = np.array([float(s.distance) for s in stars])
        lstar_arr    = np.array([_derive_lstar(teff_arr[i], rstar_sed[i]) for i in range(len(stars))])

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
