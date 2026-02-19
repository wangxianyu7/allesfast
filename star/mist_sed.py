"""
MIST + SED interfaces.

These functions are designed as stable call points for integrating EXOZIPPy
components into allesfast.
"""

import os
from typing import Any, Dict, Optional

import numpy as np

from .models import StellarInputs
from .massradius_mist import massradius_mist
from .sed_utils import mistmultised, read_sed_file

# Cache sed_data by absolute sed_file path.
# read_sed_file reads the file, allocates large numpy arrays (e.g. 24000-pt
# filter curves), and is otherwise identical on every call for the same file.
# Caching it avoids repeated I/O + allocation on every likelihood evaluation.
_sed_data_cache: Dict[str, Any] = {}


def _derive_logg(mstar: float, rstar: float) -> float:
    gravity_sun = 27420.011  # cgs
    g = gravity_sun * mstar / rstar**2
    return float(np.log10(g))


def _derive_lstar(teff: float, rstar: float) -> float:
    # L/Lsun = (R/Rsun)^2 * (Teff/Tsun)^4, Tsun ~ 5772 K
    return float((rstar**2) * (teff / 5772.0) ** 4)


def mist_chi2(star: StellarInputs, config: Optional[Dict[str, Any]] = None) -> float:
    """
    Compute MIST penalty chi2 for one star.
    """
    cfg = config or {}
    _ = cfg  # kept for interface compatibility
    if any(v is None for v in [star.mstar, star.feh, star.age, star.teff, star.rstar]):
        return np.inf
    try:
        return massradius_mist(
            mstar=float(star.mstar),
            feh=float(star.feh),
            age=float(star.age),
            teff=float(star.teff),
            rstar=float(star.rstar),
        )
    except Exception:
        return np.inf


def sed_chi2(
    star: StellarInputs,
    sed_file: str,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute SED chi2 for one star.
    """
    cfg = config or {}
    if any(v is None for v in [star.teff, star.rstar, star.feh, star.av, star.distance, star.mstar]):
        return np.inf
    if not sed_file or not os.path.exists(sed_file):
        return np.inf

    try:
        logg = _derive_logg(float(star.mstar), float(star.rstar))
        lstar = _derive_lstar(float(star.teff), float(star.rstar))

        # Load SED data once per file path; reuse on subsequent calls.
        abs_path = os.path.abspath(sed_file)
        if abs_path not in _sed_data_cache:
            _sed_data_cache[abs_path] = read_sed_file(abs_path, nstars=1)
        sed_data = _sed_data_cache[abs_path]

        chi2, _, _, _ = mistmultised(
            np.array([float(star.teff)]),
            np.array([logg]),
            np.array([float(star.feh)]),
            np.array([float(star.av)]),
            np.array([float(star.distance)]),
            np.array([lstar]),
            float(cfg.get("errscale", 1.0)),
            sed_file,
            sed_data=sed_data,
        )
        return float(chi2) if np.isfinite(chi2) else np.inf
    except Exception:
        return np.inf
