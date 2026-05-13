"""Orbit geometry helpers — ports of EXOFASTv2 exofast_getphase.pro
and exofast_getb2.pro.

We only implement the cases we need for DT:
- primary transit phase
- sky-projected (x, y, z) of a single companion in stellar radii
"""
import numpy as np
from ..rm_models.Hirano2011 import _kepler_eq


def primary_transit_phase(e, omega):
    """Phase of primary transit relative to periastron (matches
    exofast_getphase with /primary).

    Parameters
    ----------
    e : float       eccentricity, 0 ≤ e < 1
    omega : float   argument of periastron of the star (radians)

    Returns
    -------
    phase : float in [0, 1)
    """
    trueanom = np.pi / 2.0 - omega
    eccen_anom = 2.0 * np.arctan(np.sqrt((1.0 - e) / (1.0 + e))
                                  * np.tan(trueanom / 2.0))
    M = eccen_anom - e * np.sin(eccen_anom)
    phase = M / (2.0 * np.pi)
    if phase < 0.0:
        phase += 1.0
    return phase


def sky_positions(bjd, inc, ar, tperiastron, period, e, omega):
    """Compute (x, y, z, b) sky-projected positions for a single
    companion at observation times ``bjd``.

    Units: stellar radii.  Convention (EXOFASTv2 exofast_getb2):
        +x = right (on sky)
        +y = up
        +z = out of page (toward observer; positive z = primary transit)

    The mass ratio q is assumed infinite (motion of planet wrt star).

    Parameters
    ----------
    bjd : array_like
    inc : float       inclination (radians)
    ar : float        semi-major axis in stellar radii
    tperiastron : float   time of periastron (same units as bjd)
    period : float
    e : float
    omega : float     argument of periastron of star (radians)

    Returns
    -------
    x, y, z, b : ndarray   each shape (len(bjd),); b = sqrt(x²+y²)
    """
    bjd = np.atleast_1d(np.asarray(bjd, dtype=np.float64))

    # Mean anomaly
    M = (2.0 * np.pi * (1.0 + (bjd - tperiastron) / period)) % (2.0 * np.pi)

    if e != 0.0:
        E = _kepler_eq(M, e)   # IDL's exofast_keplereq equivalent
        nu = 2.0 * np.arctan(np.sqrt((1.0 + e) / (1.0 - e))
                               * np.tan(0.5 * E))
    else:
        nu = M

    # Note IDL sign convention: r2 negative for the planet (omega_*).
    # We replicate: r2 = -a*(1-e²)/(1+e cos nu)
    r2 = -ar * (1.0 - e ** 2) / (1.0 + e * np.cos(nu))
    x = r2 * np.cos(nu + omega)
    tmp = r2 * np.sin(nu + omega)
    y = tmp * np.cos(inc)
    z = tmp * np.sin(inc)
    b = np.sqrt(x ** 2 + y ** 2)
    return x, y, z, b
