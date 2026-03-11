"""
Kepler equation solver and RV computation.

Drop-in replacement for radvel.kepler.rv_drive and radvel.orbit.timetrans_to_timeperi.
Uses continuous mean anomaly (no mod 2pi) to avoid numerical discontinuities
at the periastron passage for high-eccentricity orbits.

Core loops are JIT-compiled with numba for performance.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def _rv_drive_jit(t, per, tp, ecc, omega, K):
    """Compute Keplerian RV for each time point."""
    n = len(t)
    rv = np.empty(n)
    cos_omega = np.cos(omega)
    two_pi_over_per = 2.0 * np.pi / per
    sqrt_1pe = np.sqrt(1.0 + ecc)
    sqrt_1me = np.sqrt(1.0 - ecc)

    for i in prange(n):
        # Continuous mean anomaly — no mod 2pi
        Mi = two_pi_over_per * (t[i] - tp)

        # Kepler equation: Newton-Raphson
        Ei = Mi
        for _ in range(200):
            sinE = np.sin(Ei)
            dE = (Ei - ecc * sinE - Mi) / (1.0 - ecc * np.cos(Ei))
            Ei -= dE
            if abs(dE) < 1e-12:
                break

        # True anomaly via arctan2 (no branch cuts)
        fi = 2.0 * np.arctan2(
            sqrt_1pe * np.sin(Ei * 0.5),
            sqrt_1me * np.cos(Ei * 0.5),
        )
        rv[i] = K * (np.cos(fi + omega) + ecc * cos_omega)
    return rv


def timetrans_to_timeperi(tc, per, ecc, omega):
    """Convert time of transit (conjunction) to time of periastron passage.

    Parameters
    ----------
    tc : float
        Time of transit (inferior conjunction), same units as per.
    per : float
        Orbital period.
    ecc : float
        Eccentricity.
    omega : float
        Argument of periastron in radians.

    Returns
    -------
    tp : float
        Time of periastron passage.
    """
    if ecc == 0.0 and omega == 0.5 * np.pi:
        return tc

    f_tc = 0.5 * np.pi - omega          # true anomaly at transit
    E_tc = 2.0 * np.arctan2(
        np.sqrt(1.0 - ecc) * np.sin(f_tc / 2.0),
        np.sqrt(1.0 + ecc) * np.cos(f_tc / 2.0),
    )
    M_tc = E_tc - ecc * np.sin(E_tc)    # mean anomaly at transit
    return tc - per / (2.0 * np.pi) * M_tc


def rv_drive(t, params):
    """Compute Keplerian radial velocity.

    Drop-in replacement for radvel.kepler.rv_drive.

    Parameters
    ----------
    t : array_like
        Times (same units as period).
    params : array_like
        [period, tp, ecc, omega, K]
        - period : orbital period
        - tp : time of periastron passage
        - ecc : eccentricity
        - omega : argument of periastron (radians)
        - K : RV semi-amplitude (same units as returned RV)

    Returns
    -------
    rv : ndarray
        Radial velocity at each time.
    """
    per, tp, ecc, omega, K = float(params[0]), float(params[1]), float(params[2]), float(params[3]), float(params[4])
    t = np.atleast_1d(np.asarray(t, dtype=float)).ravel()

    if ecc == 0.0:
        M = 2.0 * np.pi * (t - tp) / per
        return K * np.cos(M + omega)

    return _rv_drive_jit(t, per, tp, ecc, omega, K)
