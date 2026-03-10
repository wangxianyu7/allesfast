from dataclasses import dataclass
from typing import Optional


@dataclass
class StellarInputs:
    """
    Common stellar inputs used by Torres and MIST+SED constraints.
    Units:
    - teff: K        (spectroscopic effective temperature)
    - logg: cgs
    - feh: dex       (current/spectroscopic metallicity)
    - initfeh: dex   (initial birth metallicity for MIST track; falls back to feh if None)
    - rstar: Rsun    (spectroscopic/transit radius)
    - teffsed: K     (SED-specific Teff; falls back to teff if None — EXOFASTv2 style)
    - rstarsed: Rsun (SED-specific Rstar; falls back to rstar if None — EXOFASTv2 style)

    - mstar: Msun
    - age: Gyr
    - av: mag
    - distance: pc
    """

    teff: Optional[float] = None
    logg: Optional[float] = None
    feh: Optional[float] = None
    initfeh: Optional[float] = None   # initial metallicity for MIST track (A_initfeh); falls back to feh
    rstar: Optional[float] = None
    teffsed: Optional[float] = None   # SED Teff (A_teffsed); falls back to teff if None
    rstarsed: Optional[float] = None  # SED Rstar (A_rstarsed); falls back to rstar if None

    mstar: Optional[float] = None
    eep: Optional[float] = None   # primary MIST parameter (1-808, continuous)
    age: Optional[float] = None   # derived from EEP; kept for legacy / non-MIST use
    av: Optional[float] = None
    distance: Optional[float] = None


@dataclass
class StellarOutputs:
    """Container for derived stellar quantities and chi2-like penalties."""

    chi2: float = 0.0
    teff_model: Optional[float] = None
    rstar_model: Optional[float] = None
    mstar_model: Optional[float] = None

