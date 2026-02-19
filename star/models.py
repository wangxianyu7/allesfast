from dataclasses import dataclass
from typing import Optional


@dataclass
class StellarInputs:
    """
    Common stellar inputs used by Torres and MIST+SED constraints.
    Units:
    - teff: K
    - logg: cgs
    - feh: dex
    - rstar: Rsun
    - mstar: Msun
    - age: Gyr
    - av: mag
    - distance: pc
    """

    teff: Optional[float] = None
    logg: Optional[float] = None
    feh: Optional[float] = None
    rstar: Optional[float] = None
    mstar: Optional[float] = None
    age: Optional[float] = None
    av: Optional[float] = None
    distance: Optional[float] = None


@dataclass
class StellarOutputs:
    """Container for derived stellar quantities and chi2-like penalties."""

    chi2: float = 0.0
    teff_model: Optional[float] = None
    rstar_model: Optional[float] = None
    mstar_model: Optional[float] = None

