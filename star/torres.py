"""
Torres-relation interface.

This module is intentionally lightweight: it defines the public API that can be
wired into `computer.calculate_external_priors()` without forcing a specific
implementation yet.
"""

from .models import StellarInputs, StellarOutputs


def torres_constraints(star: StellarInputs) -> StellarOutputs:
    """
    Evaluate Torres-based stellar constraints.

    Returns
    -------
    StellarOutputs
        `chi2` should be a penalty term (smaller is better).
    """
    raise NotImplementedError("Torres relation backend is not implemented yet.")

