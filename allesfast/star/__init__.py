"""
Stellar-model interfaces for allesfast.

This package centralizes stellar constraints used as external priors, such as:
- empirical Torres relations
- evolutionary-model constraints (e.g. MIST)
- SED likelihoods
"""

from .models import StellarInputs, StellarOutputs
from .torres import torres_constraints
from .mist_sed import mist_chi2, sed_chi2
from .diagnostics import make_sed_plot, make_mist_plot
from .stellar_params import get_stellar_row, has_stellar_info, sample_stellar, summary_dict, plot_params_star

__all__ = [
    "StellarInputs",
    "StellarOutputs",
    "torres_constraints",
    "mist_chi2",
    "sed_chi2",
    "make_sed_plot",
    "make_mist_plot",
    "get_stellar_row",
    "has_stellar_info",
    "sample_stellar",
    "summary_dict",
    "plot_params_star",
]
