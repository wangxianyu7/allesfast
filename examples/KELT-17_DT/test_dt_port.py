"""Validate allesfast.dt.dopptom_chi2 against EXOFASTv2 IDL.

Run this Python script and the companion IDL script (test_dt_idl.pro) at
the same parameters; their χ² should match to better than 1%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from allesfast.dt.io import read_dt_fits
from allesfast.dt.core import dopptom_chi2


HERE = Path(__file__).parent
dat = read_dt_fits(str(HERE / 'n20160226.KELT-17b.TRES.44000.fits'))

# Best-fit parameters from EXOFASTv2 kelt17.priors2
tc       = 2457287.7456103642
period   = 10 ** 0.4885753488
p        = 0.0926333230
cosi     = 0.0817318222
secosw   = 0.2433198217
sesinw   = -0.1518641162
e        = secosw ** 2 + sesinw ** 2
omega    = np.arctan2(sesinw, secosw)
lam      = -2.0164720094

# a/Rs from Kepler's 3rd law (units: G[Rsun³/(Msun·day²)] = 2942.71377)
G_unit = 2942.71377
mstar  = 10 ** 0.2604162879
rstar  = 1.6458230572
ar     = (G_unit * mstar * period ** 2 / (4 * np.pi ** 2)) ** (1.0 / 3.0) / rstar

# V band LD from quadld(logg, teff, feh, 'V')
u1, u2 = 0.2788, 0.3468

vsini    = 44.2
vline    = 5.49
errscale = 3.39

chi2, model = dopptom_chi2(
    dat, tc, period, e, omega, cosi, p, ar, lam,
    u1, u2, vsini, vline, errscale, return_model=True,
)
print(f'allesfast DT chi^2 = {chi2:.4f}')
print(f'  (IDL EXOFASTv2 chi^2 / IndepVels = 511.10)')
print(f'  model min/max = {model.min():.6e} / {model.max():.6e}')
