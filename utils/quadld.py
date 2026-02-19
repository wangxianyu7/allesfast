"""
quadld.py

Python port of EXOFASTv2's quadld.pro (Eastman et al.)

Interpolates the quadratic limb darkening tables of Claret & Bloemen (2011):
  http://adsabs.harvard.edu/abs/2011A%26A...529A..75C

The .sav tables are read from the quadld/ directory located alongside this
package (i.e. <repo_root>/quadld/).

Usage:
    from allesfast.utils.quadld import quadld

    u1, u2 = quadld(logg=4.5, teff=5800, feh=0.0, band='Kepler')
    u1, u2 = quadld(4.5, 5800, 0.0, 'Sloanr', model='PHOENIX', method='F', vt=2)

Parameters
----------
logg : float
    Log of stellar surface gravity [dex].
teff : float
    Stellar effective temperature [K].
feh : float
    Stellar metallicity [Fe/H].
band : str
    Photometric band. Allowed values:
      Johnson/Cousins : U, B, V, R, I, J, H, K
      Sloan           : Sloanu, Sloang, Sloanr, Sloani, Sloanz
                        (or u, g, r, i, z as short aliases)
      Space missions  : Kepler, TESS, CoRoT
      Spitzer         : Spit36, Spit45, Spit58, Spit80
      Stromgren       : Stromu, Stromb, Stromv, Stromy
                        (or u, b, v, y as short aliases)
model : str
    Atmosphere model: 'ATLAS' (default) or 'PHOENIX'.
method : str
    Integration method: 'L' (luminosity, default) or 'F' (flux).
vt : int
    Microturbulent velocity: 0, 1, 2 (default), 4, or 8 km/s.

Returns
-------
u1, u2 : float
    Quadratic limb darkening coefficients.
    Returns (nan, nan) if outside table bounds or table not found.
"""

import os
import numpy as np
from scipy.io import readsav
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

# ---------------------------------------------------------------------------
# Path to the quadld/ table directory (two levels up from this file)
# ---------------------------------------------------------------------------
_QUADLD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'quadld')

# ---------------------------------------------------------------------------
# Band name aliases (short → filename stem)
# ---------------------------------------------------------------------------
_BAND_ALIASES = {
    # Stromgren short names
    'u': 'Stromu', 'b': 'Stromb', 'v': 'Stromv', 'y': 'Stromy',
    # Sloan short names
    'g': 'Sloang', 'r': 'Sloanr', 'i': 'Sloani', 'z': 'Sloanz',
}

# Boundary limits (from IDL code)
_LOGG_MIN, _LOGG_MAX   = 0.0,    5.0
_TEFF_MIN, _TEFF_MAX   = 3500.0, 50000.0
_FEH_MIN,  _FEH_MAX    = -5.0,   1.0

# ---------------------------------------------------------------------------
# Module-level cache  (equivalent to IDL COMMON LD_BLOCK)
# Key: (model, method, vt, bandname)  →  (interp_u1, interp_u2)
# ---------------------------------------------------------------------------
_cache = {}


def _fill_nans_nearest(values, fehs, teffs, loggs):
    """
    Fill NaN entries in a (feh, teff, logg) array using nearest-neighbour
    interpolation in physical-unit space.  Returns a fully finite array.
    """
    finite = np.isfinite(values)
    if finite.all():
        return values

    # Build coordinate arrays for every grid point
    fi, ti, li = np.meshgrid(np.arange(len(fehs)),
                              np.arange(len(teffs)),
                              np.arange(len(loggs)),
                              indexing='ij')
    # Physical coordinates (normalise teff so all axes have similar scale)
    coords = np.column_stack([
        fehs[fi.ravel()],
        teffs[ti.ravel()] / 1000.0,   # km/s-like scale
        loggs[li.ravel()],
    ])

    finite_flat = finite.ravel()
    nn = NearestNDInterpolator(coords[finite_flat], values.ravel()[finite_flat])

    filled = values.copy()
    nan_mask = ~finite_flat
    filled.ravel()[nan_mask] = nn(coords[nan_mask])
    return filled


def _load_table(model, method, vt, bandname):
    """
    Load a quadld .sav file and return a pair of RegularGridInterpolators
    (u1, u2), with NaNs pre-filled so interpolation always succeeds.
    """
    filename = f'{model}.{method}.{vt}.{bandname}.sav'
    filepath = os.path.join(_QUADLD_DIR, filename)

    if not os.path.isfile(filepath):
        return None, None

    data = readsav(filepath)
    fehs  = np.asarray(data['fehs'],   dtype=float)
    teffs = np.asarray(data['teffs'],  dtype=float)
    loggs = np.asarray(data['loggs'],  dtype=float)
    # Array shape from scipy.io.readsav: (feh, teff, logg)
    a = np.asarray(data['quadlda'], dtype=float)   # u1
    b = np.asarray(data['quadldb'], dtype=float)   # u2

    # Fill NaN entries with nearest-neighbour so interpolation always works
    a = _fill_nans_nearest(a, fehs, teffs, loggs)
    b = _fill_nans_nearest(b, fehs, teffs, loggs)

    # Build RegularGridInterpolators (linear, bounded extrapolation clipped)
    points = (fehs, teffs, loggs)
    interp_u1 = RegularGridInterpolator(points, a, method='linear',
                                         bounds_error=False, fill_value=np.nan)
    interp_u2 = RegularGridInterpolator(points, b, method='linear',
                                         bounds_error=False, fill_value=np.nan)
    return interp_u1, interp_u2


def quadld(logg, teff, feh, band,
           model='ATLAS', method='L', vt=2):
    """
    Interpolate quadratic limb darkening coefficients.

    Parameters
    ----------
    logg, teff, feh : float
        Stellar parameters.
    band : str
        Photometric band (see module docstring for allowed values).
    model : str
        'ATLAS' or 'PHOENIX'.
    method : str
        'L' (luminosity-weighted) or 'F' (flux-weighted).
    vt : int
        Microturbulent velocity in km/s: 0, 1, 2, 4, or 8.

    Returns
    -------
    u1, u2 : float
        Quadratic limb darkening coefficients, or (nan, nan) on failure.
    """
    nan = (np.nan, np.nan)

    # Boundary check
    if (feh  < _FEH_MIN  or feh  > _FEH_MAX  or
        teff < _TEFF_MIN or teff > _TEFF_MAX  or
        logg < _LOGG_MIN or logg > _LOGG_MAX):
        return nan

    # Resolve band alias
    bandname = _BAND_ALIASES.get(band, band)

    # Cache key
    key = (model.upper(), method.upper(), int(vt), bandname)

    if key not in _cache:
        _cache[key] = _load_table(*key)

    interp_u1, interp_u2 = _cache[key]

    if interp_u1 is None:
        return nan

    pt = np.array([[feh, teff, logg]])
    u1 = float(interp_u1(pt)[0])
    u2 = float(interp_u2(pt)[0])

    return u1, u2


def quadld_array(logg, teff, feh, band,
                 model='ATLAS', method='L', vt=2):
    """
    Vectorised version: logg/teff/feh may be arrays of the same shape.

    Returns
    -------
    u1, u2 : ndarray
    """
    logg = np.asarray(logg, dtype=float)
    teff = np.asarray(teff, dtype=float)
    feh  = np.asarray(feh,  dtype=float)

    bandname = _BAND_ALIASES.get(band, band)
    key = (model.upper(), method.upper(), int(vt), bandname)

    if key not in _cache:
        _cache[key] = _load_table(*key)

    interp_u1, interp_u2 = _cache[key]

    if interp_u1 is None:
        return np.full_like(logg, np.nan), np.full_like(logg, np.nan)

    # Out-of-bounds → NaN
    in_bounds = ((feh  >= _FEH_MIN)  & (feh  <= _FEH_MAX)  &
                 (teff >= _TEFF_MIN) & (teff <= _TEFF_MAX)  &
                 (logg >= _LOGG_MIN) & (logg <= _LOGG_MAX))

    pts = np.column_stack([feh.ravel(), teff.ravel(), logg.ravel()])
    u1 = interp_u1(pts).reshape(logg.shape)
    u2 = interp_u2(pts).reshape(logg.shape)
    u1[~in_bounds] = np.nan
    u2[~in_bounds] = np.nan

    return u1, u2


def list_bands(model='ATLAS', method='L', vt=2):
    """Return list of available band names for a given model/method/vt."""
    prefix = f'{model.upper()}.{method.upper()}.{vt}.'
    bands = []
    for fname in sorted(os.listdir(_QUADLD_DIR)):
        if fname.startswith(prefix) and fname.endswith('.sav'):
            stem = fname[len(prefix):-4]
            # exclude linear/nonlin variants
            if '.' not in stem:
                bands.append(stem)
    return bands


# ---------------------------------------------------------------------------
# ldtk backend (optional)
# ---------------------------------------------------------------------------

def get_ldc_ldtk(teff, logg, feh, band='TESS', law='quadratic',
                 teff_err=50.0, logg_err=0.1, feh_err=0.1):
    """
    Compute limb darkening coefficients using ldtk (Parviainen & Aigrain 2015).
    Requires: pip install ldtk

    ldtk queries PHOENIX stellar atmosphere models and marginalises over
    the uncertainties in teff/logg/feh, returning coefficients *and* their
    uncertainties — useful for setting Gaussian priors in an MCMC fit.

    Parameters
    ----------
    teff, logg, feh : float
        Stellar parameters.
    band : str
        SVO filter identifier, e.g. 'TESS', 'Kepler',
        'SLOAN/SDSS.r', '2MASS/2MASS.J'.
        See http://svo2.cab.inta-csic.es/theory/fps/ for all filter IDs.
    law : str
        'quadratic' (default), 'linear', or 'nonlinear'.
    teff_err, logg_err, feh_err : float
        1-sigma uncertainties on the stellar parameters.

    Returns
    -------
    coeffs : ndarray, shape (n_coeffs,)
        Limb darkening coefficients.
        quadratic → [u1, u2]
        linear    → [u1]
        nonlinear → [c1, c2, c3, c4]
    coeffs_err : ndarray, shape (n_coeffs,)
        1-sigma uncertainties on the coefficients.

    Examples
    --------
    >>> u, u_err = get_ldc_ldtk(5777, 4.44, 0.0, band='TESS')
    >>> u1, u2 = u;  su1, su2 = u_err
    """
    try:
        from ldtk import LDPSetCreator, SVOFilter
    except ImportError:
        raise ImportError(
            "ldtk is not installed. Install it with:  pip install ldtk"
        )

    filters = [SVOFilter(band)]
    sc = LDPSetCreator(
        teff=(teff, teff_err),
        logg=(logg, logg_err),
        z=(feh, feh_err),
        filters=filters,
    )
    ps = sc.create_profiles()

    law = law.lower()
    if law == 'quadratic':
        coeffs, coeffs_err = ps.coeffs_qd()
    elif law == 'linear':
        coeffs, coeffs_err = ps.coeffs_lin()
    elif law == 'nonlinear':
        coeffs, coeffs_err = ps.coeffs_nl()
    else:
        raise ValueError(f"Unsupported LD law '{law}'. Choose: quadratic, linear, nonlinear.")

    return np.array(coeffs[0]), np.array(coeffs_err[0])


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def get_ldc(teff, logg, feh, band,
            method='claret',
            law='quadratic',
            model='ATLAS', claret_method='L', vt=2,
            teff_err=50.0, logg_err=0.1, feh_err=0.1):
    """
    Unified limb darkening coefficient interface.

    Parameters
    ----------
    teff, logg, feh : float
        Stellar parameters.
    band : str
        Photometric band.
    method : str
        'claret' — interpolate Claret & Bloemen (2011) tables via quadld()
                   (returns coefficients only, no uncertainties).
        'ldtk'   — compute via ldtk / PHOENIX models (requires pip install ldtk)
                   (returns coefficients + uncertainties).
    law : str
        For ldtk: 'quadratic' (default), 'linear', 'nonlinear'.
        For claret: always quadratic (u1, u2).
    model, claret_method, vt : str/int
        Passed to quadld() when method='claret'.
    teff_err, logg_err, feh_err : float
        Passed to get_ldc_ldtk() when method='ldtk'.

    Returns
    -------
    method='claret' → (u1, u2)
    method='ldtk'   → (coeffs_array, coeffs_err_array)

    Examples
    --------
    >>> u1, u2 = get_ldc(5778, 4.44, 0.0, 'Kepler', method='claret')
    >>> u, ue  = get_ldc(5778, 4.44, 0.0, 'TESS',   method='ldtk')
    """
    method = method.lower()
    if method == 'claret':
        return quadld(logg, teff, feh, band,
                      model=model, method=claret_method, vt=vt)
    elif method == 'ldtk':
        return get_ldc_ldtk(teff, logg, feh, band=band, law=law,
                             teff_err=teff_err, logg_err=logg_err,
                             feh_err=feh_err)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'claret' or 'ldtk'.")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('Available bands (ATLAS L vt=2):')
    print(list_bands())
    print()

    test_cases = [
        (4.5, 5778, 0.0,  'Kepler'),
        (4.5, 5778, 0.0,  'TESS'),
        (4.5, 5778, 0.0,  'Sloanr'),
        (4.5, 5778, 0.0,  'V'),
        (4.5, 3800, -0.5, 'Kepler'),
    ]
    print(f'{"logg":>6} {"teff":>7} {"feh":>6} {"band":>8}  {"u1":>8} {"u2":>8}')
    print('-' * 52)
    for logg, teff, feh, band in test_cases:
        u1, u2 = quadld(logg, teff, feh, band)
        print(f'{logg:6.1f} {teff:7.0f} {feh:6.2f} {band:>8}  {u1:8.5f} {u2:8.5f}')

    print()
    print('Via unified get_ldc (claret):')
    u1, u2 = get_ldc(5778, 4.44, 0.0, 'Kepler', method='claret')
    print(f'  Kepler  u1={u1:.5f}  u2={u2:.5f}')

    print()
    print('ldtk backend (skipped if not installed):')
    try:
        u, ue = get_ldc(5778, 4.44, 0.0, 'TESS', method='ldtk')
        print(f'  TESS  u={u}  u_err={ue}')
    except ImportError as e:
        print(f'  {e}')
