"""
Diagnostic plots for stellar constraints (SED + MIST).
"""

from __future__ import annotations

import os
import glob as _glob
import pathlib
import functools

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import readsav
from scipy.ndimage import uniform_filter1d

from .models import StellarInputs
from .massradius_mist import massradius_mist, get_mist_point, _interpolate_track_for_plot
from .sed_utils import mistmultised, read_sed_file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODULE_PATH = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
_SED_PATH    = _MODULE_PATH / 'sed'

# NextGen model grid path — set via env var ALLESFAST_NEXTGEN_PATH
_NEXTGEN_PATH_STR = os.environ.get('ALLESFAST_NEXTGEN_PATH', None)
if _NEXTGEN_PATH_STR is None:
    import warnings
    warnings.warn(
        "Environment variable ALLESFAST_NEXTGEN_PATH is not set. "
        "SED diagnostic plots requiring NextGen model atmospheres will not work. "
        "Please set it to the path of the nextgenfin directory, e.g.:\n"
        "  export ALLESFAST_NEXTGEN_PATH=/path/to/EXOFASTv2/sed/nextgenfin",
        stacklevel=2,
    )
    _NEXTGEN_PATH = None
else:
    _NEXTGEN_PATH = pathlib.Path(_NEXTGEN_PATH_STR)

# Wavelength grid matching NextGen models (0.1–24 μm, 24000 points)
_WAVELENGTH = np.arange(24000) / 1000.0 + 0.1

# NextGen grid axes
_ALLOWED_TEFF = np.array([
    8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    68, 69, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94,
    96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120,
    125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185,
    190, 195, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
    310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430,
    440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560,
    570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700,
]) * 100.0  # → Kelvin

_ALLOWED_LOGG = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
_ALLOWED_FEH  = np.array([-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.3, 0.5])
_ALPHA_STR    = ['+0.0', '+0.2', '-0.2', '+0.4', '+0.6']

_TEFF_STR = [str(int(t // 100)) for t in _ALLOWED_TEFF]
_LOGG_STR = ['-0.5', '+0.0', '+0.5', '+1.0', '+1.5', '+2.0', '+2.5',
             '+3.0', '+3.5', '+4.0', '+4.5', '+5.0', '+5.5', '+6.0']
_FEH_STR  = ['-4.0', '-3.5', '-3.0', '-2.5', '-2.0', '-1.5', '-1.0',
             '-0.5', '+0.0', '+0.3', '+0.5']

# Constants (cgs)
_RSUN_CM = 6.957e10
_PC_CM   = 3.085677581e18


# ---------------------------------------------------------------------------
# NextGen atmosphere helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=128)
def _load_nextgen(filepath: str) -> np.ndarray:
    s = readsav(filepath, python_dict=True)
    return s['lamflam1']


def _interp_atmosphere(teff, logg, feh):
    """
    Trilinearly interpolate NextGen λFλ at given (Teff, logg, [Fe/H]).
    Returns ndarray shape (24000,) or None if out of range / files missing.
    """
    if _NEXTGEN_PATH is None:
        return None
    if not (_ALLOWED_TEFF[0] <= teff <= _ALLOWED_TEFF[-1]):
        return None
    if not (_ALLOWED_LOGG[0] <= logg <= _ALLOWED_LOGG[-1]):
        return None
    if not (_ALLOWED_FEH[0] <= feh <= _ALLOWED_FEH[-1]):
        return None

    ti = np.clip(np.searchsorted(_ALLOWED_TEFF, teff) - 1, 0, len(_ALLOWED_TEFF) - 2)
    li = np.clip(np.searchsorted(_ALLOWED_LOGG, logg) - 1, 0, len(_ALLOWED_LOGG) - 2)
    fi = np.clip(np.searchsorted(_ALLOWED_FEH,  feh)  - 1, 0, len(_ALLOWED_FEH)  - 2)

    lamflams = np.zeros((24000, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                found = False
                for alpha in _ALPHA_STR:
                    fname = (f"lte{_TEFF_STR[ti+i]}"
                             f"{_LOGG_STR[li+j]}"
                             f"{_FEH_STR[fi+k]}"
                             f"{alpha}.NextGen.spec.idl")
                    fpath = _NEXTGEN_PATH / fname
                    if fpath.exists():
                        lamflams[:, i, j, k] = _load_nextgen(str(fpath))
                        found = True
                        break
                if not found:
                    return None

    xt = (teff - _ALLOWED_TEFF[ti]) / (_ALLOWED_TEFF[ti+1] - _ALLOWED_TEFF[ti])
    xl = (logg - _ALLOWED_LOGG[li]) / (_ALLOWED_LOGG[li+1] - _ALLOWED_LOGG[li])
    xf = (feh  - _ALLOWED_FEH[fi])  / (_ALLOWED_FEH[fi+1]  - _ALLOWED_FEH[fi])

    c00 = lamflams[:, 0,0,0]*(1-xt) + lamflams[:, 1,0,0]*xt
    c01 = lamflams[:, 0,0,1]*(1-xt) + lamflams[:, 1,0,1]*xt
    c10 = lamflams[:, 0,1,0]*(1-xt) + lamflams[:, 1,1,0]*xt
    c11 = lamflams[:, 0,1,1]*(1-xt) + lamflams[:, 1,1,1]*xt
    c0  = c00*(1-xl) + c10*xl
    c1  = c01*(1-xl) + c11*xl
    return c0*(1-xf) + c1*xf


def _apply_extinction(lamflam, av):
    ext_file = _SED_PATH / 'extinction_law.ascii'
    if ext_file.exists():
        klam, kkap = np.loadtxt(ext_file, unpack=True)
        kapv = np.interp(0.55, klam, kkap)
        kapp = np.interp(_WAVELENGTH, klam, kkap)
        tau  = kapp / kapv / 1.086 * av
        return lamflam * np.exp(-tau)
    return lamflam


def _scale_atmosphere(lamflam, rstar, distance):
    """Scale surface λFλ to observed flux: × (R★/d)²."""
    scale = (_RSUN_CM * rstar) ** 2 / (_PC_CM * distance) ** 2
    return lamflam * scale


# ---------------------------------------------------------------------------
# Star helper
# ---------------------------------------------------------------------------

def _star_from_params(params):
    distance = params.get("A_distance", None)
    if distance is None:
        parallax = params.get("A_parallax", None)
        if parallax is not None and float(parallax) > 0:
            distance = 1000.0 / float(parallax)   # mas → pc
    # logmstar support
    mstar = params.get("A_mstar", None)
    if mstar is None:
        logmstar = params.get("A_logmstar", None)
        if logmstar is not None:
            mstar = 10.0 ** float(logmstar)
    return StellarInputs(
        teff=params.get("A_teff", None),
        logg=params.get("A_logg", None),
        feh=params.get("A_feh", None),
        rstar=params.get("A_rstar", None),
        teffsed=params.get("A_teffsed", None),
        rstarsed=params.get("A_rstarsed", None),
        mstar=mstar,
        eep=params.get("A_eep", None),
        age=params.get("A_age", None),
        av=params.get("A_av", None),
        distance=distance,
    )


def _star_X_from_params(params, letter):
    """Build a StellarInputs for star *letter* (B, C, D, …).

    Falls back to star-A values for shared properties (parallax/distance,
    feh, av) when the companion has no dedicated entry.
    """
    distance = params.get(f"{letter}_distance", None)
    if distance is None:
        parallax = params.get(f"{letter}_parallax") or params.get("A_parallax", None)
        if parallax is not None and float(parallax) > 0:
            distance = 1000.0 / float(parallax)
    mstar = params.get(f"{letter}_mstar", None)
    if mstar is None:
        logmstar = params.get(f"{letter}_logmstar", None)
        if logmstar is not None:
            mstar = 10.0 ** float(logmstar)
    return StellarInputs(
        teff=params.get(f"{letter}_teff", None),
        logg=params.get(f"{letter}_logg", None),
        feh=params.get(f"{letter}_feh") or params.get("A_feh", None),
        rstar=params.get(f"{letter}_rstar", None),
        teffsed=params.get(f"{letter}_teffsed", None),
        rstarsed=params.get(f"{letter}_rstarsed", None),
        mstar=mstar,
        eep=params.get(f"{letter}_eep", None),
        age=params.get(f"{letter}_age", None),
        av=params.get(f"{letter}_av") or params.get("A_av", None),
        distance=distance,
    )


def _star_B_from_params(params):
    return _star_X_from_params(params, 'B')


# ---------------------------------------------------------------------------
# Multi-star array builder (shared by compute_sed_modeldata & make_sed_plot)
# ---------------------------------------------------------------------------
_COMPANION_LETTERS = 'BCDEFGHIJKLMNOPQRSTUVWXYZ'
_SED_REQUIRED_KEYS = ('teff', 'rstar', 'feh', 'av', 'distance', 'mstar')
_GRAVITY_SUN = 27420.011


def _sed_val(star, sed_attr, fallback_attr):
    """Return SED-specific value if present, else the standard one."""
    v = getattr(star, sed_attr, None)
    return float(v) if v is not None else float(getattr(star, fallback_attr))


def _collect_sed_stars(params):
    """Return (all_stars, nstars) where all_stars[0] is star A.

    Dynamically detects companions B, C, D, … by checking whether all
    required SED attributes are present.
    """
    star_A = _star_from_params(params)
    all_stars = [star_A]
    for ltr in _COMPANION_LETTERS:
        comp = _star_X_from_params(params, ltr)
        if all(getattr(comp, k) is not None for k in _SED_REQUIRED_KEYS):
            all_stars.append(comp)
        else:
            break  # stop at first missing companion
    return all_stars, len(all_stars)


def _build_sed_arrays(all_stars):
    """Build the (teff, logg, feh, av, dist, lstar) arrays for mistmultised."""
    teff_arr, logg_arr, feh_arr = [], [], []
    av_arr, dist_arr, lstar_arr = [], [], []
    for s in all_stars:
        t = _sed_val(s, 'teffsed', 'teff')
        r = _sed_val(s, 'rstarsed', 'rstar')
        teff_arr.append(t)
        logg_arr.append(np.log10(_GRAVITY_SUN * float(s.mstar) / r ** 2))
        feh_arr.append(float(s.feh))
        av_arr.append(float(s.av))
        dist_arr.append(float(s.distance))
        lstar_arr.append(r ** 2 * (t / 5772.0) ** 4)
    return (np.array(teff_arr), np.array(logg_arr), np.array(feh_arr),
            np.array(av_arr), np.array(dist_arr), np.array(lstar_arr))


# ---------------------------------------------------------------------------
# Public data / plot functions
# ---------------------------------------------------------------------------

def compute_sed_modeldata(params, datadir, errscale=None, sed_file=None):
    """Compute SED band fluxes and model fluxes without creating a figure.

    Returns a dict of arrays suitable for ``np.savez``, or None if the SED
    cannot be evaluated (missing stellar params or SED data file).

    Keys
    ----
    sedbands, weff_um, widtheff_um : band metadata
    obs_flux, obs_err              : observed band-integrated fluxes
    model_flux, residuals, chi2    : model fluxes and residuals (σ)
    wave_atm_um                    : wavelength axis (μm, 0.1–24)
    flux_atm_A                     : NextGen atmosphere for star A (if loadable)
    flux_atm_B, flux_atm_combined  : binary-star atmospheres (if available)
    nstars                         : number of stars
    star_A_teff, star_A_av         : scalar metadata for labelling
    star_<X>_teff                  : companion Teff (B, C, …)
    """
    all_stars, nstars = _collect_sed_stars(params)
    star_A = all_stars[0]
    if any(v is None for v in [star_A.teff, star_A.rstar, star_A.feh,
                                star_A.av, star_A.distance, star_A.mstar]):
        return None

    if sed_file is None:
        sed_file = params.get("sed_file", None)
    if sed_file is None:
        hits = _glob.glob(os.path.join(datadir, "*.sed"))
        sed_file = hits[0] if hits else os.path.join(datadir, "sed.dat")
    elif not os.path.isabs(sed_file):
        sed_file = os.path.join(datadir, sed_file)
    if not os.path.exists(sed_file):
        return None

    if errscale is None:
        _esc = params.get('A_sed_errscale', None)
        if _esc is not None:
            errscale = float(_esc)
        else:
            try:
                from .. import config as _cfg
                _esc = _cfg.BASEMENT.settings.get('sed_errscale', 1.0)
                errscale = float(np.asarray(_esc).flat[0])
            except Exception:
                errscale = 1.0

    teff_arr, logg_arr, feh_arr, av_arr, dist_arr, lstar_arr = _build_sed_arrays(all_stars)

    sed_data = read_sed_file(sed_file, nstars=nstars)
    chi2, blendmag, _, _ = mistmultised(
        teff_arr, logg_arr, feh_arr, av_arr, dist_arr, lstar_arr,
        float(errscale), sed_file, sed_data=sed_data,
    )

    zero_point = np.asarray(sed_data["zero_point"], dtype=float)
    mags       = np.asarray(sed_data["mag"],        dtype=float)
    errmag     = np.asarray(sed_data["errmag"],     dtype=float)
    weff       = np.asarray(sed_data["weff"],       dtype=float)
    widtheff   = np.asarray(sed_data["widtheff"],   dtype=float)
    sedbands   = np.asarray(sed_data["sedbands"])
    blendmag   = np.asarray(blendmag,               dtype=float)
    obs_flux   = zero_point * 10 ** (-0.4 * mags)
    obs_err    = obs_flux * np.log(10) / 2.5 * errmag
    model_flux = zero_point * 10 ** (-0.4 * blendmag)
    residuals  = np.where(obs_err > 0, (obs_flux - model_flux) / obs_err, 0.0)

    def _get_atm(star, logg_val):
        lf = _interp_atmosphere(float(star.teff), float(logg_val), float(star.feh))
        if lf is None:
            return None
        lf = _scale_atmosphere(lf, float(star.rstar), float(star.distance))
        lf = _apply_extinction(lf, float(star.av))
        return lf

    atmospheres = [_get_atm(s, lg) for s, lg in zip(all_stars, logg_arr)]

    result = dict(
        sedbands=sedbands,
        weff_um=weff,
        widtheff_um=widtheff,
        obs_flux=obs_flux,
        obs_err=obs_err,
        model_flux=model_flux,
        residuals=residuals,
        chi2=np.array([chi2]),
        nstars=np.array([nstars]),
        has_B=np.array([nstars >= 2]),  # backward compat
        star_A_teff=np.array([float(star_A.teff)]),
        star_A_av=np.array([float(star_A.av)]),
        wave_atm_um=_WAVELENGTH.copy(),
    )
    # Per-star Teff and atmospheres
    letters = 'A' + _COMPANION_LETTERS
    for i, s in enumerate(all_stars):
        ltr = letters[i]
        if i > 0:
            result[f'star_{ltr}_teff'] = np.array([float(s.teff)])
        if atmospheres[i] is not None:
            result[f'flux_atm_{ltr}'] = atmospheres[i]
    # Combined atmosphere
    valid_atm = [a for a in atmospheres if a is not None]
    if len(valid_atm) > 1:
        result['flux_atm_combined'] = sum(valid_atm)
    return result


def make_sed_plot(params, datadir, outdir, outfile="stellar_sed_fit.pdf", errscale=None, sed_file=None):
    all_stars, nstars = _collect_sed_stars(params)
    star_A = all_stars[0]
    if any(v is None for v in [star_A.teff, star_A.rstar, star_A.feh, star_A.av, star_A.distance, star_A.mstar]):
        return None

    if errscale is None:
        errscale = float(params.get('A_sed_errscale', 1.0))

    if sed_file is None:
        sed_file = params.get("sed_file", None)
    if sed_file is None:
        hits = _glob.glob(os.path.join(datadir, "*.sed"))
        sed_file = hits[0] if hits else os.path.join(datadir, "sed.dat")
    elif not os.path.isabs(sed_file):
        sed_file = os.path.join(datadir, sed_file)
    if not os.path.exists(sed_file):
        return None

    teff_arr, logg_arr, feh_arr, av_arr, dist_arr, lstar_arr = _build_sed_arrays(all_stars)

    sed_data = read_sed_file(sed_file, nstars=nstars)
    chi2, blendmag, _, _ = mistmultised(
        teff_arr, logg_arr, feh_arr, av_arr, dist_arr, lstar_arr,
        float(errscale), sed_file, sed_data=sed_data,
    )

    zero_point = np.asarray(sed_data["zero_point"], dtype=float)
    mags       = np.asarray(sed_data["mag"],        dtype=float)
    errmag     = np.asarray(sed_data["errmag"],     dtype=float)
    weff       = np.asarray(sed_data["weff"],       dtype=float)
    widtheff   = np.asarray(sed_data["widtheff"],   dtype=float)
    blendmag   = np.asarray(blendmag,               dtype=float)

    obs_flux   = zero_point * 10 ** (-0.4 * mags)
    obs_err    = obs_flux * np.log(10) / 2.5 * errmag
    model_flux = zero_point * 10 ** (-0.4 * blendmag)
    residuals  = np.where(obs_err > 0, (obs_flux - model_flux) / obs_err, 0.0)

    # NextGen continuous atmosphere(s) — one per star
    def _get_atmosphere(teff, logg, feh, rstar_sed, distance, av):
        lf = _interp_atmosphere(float(teff), float(logg), float(feh))
        if lf is None:
            return None
        lf = _scale_atmosphere(lf, float(rstar_sed), float(distance))
        lf = _apply_extinction(lf, float(av))
        return lf

    atmospheres = []
    for i in range(nstars):
        rstar_sed_i = lstar_arr[i] ** 0.5 * (5772.0 / teff_arr[i]) ** 2
        atmospheres.append(
            _get_atmosphere(teff_arr[i], logg_arr[i], feh_arr[i],
                            rstar_sed_i, dist_arr[i], av_arr[i])
        )

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=(3, 1),
        left=0.15, right=0.95, top=0.95, bottom=0.10, hspace=0.0,
    )
    ax_data = fig.add_subplot(outer[0])
    ax_oc   = fig.add_subplot(outer[1], sharex=ax_data)

    # --- Top panel: atmosphere curves ---
    _atm_colors = ['black', 'steelblue', 'darkorange', 'forestgreen',
                   'purple', 'brown', 'teal', 'crimson']
    letters = 'A' + _COMPANION_LETTERS
    valid_atm = [a for a in atmospheres if a is not None]
    if len(valid_atm) > 1:
        combined = sum(valid_atm)
        for i, atm in enumerate(atmospheres):
            if atm is None:
                continue
            color = _atm_colors[i % len(_atm_colors)]
            label = f'Star {letters[i]}  ({teff_arr[i]:.0f} K)'
            lf_s = uniform_filter1d(atm, size=10)
            mask = lf_s > 0
            ax_data.plot(_WAVELENGTH[mask], np.log10(lf_s[mask]), '-',
                         color=color, lw=1, zorder=1, label=label)
        lf_s = uniform_filter1d(combined, size=10)
        mask = lf_s > 0
        ax_data.plot(_WAVELENGTH[mask], np.log10(lf_s[mask]), '--',
                     color='gray', lw=1, zorder=1, label='Combined')
    elif atmospheres[0] is not None:
        lf_s = uniform_filter1d(atmospheres[0], size=10)
        mask = lf_s > 0
        ax_data.plot(_WAVELENGTH[mask], np.log10(lf_s[mask]), '-',
                     color='black', lw=1, zorder=1, label='Model atmosphere')

    # Model band fluxes (blue circles)
    safe_model = np.where(model_flux > 0, model_flux, np.nan)
    ax_data.plot(weff, np.log10(safe_model), 'o',
                 color='blue', ms=8, zorder=3, label='Model bands')

    # Observed fluxes (red)
    for i in range(len(weff)):
        if obs_flux[i] <= 0:
            continue
        log_obs = np.log10(obs_flux[i])
        y_lo = np.log10(obs_flux[i] - obs_err[i]) if obs_flux[i] > obs_err[i] else log_obs - 0.5
        y_hi = np.log10(obs_flux[i] + obs_err[i])
        ax_data.plot([weff[i], weff[i]], [y_lo, y_hi], '-', color='red', lw=1.5, zorder=2)
        ax_data.plot(
            [weff[i] - widtheff[i] / 2, weff[i] + widtheff[i] / 2],
            [log_obs, log_obs], '-', color='red', lw=1.5, zorder=2,
        )
    safe_obs = np.where(obs_flux > 0, obs_flux, np.nan)
    ax_data.plot(weff, np.log10(safe_obs), 'o',
                 color='red', ms=6, zorder=4, label='Observed')

    ax_data.set_xscale('log')
    ax_data.set_xlim(0.3, 30)
    ax_data.set_ylabel(r'log $\lambda F_\lambda$ (erg s$^{-1}$ cm$^{-2}$)')
    teff_parts = ', '.join(
        rf'$T_{{\rm eff;{letters[i]}}}={teff_arr[i]:.0f}$ K'
        for i in range(nstars)
    )
    title_str = rf'SED fit  {teff_parts}, $A_V={av_arr[0]:.3f}$, $\chi^2={chi2:.2f}$'
    ax_data.set_title(title_str)
    ax_data.legend(loc='upper right', frameon=False, fontsize=9)
    plt.setp(ax_data.get_xticklabels(), visible=False)

    all_vals = np.concatenate([obs_flux[obs_flux > 0], model_flux[model_flux > 0]])
    if len(all_vals):
        all_log = np.log10(all_vals)
        ax_data.set_ylim(all_log.min() - 0.3, all_log.max() + 0.3)

    # --- Bottom panel: O-C residuals in sigma ---
    for i in range(len(weff)):
        ax_oc.plot(
            [weff[i] - widtheff[i] / 2, weff[i] + widtheff[i] / 2],
            [residuals[i], residuals[i]], '-', color='red', lw=1.5,
        )
        ax_oc.plot([weff[i], weff[i]], [residuals[i] - 1, residuals[i] + 1],
                   '-', color='red', lw=1.5)
    ax_oc.plot(weff, residuals, 'o', color='red', ms=6)
    ax_oc.axhline(0, ls='--', color='red', lw=0.8)

    ymax_oc = max(np.ceil(np.abs(residuals).max() * 2) / 2, 1.0) if len(residuals) else 1.0
    ax_oc.set_ylim(-ymax_oc / 0.7, ymax_oc / 0.7)
    ax_oc.set_yticks([-ymax_oc, 0, ymax_oc])
    ax_oc.set_xlabel(r'$\lambda$ ($\mu$m)')
    ax_oc.set_ylabel(r'Res ($\sigma$)')

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, outfile)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def make_mist_plot(params, outdir, outfile="stellar_mist_track.pdf"):
    star_A = _star_from_params(params)
    if any(v is None for v in [star_A.eep, star_A.mstar, star_A.feh, star_A.teff, star_A.rstar]):
        return None
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, outfile)

    # Collect all companion stars with MIST params
    _mist_keys = ('eep', 'mstar', 'feh', 'teff', 'rstar')
    companion_stars = []
    for ltr in _COMPANION_LETTERS:
        comp = _star_X_from_params(params, ltr)
        if all(getattr(comp, k) is not None for k in _mist_keys):
            companion_stars.append(comp)
        else:
            break

    if not companion_stars:
        _ = massradius_mist(
            eep=float(star_A.eep),
            mstar=float(star_A.mstar),
            feh=float(star_A.feh),
            teff=float(star_A.teff),
            rstar=float(star_A.rstar),
            debug=True,
            pngname=path,
        )
        return path if os.path.exists(path) else None

    # Multi-star: plot all tracks on the same HR diagram
    _make_multistar_mist_plot(star_A, companion_stars, path)
    return path if os.path.exists(path) else None


def _make_multistar_mist_plot(star_A, companion_stars, outfile):
    """Create a MIST HR diagram with evolutionary tracks for all stars."""
    gravitysun = 27420.011
    ZAMS_EEP = 202

    def _get_track_data(star):
        try:
            teffs_iso, rstars_iso, _, eeps_iso = _interpolate_track_for_plot(
                float(star.mstar), float(star.feh)
            )
        except Exception:
            return None, None
        if len(teffs_iso) == 0:
            return None, None
        safe_r = np.clip(rstars_iso, 1e-6, None)
        track_logg = np.log10((float(star.mstar) / safe_r ** 2) * gravitysun)
        finite = np.isfinite(teffs_iso) & np.isfinite(track_logg)
        eep_best = float(star.eep)
        if (eep_best + 3) > np.max(eeps_iso):
            eep_cutoff = 1
        else:
            eep_cutoff = min(ZAMS_EEP, eep_best)
        use = finite & (track_logg > 3) & (track_logg < 5) & (eeps_iso >= eep_cutoff)
        if not np.any(use):
            use = finite
        return teffs_iso[use], track_logg[use]

    def _get_point(star):
        logg_best = np.log10((float(star.mstar) / max(float(star.rstar), 1e-6) ** 2) * gravitysun)
        mistteff, mistrstar = get_mist_point(float(star.eep), float(star.mstar), float(star.feh))
        if np.isnan(mistteff):
            logg_mist = logg_best
        else:
            logg_mist = np.log10((float(star.mstar) / max(mistrstar, 1e-6) ** 2) * gravitysun)
        return float(star.teff), logg_best, mistteff, logg_mist

    all_stars = [star_A] + list(companion_stars)
    letters = 'A' + _COMPANION_LETTERS
    track_colors = ['black', 'steelblue', 'darkorange', 'forestgreen',
                    'purple', 'brown', 'teal', 'crimson']
    mist_marker_colors = ['red', 'darkorange', 'green', 'purple',
                          'brown', 'teal', 'crimson', 'gold']

    fig, ax = plt.subplots(figsize=(6, 5.5))
    all_teff_vals = []
    all_logg_vals = []

    for i, star in enumerate(all_stars):
        color = track_colors[i % len(track_colors)]
        mcolor = mist_marker_colors[i % len(mist_marker_colors)]
        ltr = letters[i]

        teff_track, logg_track = _get_track_data(star)
        teff_pt, logg_pt, mistteff, logg_mist = _get_point(star)

        if teff_track is not None:
            ax.plot(teff_track, logg_track, color=color, lw=1.0,
                    label=f'Star {ltr} track')
            all_teff_vals.extend(teff_track)
            all_logg_vals.extend(logg_track)

        ax.plot([teff_pt], [logg_pt], 'o', color=color, ms=6, zorder=5)
        ax.plot([mistteff], [logg_mist], '*', color=mcolor, ms=12, zorder=6)

        all_teff_vals.append(teff_pt)
        all_logg_vals.append(logg_pt)
        if np.isfinite(mistteff):
            all_teff_vals.append(mistteff)
        if np.isfinite(logg_mist):
            all_logg_vals.append(logg_mist)

    margin_t = (max(all_teff_vals) - min(all_teff_vals)) * 0.08 + 50
    margin_g = (max(all_logg_vals) - min(all_logg_vals)) * 0.08 + 0.05
    ax.set_xlim(max(all_teff_vals) + margin_t, min(all_teff_vals) - margin_t)
    ax.set_ylim(max(all_logg_vals) + margin_g, min(all_logg_vals) - margin_g)

    ax.set_xlabel(r'$T_{\mathrm{eff}}$ (K)')
    ax.set_ylabel(r'$\log g_\star$ (cgs)')
    ax.legend(loc='upper left', frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
