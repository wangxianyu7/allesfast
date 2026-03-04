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

# NextGen model grid path — override via env var ALLESFAST_NEXTGEN_PATH
_NEXTGEN_PATH = pathlib.Path(
    os.environ.get(
        'ALLESFAST_NEXTGEN_PATH',
        '/Users/wangxianyu/Applications/NV5/idl90/lib/EXOFASTv2/sed/nextgenfin',
    )
)

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
    return StellarInputs(
        teff=params.get("A_teff", None),
        logg=params.get("A_logg", None),
        feh=params.get("A_feh", None),
        rstar=params.get("A_rstar", None),
        mstar=params.get("A_mstar", None),
        eep=params.get("A_eep", None),
        age=params.get("A_age", None),
        av=params.get("A_av", None),
        distance=distance,
    )


def _star_B_from_params(params):
    distance = params.get("B_distance", None)
    if distance is None:
        parallax = params.get("B_parallax") or params.get("A_parallax", None)
        if parallax is not None and float(parallax) > 0:
            distance = 1000.0 / float(parallax)
    return StellarInputs(
        teff=params.get("B_teff", None),
        logg=params.get("B_logg", None),
        feh=params.get("B_feh") or params.get("A_feh", None),
        rstar=params.get("B_rstar", None),
        mstar=params.get("B_mstar", None),
        eep=params.get("B_eep", None),
        age=params.get("B_age", None),
        av=params.get("B_av") or params.get("A_av", None),
        distance=distance,
    )


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
    has_B, star_A_teff, star_A_av  : scalar metadata for labelling
    star_B_teff                    : (binary only)
    """
    star_A = _star_from_params(params)
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
        try:
            from .. import config as _cfg
            _esc = _cfg.BASEMENT.settings.get('sed_errscale', 1.0)
            errscale = float(np.asarray(_esc).flat[0])
        except Exception:
            errscale = 1.0

    gravity_sun = 27420.011
    star_B = _star_B_from_params(params)
    has_B = all(getattr(star_B, k) is not None
                for k in ('teff', 'rstar', 'feh', 'av', 'distance', 'mstar'))

    if has_B:
        nstars = 2
        teff_arr  = np.array([float(star_A.teff),  float(star_B.teff)])
        logg_arr  = np.array([
            np.log10(gravity_sun * float(star_A.mstar) / float(star_A.rstar) ** 2),
            np.log10(gravity_sun * float(star_B.mstar) / float(star_B.rstar) ** 2),
        ])
        feh_arr   = np.array([float(star_A.feh),   float(star_B.feh)])
        av_arr    = np.array([float(star_A.av),     float(star_B.av)])
        dist_arr  = np.array([float(star_A.distance), float(star_B.distance)])
        lstar_arr = np.array([
            float(star_A.rstar) ** 2 * (float(star_A.teff) / 5772.0) ** 4,
            float(star_B.rstar) ** 2 * (float(star_B.teff) / 5772.0) ** 4,
        ])
    else:
        nstars = 1
        logg_A = np.log10(gravity_sun * float(star_A.mstar) / float(star_A.rstar) ** 2)
        teff_arr  = np.array([float(star_A.teff)])
        logg_arr  = np.array([logg_A])
        feh_arr   = np.array([float(star_A.feh)])
        av_arr    = np.array([float(star_A.av)])
        dist_arr  = np.array([float(star_A.distance)])
        lstar_arr = np.array([float(star_A.rstar) ** 2 * (float(star_A.teff) / 5772.0) ** 4])

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

    def _get_atm(teff, logg, feh, rstar, distance, av):
        lf = _interp_atmosphere(float(teff), float(logg), float(feh))
        if lf is None:
            return None
        lf = _scale_atmosphere(lf, float(rstar), float(distance))
        lf = _apply_extinction(lf, float(av))
        return lf

    atm_A = _get_atm(star_A.teff, logg_arr[0], star_A.feh,
                     star_A.rstar, star_A.distance, star_A.av)
    atm_B = _get_atm(star_B.teff, logg_arr[1], star_B.feh,
                     star_B.rstar, star_B.distance, star_B.av) if has_B else None

    result = dict(
        sedbands=sedbands,
        weff_um=weff,
        widtheff_um=widtheff,
        obs_flux=obs_flux,
        obs_err=obs_err,
        model_flux=model_flux,
        residuals=residuals,
        chi2=np.array([chi2]),
        has_B=np.array([has_B]),
        star_A_teff=np.array([float(star_A.teff)]),
        star_A_av=np.array([float(star_A.av)]),
        wave_atm_um=_WAVELENGTH.copy(),
    )
    if has_B:
        result['star_B_teff'] = np.array([float(star_B.teff)])
    if atm_A is not None:
        result['flux_atm_A'] = atm_A
    if atm_B is not None:
        result['flux_atm_B'] = atm_B
        result['flux_atm_combined'] = atm_A + atm_B
    return result


def make_sed_plot(params, datadir, outdir, outfile="stellar_sed_fit.pdf", errscale=1.0, sed_file=None):
    star_A = _star_from_params(params)
    if any(v is None for v in [star_A.teff, star_A.rstar, star_A.feh, star_A.av, star_A.distance, star_A.mstar]):
        return None

    # Resolve SED file
    if sed_file is None:
        sed_file = params.get("sed_file", None)
    if sed_file is None:
        hits = _glob.glob(os.path.join(datadir, "*.sed"))
        sed_file = hits[0] if hits else os.path.join(datadir, "sed.dat")
    elif not os.path.isabs(sed_file):
        sed_file = os.path.join(datadir, sed_file)
    if not os.path.exists(sed_file):
        return None

    gravity_sun = 27420.011

    # Check for Star B
    star_B = _star_B_from_params(params)
    has_B = all(getattr(star_B, k) is not None
                for k in ('teff', 'rstar', 'feh', 'av', 'distance', 'mstar'))

    if has_B:
        nstars = 2
        teff_arr  = np.array([float(star_A.teff),  float(star_B.teff)])
        logg_arr  = np.array([
            np.log10(gravity_sun * float(star_A.mstar) / float(star_A.rstar) ** 2),
            np.log10(gravity_sun * float(star_B.mstar) / float(star_B.rstar) ** 2),
        ])
        feh_arr   = np.array([float(star_A.feh),   float(star_B.feh)])
        av_arr    = np.array([float(star_A.av),     float(star_B.av)])
        dist_arr  = np.array([float(star_A.distance), float(star_B.distance)])
        lstar_arr = np.array([
            float(star_A.rstar) ** 2 * (float(star_A.teff) / 5772.0) ** 4,
            float(star_B.rstar) ** 2 * (float(star_B.teff) / 5772.0) ** 4,
        ])
    else:
        nstars = 1
        logg_A = np.log10(gravity_sun * float(star_A.mstar) / float(star_A.rstar) ** 2)
        teff_arr  = np.array([float(star_A.teff)])
        logg_arr  = np.array([logg_A])
        feh_arr   = np.array([float(star_A.feh)])
        av_arr    = np.array([float(star_A.av)])
        dist_arr  = np.array([float(star_A.distance)])
        lstar_arr = np.array([float(star_A.rstar) ** 2 * (float(star_A.teff) / 5772.0) ** 4])

    sed_data = read_sed_file(sed_file, nstars=nstars)
    chi2, blendmag, _, _ = mistmultised(
        teff_arr, logg_arr, feh_arr, av_arr, dist_arr, lstar_arr,
        float(errscale), sed_file, sed_data=sed_data,
    )

    zero_point = np.asarray(sed_data["zero_point"], dtype=float)
    mags       = np.asarray(sed_data["mag"],        dtype=float)
    errmag     = np.asarray(sed_data["errmag"],     dtype=float)
    weff       = np.asarray(sed_data["weff"],       dtype=float)   # μm
    widtheff   = np.asarray(sed_data["widtheff"],   dtype=float)   # μm
    blendmag   = np.asarray(blendmag,               dtype=float)

    obs_flux   = zero_point * 10 ** (-0.4 * mags)
    obs_err    = obs_flux * np.log(10) / 2.5 * errmag
    model_flux = zero_point * 10 ** (-0.4 * blendmag)
    residuals  = np.where(obs_err > 0, (obs_flux - model_flux) / obs_err, 0.0)

    # NextGen continuous atmosphere(s)
    def _get_atmosphere(teff, logg, feh, rstar, distance, av):
        lf = _interp_atmosphere(float(teff), float(logg), float(feh))
        if lf is None:
            return None
        lf = _scale_atmosphere(lf, float(rstar), float(distance))
        lf = _apply_extinction(lf, float(av))
        return lf

    atm_A = _get_atmosphere(star_A.teff, logg_arr[0], star_A.feh,
                            star_A.rstar, star_A.distance, star_A.av)
    atm_B = _get_atmosphere(star_B.teff, logg_arr[1] if has_B else logg_arr[0],
                            star_B.feh if has_B else star_A.feh,
                            star_B.rstar if has_B else star_A.rstar,
                            star_B.distance if has_B else star_A.distance,
                            star_B.av if has_B else star_A.av) if has_B else None

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=(3, 1),
        left=0.15, right=0.95, top=0.95, bottom=0.10, hspace=0.0,
    )
    ax_data = fig.add_subplot(outer[0])
    ax_oc   = fig.add_subplot(outer[1], sharex=ax_data)

    # --- Top panel: atmosphere curves ---
    if has_B and atm_A is not None and atm_B is not None:
        # Show individual components and their sum
        combined = atm_A + atm_B
        for lf, color, label in [
            (atm_A,    'black',       f'Star A  ({float(star_A.teff):.0f} K)'),
            (atm_B,    'steelblue',   f'Star B  ({float(star_B.teff):.0f} K)'),
            (combined, 'gray',        'Combined'),
        ]:
            lf_s = uniform_filter1d(lf, size=10)
            mask = lf_s > 0
            ls = '--' if color == 'gray' else '-'
            ax_data.plot(_WAVELENGTH[mask], np.log10(lf_s[mask]), ls,
                         color=color, lw=1, zorder=1, label=label)
    elif atm_A is not None:
        lf_s = uniform_filter1d(atm_A, size=10)
        mask = lf_s > 0
        ax_data.plot(_WAVELENGTH[mask], np.log10(lf_s[mask]), '-',
                     color='black', lw=1, zorder=1, label='Model atmosphere')

    # Model band fluxes (blue circles — combined model from mistmultised)
    safe_model = np.where(model_flux > 0, model_flux, np.nan)
    ax_data.plot(weff, np.log10(safe_model), 'o',
                 color='blue', ms=8, zorder=3, label='Model bands')

    # Observed fluxes (red) with x=bandwidth, y=flux error bars
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
    title_str = (
        rf'SED fit  $T_{{\rm eff;A}}={float(star_A.teff):.0f}$ K'
        + (rf', $T_{{\rm eff;B}}={float(star_B.teff):.0f}$ K' if has_B else '')
        + rf', $A_V={float(star_A.av):.3f}$, $\chi^2={chi2:.2f}$'
    )
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

    star_B = _star_B_from_params(params)
    has_B = all(getattr(star_B, k) is not None
                for k in ('eep', 'mstar', 'feh', 'teff', 'rstar'))

    if not has_B:
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

    # Binary: plot both tracks on the same HR diagram
    _make_binary_mist_plot(star_A, star_B, path)
    return path if os.path.exists(path) else None


def _make_binary_mist_plot(star_A, star_B, outfile):
    """Create a MIST HR diagram with evolutionary tracks for both stars."""
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

    teff_vals_A, logg_vals_A = _get_track_data(star_A)
    teff_vals_B, logg_vals_B = _get_track_data(star_B)
    teff_A, logg_A, mistteff_A, logg_mist_A = _get_point(star_A)
    teff_B, logg_B, mistteff_B, logg_mist_B = _get_point(star_B)

    fig, ax = plt.subplots(figsize=(6, 5.5))

    if teff_vals_A is not None:
        ax.plot(teff_vals_A, logg_vals_A, color='black', lw=1.0, label='Star A track')
    if teff_vals_B is not None:
        ax.plot(teff_vals_B, logg_vals_B, color='steelblue', lw=1.0, label='Star B track')

    ax.plot([teff_A], [logg_A], 'o', color='black', ms=6, zorder=5)
    ax.plot([mistteff_A], [logg_mist_A], '*', color='red', ms=12, zorder=6)
    ax.plot([teff_B], [logg_B], 'o', color='steelblue', ms=6, zorder=5)
    ax.plot([mistteff_B], [logg_mist_B], '*', color='darkorange', ms=12, zorder=6)

    # Axis limits: encompass both tracks and marker points (Teff decreasing left→right)
    all_teff = [t for arr in [teff_vals_A, teff_vals_B] if arr is not None for t in arr]
    all_logg = [g for arr in [logg_vals_A, logg_vals_B] if arr is not None for g in arr]
    all_teff += [teff_A, teff_B]
    all_logg += [logg_A, logg_B]
    for v in [mistteff_A, mistteff_B]:
        if np.isfinite(v):
            all_teff.append(v)
    for v in [logg_mist_A, logg_mist_B]:
        if np.isfinite(v):
            all_logg.append(v)

    margin_t = (max(all_teff) - min(all_teff)) * 0.08 + 50
    margin_g = (max(all_logg) - min(all_logg)) * 0.08 + 0.05
    ax.set_xlim(max(all_teff) + margin_t, min(all_teff) - margin_t)
    ax.set_ylim(max(all_logg) + margin_g, min(all_logg) - margin_g)

    ax.set_xlabel(r'$T_{\mathrm{eff}}$ (K)')
    ax.set_ylabel(r'$\log g_\star$ (cgs)')
    ax.legend(loc='upper left', frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
