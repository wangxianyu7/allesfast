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
from .massradius_mist import massradius_mist
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
    distance = params.get("host_distance", None)
    if distance is None:
        parallax = params.get("host_parallax", None)
        if parallax is not None and float(parallax) > 0:
            distance = 1000.0 / float(parallax)   # mas → pc
    return StellarInputs(
        teff=params.get("host_teff", None),
        logg=params.get("host_logg", None),
        feh=params.get("host_feh", None),
        rstar=params.get("host_rstar", None),
        mstar=params.get("host_mstar", None),
        eep=params.get("host_eep", None),
        age=params.get("host_age", None),
        av=params.get("host_av", None),
        distance=distance,
    )


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def make_sed_plot(params, datadir, outdir, outfile="stellar_sed_fit.png", errscale=1.0, sed_file=None):
    star = _star_from_params(params)
    if any(v is None for v in [star.teff, star.rstar, star.feh, star.av, star.distance, star.mstar]):
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
    logg  = np.log10(gravity_sun * float(star.mstar) / float(star.rstar) ** 2)
    lstar = (float(star.rstar) ** 2) * (float(star.teff) / 5772.0) ** 4

    sed_data = read_sed_file(sed_file, nstars=1)
    chi2, blendmag, _, _ = mistmultised(
        np.array([float(star.teff)]),
        np.array([logg]),
        np.array([float(star.feh)]),
        np.array([float(star.av)]),
        np.array([float(star.distance)]),
        np.array([lstar]),
        float(errscale),
        sed_file,
        sed_data=sed_data,
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

    # NextGen continuous atmosphere
    atmosphere = None
    lamflam = _interp_atmosphere(float(star.teff), logg, float(star.feh))
    if lamflam is not None:
        lamflam = _scale_atmosphere(lamflam, float(star.rstar), float(star.distance))
        lamflam = _apply_extinction(lamflam, float(star.av))
        atmosphere = (_WAVELENGTH, lamflam)

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=(3, 1),
        left=0.15, right=0.95, top=0.95, bottom=0.10, hspace=0.0,
    )
    ax_data = fig.add_subplot(outer[0])
    ax_oc   = fig.add_subplot(outer[1], sharex=ax_data)

    # --- Top panel ---
    # Model atmosphere (black line)
    if atmosphere is not None:
        wav, lf = atmosphere
        lf_smooth = uniform_filter1d(lf, size=10)
        mask = lf_smooth > 0
        ax_data.plot(wav[mask], np.log10(lf_smooth[mask]), '-',
                     color='black', lw=1, zorder=1, label='Model atmosphere')

    # Model band fluxes (blue circles)
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
    ax_data.set_title(
        rf'SED fit  $T_{{\rm eff}}={float(star.teff):.0f}$ K, '
        rf'$A_V={float(star.av):.3f}$, '
        rf'$\chi^2={chi2:.2f}$'
    )
    ax_data.legend(loc='upper right', frameon=False)
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


def make_mist_plot(params, outdir, outfile="stellar_mist_track.png"):
    star = _star_from_params(params)
    if any(v is None for v in [star.eep, star.mstar, star.feh, star.teff, star.rstar]):
        return None
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, outfile)
    _ = massradius_mist(
        eep=float(star.eep),
        mstar=float(star.mstar),
        feh=float(star.feh),
        teff=float(star.teff),
        rstar=float(star.rstar),
        debug=True,
        pngname=path,
    )
    return path if os.path.exists(path) else None
