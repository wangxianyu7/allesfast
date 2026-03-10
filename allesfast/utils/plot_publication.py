#!/usr/bin/env python3
"""
Publication-quality summary figure from allesfast modelfiles.

Usage
-----
In Python::

    from allesfast.utils.plot_publication import make_summary_plot
    make_summary_plot('/path/to/fit/dir', prefix='mcmc')

From command line::

    python -m allesfast.utils.plot_publication /path/to/fit/dir [--prefix mcmc]

Panels are auto-detected from settings.csv and available modelfiles:
    SED       — if {prefix}_sed.npz exists
    Transit   — one panel per photometric instrument
    RV        — phase-folded, all non-RM RV instruments combined
    RM        — one time-series panel per RM instrument

Model files must be generated first via show_initial_guess / de_fit / mcmc_output.
"""
from __future__ import annotations

import os
import glob as _glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d


# ---------------------------------------------------------------------------
# Marker / colour cycle for different instruments
# ---------------------------------------------------------------------------
_MARKERS = ['o', 's', '^', 'v', 'D', 'X', 'H', 'd', 'p', '*', '>', '<']
_RV_MARKERSIZES = [7, 7, 7, 10, 7, 7, 7, 7, 7, 7, 7, 7]
_TRANSIT_MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X', 'H', 'd', 'p']
_TRANSIT_COLORS  = ['silver', 'darkgrey', 'grey', 'k', 'silver',
                    'darkgrey', 'grey', 'k', 'silver', 'darkgrey']

# Instrument name standardisation for transit legends
_INST_RENAME = [
    ('TESS120QLP',  'TESS 120s (QLP)'),
    ('TESS200QLP',  'TESS 200s (QLP)'),
    ('TESS600QLP',  'TESS 600s (QLP)'),
    ('TESS1800QLP', 'TESS 1800s (QLP)'),
    ('TESS120',     'TESS 120s'),
    ('TESS200',     'TESS 200s'),
    ('TESS600',     'TESS 600s'),
    ('TESS1800',    'TESS 1800s'),
    ('Kepler120',   'Kepler 120s'),
    ('Kepler200',   'Kepler 200s'),
    ('Kepler600',   'Kepler 600s'),
    ('Kepler1800',  'Kepler 1800s'),
]

# Allowed cadences (seconds) for auto-detection
_ALLOWED_CADENCES = [60, 120, 200, 600, 1800]


def _standardise_inst_name(name, time=None):
    """Standardise instrument name for display (e.g. TESS120 → TESS 120s).

    If *time* is provided and no rename rule matches, the cadence is computed
    from ``np.median(np.diff(time))`` and appended (e.g. K2 → K2 60s).
    """
    for old, new in _INST_RENAME:
        if old in name:
            return name.replace(old, new)
    # No rule matched — compute cadence from data if available
    if time is not None and len(time) > 1:
        raw = np.median(np.diff(np.sort(time))) * 86400
        # Snap to nearest allowed cadence
        cadence_s = min(_ALLOWED_CADENCES, key=lambda c: abs(c - raw))
        return f'{name} {cadence_s}s'
    return name


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _load(path):
    return dict(np.load(path, allow_pickle=True))


def _read_settings(datadir):
    settings = {}
    path = os.path.join(datadir, 'settings.csv')
    if not os.path.exists(path):
        return settings
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ';' in line:
                line = line[:line.index(';')]
            line = line.strip()
            if ',' not in line:
                continue
            k, v = line.split(',', 1)
            settings[k.strip()] = v.strip()
    return settings


def _read_params(datadir, mode='mcmc'):
    """Return {param: median_value} from mcmc_table.csv or params.csv."""
    # Try mcmc_table.csv first
    tpath = os.path.join(datadir, 'results', f'{mode}_table.csv')
    if not os.path.exists(tpath):
        tpath = os.path.join(datadir, 'results', 'mcmc_table.csv')
    if os.path.exists(tpath):
        out = {}
        with open(tpath) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                try:
                    out[parts[0].strip()] = float(parts[1].strip())
                except ValueError:
                    pass
        return out

    # Fallback: params.csv (use value column)
    ppath = os.path.join(datadir, 'params.csv')
    if not os.path.exists(ppath):
        return {}
    out = {}
    with open(ppath) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            try:
                out[parts[0].strip()] = float(parts[1].strip())
            except ValueError:
                pass
    return out


def _read_params_with_errors(datadir, mode='mcmc'):
    """Return {param: (median, lower_err, upper_err)} from mcmc_table.csv."""
    tpath = os.path.join(datadir, 'results', f'{mode}_table.csv')
    if not os.path.exists(tpath):
        tpath = os.path.join(datadir, 'results', 'mcmc_table.csv')
    if not os.path.exists(tpath):
        return {}
    out = {}
    with open(tpath) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split(',')
            if len(parts) < 4:
                continue
            name = parts[0].strip()
            try:
                med = float(parts[1].strip())
                lo = parts[2].strip()
                hi = parts[3].strip()
                if lo == '(fixed)' or hi == '(fixed)':
                    out[name] = (med, None, None)
                else:
                    out[name] = (med, float(lo), float(hi))
            except ValueError:
                pass
    return out


def _format_val_err(val, err_lo, err_hi, ndig=None):
    """Format a value with asymmetric or symmetric errors as a LaTeX string."""
    import math
    if err_lo is None or err_hi is None:
        # Fixed parameter
        if val == int(val):
            return f'{int(val)} (fixed)'
        return f'{val} (fixed)'

    # Determine number of significant decimals from the errors
    if ndig is None:
        min_err = min(abs(err_lo), abs(err_hi))
        if min_err == 0:
            ndig = 2
        else:
            ndig = max(0, -math.floor(math.log10(min_err)) + 1)

    fmt = f'{{:.{ndig}f}}'
    v_str = fmt.format(val)
    lo_str = fmt.format(err_lo)
    hi_str = fmt.format(err_hi)

    if lo_str == hi_str:
        return f'${{{v_str}}} \\pm {{{lo_str}}}$'
    else:
        return f'${{{v_str}}}^{{+{hi_str}}}_{{-{lo_str}}}$'


def _read_derived_params(datadir, mode='mcmc'):
    """Return {description_key: (median, lower_err, upper_err)} from derived table.

    Keys are matched by searching for substrings in the description column.
    """
    tpath = os.path.join(datadir, 'results', f'{mode}_derived_table.csv')
    if not os.path.exists(tpath):
        return {}
    out = {}
    with open(tpath) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split(',')
            if len(parts) < 4:
                continue
            desc = parts[0].strip()
            try:
                med = float(parts[1].strip())
                lo  = float(parts[2].strip())
                hi  = float(parts[3].strip())
                out[desc] = (med, lo, hi)
            except ValueError:
                pass
    return out


def _find_derived(derived, substring):
    """Find a derived parameter by substring match. Returns (med, lo, hi) or None."""
    for key, val in derived.items():
        if substring in key:
            return val
    return None


def _build_param_text(params_we, companion='b', derived=None):
    """Build a multi-line LaTeX string of key parameters for annotation.

    Parameters
    ----------
    params_we : dict
        {param: (median, lower_err, upper_err)} from _read_params_with_errors.
    companion : str
        Companion letter, e.g. 'b'.
    derived : dict or None
        {description: (median, lower_err, upper_err)} from _read_derived_params.
    """
    if derived is None:
        derived = {}
    lines = []

    # Teff
    for key in ['A_teff', 'A_teffsed']:
        if key in params_we:
            med, lo, hi = params_we[key]
            lo_i = int(round(lo)) if lo is not None else None
            hi_i = int(round(hi)) if hi is not None else None
            lines.append('$T_{\\rm eff}$: ' + _format_val_err(
                int(round(med)), lo_i, hi_i, ndig=0) + ' K')
            break

    # M_star (prefer derived table, fallback to logmstar)
    mstar_d = _find_derived(derived, f'Stellar mass; $M_')
    if mstar_d is not None:
        med, lo, hi = mstar_d
        lines.append('$M_*$: ' + _format_val_err(med, lo, hi) + ' $M_\\odot$')
    elif 'A_logmstar' in params_we:
        med, lo, hi = params_we['A_logmstar']
        mstar = 10**med
        if lo is not None and hi is not None:
            mstar_lo = mstar * np.log(10) * lo
            mstar_hi = mstar * np.log(10) * hi
            lines.append('$M_*$: ' + _format_val_err(mstar, mstar_lo, mstar_hi)
                          + ' $M_\\odot$')
        else:
            lines.append(f'$M_*$: {mstar:.2f} (fixed) $M_\\odot$')

    # R_star
    key = 'A_rstar'
    if key in params_we:
        med, lo, hi = params_we[key]
        lines.append('$R_*$: ' + _format_val_err(med, lo, hi) + ' $R_\\odot$')

    # Period
    key = f'{companion}_period'
    if key in params_we:
        med, lo, hi = params_we[key]
        lines.append(f'$P$: {med:.2f} d')

    # Rp (from derived table, in R_Jup)
    rp_d = _find_derived(derived, f'$R_\\mathrm{{{companion}}}$ ($\\mathrm{{R_{{jup}}}}$)')
    if rp_d is not None:
        med, lo, hi = rp_d
        lines.append('$R_{\\rm p}$: ' + _format_val_err(med, lo, hi)
                      + ' $R_{\\rm J}$')

    # Mp (from derived table, in M_Jup)
    mp_d = _find_derived(derived, f'$M_\\mathrm{{{companion}}}$ ($\\mathrm{{M_{{jup}}}}$)')
    if mp_d is not None:
        med, lo, hi = mp_d
        lines.append('$M_{\\rm p}$: ' + _format_val_err(med, lo, hi)
                      + ' $M_{\\rm J}$')

    # Eccentricity (from derived table)
    e_d = _find_derived(derived, f'Eccentricity {companion}')
    if e_d is not None:
        med, lo, hi = e_d
        lines.append('$e$: ' + _format_val_err(med, lo, hi))
    elif f'{companion}_f_c' in params_we and f'{companion}_f_s' in params_we:
        fc_med, fc_lo, fc_hi = params_we[f'{companion}_f_c']
        fs_med, fs_lo, fs_hi = params_we[f'{companion}_f_s']
        e_med = fc_med**2 + fs_med**2
        if fc_lo is not None and fs_lo is not None:
            e_lo = np.sqrt((2 * abs(fc_med) * fc_lo)**2
                           + (2 * abs(fs_med) * fs_lo)**2)
            e_hi = np.sqrt((2 * abs(fc_med) * fc_hi)**2
                           + (2 * abs(fs_med) * fs_hi)**2)
            lines.append('$e$: ' + _format_val_err(e_med, e_lo, e_hi))
        else:
            lines.append(f'$e$: {e_med:.4f} (fixed)')

    # a/R* (from derived table)
    ar_d = _find_derived(derived, f'$a_\\mathrm{{{companion}}}/R_')
    if ar_d is not None:
        med, lo, hi = ar_d
        lines.append('$a/R_*$: ' + _format_val_err(med, lo, hi))

    # Lambda (from derived table)
    lam_d = _find_derived(derived, 'Spin-orbit angle')
    if lam_d is not None:
        med, lo, hi = lam_d
        lines.append('$\\lambda$: ' + _format_val_err(med, lo, hi)
                      + '$^\\circ$')

    # vsini (from derived table)
    vsini_d = _find_derived(derived, 'Projected stellar rotation')
    if vsini_d is not None:
        med, lo, hi = vsini_d
        lines.append('$v\\sin{i_*}$: ' + _format_val_err(med, lo, hi)
                      + ' km/s')

    return '\n'.join(lines)


def _phase_fold(time, period, epoch):
    phase = ((time - epoch) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    return phase


def _compute_t14(params, companion='b'):
    """Compute total transit duration T14 in hours from orbital parameters.

    If ``rsuma`` = (R* + Rp)/a is not directly available in *params*, it is
    derived from ``A_rstar``, ``A_logmstar`` (or ``A_mstar``), ``rr``, and
    ``period`` via Kepler's third law.
    """
    # Physical constants (SI)
    _G     = 6.67430e-11       # m^3 kg^-1 s^-2
    _M_sun = 1.98892e30        # kg
    _R_sun = 6.9570e8          # m

    per   = params.get(f'{companion}_period', 1.0)
    rr    = params.get(f'{companion}_rr', 0.1)
    cosi  = params.get(f'{companion}_cosi', 0.0)
    f_c   = params.get(f'{companion}_f_c', 0.0)
    f_s   = params.get(f'{companion}_f_s', 0.0)

    rsuma = params.get(f'{companion}_rsuma', None)
    if rsuma is None:
        # Derive from Kepler's third law: a^3 = G M P^2 / (4 pi^2)
        rstar = params.get('A_rstar', 1.0)                     # R_sun
        if 'A_logmstar' in params:
            mstar = 10.0 ** params['A_logmstar']                # M_sun
        else:
            mstar = params.get('A_mstar', 1.0)                 # M_sun
        P_sec = per * 86400.0
        a_m   = (_G * mstar * _M_sun * P_sec**2
                 / (4.0 * np.pi**2)) ** (1.0 / 3.0)            # metres
        a_Rsun = a_m / _R_sun                                   # R_sun
        rsuma  = rstar * (1.0 + rr) / a_Rsun

    e = f_c**2 + f_s**2
    w = np.arctan2(f_s, f_c)
    sini = np.sqrt(1.0 - cosi**2)
    R_star_over_a = rsuma / (1.0 + rr)
    b = cosi / R_star_over_a  # impact parameter

    arg = R_star_over_a * np.sqrt((1.0 + rr)**2 - b**2) / sini
    arg = np.clip(arg, 0, 1)
    ecc_corr = np.sqrt(1.0 - e**2) / (1.0 + e * np.sin(w))
    return per / np.pi * np.arcsin(arg) * ecc_corr * 24.0  # hours


def _bin_data(x, y, yerr, bin_minutes):
    """Bin time-series data into bins of given width (in minutes of the x-axis unit)."""
    if bin_minutes is None or bin_minutes <= 0:
        return x, y, yerr
    bin_days = bin_minutes / 24. / 60.
    edges = np.arange(x.min(), x.max() + bin_days, bin_days)
    xb, yb, eb = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if not np.any(mask):
            continue
        w = 1. / yerr[mask] ** 2
        yb.append(np.sum(w * y[mask]) / np.sum(w))
        xb.append(np.mean(x[mask]))
        eb.append(1. / np.sqrt(np.sum(w)))
    return np.array(xb), np.array(yb), np.array(eb)


def _symmetrise_residual_ylim(ax):
    """Make residual axis symmetric and set only two yticks at ±half-range."""
    ymin, ymax = ax.get_ylim()
    ylimax = max(abs(ymin), abs(ymax))
    ax.set_ylim(-ylimax, ylimax)
    halfylimax = ylimax / 2.0
    # Round to a sensible number of decimals
    if halfylimax != 0:
        import math
        order = math.floor(math.log10(abs(halfylimax)))
        ndig = max(0, -order + 1)
        halfylimax = round(halfylimax, ndig)
    ax.set_yticks([-halfylimax, halfylimax])


# ---------------------------------------------------------------------------
# Panel plotters
# ---------------------------------------------------------------------------
def _plot_sed(ax_top, ax_bot, d):
    weff     = d['weff_um'].astype(float)
    widtheff = d['widtheff_um'].astype(float)
    obs      = d['obs_flux'].astype(float)
    err      = d['obs_err'].astype(float)
    model    = d['model_flux'].astype(float)
    resid    = d['residuals'].astype(float)
    wave_atm = d['wave_atm_um'].astype(float)

    # Atmosphere curve(s)
    has_B = bool(d.get('has_B', np.array([False]))[0])
    for key, color, label, ls in [
        ('flux_atm_A',        'black',     'Star A',   '-'),
        ('flux_atm_B',        'steelblue', 'Star B',   '-'),
        ('flux_atm_combined', 'dimgray',   'Combined', '--'),
    ]:
        if key not in d:
            continue
        if key == 'flux_atm_combined' and not has_B:
            continue
        atm_s = uniform_filter1d(d[key].astype(float), size=10)
        mask = atm_s > 0
        ax_top.plot(wave_atm[mask], np.log10(atm_s[mask]), ls,
                    color=color, lw=1, zorder=1, label=label)

    # Model band fluxes
    safe_model = np.where(model > 0, model, np.nan)
    ax_top.plot(weff, np.log10(safe_model), 'o', color='royalblue',
                ms=7, zorder=3, label='Model')

    # Observed fluxes
    for i in range(len(weff)):
        if obs[i] <= 0:
            continue
        log_obs = np.log10(obs[i])
        y_lo = (np.log10(obs[i] - err[i]) if obs[i] > err[i] else log_obs - 0.5)
        y_hi = np.log10(obs[i] + err[i])
        ax_top.plot([weff[i], weff[i]], [y_lo, y_hi], '-', color='crimson', lw=1.5, zorder=2)
        ax_top.plot([weff[i] - widtheff[i] / 2, weff[i] + widtheff[i] / 2],
                    [log_obs, log_obs], '-', color='crimson', lw=1.5, zorder=2)
    safe_obs = np.where(obs > 0, obs, np.nan)
    ax_top.plot(weff, np.log10(safe_obs), 'o', color='crimson', ms=5, zorder=4,
                label='Observed')

    ax_top.set_xscale('log')
    ax_top.set_xlim(0.3, 30)
    valid_obs = obs[obs > 0]
    if len(valid_obs) > 0:
        log_obs_vals = np.log10(valid_obs)
        max_y = np.max(log_obs_vals)
        min_y = np.min(log_obs_vals)
        height = max_y - min_y
        ax_top.set_ylim(min_y - 0.1 * height, max_y + 0.1 * height)
    ax_top.set_ylabel(r'$\log(\lambda F_\lambda)$' '\n' r'(erg s$^{-1}$ cm$^{-2}$)',
                      fontsize=20, fontweight='bold')
    ax_top.legend(loc='upper right', frameon=False, fontsize=15, prop={'weight': 'bold'}, ncol=1)

    # Residuals
    for i in range(len(weff)):
        ax_bot.plot([weff[i] - widtheff[i] / 2, weff[i] + widtheff[i] / 2],
                    [resid[i], resid[i]], '-', color='crimson', lw=1.5)
        ax_bot.plot([weff[i], weff[i]],
                    [resid[i] - 1, resid[i] + 1], '-', color='crimson', lw=1.5)
    ax_bot.plot(weff, resid, 'o', color='crimson', ms=5)
    ax_bot.axhline(0, ls='--', color='crimson', lw=0.8)
    ax_bot.set_xscale('log')
    ax_bot.set_xlim(0.3, 30)
    ax_bot.set_xlabel(r'$\lambda$ ($\mu$m)', fontsize=20, fontweight='bold')
    ax_bot.set_ylabel(r'Res ($\sigma$)', fontsize=20, fontweight='bold')
    _symmetrise_residual_ylim(ax_bot)


def _plot_transit(ax_top, ax_bot, d, period, epoch, t14_hrs=None,
                  inst_label=None, bin_minutes=None, transit_index=0):
    if inst_label:
        inst_label = _standardise_inst_name(inst_label, time=d['time'])
    # Phase-fold all transit epochs
    phase     = _phase_fold(d['time'], period, epoch)
    phase_hrs = phase * period * 24.  # convert phase to hours from T0

    phase_d     = _phase_fold(d['time_dense'], period, epoch)
    phase_d_hrs = phase_d * period * 24.

    # Detrended data (subtract baseline and stellar variability)
    y    = d['data'] - d['baseline'] - d['stellar_var']
    yerr = d['err']
    resid = d['residuals']

    # Filter to near-transit data only (within ±T14 window)
    if t14_hrs is not None:
        xlim = t14_hrs * 0.7
        mask   = np.abs(phase_hrs)   <= xlim
        mask_d = np.abs(phase_d_hrs) <= xlim
    else:
        xlim = None
        mask   = np.ones(len(phase_hrs), dtype=bool)
        mask_d = np.ones(len(phase_d_hrs), dtype=bool)

    phase_hrs   = phase_hrs[mask]
    y           = y[mask]
    yerr        = yerr[mask]
    resid       = resid[mask]
    phase_d_hrs = phase_d_hrs[mask_d]
    model_dense = d['model_dense'][mask_d]

    # Sort dense model by phase for a clean curve
    idx_d = np.argsort(phase_d_hrs)

    # Optionally bin
    idx_data = np.argsort(phase_hrs)
    t_plot, y_plot, ye_plot = _bin_data(phase_hrs[idx_data], y[idx_data],
                                        yerr[idx_data], bin_minutes)
    t_res, r_plot, re_plot  = _bin_data(phase_hrs[idx_data], resid[idx_data],
                                        yerr[idx_data], bin_minutes)

    mk  = _TRANSIT_MARKERS[transit_index % len(_TRANSIT_MARKERS)]
    clr = _TRANSIT_COLORS[transit_index % len(_TRANSIT_COLORS)]
    ax_top.scatter(t_plot, y_plot, marker=mk, color=clr, s=20,
                   zorder=1, label=inst_label)
    ax_top.plot(phase_d_hrs[idx_d], model_dense[idx_d], color='firebrick',
                lw=2, zorder=2)
    ax_top.set_ylabel('Relative Flux', fontsize=20, fontweight='bold')
    if inst_label:
        ax_top.legend(loc='upper right', frameon=False, fontsize=15, prop={'weight': 'bold'})

    ax_bot.scatter(t_res, r_plot * 1e3, marker=mk, color=clr, s=20)
    ax_bot.axhline(0, ls='--', color='red', lw=0.8)
    ax_bot.set_xlabel(r'Time $-$ $T_0$ (hrs)', fontsize=20, fontweight='bold')
    ax_bot.set_ylabel('Res (ppt)', fontsize=20, fontweight='bold')
    _symmetrise_residual_ylim(ax_bot)

    if xlim is not None:
        ax_top.set_xlim(-xlim, xlim)


def _plot_rv_phased(ax_top, ax_bot, modelfiles_dict, rv_insts, period, epoch):
    dense_done = False
    for i, inst in enumerate(rv_insts):
        if inst not in modelfiles_dict:
            continue
        d     = modelfiles_dict[inst]
        mk    = _MARKERS[i % len(_MARKERS)]
        ms    = _RV_MARKERSIZES[i % len(_RV_MARKERSIZES)]

        ph    = _phase_fold(d['time'].copy(), period, epoch)
        y     = (d['data'] - d['baseline'] - d['stellar_var']) * 1e3   # km/s → m/s
        yerr  = d['err'] * 1e3
        resid = d['residuals'] * 1e3

        ax_top.errorbar(ph, y, yerr=yerr, fmt=mk, markerfacecolor='white',
                        markeredgecolor='black', ecolor='k',
                        capsize=3, capthick=1, ms=ms, zorder=2, label=inst)
        ax_bot.errorbar(ph, resid, yerr=yerr, fmt=mk, markerfacecolor='white',
                        markeredgecolor='black', ecolor='k',
                        capsize=3, capthick=1, ms=ms, zorder=2)

        if not dense_done:
            ph_d  = _phase_fold(d['time_dense'].copy(), period, epoch)
            mod_d = d['model_dense'] * 1e3
            idx   = np.argsort(ph_d)
            ax_top.plot(ph_d[idx],     mod_d[idx], color='firebrick', lw=2, zorder=3)
            ax_top.plot(ph_d[idx] + 1, mod_d[idx], color='firebrick', lw=2, zorder=3)
            ax_top.plot(ph_d[idx] - 1, mod_d[idx], color='firebrick', lw=2, zorder=3)
            dense_done = True

    ax_top.set_xlim(-0.6, 0.6)
    ax_top.set_ylabel('RV (m/s)', fontsize=20, fontweight='bold')
    ax_top.legend(loc='upper right', frameon=False, fontsize=15, prop={'weight': 'bold'})

    ax_bot.axhline(0, ls='--', color='red', lw=0.8)
    ax_bot.set_xlabel(r'Phase $+\,(T_p-T_0)/P$', fontsize=20, fontweight='bold')
    ax_bot.set_ylabel('Res (m/s)', fontsize=20, fontweight='bold')
    _symmetrise_residual_ylim(ax_bot)


def _compute_keplerian(time, params, companion='b'):
    """Compute pure Keplerian RV signal (km/s) from fitted parameters."""
    from radvel.kepler import rv_drive
    from radvel.orbit import timetrans_to_timeperi

    per = params[f'{companion}_period']
    tc  = params[f'{companion}_epoch']
    f_c = params.get(f'{companion}_f_c', 0.0)
    f_s = params.get(f'{companion}_f_s', 0.0)
    K   = params.get(f'{companion}_K', 0.0)

    e = f_c**2 + f_s**2
    w = np.arctan2(f_s, f_c)
    tp = timetrans_to_timeperi(tc, per, e, w)
    return rv_drive(time, np.array([per, tp, e, w, K * 1e3])) / 1e3  # km/s


def _plot_rm(ax_top, ax_bot, d, epoch, params, companion='b',
             t14_hrs=None, inst_label=None):
    # Use the nearest transit epoch to the observed data so that t=0 is
    # centred on the transit even when the data are from a later epoch.
    period = params.get(f'{companion}_period', 1.0)
    mid_time = (float(d['time'].min()) + float(d['time'].max())) / 2.0
    n_transits = round((mid_time - epoch) / period)
    local_epoch = epoch + n_transits * period

    t_hrs  = (d['time']       - local_epoch) * 24.
    td_hrs = (d['time_dense'] - local_epoch) * 24.

    # Subtract Keplerian orbit to isolate RM anomaly
    kep_data  = _compute_keplerian(d['time'],       params, companion)
    kep_dense = _compute_keplerian(d['time_dense'], params, companion)

    # Baseline-subtracted, Keplerian-removed RM signal in m/s
    y     = (d['data'] - d['baseline'] - d['stellar_var'] - kep_data) * 1e3
    yerr  = d['err'] * 1e3
    mod_d = (d['model_dense'] - kep_dense) * 1e3
    resid = d['residuals'] * 1e3

    # Dense model now covers full ±T14*0.7 range (extended in save_modelfiles)
    if t14_hrs is not None:
        xlim = t14_hrs * 0.7
    else:
        xlim = max(abs(td_hrs.min()), abs(td_hrs.max()))

    ax_top.errorbar(t_hrs, y, yerr=yerr, fmt='o', markerfacecolor='white',
                    markeredgecolor='black', ecolor='k',
                    capsize=3, capthick=1, ms=7, zorder=2, label=inst_label)
    ax_top.plot(td_hrs, mod_d, color='firebrick', lw=2, zorder=3)
    ax_top.set_ylabel('RM (m/s)', fontsize=20, fontweight='bold')
    if inst_label:
        ax_top.legend(loc='upper right', frameon=False, fontsize=15, prop={'weight': 'bold'})

    ax_bot.errorbar(t_hrs, resid, yerr=yerr, fmt='o', markerfacecolor='white',
                    markeredgecolor='black', ecolor='k',
                    capsize=3, capthick=1, ms=7)
    ax_bot.axhline(0, ls='--', color='red', lw=0.8)
    ax_bot.set_xlabel(r'Time $-$ $T_0$ (hrs)', fontsize=20, fontweight='bold')
    ax_bot.set_ylabel('Res (m/s)', fontsize=20, fontweight='bold')
    _symmetrise_residual_ylim(ax_bot)

    ax_top.set_xlim(-xlim, xlim)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def make_summary_plot(
    datadir,
    prefix='mcmc',
    companion='b',
    phot_insts=None,
    rv_insts=None,
    rm_insts=None,
    bin_phot_minutes=None,
    outfile=None,
    figsize_per_panel=(6, 5),
    show_params=True,
):
    """
    Generate a publication-quality summary figure from allesfast modelfiles.

    Parameters
    ----------
    datadir : str
        Path to the allesfast run directory.
    prefix : str
        Modelfile prefix: 'initial_guess', 'optimized', or 'mcmc'.
    companion : str
        Companion letter, e.g. 'b'.
    phot_insts : list[str] or None
        Photometric instruments (one transit panel each). None = auto.
    rv_insts : list[str] or None
        RV-only instruments (phase-folded together). None = auto.
    rm_insts : list[str] or None
        RM instruments (one time-series panel each). None = auto.
    bin_phot_minutes : float or None
        Bin transit data to this cadence in minutes for display. None = no binning.
    outfile : str or None
        Output path. None → results/{prefix}_summary.pdf.
    figsize_per_panel : (float, float)
        (width, height) in inches per panel column.
    show_params : bool
        If True (default), annotate the figure with key fitted parameters
        (Teff, M*, R*, P, Rp/R*, K, e, lambda, vsini) and their uncertainties.

    Returns
    -------
    fig : matplotlib.Figure
    """
    modeldir = os.path.join(datadir, 'results', 'modelfiles')
    if not os.path.isdir(modeldir):
        raise FileNotFoundError(
            f'modelfiles directory not found: {modeldir}\n'
            'Run show_initial_guess / de_fit / mcmc_output first.'
        )

    settings = _read_settings(datadir)
    params   = _read_params(datadir, mode=prefix)
    period   = params.get(f'{companion}_period', 1.0)
    epoch    = params.get(f'{companion}_epoch',  0.0)
    t14_hrs  = _compute_t14(params, companion)

    s_phot = settings.get('inst_phot', '').split() if settings.get('inst_phot') else []
    s_rv   = settings.get('inst_rv',   '').split() if settings.get('inst_rv')   else []

    rm_set = set()
    for k in settings:
        if k.startswith(f'{companion}_flux_weighted_'):
            rm_set.add(k[len(f'{companion}_flux_weighted_'):])

    def _avail(inst):
        return os.path.exists(os.path.join(modeldir, f'{prefix}_{inst}.npz'))

    if phot_insts is None:
        phot_insts = [i for i in s_phot if _avail(i)]
    if rv_insts is None:
        rv_insts = [i for i in s_rv if i not in rm_set and _avail(i)]
    if rm_insts is None:
        rm_insts = [i for i in s_rv if i in rm_set and _avail(i)]

    has_sed = os.path.exists(os.path.join(modeldir, f'{prefix}_sed.npz'))
    has_rv  = bool(rv_insts)

    # Fixed panels: SED, RV, RM are always present (placeholder if no data)
    panels = []
    panels.append(('sed', None) if has_sed else ('empty', 'No SED data'))
    panels.append(('rv', rv_insts) if has_rv else ('empty', 'No RV data'))
    if rm_insts:
        for inst in rm_insts:
            panels.append(('rm', inst))
    else:
        panels.append(('empty', 'No RM data'))
    for inst in phot_insts:
        panels.append(('phot', inst))

    # Load modelfiles
    mf = {}
    for ptype, pval in panels:
        if ptype == 'sed':
            mf['sed'] = _load(os.path.join(modeldir, f'{prefix}_sed.npz'))
        elif ptype in ('phot', 'rm'):
            mf[pval] = _load(os.path.join(modeldir, f'{prefix}_{pval}.npz'))
        elif ptype == 'rv':
            for inst in pval:
                mf[inst] = _load(os.path.join(modeldir, f'{prefix}_{inst}.npz'))

    ncols = 3
    n     = len(panels)
    nrows = (n + ncols - 1) // ncols
    fw, fh = figsize_per_panel
    # When showing params, widen figure and reserve right margin for text
    extra_w = 3.5 if show_params else 0
    fig = plt.figure(figsize=(fw * ncols + extra_w, fh * nrows))
    right_margin = 1.0 - extra_w / (fw * ncols + extra_w) if show_params else 0.98
    outer = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.30,
                              hspace=0.35,
                              left=0.09, right=right_margin, top=0.95, bottom=0.08)

    transit_axes_top = []
    transit_axes_bot = []
    transit_count = 0

    for i, (ptype, pval) in enumerate(panels):
        row, col = divmod(i, ncols)
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row, col],
            height_ratios=[2, 1], hspace=0.0)
        ax_top = fig.add_subplot(inner[0])
        ax_bot = fig.add_subplot(inner[1], sharex=ax_top)
        plt.setp(ax_top.get_xticklabels(), visible=False)

        if ptype == 'sed':
            _plot_sed(ax_top, ax_bot, mf['sed'])
        elif ptype == 'phot':
            _plot_transit(ax_top, ax_bot, mf[pval], period, epoch,
                          t14_hrs=t14_hrs, inst_label=pval,
                          bin_minutes=bin_phot_minutes,
                          transit_index=transit_count)
            transit_count += 1
            transit_axes_top.append(ax_top)
            transit_axes_bot.append(ax_bot)
        elif ptype == 'rv':
            _plot_rv_phased(ax_top, ax_bot, mf, pval, period, epoch)
        elif ptype == 'rm':
            _plot_rm(ax_top, ax_bot, mf[pval], epoch, params,
                     companion=companion, t14_hrs=t14_hrs,
                     inst_label=pval)
        elif ptype == 'empty':
            ax_top.text(0.5, 0.5, pval, transform=ax_top.transAxes,
                        ha='center', va='center', fontsize=10, color='gray')
            ax_top.set_xticks([])
            ax_top.set_yticks([])
            ax_bot.set_xticks([])
            ax_bot.set_yticks([])

        # Remove duplicate y-tick labels for adjacent panels
        if i > 0:
            ax_top.yaxis.set_tick_params(labelleft=True)
            ax_bot.yaxis.set_tick_params(labelleft=True)
        for ax in (ax_top, ax_bot):
            ax.tick_params(direction='in', which='both', width=2,
                           bottom=True, top=True, left=True, right=True,
                           labelsize=15)
            for spine in ax.spines.values():
                spine.set_linewidth(2)

    # Share ylim across all transit panels (top/data only, not residuals)
    if len(transit_axes_top) > 1:
        ylims_top = [ax.get_ylim() for ax in transit_axes_top]
        shared_top = (min(y[0] for y in ylims_top), max(y[1] for y in ylims_top))
        for ax in transit_axes_top:
            ax.set_ylim(shared_top)

    # Add parameter annotation text
    if show_params:
        params_we = _read_params_with_errors(datadir, mode=prefix)
        derived = _read_derived_params(datadir, mode=prefix)
        if params_we or derived:
            param_text = _build_param_text(params_we, companion=companion,
                                           derived=derived)
            if param_text:
                # Place target name + parameters in the right margin
                text_x = right_margin + 0.02
                target_name = os.path.basename(os.path.normpath(datadir))
                target_name = target_name.replace('_', ' ')
                fig.text(text_x, 0.95, target_name,
                         fontsize=20, ha='left', va='top',
                         fontweight='bold',
                         transform=fig.transFigure)
                fig.text(text_x, 0.88, param_text,
                         fontsize=16, ha='left', va='top',
                         transform=fig.transFigure,
                         linespacing=1.6)

    if outfile is None:
        outfile = os.path.join(datadir, 'results', f'{prefix}_summary.pdf')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, bbox_inches='tight', dpi=150)
    print(f'Saved: {outfile}')
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description='Publication summary figure from allesfast modelfiles.')
    p.add_argument('datadir', help='allesfast run directory')
    p.add_argument('--prefix', default='mcmc',
                   choices=['initial_guess', 'optimized', 'mcmc'])
    p.add_argument('--companion', default='b')
    p.add_argument('--bin-phot-minutes', type=float, default=None,
                   dest='bin_phot_minutes',
                   help='Bin transit data to this cadence in minutes')
    p.add_argument('--outfile', default=None)
    p.add_argument('--no-params', action='store_true', default=False,
                   help='Do not show parameter annotations on the figure')
    args = p.parse_args()
    make_summary_plot(args.datadir, prefix=args.prefix,
                      companion=args.companion,
                      bin_phot_minutes=args.bin_phot_minutes,
                      outfile=args.outfile,
                      show_params=not args.no_params)
