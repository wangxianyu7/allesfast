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
_MARKERS = ['o', 's', 'D', '^', 'v', 'p', 'h', 'P', 'X', '*']
_COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


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


def _phase_fold(time, period, epoch):
    phase = ((time - epoch) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    return phase


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
    ax_top.set_ylabel(r'$\log(\lambda F_\lambda)$' '\n' r'(erg s$^{-1}$ cm$^{-2}$)',
                      fontsize=8)
    ax_top.legend(loc='upper right', frameon=False, fontsize=7, ncol=1)

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
    ax_bot.set_xlabel(r'$\lambda$ ($\mu$m)', fontsize=8)
    ax_bot.set_ylabel(r'Res ($\sigma$)', fontsize=8)


def _plot_transit(ax_top, ax_bot, d, epoch, inst_label=None, bin_minutes=None):
    t_hrs  = (d['time']       - epoch) * 24.
    td_hrs = (d['time_dense'] - epoch) * 24.

    # Detrended data (subtract baseline and stellar variability)
    y    = d['data'] - d['baseline'] - d['stellar_var']
    yerr = d['err']
    resid = d['residuals']

    # Optionally bin
    t_plot, y_plot, ye_plot = _bin_data(t_hrs, y, yerr, bin_minutes)
    t_res, r_plot, re_plot  = _bin_data(t_hrs, resid, yerr, bin_minutes)

    ax_top.errorbar(t_plot, y_plot, yerr=ye_plot, fmt='.', color='#555555',
                    ms=3, capsize=0, zorder=1, label=inst_label)
    ax_top.plot(td_hrs, d['model_dense'], 'r-', lw=1.5, zorder=2)
    ax_top.set_ylabel('Normalized Flux', fontsize=8)
    if inst_label:
        ax_top.legend(loc='upper right', frameon=False, fontsize=7)

    ax_bot.errorbar(t_res, r_plot * 1e3, yerr=re_plot * 1e3,
                    fmt='.', color='#555555', ms=3, capsize=0)
    ax_bot.axhline(0, ls='--', color='red', lw=0.8)
    ax_bot.set_xlabel(r'Time $-$ $T_0$ (hrs)', fontsize=8)
    ax_bot.set_ylabel(r'Res ($\times 10^{-3}$)', fontsize=8)


def _plot_rv_phased(ax_top, ax_bot, modelfiles_dict, rv_insts, period, epoch):
    dense_done = False
    for i, inst in enumerate(rv_insts):
        if inst not in modelfiles_dict:
            continue
        d     = modelfiles_dict[inst]
        color = _COLORS[i % len(_COLORS)]
        mk    = _MARKERS[i % len(_MARKERS)]

        ph    = _phase_fold(d['time'].copy(), period, epoch)
        y     = (d['data'] - d['baseline'] - d['stellar_var']) * 1e3   # km/s → m/s
        yerr  = d['err'] * 1e3
        resid = d['residuals'] * 1e3

        ax_top.errorbar(ph, y, yerr=yerr, fmt=mk, color=color,
                        ms=4, capsize=0, zorder=2, label=inst)
        ax_bot.errorbar(ph, resid, yerr=yerr, fmt=mk, color=color,
                        ms=4, capsize=0, zorder=2)

        if not dense_done:
            ph_d  = _phase_fold(d['time_dense'].copy(), period, epoch)
            mod_d = d['model_dense'] * 1e3
            idx   = np.argsort(ph_d)
            ax_top.plot(ph_d[idx], mod_d[idx], 'r-', lw=1.5, zorder=3)
            dense_done = True

    ax_top.axhline(0, ls='--', color='gray', lw=0.5, zorder=0)
    ax_top.set_ylabel('RV (m/s)', fontsize=8)
    ax_top.legend(loc='upper right', frameon=False, fontsize=7)

    ax_bot.axhline(0, ls='--', color='red', lw=0.8)
    ax_bot.set_xlabel(r'Phase $+\,(T_p-T_0)/P$', fontsize=8)
    ax_bot.set_ylabel('Res (m/s)', fontsize=8)


def _plot_rm(ax_top, ax_bot, d, epoch, inst_label=None):
    t_hrs  = (d['time']       - epoch) * 24.
    td_hrs = (d['time_dense'] - epoch) * 24.

    # Baseline-subtracted RM signal in m/s
    y     = (d['data'] - d['baseline'] - d['stellar_var']) * 1e3
    yerr  = d['err'] * 1e3
    mod_d = d['model_dense'] * 1e3
    resid = d['residuals'] * 1e3

    ax_top.errorbar(t_hrs, y, yerr=yerr, fmt='o', color='black',
                    ms=4, capsize=0, zorder=2, label=inst_label)
    ax_top.plot(td_hrs, mod_d, 'r-', lw=1.5, zorder=3)
    ax_top.axhline(0, ls='--', color='gray', lw=0.5, zorder=0)
    ax_top.set_ylabel('RM (m/s)', fontsize=8)
    if inst_label:
        ax_top.legend(loc='upper right', frameon=False, fontsize=7)

    ax_bot.errorbar(t_hrs, resid, yerr=yerr, fmt='o', color='black',
                    ms=4, capsize=0)
    ax_bot.axhline(0, ls='--', color='red', lw=0.8)
    ax_bot.set_xlabel(r'Time $-$ $T_0$ (hrs)', fontsize=8)
    ax_bot.set_ylabel('Res (m/s)', fontsize=8)


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
    figsize_per_panel=(3.5, 4.5),
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

    panels = []
    if has_sed:
        panels.append(('sed', None))
    for inst in phot_insts:
        panels.append(('phot', inst))
    if has_rv:
        panels.append(('rv', rv_insts))
    for inst in rm_insts:
        panels.append(('rm', inst))

    if not panels:
        raise RuntimeError(
            'No panels to plot. Check that modelfiles exist and instruments '
            'are listed in settings.csv.'
        )

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

    n  = len(panels)
    fw, fh = figsize_per_panel
    fig = plt.figure(figsize=(fw * n, fh))
    outer = gridspec.GridSpec(1, n, figure=fig, wspace=0.10,
                              left=0.09, right=0.98, top=0.95, bottom=0.13)

    for i, (ptype, pval) in enumerate(panels):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[i], height_ratios=[3, 1], hspace=0.0)
        ax_top = fig.add_subplot(inner[0])
        ax_bot = fig.add_subplot(inner[1], sharex=ax_top)
        plt.setp(ax_top.get_xticklabels(), visible=False)

        if ptype == 'sed':
            _plot_sed(ax_top, ax_bot, mf['sed'])
        elif ptype == 'phot':
            _plot_transit(ax_top, ax_bot, mf[pval], epoch,
                          inst_label=pval, bin_minutes=bin_phot_minutes)
        elif ptype == 'rv':
            _plot_rv_phased(ax_top, ax_bot, mf, pval, period, epoch)
        elif ptype == 'rm':
            _plot_rm(ax_top, ax_bot, mf[pval], epoch, inst_label=pval)

        # Remove duplicate y-tick labels for adjacent panels
        if i > 0:
            ax_top.yaxis.set_tick_params(labelleft=True)
            ax_bot.yaxis.set_tick_params(labelleft=True)
        for ax in (ax_top, ax_bot):
            ax.tick_params(direction='in', top=True, right=True, which='both')

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
    args = p.parse_args()
    make_summary_plot(args.datadir, prefix=args.prefix,
                      companion=args.companion,
                      bin_phot_minutes=args.bin_phot_minutes,
                      outfile=args.outfile)
