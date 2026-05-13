"""DT plotting: 3-panel Data | Model | Residual figure with shared
colorbar (style matches user reference: ../20240702_Jace_XO3/main.ipynb).

Public entry points:

- :func:`plot_dt`        — pure plotting given (data, model, residual) +
                          phase array.
- :func:`make_dt_plot`   — convenience wrapper that takes the params
                          dict, computes the model via dopptom_chi2,
                          and saves a PDF.  Mirrors make_sed_plot in
                          api shape (params, datadir, outdir, outfile).
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_dt(data, model, phase, *,
            vel=None, vel_range=None,
            label_fontsize=15, cmap='Greys_r',
            savepath=None, title=None):
    """Three-panel Data | Model | Residual figure with shared colorbar.

    Parameters
    ----------
    data, model : (nvels, ntimes) ndarray
        Observed CCF residual and model.
    phase : (ntimes,) ndarray
        Orbital phase corresponding to each row.  The y-axis spans
        ``[phase.min(), phase.max()]``.
    vel : (nvels,) ndarray, optional
        Velocity grid (km/s).  If not provided, pixel indices are used.
    vel_range : (vmin, vmax), optional
        x-axis limits.  Default: full extent of ``vel``.
    label_fontsize : int
    cmap : str
        Matplotlib colormap name.  Default 'Greys_r' (white = bright).
    savepath : str, optional
        If provided, save the figure to this path; otherwise return it.
    title : str, optional
        Suptitle for the figure.
    """
    residual = data - model
    if vel is None:
        vel = np.arange(data.shape[0])
    if vel_range is None:
        vel_range = (float(vel.min()), float(vel.max()))
    y_range = (float(phase.min()), float(phase.max()))

    # Shared colormap range across all 3 panels
    vmin = float(min(np.min(data), np.min(model), np.min(residual)))
    vmax = float(max(np.max(data), np.max(model), np.max(residual)))

    fig = plt.figure(figsize=(12, 4))
    ax_data = fig.add_axes([0.05, 0.10, 0.28, 0.80])
    ax_mod  = fig.add_axes([0.33, 0.10, 0.28, 0.80])
    ax_res  = fig.add_axes([0.61, 0.10, 0.28, 0.80])
    ax_cbar = fig.add_axes([0.92, 0.10, 0.02, 0.80])

    imshow_kw = dict(
        aspect='auto', origin='lower',
        extent=[vel_range[0], vel_range[1], y_range[0], y_range[1]],
        vmin=vmin, vmax=vmax, cmap=cmap,
    )

    ax_data.imshow(data.T, **imshow_kw)
    ax_data.set_xlabel('Velocity (km/s)', fontsize=label_fontsize)
    ax_data.set_ylabel('Orbital Phase', fontsize=label_fontsize)
    ax_data.text(0.95, 0.9, 'Data', ha='right', va='center',
                 transform=ax_data.transAxes, fontsize=label_fontsize)

    ax_mod.imshow(model.T, **imshow_kw)
    ax_mod.set_xlabel('Velocity (km/s)', fontsize=label_fontsize)
    ax_mod.text(0.95, 0.9, 'Model', ha='right', va='center',
                transform=ax_mod.transAxes, fontsize=label_fontsize)
    ax_mod.set_yticks([])

    im = ax_res.imshow(residual.T, **imshow_kw)
    ax_res.set_xlabel('Velocity (km/s)', fontsize=label_fontsize)
    ax_res.text(0.95, 0.9, 'Residual', ha='right', va='center',
                transform=ax_res.transAxes, fontsize=label_fontsize)
    ax_res.set_yticks([])

    cbar = fig.colorbar(im, cax=ax_cbar, orientation='vertical')
    cbar.set_label('Fractional Variation', fontsize=label_fontsize)

    if title:
        fig.suptitle(title, fontsize=label_fontsize, y=1.02)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')
        plt.close(fig)
        return savepath
    return fig


# ---------------------------------------------------------------------------
#  Pipeline-stage entry point (mirror of make_sed_plot)
# ---------------------------------------------------------------------------
def make_dt_plot(params, datadir, outdir, *, outfile, inst,
                 dt_data=None, basement=None):
    """Compute model + save 3-panel DT figure for one instrument.

    Used at the initial_guess / optimized / mcmc pipeline stages, mirroring
    make_sed_plot / make_mist_plot.

    Parameters
    ----------
    params : dict
        Fully-resolved parameter dict (output of update_params + extras).
    datadir : str
        Fit directory (unused here but kept for API consistency).
    outdir : str
        Where to write the PDF.
    outfile : str
        Filename within ``outdir`` (e.g. 'mcmc_dt_TRES.pdf').
    inst : str
        DT instrument label (key in basement.dt_data).
    dt_data : dict, optional
        Pre-loaded DT data (read_dt_fits output).  If None, pulled from
        ``basement.dt_data[inst]``.
    basement : Basement, optional
        Fallback when ``dt_data`` is not supplied.

    Returns
    -------
    path or None
        Full path of the saved PDF, or None if the model could not be
        computed (e.g. parameters invalid).
    """
    from .core import dopptom_chi2
    from ..utils.quadld import quadld

    if dt_data is None:
        if basement is None:
            from .. import config
            basement = config.BASEMENT
        dt_data = basement.dt_data[inst]
    if basement is None:
        from .. import config
        basement = config.BASEMENT

    companion = basement.settings['companions_phot'][0]

    # Stellar logg
    _Msun_kg = 1.989e30; _Rsun_m = 6.957e8; _G_si = 6.674e-11
    try:
        mstar = float(params['A_mstar'])
        rstar = float(params['A_rstar'])
        g_cgs = _G_si * mstar * _Msun_kg / (rstar * _Rsun_m) ** 2 * 100.0
        logg = float(np.log10(g_cgs))
    except Exception:
        return None
    teff = float(params.get('A_teff', np.nan))
    feh  = float(params.get('A_feh',  0.0))

    rr   = float(params[companion + '_rr'])
    rsuma = params.get(companion + '_rsuma')
    if rsuma is None or not np.isfinite(rsuma) or rsuma <= 0:
        return None
    ar   = (1.0 + rr) / float(rsuma)
    cosi = float(params[companion + '_cosi'])
    _fc  = float(params[companion + '_f_c'])
    _fs  = float(params[companion + '_f_s'])
    e    = _fc * _fc + _fs * _fs
    w    = float(np.mod(np.arctan2(_fs, _fc), 2 * np.pi))
    tc   = float(params[companion + '_epoch'])
    per  = float(params[companion + '_period'])
    lam  = float(params.get(companion + '_lambda',
                            params.get('A_lambda', 0.0)))

    band = basement.settings.get(f'dt_ld_band_{inst}', 'V')
    u1, u2 = quadld(logg, teff, feh, band)
    if not (np.isfinite(u1) and np.isfinite(u2)):
        return None

    vsini = float(params.get('A_vsini', np.nan))
    vline = float(params.get(f'A_vline_{inst}', np.nan))
    errs  = float(params.get(f'A_dt_errscale_{inst}', 1.0))
    if not (np.isfinite(vsini) and np.isfinite(vline)):
        return None

    chi2, model = dopptom_chi2(
        dt_data, tc, per, e, w, cosi, rr, ar, lam,
        float(u1), float(u2), vsini, vline, errs,
        return_model=True,
    )
    if model is None or not np.all(np.isfinite(model)):
        return None

    # Per-frame orbital phase
    phase = ((dt_data['bjd'] - tc) / per) % 1.0
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    # Mean-subtract data so colormap centres on zero (matches Beatty plots)
    data_centered = dt_data['ccf2d'] - dt_data['median_ccf']
    model_centered = model - dt_data['median_ccf']

    path = os.path.join(outdir, outfile)
    title = dt_data.get('label', f'DT {inst}')
    plot_dt(data_centered, model_centered, phase,
            vel=dt_data['vel'], savepath=path, title=title)
    return path
