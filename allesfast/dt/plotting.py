"""DT plotting: 2D shadow images (observed / model / residual) + 1D summary.

Output format: a single multi-panel figure per DT instrument.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize


def plot_dt(dt_data, model, *,
            label=None, savepath=None, vlim=None,
            cmap_data='RdBu_r', cmap_resid='RdBu_r',
            show_velocity_marks=None,
            ):
    """Three-panel DT figure: observed CCF residual, model, residual.

    Parameters
    ----------
    dt_data : dict
        Output of :func:`allesfast.dt.io.read_dt_fits`.
    model : (nvels, ntimes) ndarray
        Model shadow returned by :func:`allesfast.dt.core.dopptom_chi2`.
    label : str, optional
        Title prefix.  Falls back to dt_data['label'].
    savepath : str, optional
        If provided, write to this file.  Otherwise the figure is returned.
    vlim : float, optional
        Symmetric colour limit (km/s normalisation in shadow strength).
        Default: 5 × stddev of the data.
    show_velocity_marks : iterable of float, optional
        Vertical dashed lines at these velocities (km/s).
        Default: [-vsini, 0, +vsini] from data (none if not known).
    """
    ccf = dt_data['ccf2d']
    vel = dt_data['vel']
    bjd = dt_data['bjd']

    if vlim is None:
        vlim = 5.0 * float(np.std(ccf))
    if label is None:
        label = dt_data.get('label', '')

    resid = ccf - model

    # Time axis: use BJD offset
    bjd_ref = float(bjd.min())
    t_min = bjd_ref
    t_off = (bjd - bjd_ref) * 24.0   # hours since first frame
    t_centres = t_off
    # Use frame index for the y-axis (each row = one CCF) since spacing
    # may be uneven.

    extent_v = [vel.min(), vel.max()]

    fig = plt.figure(figsize=(11, 8))
    gs = GridSpec(2, 3, height_ratios=[3, 1], hspace=0.30, wspace=0.22)

    panel_titles = ['Observed', 'Model', 'Residual']
    panel_arrays = [ccf, model, resid]
    panel_cmap   = [cmap_data, cmap_data, cmap_resid]

    norm = Normalize(vmin=-vlim, vmax=vlim)
    for i, (arr, ttl, cmap) in enumerate(zip(panel_arrays, panel_titles, panel_cmap)):
        ax = fig.add_subplot(gs[0, i])
        # imshow: rows = velocity (we'll transpose so y is time), origin lower
        im = ax.imshow(arr.T, aspect='auto', origin='lower',
                        extent=[extent_v[0], extent_v[1],
                                t_off.min(), t_off.max()],
                        norm=norm, cmap=cmap, interpolation='nearest')
        ax.set_xlabel('Velocity (km/s)')
        if i == 0:
            ax.set_ylabel(f'Time − {bjd_ref:.4f} (hr)')
        ax.set_title(ttl, fontsize=11)
        if show_velocity_marks:
            for v in show_velocity_marks:
                ax.axvline(v, color='k', lw=0.4, ls='--', alpha=0.5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Bottom row: 1D summaries
    ax_sum = fig.add_subplot(gs[1, :])
    # Time-collapsed: stack the in-transit frames (where model has signal)
    has_signal = np.any(np.abs(model - np.median(model)) > 1e-4, axis=0)
    if not np.any(has_signal):
        has_signal = np.ones_like(bjd, dtype=bool)
    obs_stack   = np.mean(ccf[:, has_signal], axis=1)
    model_stack = np.mean(model[:, has_signal], axis=1)
    resid_stack = obs_stack - model_stack
    ax_sum.plot(vel, obs_stack, color='C0', alpha=0.8, label='observed (mean of in-transit)')
    ax_sum.plot(vel, model_stack, color='C3', lw=1.0, label='model')
    ax_sum.plot(vel, resid_stack, color='gray', lw=0.5, label='residual')
    ax_sum.set_xlabel('Velocity (km/s)')
    ax_sum.set_ylabel('CCF residual')
    ax_sum.axhline(0, color='k', lw=0.4)
    ax_sum.legend(fontsize=8, frameon=False, loc='upper right')
    ax_sum.set_xlim(extent_v)

    fig.suptitle(label, fontsize=12)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight', dpi=150)
        return savepath
    return fig
