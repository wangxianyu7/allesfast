"""Reader for Doppler-Tomography fits files.

Format (matches EXOFASTv2 exofast_readdt.pro):
    HDU 0 : ccf2d (nvels, ntimes) — observed CCF residual
    HDU 1 : bjd (ntimes,) — observation times
    HDU 2 : vel (nvels,) — velocity grid (km/s)

The filename encodes the spectrograph resolving power as the 4th dot-
separated field, e.g. ``n20160226.KELT-17b.TRES.44000.fits`` → R=44000.
"""
import os
import numpy as np


def read_dt_fits(filename):
    """Read a Doppler-Tomography FITS file (EXOFASTv2 format).

    Returns a dict with keys:
        ccf2d   : (nvels, ntimes) float64
        bjd     : (ntimes,)
        vel     : (nvels,)        km/s
        stepsize: (nvels,)        Δv per pixel
        rms     : float           pixel RMS (estimated from data)
        rspec   : float           spectrograph resolving power R
        chisqr0 : float           χ² of a flat-zero model (sanity check)
        label, tel, night         strings for plotting
    """
    from astropy.io import fits

    basename = os.path.basename(filename)
    with fits.open(filename) as hdul:
        ccf2d = np.asarray(hdul[0].data, dtype=np.float64)
        bjd   = np.asarray(hdul[1].data, dtype=np.float64).ravel()
        vel   = np.asarray(hdul[2].data, dtype=np.float64).ravel()

    if ccf2d.ndim != 2:
        raise ValueError(f'ccf2d should be 2D, got shape {ccf2d.shape}')

    # IDL stores arrays with vel as fastest axis; verify:
    if ccf2d.shape != (vel.size, bjd.size):
        # Try a transpose if dimensions are flipped (astropy reads in C-order)
        if ccf2d.shape == (bjd.size, vel.size):
            ccf2d = ccf2d.T
        else:
            raise ValueError(
                f'ccf2d shape {ccf2d.shape} incompatible with '
                f'(nvels={vel.size}, ntimes={bjd.size})'
            )

    # Filename parsing: <date>.<planet>.<telescope>.<rspec>.fits
    parts = basename.split('.')
    if len(parts) < 5:
        raise ValueError(
            f'DT filename "{basename}" must follow '
            '"<date>.<planet>.<telescope>.<rspec>.fits" convention.'
        )
    telescope = parts[2]
    rspec = float(parts[3])
    if rspec <= 0:
        raise ValueError(
            f'rspec={rspec} ≤ 0 — encode the resolving power in the '
            f'filename (4th dot-separated field).'
        )

    night = basename[1:5] + '-' + basename[5:7] + '-' + basename[7:9]
    label = f'UT {night} {telescope}'

    # stepsize: shift forward; final element = mean of the rest
    stepsize = np.empty_like(vel)
    stepsize[:-1] = vel[1:] - vel[:-1]
    stepsize[-1] = float(np.mean(stepsize[:-1]))

    # RMS estimation (matches IDL):
    #   rms0 = stddev(ccf2d)
    #   errscale = sqrt(sum(((ccf2d-median)/rms0)^2) / (N-3))
    #   rms = rms0 * errscale
    median_ccf = float(np.median(ccf2d))
    rms0 = float(np.std(ccf2d, ddof=1))
    n = ccf2d.size
    errscale = float(np.sqrt(np.sum(((ccf2d - median_ccf) / rms0) ** 2)
                              / (n - 3.0)))
    rms = rms0 * errscale

    chisqr0 = float(np.sum(((ccf2d - median_ccf) / rms) ** 2))

    return dict(
        ccf2d=ccf2d, bjd=bjd, vel=vel, stepsize=stepsize,
        rms=rms, rspec=rspec, chisqr0=chisqr0,
        label=label, tel=telescope, night=night,
        median_ccf=median_ccf,
    )
