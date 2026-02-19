import numpy as np
from pathlib import Path
from .mist_utils import readeep

# Persistent track storage (mass x feh x vvcrit x alpha)
tracks = None
ALLOWED_MASS = np.array([
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
    0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04,
    1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20, 1.22, 1.24, 1.26, 1.28,
    1.30, 1.32, 1.34, 1.36, 1.38, 1.40, 1.42, 1.44, 1.46, 1.48, 1.50, 1.52,
    1.54, 1.56, 1.58, 1.60, 1.62, 1.64, 1.66, 1.68, 1.70, 1.72, 1.74, 1.76,
    1.78, 1.80, 1.82, 1.84, 1.86, 1.88, 1.90, 1.92, 1.94, 1.96, 1.98, 2.00,
    2.02, 2.04, 2.06, 2.08, 2.10, 2.12, 2.14, 2.16, 2.18, 2.20, 2.22, 2.24,
    2.26, 2.28, 2.30, 2.32, 2.34, 2.36, 2.38, 2.40, 2.42, 2.44, 2.46, 2.48,
    2.50, 2.52, 2.54, 2.56, 2.58, 2.60, 2.62, 2.64, 2.66, 2.68, 2.70, 2.72,
    2.74, 2.76, 2.78, 2.80, 3.00, 3.20, 3.40, 3.60, 3.80, 4.00, 4.20, 4.40,
    4.60, 4.80, 5.00, 5.20, 5.40, 5.60, 5.80, 6.00, 6.20, 6.40, 6.60, 6.80,
    7.00, 7.20, 7.40, 7.60, 7.80, 8.00, 9.00, 10.00, 11.00, 12.00, 13.00,
    14.00, 15.00, 16.00, 17.00, 18.00, 19.00, 20.00, 22.00, 24.00, 26.00,
    28.00, 30.00, 32.00, 34.00, 36.00, 38.00, 40.00, 45.00, 50.00, 55.00,
    60.00, 65.00, 70.00, 75.00, 80.00, 85.00, 90.00, 95.00, 100.00, 105.00,
    110.00, 115.00, 120.00, 125.00, 130.00, 135.00, 140.00, 145.00, 150.00,
    175.00, 200.00, 225.00, 250.00, 275.00, 300.00
])
ALLOWED_INITFEH = np.array([
    -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.25, -1.0, -0.75, -0.5,
    -0.25, 0.0, 0.25, 0.5
])
ALLOWED_VVCRIT = np.array([0.0, 0.4])
ALLOWED_ALPHA = np.array([0.0])


def _init_tracks():
    """Allocate the cache array on first use."""
    global tracks
    if tracks is None:
        tracks = np.empty(
            (len(ALLOWED_MASS), len(ALLOWED_INITFEH),
             len(ALLOWED_VVCRIT), len(ALLOWED_ALPHA)),
            dtype=object,
        )
        tracks.fill(None)


def _mass_index(mstar):
    return int(np.argmin(np.abs(ALLOWED_MASS - mstar)))


def _feh_index(feh):
    return int(np.argmin(np.abs(ALLOWED_INITFEH - feh)))


def _vvcrit_index(vvcrit):
    matches = np.where(np.isclose(ALLOWED_VVCRIT, vvcrit))[0]
    if len(matches) == 0:
        raise ValueError(f"vvcrit ({vvcrit}) not allowed")
    return int(matches[0])


def _alpha_index(alpha):
    matches = np.where(np.isclose(ALLOWED_ALPHA, alpha))[0]
    if len(matches) == 0:
        raise ValueError(f"alpha ({alpha}) not allowed")
    return int(matches[0])


def _get_track_tuple_cached(mass_idx, feh_idx, vvcrit_idx, alpha_idx):
    """Return (ages, rstars, teffs, fehs, ageweights) for a cached grid point."""
    _init_tracks()
    global tracks
    if tracks[mass_idx, feh_idx, vvcrit_idx, alpha_idx] is None:
        tracks[mass_idx, feh_idx, vvcrit_idx, alpha_idx] = _load_track_tuple(
            ALLOWED_MASS[mass_idx],
            ALLOWED_INITFEH[feh_idx],
            ALLOWED_VVCRIT[vvcrit_idx],
            ALLOWED_ALPHA[alpha_idx],
        )
    return tracks[mass_idx, feh_idx, vvcrit_idx, alpha_idx]


def _load_track_tuple(mstar, feh, vvcrit, alpha):
    """Return (ages, rstars, teffs, fehs, ageweights) arrays for a grid point."""
    track = readeep(mstar, feh, vvcrit=vvcrit, alpha=alpha)
    if isinstance(track, dict):
        track_arr = track['track']
    else:
        track_arr = np.asarray(track)
    if track_arr.ndim != 2 or (5 not in track_arr.shape):
        raise ValueError("Unexpected track array shape")
    if track_arr.shape[0] != 5:
        track_arr = track_arr.T
    if track_arr.shape[0] != 5:
        raise ValueError("Track array has invalid orientation")
    ages, rstars, teffs, fehs, ageweights = track_arr
    return ages.astype(float), rstars.astype(float), teffs.astype(float), fehs.astype(float), ageweights.astype(float)

def massradius_mist(eep, mstar, feh, teff, rstar, vvcrit=None, alpha=None, span=1, epsname=None, debug=False,
                    gravitysun=27420.011, fitage=False, ageweight=None, verbose=False, logname=None,
                    trackfile=None, allowold=False, tefffloor=None, fehfloor=None, rstarfloor=None,
                    agefloor=None, pngname=None, range=None):
    '''
    ;+
; NAME:
;   massradius_mist
;
; PURPOSE: 
;   Interpolate the MIST stellar evolutionary models to derive Teff
;   and Rstar from mass, metallicity, and age. Intended to be a drop in
;   replacement for the Yonsie Yale model interpolation
;   (massradius_yy3.pro).
;
; CALLING SEQUENCE:
;   chi2 = massradius_mist(eep, mstar, feh, teff, rstar, $
;                          VVCRIT=vvcrit, ALPHA=alpha, SPAN=span,$
;                          MISTRSTAR=mistrstar, MISTTEFF=mistteff)
; INPUTS:
;
;    EEP    - Equivalent Evolutionary Phase (1–808, continuous float).
;             Primary fitted parameter replacing AGE.  AGE is derived
;             from the track at this EEP and is no longer fitted.
;    MSTAR  - The mass of the star, in m_sun
;    FEH    - The initial metallicity of the star [Fe/H]
;    RSTAR  - The radius you expect; used to calculate a chi^2
;    TEFF   - The Teff you expect; used to calculate a chi^2
;    
; OPTIONAL INPUTS:
;   VVCRIT    - The rotational velocity normalized by the critical
;               rotation speed. Must be 0.0d0 or 0.4d0 (default 0.0d0).
;   ALPHA     - The alpha abundance. Must be 0.0 (default 0.0). A
;               placeholder for future improvements to MIST models.
;   SPAN      - The interpolation is done at the closest value +/-
;               SPAN grid points in the evolutionary tracks in mass,
;               age, metallicity. The larger this number, the longer it
;               takes. Default=1. Change with care.
;   EPSNAME   - A string specifying the name of postscript file to plot
;               the evolutionary track. If not specified, no plot is
;               generated.
;
; OPTIONAL KEYWORDS:
;   DEBUG     - If set, will plot the teff and rstar over the MIST
;               Isochrone.
;
; OPTIONAL OUTPUTS:
;   MISTRSTAR - The rstar interpolated from the MIST models.
;   MISTTEFF  - The Teff interpolated from the MIST models.
;
; RESULT:
;   The chi^2 penalty due to the departure from the MIST models,
;   assuming 3% errors in the MIST model values.
;
; COMMON BLOCKS:
;   MIST_BLOCK:
;     Loading EEPs (model tracks) is very slow. This common block
;     allows us to store the tracks in memory between calls. The first
;     call will take ~3 seconds. Subsequent calls that use the same
;     EEP files take 1 ms.
;
; EXAMPLE: 
;   ;; penalize a model for straying from the MIST models 
;   chi2 += massradius_mist(mstar, feh, age, rstar=rstar, teff=teff)
;
; MODIFICATION HISTORY
; 
;  2018/01 -- Written, JDE
;-
    '''
    global tracks

    if tefffloor is None:
        tefffloor = -1
    if fehfloor is None:
        fehfloor = -1
    if rstarfloor is None:
        rstarfloor = -1
    if agefloor is None:
        agefloor = -1

    if not (ALLOWED_MASS.min() <= mstar <= ALLOWED_MASS.max()):
        if verbose:
            print(f"Mstar ({mstar}) is out of range [0.1, 300]", file=logname or None)
        return np.inf

    if not (ALLOWED_INITFEH.min() <= feh <= ALLOWED_INITFEH.max()):
        if verbose:
            print(f"initfeh ({feh}) is out of range [-4, 0.5]", file=logname or None)
        return np.inf

    try:
        massndx = _mass_index(mstar)
        fehndx = _feh_index(feh)
        vvcritndx = 0 if vvcrit is None else _vvcrit_index(vvcrit)
        alphandx = 0 if alpha is None else _alpha_index(alpha)
    except ValueError as exc:
        if verbose:
            print(str(exc), file=logname or None)
        return np.inf

    ages, rstars, teffs, fehs, ageweights_data = _get_track_tuple_cached(
        massndx, fehndx, vvcritndx, alphandx
    )

    # EEP is 1-indexed and continuous; convert to 0-based bracket indices.
    neep = len(ages)
    eep_lo = int(np.floor(eep)) - 1   # 0-based lower bracket
    eep_hi = eep_lo + 1                # 0-based upper bracket
    frac = eep - np.floor(eep)         # fractional part for linear interpolation

    if eep_lo < 0:
        if verbose:
            print(f"EEP ({eep}) is below minimum (1)", file=logname or None)
        return np.inf
    if eep_hi >= neep:
        if verbose:
            print(f"EEP ({eep}) is out of bounds for track with {neep} EEPs", file=logname or None)
        return np.inf

    # Interpolate all quantities directly from the EEP index (no searchsorted).
    mistage   = (1 - frac) * ages[eep_lo]   + frac * ages[eep_hi]
    mistrstar = (1 - frac) * rstars[eep_lo] + frac * rstars[eep_hi]
    mistteff  = (1 - frac) * teffs[eep_lo]  + frac * teffs[eep_hi]
    mistfeh   = (1 - frac) * fehs[eep_lo]   + frac * fehs[eep_hi]

    if mistage < 0 or (not allowold and mistage > 13.82):
        if verbose:
            print(f"Derived age ({mistage:.3f} Gyr) is out of range", file=logname or None)
        return np.inf

    percenterror = 0.03 - 0.025 * np.log10(mstar) + 0.045 * (np.log10(mstar))**2

    # chi2 penalises departures of the *fitted* (teff, rstar, feh) from the
    # MIST track prediction at this EEP.  Age is now a derived quantity, so
    # there is no chi2_age term.
    chi2_rstar = ((mistrstar - rstar) / (rstarfloor * mistrstar if rstarfloor > 0 else percenterror * mistrstar))**2
    chi2_teff  = ((mistteff  - teff)  / (tefffloor  * mistteff  if tefffloor  > 0 else percenterror * mistteff))**2
    chi2_feh   = ((mistfeh   - feh)   / (fehfloor                if fehfloor   > 0 else percenterror))**2

    chi2 = chi2_rstar + chi2_teff + chi2_feh

    if trackfile:
        _write_track_file(trackfile, teffs, rstars, ages)

    plot_target = pngname or epsname
    if debug and not plot_target:
        plot_target = Path.cwd() / "mist_track_debug.png"
    if plot_target:
        try:
            teffs_iso, rstars_iso, ages_iso, eeps_iso = _interpolate_track_for_plot(
                mstar, feh, vvcrit=vvcrit, alpha=alpha
            )
            if len(teffs_iso) > 0:
                _plot_mist_track(
                    teffs_iso,
                    rstars_iso,
                    ages_iso,
                    eeps_iso,
                    mstar,
                    feh,
                    age,
                    teff,
                    rstar,
                    gravitysun,
                    mistteff=mistteff,
                    mistrstar=mistrstar,
                    eep_best=eep,
                    outfile=str(plot_target),
                    range_vals=range,
                )
        except (ValueError, IndexError):
            pass
    return chi2


def _write_track_file(path, teffs, rstars, ages):
    path = Path(path)
    data = np.column_stack((teffs, rstars, ages))
    header = "teff[K] rstar[Rsun] age[Gyr]"
    np.savetxt(path, data, header=header)


def _interpolate_track_for_plot(mstar, feh, vvcrit=None, alpha=None):
    """
    Interpolate an evolutionary track at exact (mstar, feh) for all EEPs.

    Matches IDL massradius_mist.pro plotting logic: for each EEP from 1 to 808,
    trilinearly interpolate over (EEP, mass, [Fe/H]) using the 4 surrounding
    grid tracks (2 mass x 2 feh).

    Returns
    -------
    teffs_iso, rstars_iso, ages_iso, eeps : ndarray
        Interpolated Teff, Rstar, age, and EEP number at each valid point.
    """
    vvcritndx = 0 if vvcrit is None else _vvcrit_index(vvcrit)
    alphandx = 0 if alpha is None else _alpha_index(alpha)

    # Find bracketing mass indices (matches IDL logic)
    massndx = _mass_index(mstar)
    if mstar < ALLOWED_MASS[massndx]:
        minmassndx = max(massndx - 1, 0)
    else:
        minmassndx = min(massndx, len(ALLOWED_MASS) - 2)

    fehndx = _feh_index(feh)
    if feh < ALLOWED_INITFEH[fehndx]:
        minfehndx = max(fehndx - 1, 0)
    else:
        minfehndx = min(fehndx, len(ALLOWED_INITFEH) - 2)

    mstarbox = ALLOWED_MASS[minmassndx:minmassndx + 2]
    fehbox = ALLOWED_INITFEH[minfehndx:minfehndx + 2]

    y_mass = (mstar - mstarbox[0]) / (mstarbox[1] - mstarbox[0]) if mstarbox[1] != mstarbox[0] else 0.0
    z_feh = (feh - fehbox[0]) / (fehbox[1] - fehbox[0]) if fehbox[1] != fehbox[0] else 0.0

    # Load the 4 corner tracks
    corner_tracks = []
    for i in range(2):
        for j in range(2):
            ages_c, rstars_c, teffs_c, _, _ = _get_track_tuple_cached(
                minmassndx + i, minfehndx + j, vvcritndx, alphandx
            )
            corner_tracks.append((ages_c, rstars_c, teffs_c))

    # Sweep EEP from 1 to 808 (IDL convention)
    max_eep = 808
    teffs_iso = np.full(max_eep, np.nan)
    rstars_iso = np.full(max_eep, np.nan)
    ages_iso = np.full(max_eep, np.nan)

    for eep_val in range(1, max_eep + 1):
        eepndx = eep_val - 1  # 0-based index into track arrays

        # Build 2x2x2 cube for trilinear interpolation (eep x mass x feh)
        vals_ok = True
        all_ages = np.zeros((2, 2, 2))
        all_rstars = np.zeros((2, 2, 2))
        all_teffs = np.zeros((2, 2, 2))

        idx = 0
        for i in range(2):
            for j in range(2):
                ages_c, rstars_c, teffs_c = corner_tracks[idx]
                idx += 1
                neep_c = len(ages_c)
                mineepndx = min(eepndx, neep_c - 2)
                if mineepndx < 0 or mineepndx + 1 >= neep_c:
                    vals_ok = False
                    break
                all_ages[:, i, j] = ages_c[mineepndx:mineepndx + 2]
                all_rstars[:, i, j] = rstars_c[mineepndx:mineepndx + 2]
                all_teffs[:, i, j] = teffs_c[mineepndx:mineepndx + 2]
            if not vals_ok:
                break

        if not vals_ok:
            continue

        eepbox_lo = mineepndx + 1  # 1-based
        eepbox_hi = mineepndx + 2
        x_eep = (eep_val - eepbox_lo) / (eepbox_hi - eepbox_lo) if eepbox_hi != eepbox_lo else 0.0

        # Trilinear interpolation (matches IDL interpolate)
        def _trilinear(cube, x, y, z):
            c00 = cube[0, 0, 0] * (1 - x) + cube[1, 0, 0] * x
            c01 = cube[0, 0, 1] * (1 - x) + cube[1, 0, 1] * x
            c10 = cube[0, 1, 0] * (1 - x) + cube[1, 1, 0] * x
            c11 = cube[0, 1, 1] * (1 - x) + cube[1, 1, 1] * x
            c0 = c00 * (1 - y) + c10 * y
            c1 = c01 * (1 - y) + c11 * y
            return c0 * (1 - z) + c1 * z

        teffs_iso[eep_val - 1] = _trilinear(all_teffs, x_eep, y_mass, z_feh)
        rstars_iso[eep_val - 1] = _trilinear(all_rstars, x_eep, y_mass, z_feh)
        ages_iso[eep_val - 1] = _trilinear(all_ages, x_eep, y_mass, z_feh)

    eeps = np.arange(1, max_eep + 1, dtype=float)
    good = np.isfinite(teffs_iso) & np.isfinite(rstars_iso) & np.isfinite(ages_iso)
    return teffs_iso[good], rstars_iso[good], ages_iso[good], eeps[good]


def _plot_mist_track(teffs, rstars, ages, eeps, mstar, feh, age, teff, rstar,
                     gravitysun, mistteff, mistrstar, eep_best=None,
                     outfile=None, range_vals=None):
    """
    Plot the MIST HR diagram (Teff vs logg).

    Matches IDL massradius_mist.pro plotting:
    - Black line: interpolated evolutionary track at exact (mstar, feh)
    - Black dot: best-fit (input) point
    - Red asterisk: MIST model point (interpolated at the best-fit EEP)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    teffs = np.asarray(teffs, dtype=float)
    rstars = np.asarray(rstars, dtype=float)
    ages = np.asarray(ages, dtype=float)
    eeps = np.asarray(eeps, dtype=float)

    safe_rstars = np.clip(rstars, 1e-6, None)
    track_logg = np.log10((mstar / (safe_rstars**2)) * gravitysun)
    finite = np.isfinite(teffs) & np.isfinite(track_logg)
    if not np.any(finite):
        return

    # IDL line 348-350: filter by EEP >= 202 (ZAMS) unless best-fit EEP is
    # near the end of the track, and logg in [3, 5]
    ZAMS_EEP = 202
    if eep_best is not None and (eep_best + 3) > np.max(eeps):
        min_eep = 1
    else:
        min_eep = ZAMS_EEP
    # IDL: eepplot >= min([mineep, eep])
    eep_cutoff = min(min_eep, eep_best) if eep_best is not None else min_eep
    use = finite & (track_logg > 3) & (track_logg < 5) & (eeps >= eep_cutoff)
    if not np.any(use):
        use = finite

    teff_vals = teffs[use]
    logg_vals = track_logg[use]

    logg_best = np.log10((mstar / (max(rstar, 1e-6)**2)) * gravitysun)
    logg_mist = np.log10((mstar / (max(mistrstar, 1e-6)**2)) * gravitysun)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(teff_vals, logg_vals, color="black", lw=1.0)

    # Black dot: input (best-fit) point (IDL: psym=8, black)
    ax.plot([teff], [logg_best], 'o', color='black', ms=6, zorder=5)
    # Red asterisk: MIST interpolated point (IDL: psym=2, red)
    ax.plot([mistteff], [logg_mist], '*', color='red', ms=12, zorder=6)

    if range_vals and len(range_vals) >= 4:
        xmin, xmax, ymin, ymax = range_vals[:4]
    else:
        # IDL convention (line 353-354)
        xmin = max(np.max(teff_vals), teff * 1.1, teff * 0.9)
        xmax = min(np.min(teff_vals), teff * 0.9, teff * 1.1)
        xmin = int(np.ceil(xmin / 100)) * 100
        xmax = int(np.floor(xmax / 100)) * 100
        # IDL: ymax = min([loggplot,3,5,loggplottrack]), ymin = max(...)
        ymax = min(logg_best, 3.0, np.min(logg_vals))
        ymin = max(logg_best, 5.0, np.max(logg_vals))
        # IDL tick spacing: expand to 4 equally spaced ticks on 100 K boundaries
        spacing = int(np.ceil((xmin - xmax) / 3 / 100)) * 100
        if spacing > 0:
            while (xmin - xmax) / spacing != 3:
                xmin += 100
                xmax -= 100
                spacing = int(np.ceil((xmin - xmax) / 3 / 100)) * 100
                if xmin - xmax > 5000:
                    break

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel(r"$T_{\mathrm{eff}}$ (K)")
    ax.set_ylabel(r"$\log g_\star$ (cgs)")

    outfile = outfile or "mist_track.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mist_track(bestfit, outfile, vvcrit=None, alpha=None, range_vals=None):
    """
    Convenience wrapper to render a Teff-logg plot for a best-fit solution.

    Matches IDL: interpolates a full evolutionary track at the exact
    (mstar, feh) by trilinear interpolation over the MIST grid, then plots.

    For multi-star systems, overlays each star's track on the same axes.
    """
    nstars = getattr(bestfit, 'nstars', 1)

    try:
        if nstars == 1:
            _plot_single_star_track(bestfit, '', outfile, vvcrit, alpha, range_vals)
        else:
            # Multi-star: overlay all tracks on a single plot
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
            gravitysun = 27420.011
            colors = ['blue', 'red', 'green', 'purple', 'orange']

            for i in range(nstars):
                suffix = f'_{i}'
                mstar = bestfit[f'mstar{suffix}']
                feh = bestfit[f'feh{suffix}']
                age = bestfit[f'age{suffix}']
                teff = bestfit[f'teff{suffix}']
                rstar = bestfit[f'rstar{suffix}']
                label = f'Star {chr(65+i)}'

                if not (ALLOWED_MASS.min() <= mstar <= ALLOWED_MASS.max()):
                    continue
                if not (ALLOWED_INITFEH.min() <= feh <= ALLOWED_INITFEH.max()):
                    continue

                try:
                    teffs_iso, rstars_iso, ages_iso, eeps_iso = _interpolate_track_for_plot(
                        mstar, feh, vvcrit=vvcrit, alpha=alpha
                    )
                except (ValueError, IndexError):
                    continue

                if len(teffs_iso) == 0:
                    continue

                loggs_iso = np.log10(gravitysun * mstar / rstars_iso**2)

                # Track line
                color = colors[i % len(colors)]
                ax.plot(teffs_iso, loggs_iso, '-', color=color, lw=1, label=f'{label} track')

                # Best-fit point
                logg_fit = np.log10(gravitysun * mstar / rstar**2)
                ax.plot(teff, logg_fit, 'o', color=color, ms=8, mfc=color,
                        label=f'{label} ({mstar:.2f} M$_\\odot$, {age:.1f} Gyr)')

                # MIST model point at best-fit age
                eep_idx = np.searchsorted(ages_iso, age)
                eep_idx = np.clip(eep_idx, 1, len(ages_iso) - 1)
                x = (age - ages_iso[eep_idx - 1]) / (ages_iso[eep_idx] - ages_iso[eep_idx - 1]) \
                    if ages_iso[eep_idx] != ages_iso[eep_idx - 1] else 0.0
                mistteff = (1 - x) * teffs_iso[eep_idx - 1] + x * teffs_iso[eep_idx]
                mistrstar = (1 - x) * rstars_iso[eep_idx - 1] + x * rstars_iso[eep_idx]
                mistlogg = np.log10(gravitysun * mstar / mistrstar**2)
                ax.plot(mistteff, mistlogg, 's', color=color, ms=6, mfc='none', mew=1.5)

            ax.set_xlabel(r'$T_{\rm eff}$ (K)')
            ax.set_ylabel(r'$\log g$ (cgs)')
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.legend(fontsize=8, loc='best')

            fig.savefig(outfile, dpi=150, bbox_inches='tight')
            plt.close(fig)
    except (FileNotFoundError, TypeError) as e:
        print(f"Warning: Skipping MIST track plot ({e}). "
              "MIST EEP track files may not be installed.")


def _plot_single_star_track(bestfit, suffix, outfile, vvcrit, alpha, range_vals):
    """Plot MIST track for a single star (original logic)."""
    mstar = bestfit[f'mstar{suffix}'] if suffix else bestfit['mstar']
    feh = bestfit[f'feh{suffix}'] if suffix else bestfit['feh']
    age = bestfit[f'age{suffix}'] if suffix else bestfit['age']
    teff = bestfit[f'teff{suffix}'] if suffix else bestfit['teff']
    rstar = bestfit[f'rstar{suffix}'] if suffix else bestfit['rstar']
    gravitysun = 27420.011

    if not (ALLOWED_MASS.min() <= mstar <= ALLOWED_MASS.max()):
        return
    if not (ALLOWED_INITFEH.min() <= feh <= ALLOWED_INITFEH.max()):
        return

    try:
        teffs_iso, rstars_iso, ages_iso, eeps_iso = _interpolate_track_for_plot(
            mstar, feh, vvcrit=vvcrit, alpha=alpha
        )
    except (ValueError, IndexError):
        return

    if len(teffs_iso) == 0:
        return

    # Find the MIST model point at the best-fit EEP
    eep = np.searchsorted(ages_iso, age)
    eep = np.clip(eep, 1, len(ages_iso) - 1)
    x = (age - ages_iso[eep - 1]) / (ages_iso[eep] - ages_iso[eep - 1]) \
        if ages_iso[eep] != ages_iso[eep - 1] else 0.0
    mistteff = (1 - x) * teffs_iso[eep - 1] + x * teffs_iso[eep]
    mistrstar = (1 - x) * rstars_iso[eep - 1] + x * rstars_iso[eep]
    eep_best = (1 - x) * eeps_iso[eep - 1] + x * eeps_iso[eep]

    _plot_mist_track(
        teffs_iso,
        rstars_iso,
        ages_iso,
        eeps_iso,
        mstar,
        feh,
        age,
        teff,
        rstar,
        gravitysun,
        mistteff=mistteff,
        mistrstar=mistrstar,
        eep_best=eep_best,
        outfile=outfile,
        range_vals=range_vals,
    )
