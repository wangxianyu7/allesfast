#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 01:03:21 2018

@author:
Dr. Maximilian N. Günther
European Space Agency (ESA)
European Space Research and Technology Centre (ESTEC)
Keplerlaan 1, 2201 AZ Noordwijk, The Netherlands
Email: maximilian.guenther@esa.int
GitHub: mnguenther
Twitter: m_n_guenther
Web: www.mnguenther.com
"""
# /N/u/xwa5/Quartz/.conda/envs/normal/lib/python3.8/site-packages/pytransit/
from __future__ import print_function, division, absolute_import

#::: modules
import numpy as np
import os
import emcee
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
#solves python>=3.8 issues, see https://stackoverflow.com/questions/60518386/error-with-module-multiprocessing-under-python3-8
from multiprocessing import Pool
from contextlib import closing
from time import time as timer
#::: warnings
import warnings
# warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# warnings.filterwarnings('ignore', category=np.RankWarning)

#::: allesfast modules
from .. import config
from ..computer import update_params, calculate_lnlike_total
from ..general_output import logprint
from .mcmc_output import print_autocorr
from .demcpt import DEMCPTSampler




###############################################################################
#::: MCMC log likelihood
###############################################################################
def mcmc_lnlike(theta):

    params = update_params(theta)
    lnlike = calculate_lnlike_total(params)

#    lnlike = 0
#
#    for inst in config.BASEMENT.settings['inst_phot']:
#        lnlike += calculate_lnlike(params, inst, 'flux')
#
#    for inst in config.BASEMENT.settings['inst_rv']:
#        lnlike += calculate_lnlike(params, inst, 'rv')
#
#    if np.isnan(lnlike) or np.isinf(lnlike):
#        lnlike = -np.inf

    return lnlike



###############################################################################
#::: MCMC log prior
###############################################################################
def mcmc_lnprior(theta):
    '''
    bounds has to be list of len(theta), containing tuples of form
    ('none'), ('uniform', lower bound, upper bound), or ('normal', mean, std)
    '''
    lnp = 0.

    for th, b in zip(theta, config.BASEMENT.bounds):
        if b[0] == 'uniform':
            if not (b[1] <= th <= b[2]):
                return -np.inf
        elif b[0] == 'normal':
            lnp += np.log( 1./(np.sqrt(2*np.pi) * b[2]) * np.exp( - (th - b[1])**2 / (2.*b[2]**2) ) )
        elif b[0] == 'trunc_normal':
            if not (b[1] <= th <= b[2]):
                return -np.inf
            lnp += np.log( 1./(np.sqrt(2*np.pi) * b[4]) * np.exp( - (th - b[3])**2 / (2.*b[4]**2) ) )
        else:
            raise ValueError('Bounds have to be "uniform" or "normal". Input from "params.csv" was "'+b[0]+'".')
    return lnp



###############################################################################
#::: MCMC log probability
###############################################################################
def mcmc_lnprob(theta):
    '''
    has to be top-level for  for multiprocessing pickle
    '''
    lp = mcmc_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
#        try:
        ln = mcmc_lnlike(theta)
        # print(lp + ln)
        return lp + ln
#        except:
#            return -np.inf



###########################################################################
#::: EXOFASTv2-style MCMC scale (Δχ²=1 per parameter)
###########################################################################
def getmcmcscale(bestpars, lnprob_func, seedscale=None, maxiter=10000,
                 debug=False):
    """Determine per-parameter MCMC step scale via Δχ²=1 binary search.

    Faithful Python port of EXOFASTv2's ``exofast_getmcmcscale.pro``.

    For each parameter, takes a positive and negative step from *bestpars*,
    doubling the step until Δχ² > 1, then binary-searches for the step that
    gives Δχ² ≈ 1.  The average of the positive and negative excursions is
    returned as the optimal MCMC scale.

    Parameters
    ----------
    bestpars : 1-D array (nfit,)
        Best-fit parameter vector.
    lnprob_func : callable
        ``theta -> float`` returning log-posterior.
    seedscale : 1-D array (nfit,) or None
        Initial small step per parameter.  Default ``1e-3`` for each.
    maxiter : int
        Max iterations per direction per parameter.
    debug : bool
        Print progress.

    Returns
    -------
    scale : 1-D array (nfit,)
        Per-parameter step scale (≈ 1σ).
    """
    nfit = len(bestpars)
    if seedscale is None:
        seedscale = np.full(nfit, 1e-3)

    bestchi2 = -2.0 * lnprob_func(bestpars)
    if not np.isfinite(bestchi2):
        raise RuntimeError('getmcmcscale: bestpars has non-finite chi²; '
                           'refine starting values.')

    origbestchi2 = bestchi2
    bestpars = bestpars.copy()

    # mcmcscale[i, 0] = positive direction, [i, 1] = negative direction
    mcmcscale = np.column_stack([seedscale.copy(), seedscale.copy()])

    for i in range(nfit):
        for j in range(2):  # 0 = positive, 1 = negative
            testpars = bestpars.copy()
            minstep = 0.0
            maxstep = 0.0
            niter = 0
            bestdeltachi2 = np.inf
            bestscale = 0.0

            while True:
                # Check for infinite step
                if (not np.isfinite(bestpars[i] + mcmcscale[i, j]) or
                        not np.isfinite(bestpars[i] - mcmcscale[i, j])):
                    mcmcscale[i, j] = np.nan
                    break

                # Apply offset
                if j == 0:
                    testpars[i] = bestpars[i] + mcmcscale[i, j]
                else:
                    testpars[i] = bestpars[i] - mcmcscale[i, j]

                chi2 = -2.0 * lnprob_func(testpars)
                chi2changed = False

                if (chi2 - bestchi2) >= 1.0:
                    # Too large → set max, bisect
                    maxstep = mcmcscale[i, j]
                    mcmcscale[i, j] = (maxstep + minstep) / 2.0
                elif (chi2 - bestchi2) >= 0.0:
                    # Too small → set min, double or bisect
                    minstep = mcmcscale[i, j]
                    if maxstep == 0.0:
                        mcmcscale[i, j] *= 2.0
                    else:
                        mcmcscale[i, j] = (maxstep + minstep) / 2.0
                else:
                    # Found a better chi²!
                    if debug:
                        logprint(f'  getmcmcscale: better chi² found for '
                                 f'param {i}: {chi2:.6f} < {bestchi2:.6f}')
                    bestpars = testpars.copy()
                    mcmcscale[i, j] *= 2.0
                    bestchi2 = chi2
                    niter = 0
                    chi2changed = True

                deltachi2 = chi2 - bestchi2
                # Track best Δχ² closest to 1
                if abs(deltachi2 - 1.0) < abs(bestdeltachi2 - 1.0):
                    bestdeltachi2 = deltachi2
                    bestscale = mcmcscale[i, j]

                # Convergence: binary search exhausted or maxiter
                if abs(minstep - maxstep) < 1e-12 or niter > maxiter:
                    # Check if near a boundary
                    testpars[i] = bestpars[i] - 2.0 * mcmcscale[i, j]
                    lowchi2 = -2.0 * lnprob_func(testpars)
                    testpars[i] = bestpars[i] + 2.0 * mcmcscale[i, j]
                    hichi2 = -2.0 * lnprob_func(testpars)

                    if (not np.isfinite(chi2) or not np.isfinite(lowchi2)
                            or not np.isfinite(hichi2)):
                        # Hit a boundary
                        if bestscale != 0.0:
                            mcmcscale[i, j] = bestscale / 100.0
                        else:
                            mcmcscale[i, j] = maxstep / 100.0
                        break
                    elif not chi2changed:
                        if abs(bestdeltachi2 - 1.0) < 0.75:
                            mcmcscale[i, j] = bestscale
                        else:
                            if bestdeltachi2 == 0.0:
                                mcmcscale[i, j] = bestscale / 100.0
                            else:
                                mcmcscale[i, j] = bestscale / bestdeltachi2 / 10.0
                            if debug:
                                logprint(
                                    f'  getmcmcscale: param {i} rough surface; '
                                    f'Δχ²={bestdeltachi2:.4f}, '
                                    f'scale={mcmcscale[i, j]:.6e}')
                    break

                niter += 1

                if debug and niter % 500 == 0:
                    direction = 'hi' if j == 0 else 'lo'
                    logprint(f'  getmcmcscale: param {i} ({direction}) '
                             f'iter={niter} min={minstep:.6e} max={maxstep:.6e} '
                             f'Δχ²={deltachi2:.6f}')

                # Check convergence
                if abs(chi2 - bestchi2 - 1.0) < 1e-8:
                    break

    # Replace undefined scales with the other direction
    for i in range(nfit):
        if not np.isfinite(mcmcscale[i, 0]) or mcmcscale[i, 0] == 0:
            mcmcscale[i, 0] = mcmcscale[i, 1]
        if not np.isfinite(mcmcscale[i, 1]) or mcmcscale[i, 1] == 0:
            mcmcscale[i, 1] = mcmcscale[i, 0]

    # Check for completely undefined parameters
    bad = ~np.isfinite(mcmcscale) | (mcmcscale == 0)
    if np.any(bad):
        bad_params = np.where(bad.any(axis=1))[0]
        logprint(f'  WARNING: getmcmcscale failed for params {bad_params}; '
                 f'using 1%% of |bestpars| as fallback')
        for idx in bad_params:
            fallback = abs(bestpars[idx]) * 0.01
            if fallback == 0:
                fallback = 1e-8
            mcmcscale[idx, :] = fallback

    # Return average of positive and negative excursions
    return mcmcscale.sum(axis=1) / 2.0


###########################################################################
#::: Automatic burn-in detection (EXOFASTv2 getburnndx logic)
###########################################################################
def get_burnndx(log_prob):
    """
    Determine the burn-in index from the log-probability chain.

    Delegates to the EXOFASTv2-faithful ``_getburnndx`` in demcpt.py,
    which also identifies good chains.  This wrapper returns only the
    burn-in index for backward compatibility.

    Parameters
    ----------
    log_prob : ndarray, shape (nsteps, nwalkers)
        Log-probability at each step for each walker.

    Returns
    -------
    burnndx : int
        Index into the stored-step array.  Discard steps 0..burnndx-1.
    """
    from .demcpt import _getburnndx
    chi2 = -2.0 * log_prob
    burnndx, _good = _getburnndx(chi2)
    return burnndx


###########################################################################
#::: DE pre-optimization
###########################################################################
def run_de_optimization(s):
    """Run Differential Evolution to find a good starting point for MCMC.

    Controlled by settings keys:
      de_ngen  : number of DE generations (default 0 = skip)
      de_npop  : population size (default max(5*ndim, mcmc_nwalkers))

    Returns (best_theta, population) array shaped (npop, ndim), or None.
    """
    try:
        from pytransit.utils.de import DiffEvol
    except ImportError:
        logprint("WARNING: pytransit not found – skipping DE pre-optimization.")
        return None

    ngen = int(s.get('de_ngen', 0))
    if ngen <= 0:
        return None

    # --- resume: skip DE if results already exist ---
    pop_file  = os.path.join(config.BASEMENT.outdir, 'optimized_population.csv')
    best_file = os.path.join(config.BASEMENT.outdir, 'optimized_best.csv')
    if os.path.exists(pop_file) and os.path.exists(best_file):
        import pandas as pd
        logprint("\nFound existing DE results – skipping DE optimisation.")
        logprint(f"  Loading: {pop_file}")
        df  = pd.read_csv(pop_file)
        pop = df[config.BASEMENT.fitkeys].values
        # optimized_best.csv is saved in params.csv format (name,value,...)
        best_map = {}
        with open(best_file, 'r') as fh:
            for line in fh:
                stripped = line.strip()
                if stripped == '' or stripped.startswith('#'):
                    continue
                parts = stripped.split(',', 2)
                if len(parts) >= 2:
                    best_map[parts[0]] = float(parts[1])
        best = np.array([best_map[k] for k in config.BASEMENT.fitkeys])
        return best, pop

    ndim = config.BASEMENT.ndim
    npop = int(s.get('de_npop', max(5 * ndim, s['mcmc_nwalkers'])))

    logprint(f"\nRunning DE pre-optimization ({npop} population, {ngen} generations)...")
    logprint('--------------------------')

    start_best_lnprob = mcmc_lnprob(config.BASEMENT.theta_0)
    logprint(f"  Starting lnprob: {start_best_lnprob:.4f}")
    if not np.isfinite(start_best_lnprob):
        logprint('\n[DEBUG] theta_0 has lnprob = -inf. Running diagnostics...')
        _test_params = update_params(config.BASEMENT.theta_0)
        calculate_lnlike_total(_test_params, debug=True)
        logprint('[DEBUG] Parameter values for theta_0:')
        for _k, _v in zip(config.BASEMENT.fitkeys, config.BASEMENT.theta_0):
            logprint(f'  {_k} = {_v}')
        logprint('')
    # Build [npar, 2] bounds array for DiffEvol
    de_bounds = []
    for b in config.BASEMENT.bounds:
        if b[0] == 'uniform':
            de_bounds.append([b[1], b[2]])
        elif b[0] == 'normal':
            de_bounds.append([b[1] - 5. * b[2], b[1] + 5. * b[2]])
        elif b[0] == 'trunc_normal':
            de_bounds.append([b[1], b[2]])
        else:
            raise ValueError(f'Unknown bound type for DE: {b[0]}')
    de_bounds = np.array(de_bounds)

    # # --- freeze sed_errscale during DE (fix to initial value) ---
    # # Prevents errscale from inflating and effectively disabling the SED
    # # constraint.  The parameter is freed again during MCMC.
    # _errscale_freeze = {}  # {dim_index: frozen_value}
    # for i, k in enumerate(config.BASEMENT.fitkeys):
    #     if 'sed_errscale' in k:
    #         _errscale_freeze[i] = config.BASEMENT.theta_0[i]
    #         logprint(f"  Freezing {k} = {_errscale_freeze[i]} during DE")

    # --- multiprocessing pool (reuse settings from MCMC) ---
    pool = None
    if s.get('multiprocess', False):
        cores = s.get('multiprocess_cores', 1)
        nworkers = (multiprocessing.cpu_count()
                    if str(cores).lower() == 'all' else int(cores))
        logprint(f"  Using {nworkers} CPUs for DE.")
        pool = Pool(processes=nworkers)

    try:
        de_dlnprob = float(s.get('de_dlnprob', 0.01))   # early-stop threshold
        de_nconv   = int(s.get('de_nconv', 6))         # consecutive converged generations required
        de = DiffEvol(mcmc_lnprob, de_bounds, npop=npop, maximize=True,
                      pool=pool, min_ptp=0)            # disable internal break; we check externally
        # Seed member 0 with theta_0 so the known starting point is never lost
        de._population[0] = np.clip(config.BASEMENT.theta_0,
                                    de_bounds[:, 0], de_bounds[:, 1])

        # Narrow the initial DE population around theta_0 using prior-width
        # based scale (same logic as Amoeba), so individuals start in a
        # sensible region instead of spanning the full prior bounds.
        ndim = len(config.BASEMENT.theta_0)
        _init_scale = np.empty(ndim)
        for k, b in enumerate(config.BASEMENT.bounds):
            if b[0] == 'uniform':
                _init_scale[k] = (b[2] - b[1]) / 10.0
            else:  # normal or trunc_normal
                _init_scale[k] = 3.0 * b[2]
        _init_scale = np.abs(_init_scale)
        _init_scale[_init_scale == 0] = 1e-3

        for idx in range(ndim):
            center = config.BASEMENT.theta_0[idx]
            lo = max(de_bounds[idx, 0], center - _init_scale[idx])
            hi = min(de_bounds[idx, 1], center + _init_scale[idx])
            de._population[1:, idx] = np.random.uniform(lo, hi, npop - 1)

        # # Pin frozen errscale in initial population
        # for _fi, _fv in _errscale_freeze.items():
        #     de._population[:, _fi] = _fv

        # --- iterate generation-by-generation so tqdm can track progress ---
        show_progress = s.get('print_progress', True)
        bar = None
        if show_progress:
            from tqdm import tqdm
            bar = tqdm(total=ngen, desc='DE', unit='gen')
        deadline = s.get('_deadline', None)
        ngen_done = 0
        timed_out = False
        converged = False
        _conv_count = 0
        for _, best_fit in de(ngen):
            # # Reset frozen sed_errscale after each DE generation
            # for _fi, _fv in _errscale_freeze.items():
            #     de._population[:, _fi] = _fv
            ngen_done += 1
            _fit = de._fitness
            _fin = _fit[np.isfinite(_fit)]
            _dlnp = np.ptp(_fin) if len(_fin) > 1 else np.inf
            if bar is not None:
                _n_inf = len(_fit) - len(_fin)
                _postfix = dict(lnprob=f'{-best_fit:.4f}', dlnp=f'{_dlnp:.4f}',
                                conv=f'{_conv_count}/{de_nconv}')
                if _n_inf > 0:
                    _postfix['n_inf'] = _n_inf
                bar.set_postfix(**_postfix)
                bar.update(1)
            if _dlnp < de_dlnprob:
                _conv_count += 1
                if _conv_count >= de_nconv:
                    converged = True
                    break
            else:
                _conv_count = 0
            if deadline is not None and timer() >= deadline:
                timed_out = True
                break
        if bar is not None:
            bar.close()
        if timed_out:
            logprint(f"  DE stopped early at generation {ngen_done}/{ngen}: time limit reached.")
        elif converged:
            logprint(f"  DE converged at generation {ngen_done}/{ngen} "
                     f"(Δlnprob < {de_dlnprob} for {de_nconv} consecutive generations)")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    best = de.minimum_location          # best parameter vector
    best_lnprob = -de.minimum_value     # DiffEvol stores -lnprob internally
    n_inf = np.sum(~np.isfinite(de._fitness))
    delta_lnprob = np.ptp(de._fitness[np.isfinite(de._fitness)]) if n_inf < len(de._fitness) else np.inf
    logprint(f"  DE best lnprob : {best_lnprob:.4f}")
    logprint(f"  DE Δlnprob     : {delta_lnprob:.4f}  (population spread; smaller → more converged)")
    if n_inf > 0:
        logprint(f"  WARNING: {n_inf}/{len(de._fitness)} DE members have lnprob=-inf "
                 f"(stuck in invalid region; check jittervar/variance bounds)")

    _save_de_results(de)

    return best, de.population.copy()


def _save_de_results(de):
    """Save DE population to CSV, generate fit plots, and optionally a corner plot."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import arviz as az
    import seaborn as sns
    sns.set(context='paper', style='ticks', palette='deep',
            font='sans-serif', font_scale=1.5, color_codes=True)
    sns.set_style({'xtick.direction': 'in', 'ytick.direction': 'in'})

    outdir   = config.BASEMENT.outdir
    fitkeys  = config.BASEMENT.fitkeys
    pop      = de.population.copy()          # (npop, ndim)
    lnprobs  = -de._fitness.copy()           # stored as -lnprob

    # --- population CSV  (easy to reload: pd.read_csv) ---
    df = pd.DataFrame(pop, columns=fitkeys)
    df['lnprob'] = lnprobs
    pop_file = os.path.join(outdir, 'optimized_population.csv')
    df.to_csv(pop_file, index=False)
    logprint(f"  Saved: {pop_file}")

    # --- best-params CSV ---
    best_idx  = int(np.argmax(lnprobs))
    best_file = os.path.join(outdir, 'optimized_best.csv')
    _save_optimized_best(dict(zip(fitkeys, pop[best_idx])), best_file)
    logprint(f"  Saved: {best_file}")

    # --- fit diagnostic plots at best DE solution ---
    _plot_de_fit(pop[best_idx])

    # --- model data files at best DE solution ---
    try:
        from ..general_output import save_modelfiles
        save_modelfiles(pop[best_idx][np.newaxis, :], 'optimized')
    except Exception as e:
        logprint(f"  WARNING: save_modelfiles failed – {e}")

    if config.BASEMENT.settings.get('cornerplot', False):
        # --- corner plot of the final population (arviz hexbin) ---
        try:
            ndim = len(fitkeys)
            az.rcParams['plot.max_subplots'] = max(ndim * ndim + 1, 40)
            var_dict = {key: pop[np.newaxis, :, i] for i, key in enumerate(fitkeys)}
            idata = az.from_dict(posterior=var_dict)
            axs = az.plot_pair(
                idata,
                kind='hexbin',
                marginals=True,
                gridsize=30,
                hexbin_kwargs={'cmap': 'Blues'},
                marginal_kwargs={'color': '#1f77b4'},
                point_estimate='median',
            )
            fig = axs.ravel()[0].get_figure() if hasattr(axs, 'ravel') else plt.gcf()
            fig.suptitle('DE pre-optimisation population\n'
                         f'(best lnprob = {lnprobs[best_idx]:.4f}, '
                         f'Δlnprob = {np.ptp(lnprobs):.4f})',
                         y=1.01, fontsize=10)
            plot_file = os.path.join(outdir, 'optimized_corner.pdf')
            fig.savefig(plot_file, bbox_inches='tight')
            plt.close(fig)
            logprint(f"  Saved: {plot_file}")
        except Exception as e:
            logprint(f"  WARNING: DE corner plot failed – {e}")


def _save_optimized_best(best_map, dest_path):
    """
    Write optimized best parameters as a params.csv-compatible file.

    Copies params.csv line by line, replacing the value field for any
    parameter in best_map (fit parameters). Fixed rows are preserved
    unchanged.  The result can be copied directly over params.csv to
    warm-start the next run.

    Parameters
    ----------
    best_map : dict  {param_name: best_value}
    dest_path : str  output file path
    """
    src = os.path.join(config.BASEMENT.datadir, 'params.csv')
    out_lines = []
    with open(src, 'r') as fh:
        for line in fh:
            stripped = line.rstrip('\n')
            # preserve comment / empty lines as-is
            if stripped.lstrip().startswith('#') or stripped.strip() == '':
                out_lines.append(stripped)
                continue
            # split on first comma only to get the parameter name
            parts = stripped.split(',', 2)
            if len(parts) >= 2 and parts[0] in best_map:
                parts[1] = f'{best_map[parts[0]]:.10e}'
                out_lines.append(','.join(parts))
            else:
                out_lines.append(stripped)
    with open(dest_path, 'w') as fh:
        fh.write('\n'.join(out_lines) + '\n')


def _plot_de_fit(best_theta):
    """Generate model fit plots at the best DE solution (light curves, RVs, SED, MIST)."""
    import matplotlib.pyplot as plt
    from ..general_output import afplot, afplot_per_transit, get_params_from_samples
    from ..star import make_sed_plot, make_mist_plot

    outdir  = config.BASEMENT.outdir
    samples = best_theta[np.newaxis, :]          # shape (1, ndim)

    for companion in config.BASEMENT.settings['companions_all']:
        try:
            fig, _ = afplot(samples, companion)
            if fig is not None:
                path = os.path.join(outdir, f'optimized_{companion}.pdf')
                fig.savefig(path, bbox_inches='tight')
                plt.close(fig)
                logprint(f"  Saved: {path}")
        except Exception as e:
            logprint(f"  WARNING: DE fit plot failed for {companion} – {e}")

    for companion in config.BASEMENT.settings['companions_phot']:
        for inst in config.BASEMENT.settings['inst_phot']:
            first_transit = 0
            while first_transit >= 0:
                try:
                    fig, _, last_transit, total_transits = afplot_per_transit(
                        samples, inst, companion,
                        kwargs_dict={'first_transit': first_transit},
                    )
                    path = os.path.join(outdir,
                        f'optimized_per_transit_{inst}_{companion}_{last_transit}th.pdf')
                    fig.savefig(path, bbox_inches='tight')
                    plt.close(fig)
                    logprint(f"  Saved: {path}")
                    if total_transits > 0 and last_transit < total_transits - 1:
                        first_transit = last_transit
                    else:
                        first_transit = -1
                except Exception:
                    first_transit = -1

    params_median, _, _ = get_params_from_samples(samples)
    _sed_file = config.BASEMENT.settings.get('sed_file', None)
    try:
        path = make_sed_plot(params_median, config.BASEMENT.datadir, outdir,
                             outfile='optimized_sed_fit.pdf', sed_file=_sed_file)
        if path:
            logprint(f"  Saved: {path}")
    except Exception as e:
        logprint(f"  WARNING: DE SED plot failed – {e}")
    try:
        path = make_mist_plot(params_median, outdir,
                              outfile='optimized_mist_track.pdf')
        if path:
            logprint(f"  Saved: {path}")
    except Exception as e:
        logprint(f"  WARNING: DE MIST plot failed – {e}")


###########################################################################
#::: Amoeba (Nelder-Mead) optimization
###########################################################################
def _amoeba_nm(func, p0, scale, ftol=1e-5, nmax=100000, nconv=6, progress_cb=None):
    """Nelder-Mead minimization — direct Python port of EXOFASTv2's exofast_amoeba.pro.

    Reference: Numerical Recipes, 2nd ed., Section 10.4.

    Convergence criterion (identical to EXOFASTv2):
      rtol = 2 * |y_hi - y_lo| / (|y_hi| + |y_lo|) < ftol
    Must be satisfied for ``nconv`` consecutive iterations.

    Parameters
    ----------
    func        : callable  f(x) -> scalar to MINIMISE
    p0          : (ndim,)   starting point
    scale       : (ndim,)   initial simplex displacement per dimension
    ftol        : float     fractional tolerance on function value spread
    nmax        : int       max function evaluations
    nconv       : int       consecutive converged iterations required (default 6)
    progress_cb : callable  progress_cb(x_best, y_best, ncalls, rtol),
                            called every 100 function evaluations

    Returns
    -------
    x_best : (ndim,) best vertex found
    y_best : float   function value at x_best
    ncalls : int     total function evaluations used
    """
    ndim = len(p0)
    mpts = ndim + 1

    # Initialise simplex: row 0 at p0, row i+1 displaced along dimension i
    p = np.tile(np.asarray(p0, dtype=float), (mpts, 1))   # (mpts, ndim)
    for i in range(ndim):
        p[i + 1, i] += scale[i]

    y      = np.array([func(p[i]) for i in range(mpts)], dtype=float)
    ncalls = mpts
    psum   = p.sum(axis=0)   # sum over ALL mpts vertices, shape (ndim,)

    def amotry(ihi, fac):
        """Move vertex ihi through the centroid by factor fac; update in place."""
        nonlocal ncalls
        fac1   = (1.0 - fac) / ndim
        fac2   = fac1 - fac
        ptry   = psum * fac1 - p[ihi] * fac2
        ytry   = func(ptry)
        ncalls += 1
        if ytry < y[ihi]:          # accept: replace worst vertex
            y[ihi]  = ytry
            psum[:] += ptry - p[ihi]
            p[ihi]  = ptry
        return ytry

    last_report = 0
    _conv_count = 0
    while ncalls <= nmax:
        s    = np.argsort(y)
        ilo  = s[0]    # best (lowest)
        ihi  = s[-1]   # worst (highest)
        inhi = s[-2]   # second-worst

        d    = abs(y[ihi]) + abs(y[ilo])
        rtol = 2.0 * abs(y[ihi] - y[ilo]) / d if d != 0.0 else ftol / 2.0

        if progress_cb is not None and ncalls - last_report >= 100:
            progress_cb(p[ilo], y[ilo], ncalls, rtol)
            last_report = ncalls

        if rtol < ftol:
            _conv_count += 1
            if _conv_count >= nconv:
                break
        else:
            _conv_count = 0

        # 1. Reflect worst vertex through centroid
        ytry = amotry(ihi, -1.0)
        if ytry <= y[ilo]:
            # Better than best → expand in the same direction
            amotry(ihi, 2.0)
        elif ytry >= y[inhi]:
            # Still worse than second-worst → contract
            ysave = y[ihi]
            ytry  = amotry(ihi, 0.5)
            if ytry >= ysave:
                # Contraction failed → shrink all toward best
                for i in range(mpts):
                    if i != ilo:
                        p[i]  = 0.5 * (p[i] + p[ilo])
                        y[i]  = func(p[i])
                        ncalls += 1
                psum[:] = p.sum(axis=0)

    ilo = np.argmin(y)
    return p[ilo], y[ilo], ncalls


def run_amoeba_optimization(s, p0=None):
    """Run Amoeba (Nelder-Mead) optimisation — direct port of EXOFASTv2.

    Controlled by settings keys:
      amoeba_nmax : max function evaluations (default 0 = skip)
      amoeba_ftol : fractional convergence tolerance (default 1e-5, same as
                    EXOFASTv2's optcriteria)

    Convergence follows EXOFASTv2's exofast_amoeba.pro exactly:
      rtol = 2 * |y_hi - y_lo| / (|y_hi| + |y_lo|) < ftol

    Simplex scale follows EXOFASTv2's philosophy
    ("amoeba stepping scale = 3× Gaussian width"):
      uniform [lo, hi]  → scale = (hi − lo) / 10
      normal  [μ, σ]    → scale = 3 σ

    Parameters
    ----------
    s  : settings dict
    p0 : 1-D array, optional — starting point (defaults to config.BASEMENT.theta_0)

    Returns
    -------
    best_theta : 1-D ndarray or None if skipped.
    """
    nmax = int(s.get('amoeba_nmax', 0))
    if nmax <= 0:
        return None

    ftol = float(s.get('amoeba_ftol', 1e-5))

    outdir    = config.BASEMENT.outdir
    best_file = os.path.join(outdir, 'amoeba_best.csv')

    # --- resume: skip if results already exist ---
    if os.path.exists(best_file):
        logprint("\nFound existing Amoeba results – skipping Amoeba optimisation.")
        logprint(f"  Loading: {best_file}")
        # amoeba_best.csv is saved in params.csv format (name,value,...)
        best_map = {}
        with open(best_file, 'r') as fh:
            for line in fh:
                stripped = line.strip()
                if stripped == '' or stripped.startswith('#'):
                    continue
                parts = stripped.split(',', 2)
                if len(parts) >= 2:
                    best_map[parts[0]] = float(parts[1])
        best = np.array([best_map[k] for k in config.BASEMENT.fitkeys])
        return best

    theta_0 = np.asarray(p0) if p0 is not None else config.BASEMENT.theta_0.copy()

    logprint(f"\nRunning Amoeba optimisation (max {nmax} evals, ftol={ftol})...")
    logprint('--------------------------')
    logprint(f"  Starting lnprob: {mcmc_lnprob(theta_0):.4f}")

    # Minimise −lnprob; return 1e100 at boundaries so NM can move away
    def neg_lnprob(theta):
        lp = mcmc_lnprob(theta)
        return -lp if np.isfinite(lp) else 1e100

    # Simplex scale from prior widths (matches EXOFASTv2 philosophy)
    ndim  = len(theta_0)
    scale = np.empty(ndim)
    for k, b in enumerate(config.BASEMENT.bounds):
        if b[0] == 'uniform':
            scale[k] = (b[2] - b[1]) / 10.0
        else:  # normal or trunc_normal
            scale[k] = 3.0 * b[2]
    scale = np.abs(scale)
    scale[scale == 0] = 1e-3

    # # --- freeze sed_errscale during Amoeba (zero simplex step) ---
    # for i, k in enumerate(config.BASEMENT.fitkeys):
    #     if 'sed_errscale' in k:
    #         scale[i] = 0.0
    #         logprint(f"  Freezing {k} = {theta_0[i]} during Amoeba")

    show_progress = s.get('print_progress', True)
    bar           = None
    if show_progress:
        from tqdm import tqdm
        bar = tqdm(desc='Amoeba', unit=' evals', dynamic_ncols=True)

    best_lp_seen = mcmc_lnprob(theta_0)
    best_x_seen  = theta_0.copy()
    last_ncalls  = [0]

    def _progress(x_lo, y_lo, ncalls, rtol):
        nonlocal best_lp_seen, best_x_seen
        lp = -y_lo
        if lp > best_lp_seen:
            best_lp_seen = lp
            best_x_seen  = x_lo.copy()
        if bar is not None:
            bar.update(ncalls - last_ncalls[0])
            last_ncalls[0] = ncalls
            bar.set_postfix(lnprob=f'{best_lp_seen:.4f}', rtol=f'{rtol:.2e}')

    amoeba_nconv = int(s.get('amoeba_nconv', 6))
    x_best, y_best, ncalls = _amoeba_nm(
        neg_lnprob, theta_0, scale,
        ftol=ftol, nmax=nmax, nconv=amoeba_nconv, progress_cb=_progress,
    )

    if bar is not None:
        bar.close()

    # Return the best point seen overall (simplex best might not be global best
    # if we hit nmax before convergence)
    lp_final = -y_best
    if best_lp_seen > lp_final:
        best     = best_x_seen
        lp_final = best_lp_seen
    else:
        best = x_best

    logprint(f"  Amoeba done: {ncalls} evals, best lnprob = {lp_final:.4f}")

    # --- save best params CSV ---
    _save_optimized_best(dict(zip(config.BASEMENT.fitkeys, best)), best_file)
    logprint(f"  Saved: {best_file}")

    # --- fit diagnostic plots at best Amoeba solution ---
    _plot_de_fit(best)

    return best


###########################################################################
#::: emcee backend
###########################################################################
def _run_emcee(s, p0_de=None, p0_best=None):
    """Run emcee EnsembleSampler. Returns the sampler.

    Parameters
    ----------
    p0_de   : (npop, ndim) array from DE population, used for walker seeding.
    p0_best : (ndim,) array — best single point (from Amoeba or DE best).
              Used as centre for random perturbations when p0_de is None.
    """
    outdir = config.BASEMENT.outdir
    save_h5 = os.path.join(outdir, 'mcmc_save.h5')
    continue_old_run = os.path.exists(save_h5)

    backend = emcee.backends.HDFBackend(save_h5)

    # Compute init_scale: DE pop std > getmcmcscale > init_err fallback
    if p0_de is not None and len(p0_de) > 2:
        _init_scale = np.std(p0_de, axis=0)
        _init_scale[_init_scale == 0] = 1e-8
    elif p0_best is not None:
        logprint('  Computing MCMC scale via Δχ²=1 search (emcee init)...')
        _init_scale = getmcmcscale(p0_best, mcmc_lnprob, debug=False)
    else:
        _init_scale = config.BASEMENT.init_err

    def _run(sampler):
        if continue_old_run:
            p0 = backend.get_chain()[-1, :, :]
            already_completed_steps = backend.get_chain().shape[0] * s['mcmc_thin_by']
        else:
            nwalkers = s['mcmc_nwalkers']
            if p0_de is not None:
                # Seed walkers from the DE population
                npop = p0_de.shape[0]
                if npop >= nwalkers:
                    idx = np.random.choice(npop, nwalkers, replace=False)
                    p0 = p0_de[idx, :]
                else:
                    repeats = (nwalkers // npop) + 1
                    p0 = np.tile(p0_de, (repeats, 1))[:nwalkers, :]
                # Add small perturbation so walkers are not identical
                p0 = p0 + _init_scale * 0.01 * np.random.randn(nwalkers, config.BASEMENT.ndim)
            else:
                # Use p0_best (Amoeba/DE best) if available, else theta_0
                centre = p0_best if p0_best is not None else config.BASEMENT.theta_0
                # EXOFASTv2 style: scatter walkers by factor*scale*randn
                factor = min(np.sqrt(500.0 / config.BASEMENT.ndim), 3.0)
                p0 = (centre
                      + factor * _init_scale
                      * np.random.randn(nwalkers, config.BASEMENT.ndim))
            already_completed_steps = 0

        # --- EXOFASTv2-style walker initialization ---
        # (see exofast_demcpt_multi.pro lines 336-362)
        ndim = config.BASEMENT.ndim
        nwalkers = p0.shape[0]
        centre = p0_best if p0_best is not None else config.BASEMENT.theta_0
        factor = min(np.sqrt(500.0 / ndim), 3.0)

        if p0_de is not None and not continue_old_run:
            # DE population: walkers are already good, just verify finite
            n_bad = 0
            for j in range(nwalkers):
                lnp = mcmc_lnprob(p0[j])
                if not np.isfinite(lnp):
                    # Replace bad DE walker with decaying scatter around centre
                    n_bad += 1
                    niter = 0
                    while True:
                        trial = (centre
                                 + factor / np.exp(niter / 1000.0)
                                 * _init_scale * np.random.randn(ndim))
                        for i, b in enumerate(config.BASEMENT.bounds):
                            if b[0] == 'uniform':
                                trial[i] = np.clip(trial[i], b[1], b[2])
                        if np.isfinite(mcmc_lnprob(trial)):
                            p0[j] = trial
                            break
                        niter += 1
                        if niter > 10000:
                            logprint(f'[WARNING] Walker {j}: could not find finite lnprob')
                            break
            logprint(f'  DE walkers: {nwalkers - n_bad}/{nwalkers} finite, replaced {n_bad}')

        elif not continue_old_run:
            # No DE population: EXOFASTv2-style init from centre with decaying scatter
            logprint(f'  Initialising {nwalkers} walkers (EXOFASTv2 style, factor={factor:.2f}) ...')
            for j in range(nwalkers):
                niter = 0
                while True:
                    if j == 0 and niter == 0:
                        trial = centre.copy()  # first walker at best-fit
                    else:
                        trial = (centre
                                 + factor / np.exp(niter / 1000.0)
                                 * _init_scale * np.random.randn(ndim))
                    for i, b in enumerate(config.BASEMENT.bounds):
                        if b[0] == 'uniform':
                            trial[i] = np.clip(trial[i], b[1], b[2])
                    if np.isfinite(mcmc_lnprob(trial)):
                        p0[j] = trial
                        break
                    niter += 1
                    if niter > 10000:
                        logprint(f'[WARNING] Walker {j}: could not find finite lnprob after 10000 tries')
                        p0[j] = trial
                        break
            logprint(f'  Walker initialisation done.')

        if not continue_old_run:
            for i in range(s['mcmc_pre_run_loops']):
                logprint("\nRunning pre-run loop", i + 1, '/', s['mcmc_pre_run_loops'])
                sampler.run_mcmc(p0, s['mcmc_pre_run_steps'], progress=s['print_progress'])
                log_prob = sampler.get_log_prob(flat=True)
                posterior_samples = sampler.get_chain(flat=True)
                ind_max = np.argmax(log_prob)
                factor = min(np.sqrt(500.0 / config.BASEMENT.ndim), 3.0)
                p0 = (posterior_samples[ind_max, :]
                      + factor * _init_scale
                      * np.random.randn(s['mcmc_nwalkers'], config.BASEMENT.ndim))
                os.remove(save_h5)
                sampler.reset()

        logprint("\nRunning full MCMC")
        deadline = s.get('_deadline', None)
        remaining_steps = int((s['mcmc_total_steps'] - already_completed_steps) / s['mcmc_thin_by'])
        thin_by = int(s['mcmc_thin_by'])
        tau_factor = float(s.get('mcmc_convergence_factor', 50))
        check_every = int(s.get('mcmc_check_every', 500))  # thinned steps between convergence checks
        min_steps_before_check = max(check_every, 200)  # don't check too early

        # Unified chunked loop: checks deadline + autocorrelation convergence
        chunk = 1
        done = 0
        pos = p0
        converged = False
        steps_since_check = 0

        show_progress = s.get('print_progress', True)
        bar = None
        if show_progress:
            from tqdm import tqdm
            bar = tqdm(total=remaining_steps, desc='emcee', unit=' steps',
                       dynamic_ncols=True)

        while done < remaining_steps:
            if deadline is not None and timer() >= deadline:
                logprint(f"\n  emcee stopped early at {done}/{remaining_steps} thinned steps: time limit reached.")
                break
            n = min(chunk, remaining_steps - done)
            sampler.run_mcmc(pos, n, thin_by=thin_by,
                             progress=False, skip_initial_state_check=True)
            pos = sampler.get_last_sample()
            done += n
            steps_since_check += n

            if bar is not None:
                bar.update(n)

            # Convergence check via autocorrelation time
            if steps_since_check >= check_every and done >= min_steps_before_check:
                steps_since_check = 0
                try:
                    # Total thinned steps in backend (old + new)
                    total_thinned = sampler.get_chain().shape[0]
                    # Discard first half to exclude burn-in when estimating tau
                    discard = total_thinned // 2
                    tau = sampler.get_autocorr_time(tol=0, discard=discard)
                    post_burn_steps = total_thinned - discard
                    chain_lengths = post_burn_steps / tau
                    min_Ntau = np.min(chain_lengths) if np.all(np.isfinite(chain_lengths)) else 0.0
                    if bar is not None:
                        bar.set_postfix_str(f'worst {min_Ntau:.1f}x tau (need {tau_factor:.0f}x)')
                    if np.all(chain_lengths > tau_factor) and np.all(np.isfinite(tau)):
                        logprint(f"\n  emcee converged at {done}/{remaining_steps} thinned steps: "
                                 f"all chains > {tau_factor:.0f}x tau "
                                 f"(min={np.min(chain_lengths):.1f}x, max tau={np.max(tau):.1f})")
                        converged = True
                        break
                except emcee.autocorr.AutocorrError:
                    pass  # not enough samples yet

        if bar is not None:
            bar.close()
        if not converged and done >= remaining_steps:
            logprint(f"  emcee completed {done} thinned steps (max reached).")
        return sampler

    _moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    if s['multiprocess']:
        logprint('\nRunning on', s['multiprocess_cores'], 'CPUs.')
        with closing(Pool(processes=s['multiprocess_cores'])) as pool:
            sampler = emcee.EnsembleSampler(
                s['mcmc_nwalkers'], config.BASEMENT.ndim, mcmc_lnprob,
                moves=_moves, pool=pool, backend=backend,
            )
            sampler = _run(sampler)
    else:
        sampler = emcee.EnsembleSampler(
            s['mcmc_nwalkers'], config.BASEMENT.ndim, mcmc_lnprob,
            moves=s.get('mcmc_moves', _moves), backend=backend,
        )
        sampler = _run(sampler)

    print_autocorr(sampler)
    return sampler


###########################################################################
#::: DEMCPT backend
###########################################################################
def _run_demcpt(s, p0_de=None, de_pop=None):
    """Run DEMCPTSampler. Returns the sampler."""
    outdir   = config.BASEMENT.outdir
    nchains  = s['mcmc_nwalkers']
    ntemps   = int(s.get('mcmc_ntemps', 1))
    maxgr    = float(s.get('mcmc_maxgr', 1.01))
    mintz    = float(s.get('mcmc_mintz', 1000))
    nsteps   = int(s['mcmc_total_steps'] / s['mcmc_thin_by'])
    nthin    = int(s['mcmc_thin_by'])

    cores    = s.get('multiprocess_cores', 1)
    nworkers = (multiprocessing.cpu_count()
                if str(cores).lower() == 'all' else int(cores))

    save_file        = os.path.join(outdir, 'demcpt_save.h5')
    continue_old_run = os.path.exists(save_file)

    # --- Determine MCMC scale ------------------------------------------------
    # Priority: 1) DE population std  2) Δχ²=1 binary search (EXOFASTv2)
    if de_pop is not None and len(de_pop) > 2:
        mcmc_scale = np.std(de_pop, axis=0)
        mcmc_scale[mcmc_scale == 0] = 1e-8
        logprint(f'  MCMC scale: from DE population std (median={np.median(mcmc_scale):.4e})')
    else:
        p0_start = p0_de if p0_de is not None else config.BASEMENT.theta_0
        logprint('  Computing MCMC scale via Δχ²=1 search (EXOFASTv2 getmcmcscale)...')
        mcmc_scale = getmcmcscale(p0_start, mcmc_lnprob, debug=False)
        logprint(f'  MCMC scale: from Δχ²=1 search (median={np.median(mcmc_scale):.4e})')

    logprint(f'  nchains={nchains}  ntemps={ntemps}  target_thinned_steps={nsteps}  nthin={nthin}')
    logprint(f'  maxgr={maxgr}  mintz={mintz}  nworkers={nworkers}')

    deadline = s.get('_deadline', None)
    if continue_old_run:
        logprint(f'\nResuming from {save_file}')
        sampler   = DEMCPTSampler.load(save_file, mcmc_lnprob)
        converged = sampler._run_continue(
            nsteps=nsteps, nthin=nthin, scale=mcmc_scale,
            progress=s.get('print_progress', True), nworkers=nworkers,
            save_every=max(nsteps // 10, 1), save_file=save_file,
            deadline=deadline,
        )
    else:
        p0_start = p0_de if p0_de is not None else config.BASEMENT.theta_0
        sampler   = DEMCPTSampler(
            mcmc_lnprob, ndim=config.BASEMENT.ndim,
            nchains=nchains, ntemps=ntemps, maxgr=maxgr, mintz=mintz,
        )
        converged = sampler.run(
            p0=p0_start, nsteps=nsteps, nthin=nthin,
            scale=mcmc_scale, population=de_pop,
            progress=s.get('print_progress', True), nworkers=nworkers,
            save_every=max(nsteps // 10, 1), save_file=save_file,
            deadline=deadline,
        )

    if converged:
        logprint("DEMCPT converged.")
    else:
        logprint("WARNING: DEMCPT did NOT converge within the step limit.")

    s['mcmc_total_steps'] = sampler._chain.shape[0] * nthin

    mcmc_save_h5 = os.path.join(outdir, 'mcmc_save.h5')
    sampler.save_as_emcee_backend(mcmc_save_h5)
    logprint(f'\nSaved emcee-compatible backend: {mcmc_save_h5}')

    sampler.summary(param_names=config.BASEMENT.fitkeys, logger=logprint)
    return sampler


###########################################################################
#::: Unified entry point
###########################################################################
def mcmc_fit(datadir, method=None):
    """
    Run MCMC sampling.

    Parameters
    ----------
    datadir : str
        Path to the data directory containing params.csv and settings.csv.
    method : {'emcee', 'demcpt'} or None
        Sampler backend.  If None, reads ``mcmc_sampler`` from settings.csv
        (defaults to 'emcee' when not set).
    """
    config.init(datadir)
    s = config.BASEMENT.settings

    if method is None:
        method = s.get('mcmc_sampler', 'emcee')

    t0 = timer()

    # Set a wall-clock deadline if max_hours is specified
    max_hours = s.get('max_hours', None)
    if max_hours is not None:
        s['_deadline'] = t0 + max_hours * 3600.
        logprint(f"\nTime limit: {max_hours} hours (deadline in {max_hours:.2f} h)")
    else:
        s['_deadline'] = None

    de_result = run_de_optimization(s)
    de_best   = de_result[0] if de_result is not None else None
    de_pop    = de_result[1] if de_result is not None else None

    # Amoeba: polish DE best (or theta_0 if DE was skipped)
    amoeba_best = run_amoeba_optimization(s, p0=de_best)

    # Best single starting point: amoeba > DE > None (falls back to theta_0 inside sampler)
    best_p0 = amoeba_best if amoeba_best is not None else de_best

    logprint(f"\nRunning MCMC ({method})...")
    logprint('--------------------------')

    if method == 'emcee':
        sampler = _run_emcee(s, p0_de=de_pop, p0_best=best_p0)
    elif method == 'demcpt':
        sampler = _run_demcpt(s, p0_de=best_p0, de_pop=de_pop)
    else:
        raise ValueError(
            f"Unknown mcmc_sampler: {method!r}. Choose 'emcee' or 'demcpt'."
        )

    # Update mcmc_total_steps to reflect actual chain length (may differ from
    # requested if early stop or time limit kicked in)
    actual_thinned = sampler.get_chain().shape[0]
    nthin = int(s['mcmc_thin_by'])
    s['mcmc_total_steps'] = actual_thinned * nthin

    t1 = timer()
    logprint(f"\nTime taken: {(t1 - t0) / 3600:.2f} hours")

    #::: shared epilogue: auto burn-in + acceptance fractions
    log_prob = sampler.get_log_prob()       # (nsteps, nwalkers/nchains)
    burnndx  = get_burnndx(log_prob)
    s['mcmc_burn_steps'] = burnndx * nthin
    logprint(f'\nAuto burn-in: stored step {burnndx} → {s["mcmc_burn_steps"]} total steps '
             f'({100. * burnndx / log_prob.shape[0]:.0f}% of chain)')

    if method == 'emcee':
        logprint('\nAcceptance fractions:')
        logprint('--------------------------')
        logprint(sampler.acceptance_fraction)


def demcpt_fit(datadir):
    """Backward-compatible alias for ``mcmc_fit(datadir, method='demcpt')``."""
    return mcmc_fit(datadir, method='demcpt')


def de_fit(datadir):
    """Run DE + amoeba optimization only, without MCMC.

    Saves results to the output directory (optimized_best.csv,
    optimized_population.csv) and generates fit plots, then returns.
    Useful for checking the optimizer converges before committing to a
    full MCMC run.
    """
    config.init(datadir)
    s = config.BASEMENT.settings

    t0 = timer()

    max_hours = s.get('max_hours', None)
    if max_hours is not None:
        s['_deadline'] = t0 + max_hours * 3600.
        logprint(f"\nTime limit: {max_hours} hours (deadline in {max_hours:.2f} h)")
    else:
        s['_deadline'] = None

    de_result = run_de_optimization(s)
    de_best   = de_result[0] if de_result is not None else None

    amoeba_best = run_amoeba_optimization(s, p0=de_best)

    best_p0 = amoeba_best if amoeba_best is not None else de_best

    if best_p0 is not None:
        lnp = mcmc_lnprob(best_p0)
        logprint(f"\nFinal best lnprob after optimization: {lnp:.4f}")
    else:
        logprint("\nNo optimization was run (de_ngen=0 and amoeba skipped).")

    t1 = timer()
    logprint(f"\nTime taken: {(t1 - t0) / 3600:.2f} hours")

