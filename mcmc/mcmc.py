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
#::: Automatic burn-in detection (EXOFASTv2 getburnndx logic)
###########################################################################
def get_burnndx(log_prob):
    """
    Determine the burn-in index from the log-probability chain.

    Mirrors EXOFASTv2's getburnndx: burn-in ends at the last stored step
    before ALL walkers have crossed above the median log-probability.

    Parameters
    ----------
    log_prob : ndarray, shape (nsteps, nwalkers)
        Log-probability at each step for each walker.

    Returns
    -------
    burnndx : int
        Index into the stored-step array.  Discard steps 0..burnndx-1.
    """
    median_lp = np.median(log_prob)
    nsteps, nwalkers = log_prob.shape

    last_below = np.zeros(nwalkers, dtype=int)
    for w in range(nwalkers):
        below = np.where(log_prob[:, w] < median_lp)[0]
        last_below[w] = below[-1] if len(below) else 0

    burnndx = int(np.max(last_below)) + 1
    # Always keep at least 25 % of the chain (or at least 50 steps) for
    # post-burn-in analysis.  Without this cap, an unconverged chain where a
    # single walker last dips below the median on the final step would set
    # burnndx = nsteps-1, leaving only 1 thinned step and making corner plots,
    # autocorrelation estimates, and parameter summaries meaningless.
    min_eval = max(50, nsteps // 4)
    return min(burnndx, nsteps - min_eval)


###########################################################################
#::: emcee backend
###########################################################################
def _run_emcee(s):
    """Run emcee EnsembleSampler. Returns the sampler."""
    outdir = config.BASEMENT.outdir
    save_h5 = os.path.join(outdir, 'mcmc_save.h5')
    continue_old_run = os.path.exists(save_h5)

    backend = emcee.backends.HDFBackend(save_h5)

    def _run(sampler):
        if continue_old_run:
            p0 = backend.get_chain()[-1, :, :]
            already_completed_steps = backend.get_chain().shape[0] * s['mcmc_thin_by']
        else:
            p0 = (config.BASEMENT.theta_0
                  + config.BASEMENT.init_err
                  * np.random.randn(s['mcmc_nwalkers'], config.BASEMENT.ndim))
            already_completed_steps = 0

        for i, b in enumerate(config.BASEMENT.bounds):
            if b[0] == 'uniform':
                p0[:, i] = np.clip(p0[:, i], b[1], b[2])

        if not continue_old_run:
            for i in range(s['mcmc_pre_run_loops']):
                logprint("\nRunning pre-run loop", i + 1, '/', s['mcmc_pre_run_loops'])
                sampler.run_mcmc(p0, s['mcmc_pre_run_steps'], progress=s['print_progress'])
                log_prob = sampler.get_log_prob(flat=True)
                posterior_samples = sampler.get_chain(flat=True)
                ind_max = np.argmax(log_prob)
                p0 = (posterior_samples[ind_max, :]
                      + config.BASEMENT.init_err
                      * np.random.randn(s['mcmc_nwalkers'], config.BASEMENT.ndim))
                os.remove(save_h5)
                sampler.reset()

        logprint("\nRunning full MCMC")
        sampler.run_mcmc(
            p0,
            int((s['mcmc_total_steps'] - already_completed_steps) / s['mcmc_thin_by']),
            thin_by=int(s['mcmc_thin_by']),
            progress=s['print_progress'],
        )
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
def _run_demcpt(s):
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

    logprint(f'  nchains={nchains}  ntemps={ntemps}  nsteps={nsteps}  nthin={nthin}')
    logprint(f'  maxgr={maxgr}  mintz={mintz}  nworkers={nworkers}')

    if continue_old_run:
        logprint(f'\nResuming from {save_file}')
        sampler   = DEMCPTSampler.load(save_file, mcmc_lnprob)
        converged = sampler._run_continue(
            nsteps=nsteps, nthin=nthin, scale=config.BASEMENT.init_err,
            progress=s.get('print_progress', True), nworkers=nworkers,
            save_every=max(nsteps // 10, 1), save_file=save_file,
        )
    else:
        sampler   = DEMCPTSampler(
            mcmc_lnprob, ndim=config.BASEMENT.ndim,
            nchains=nchains, ntemps=ntemps, maxgr=maxgr, mintz=mintz,
        )
        converged = sampler.run(
            p0=config.BASEMENT.theta_0, nsteps=nsteps, nthin=nthin,
            scale=config.BASEMENT.init_err,
            progress=s.get('print_progress', True), nworkers=nworkers,
            save_every=max(nsteps // 10, 1), save_file=save_file,
        )

    if converged:
        logprint("DEMCPT converged.")
    else:
        logprint("WARNING: DEMCPT did NOT converge within the step limit.")

    s['mcmc_total_steps'] = sampler._chain.shape[0] * nthin

    mcmc_save_h5 = os.path.join(outdir, 'mcmc_save.h5')
    sampler.save_as_emcee_backend(mcmc_save_h5)
    logprint(f'\nSaved emcee-compatible backend: {mcmc_save_h5}')

    sampler.summary(param_names=config.BASEMENT.fitkeys)
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

    logprint(f"\nRunning MCMC ({method})...")
    logprint('--------------------------')
    t0 = timer()

    if method == 'emcee':
        sampler = _run_emcee(s)
    elif method == 'demcpt':
        sampler = _run_demcpt(s)
    else:
        raise ValueError(
            f"Unknown mcmc_sampler: {method!r}. Choose 'emcee' or 'demcpt'."
        )

    t1 = timer()
    logprint(f"\nTime taken: {(t1 - t0) / 3600:.2f} hours")

    #::: shared epilogue: auto burn-in + acceptance fractions
    log_prob = sampler.get_log_prob()       # (nsteps, nwalkers/nchains)
    nthin    = int(s['mcmc_thin_by'])
    burnndx  = get_burnndx(log_prob)
    s['mcmc_burn_steps'] = burnndx * nthin
    logprint(f'\nAuto burn-in: stored step {burnndx} → {s["mcmc_burn_steps"]} total steps '
             f'({100. * burnndx / log_prob.shape[0]:.0f}% of chain)')

    logprint('\nAcceptance fractions:')
    logprint('--------------------------')
    logprint(sampler.acceptance_fraction)


def demcpt_fit(datadir):
    """Backward-compatible alias for ``mcmc_fit(datadir, method='demcpt')``."""
    return mcmc_fit(datadir, method='demcpt')

