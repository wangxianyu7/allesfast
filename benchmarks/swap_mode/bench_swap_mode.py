#!/usr/bin/env python
"""
Benchmark: DEMCPT (bidirectional vs unidirectional PT swap) vs emcee.

Three test distributions:
  1. Bimodal Gaussian (5D) — tests mode-hopping
  2. Rosenbrock banana (2D) — tests curved degeneracies
  3. Correlated Gaussian (20D) — tests high-dim efficiency

Metrics: wall-clock time, min ESS, ESS/s, mode discovery.
"""
import time
import numpy as np
from scipy.special import logsumexp
import emcee
import ptemcee
import reddemcee

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from allesfast.mcmc.demcpt import DEMCPTSampler


# ---- target distributions --------------------------------------------------
NDIM = 5
SEP = 6.0
SIGMA = 1.0
MU1 = np.zeros(NDIM)
MU2 = np.full(NDIM, SEP)


def log_bimodal(theta):
    lp1 = -0.5 * np.sum(((theta - MU1) / SIGMA) ** 2)
    lp2 = -0.5 * np.sum(((theta - MU2) / SIGMA) ** 2)
    return logsumexp([lp1, lp2]) - np.log(2)


def log_rosenbrock(theta):
    x, y = theta[0], theta[1]
    return -0.5 * ((1 - x)**2 + 100 * (y - x**2)**2) / 10.0


# correlated Gaussian (set up in main)
_cov_inv_20d = None
_log_det_20d = None

def log_corr_gauss(theta):
    return -0.5 * (theta @ _cov_inv_20d @ theta + _log_det_20d
                   + len(theta) * np.log(2 * np.pi))


# ---- ESS estimator (batch means) ------------------------------------------
def compute_min_ess(flat):
    n, ndim = flat.shape
    batch_size = max(n // 20, 1)
    nbatch = n // batch_size
    ess_per_dim = []
    for d in range(ndim):
        bm = [flat[i*batch_size:(i+1)*batch_size, d].mean()
              for i in range(nbatch)]
        var_bm = np.var(bm, ddof=1)
        var_all = np.var(flat[:, d], ddof=1)
        if var_bm > 0 and var_all > 0:
            ess_d = n * var_all / (batch_size * var_bm)
        else:
            ess_d = n
        ess_per_dim.append(min(ess_d, n))
    return min(ess_per_dim)


def check_bimodal(flat):
    near1 = np.sum(np.linalg.norm(flat - MU1, axis=1) < 3 * SIGMA * np.sqrt(NDIM))
    near2 = np.sum(np.linalg.norm(flat - MU2, axis=1) < 3 * SIGMA * np.sqrt(NDIM))
    found = near1 > 0 and near2 > 0
    return found, near1 / len(flat), near2 / len(flat)


# ---- DEMCPT runner ---------------------------------------------------------
def run_demcpt(logpost, ndim, p0, nsteps, ntemps, swap_mode, seed=42):
    nchains = max(2 * ndim, 10)
    sampler = DEMCPTSampler(
        logpost, ndim, nchains=nchains, ntemps=ntemps,
        Tf=100.0, swap_mode=swap_mode, seed=seed,
        maxgr=1.01, mintz=500,
    )
    t0 = time.perf_counter()
    sampler.run(nsteps=nsteps, p0=p0, progress=False)
    elapsed = time.perf_counter() - t0

    chain = sampler._chain
    burn = int(0.3 * chain.shape[0])
    flat = chain[burn:].reshape(-1, ndim)
    min_ess = compute_min_ess(flat)

    return {'elapsed': elapsed, 'min_ess': min_ess,
            'ess_per_sec': min_ess / elapsed, 'flat': flat}


# ---- emcee runner ----------------------------------------------------------
def run_emcee(logpost, ndim, p0, nsteps, seed=42, moves=None, label=None):
    nwalkers = max(2 * ndim, 10)
    # make even
    if nwalkers % 2 == 1:
        nwalkers += 1
    rng = np.random.default_rng(seed)
    p0_ens = p0 + 0.1 * rng.standard_normal((nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, moves=moves)
    t0 = time.perf_counter()
    sampler.run_mcmc(p0_ens, nsteps, progress=False)
    elapsed = time.perf_counter() - t0

    chain = sampler.get_chain()  # (nsteps, nwalkers, ndim)
    burn = int(0.3 * chain.shape[0])
    flat = chain[burn:].reshape(-1, ndim)
    min_ess = compute_min_ess(flat)

    return {'elapsed': elapsed, 'min_ess': min_ess,
            'ess_per_sec': min_ess / elapsed, 'flat': flat}


# ---- ptemcee runner --------------------------------------------------------
def _flat_prior(theta):
    return 0.0


def run_ptemcee(loglike, ndim, p0, nsteps, ntemps=8, seed=42):
    nwalkers = max(2 * ndim, 10)
    if nwalkers % 2 == 1:
        nwalkers += 1
    rng = np.random.default_rng(seed)
    p0_ens = np.empty((ntemps, nwalkers, ndim))
    for m in range(ntemps):
        p0_ens[m] = p0 + 0.1 * rng.standard_normal((nwalkers, ndim))

    sampler = ptemcee.Sampler(
        nwalkers, ndim, loglike, _flat_prior,
        ntemps=ntemps, Tmax=100.0, random=np.random.RandomState(seed),
    )
    t0 = time.perf_counter()
    sampler.run_mcmc(p0_ens, nsteps)
    elapsed = time.perf_counter() - t0

    # chain shape: (ntemps, nwalkers, nsteps, ndim)
    cold = sampler.chain[0, :, :, :]  # (nwalkers, nsteps, ndim)
    cold = cold.transpose(1, 0, 2)     # (nsteps, nwalkers, ndim)
    burn = int(0.3 * cold.shape[0])
    flat = cold[burn:].reshape(-1, ndim)
    min_ess = compute_min_ess(flat)

    return {'elapsed': elapsed, 'min_ess': min_ess,
            'ess_per_sec': min_ess / elapsed, 'flat': flat}


# ---- reddemcee runner ------------------------------------------------------
def run_reddemcee(loglike, ndim, p0, nsteps, ntemps=8, seed=42):
    nwalkers = max(2 * ndim, 10)
    if nwalkers % 2 == 1:
        nwalkers += 1
    rng = np.random.default_rng(seed)
    # reddemcee expects p0 shape (ntemps, nwalkers, ndim)
    p0_ens = np.empty((ntemps, nwalkers, ndim))
    for m in range(ntemps):
        p0_ens[m] = p0 + 0.1 * (m + 1) * rng.standard_normal((nwalkers, ndim))

    sampler = reddemcee.PTSampler(
        nwalkers, ndim, loglike, _flat_prior, ntemps=ntemps,
    )
    # reddemcee uses nsteps * nsweeps total; we do nsweeps=nsteps, nsteps=1
    t0 = time.perf_counter()
    sampler.run_mcmc(p0_ens, nsteps=1, nsweeps=nsteps, progress=False)
    elapsed = time.perf_counter() - t0

    # cold chain: temperature index 0
    chain = sampler.get_chain()  # (ntemps, nsteps*nsweeps, nwalkers, ndim)
    cold = chain[0, :, :, :]     # (nsteps*nsweeps, nwalkers, ndim)
    burn = int(0.3 * cold.shape[0])
    flat = cold[burn:].reshape(-1, ndim)
    min_ess = compute_min_ess(flat)

    return {'elapsed': elapsed, 'min_ess': min_ess,
            'ess_per_sec': min_ess / elapsed, 'flat': flat}


# ---- print helper ----------------------------------------------------------
def print_row(label, res, extra=''):
    print(f"  {label:25s} | time={res['elapsed']:6.2f}s | "
          f"min_ESS={res['min_ess']:7.0f} | ESS/s={res['ess_per_sec']:7.1f}"
          f"{extra}")


# ---- main ------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 75)
    print("DEMCPT (bidirectional / unidirectional) vs emcee  —  Benchmark")
    print("=" * 75)

    # --- Test 1: Bimodal Gaussian ---
    print("\n--- Test 1: Bimodal Gaussian (5D, separation=6σ) ---\n")
    p0_bm = MU1 + 0.1 * np.random.default_rng(0).standard_normal(NDIM)

    for mode in ['bidirectional', 'unidirectional']:
        res = run_demcpt(log_bimodal, NDIM, p0_bm, 4000, 8, mode)
        found, f1, f2 = check_bimodal(res['flat'])
        print_row(f'demcpt-{mode[:5]}',  res,
                  f" | modes={'YES' if found else 'NO'} ({f1:.0%}/{f2:.0%})")

    # emcee with different moves
    emcee_moves = [
        ('emcee-stretch', None),
        ('emcee-DE', emcee.moves.DEMove()),
        ('emcee-DESnooker', emcee.moves.DESnookerMove()),
        ('emcee-DE+Snooker', [(emcee.moves.DEMove(), 0.8),
                               (emcee.moves.DESnookerMove(), 0.2)]),
    ]
    for label, moves in emcee_moves:
        res = run_emcee(log_bimodal, NDIM, p0_bm, 4000, moves=moves)
        found, f1, f2 = check_bimodal(res['flat'])
        print_row(label, res,
                  f" | modes={'YES' if found else 'NO'} ({f1:.0%}/{f2:.0%})")

    res = run_ptemcee(log_bimodal, NDIM, p0_bm, 4000, ntemps=8)
    found, f1, f2 = check_bimodal(res['flat'])
    print_row('ptemcee', res,
              f" | modes={'YES' if found else 'NO'} ({f1:.0%}/{f2:.0%})")

    res = run_reddemcee(log_bimodal, NDIM, p0_bm, 4000, ntemps=8)
    found, f1, f2 = check_bimodal(res['flat'])
    print_row('reddemcee', res,
              f" | modes={'YES' if found else 'NO'} ({f1:.0%}/{f2:.0%})")

    # --- Test 2: Rosenbrock banana ---
    print("\n--- Test 2: Rosenbrock banana (2D) ---\n")
    p0_r = np.array([0.0, 0.0])

    for mode in ['bidirectional', 'unidirectional']:
        res = run_demcpt(log_rosenbrock, 2, p0_r, 5000, 8, mode)
        mx, my = res['flat'][:, 0].mean(), res['flat'][:, 1].mean()
        print_row(f'demcpt-{mode[:5]}', res, f" | mean=({mx:.2f}, {my:.2f})")

    for label, moves in emcee_moves:
        res = run_emcee(log_rosenbrock, 2, p0_r, 5000, moves=moves)
        mx, my = res['flat'][:, 0].mean(), res['flat'][:, 1].mean()
        print_row(label, res, f" | mean=({mx:.2f}, {my:.2f})")

    res = run_ptemcee(log_rosenbrock, 2, p0_r, 5000, ntemps=8)
    mx, my = res['flat'][:, 0].mean(), res['flat'][:, 1].mean()
    print_row('ptemcee', res, f" | mean=({mx:.2f}, {my:.2f})")

    res = run_reddemcee(log_rosenbrock, 2, p0_r, 5000, ntemps=8)
    mx, my = res['flat'][:, 0].mean(), res['flat'][:, 1].mean()
    print_row('reddemcee', res, f" | mean=({mx:.2f}, {my:.2f})")

    # --- Test 3: Correlated Gaussian 20D ---
    print("\n--- Test 3: Correlated Gaussian (20D) ---\n")
    ndim20 = 20
    rng = np.random.default_rng(123)
    A = rng.standard_normal((ndim20, ndim20))
    cov = A @ A.T / ndim20
    _cov_inv_20d = np.linalg.inv(cov)
    _log_det_20d = np.linalg.slogdet(cov)[1]
    p0_20 = 0.1 * rng.standard_normal(ndim20)

    for mode in ['bidirectional', 'unidirectional']:
        res = run_demcpt(log_corr_gauss, ndim20, p0_20, 5000, 8, mode)
        print_row(f'demcpt-{mode[:5]}', res)

    for label, moves in emcee_moves:
        res = run_emcee(log_corr_gauss, ndim20, p0_20, 5000, moves=moves)
        print_row(label, res)

    res = run_ptemcee(log_corr_gauss, ndim20, p0_20, 5000, ntemps=8)
    print_row('ptemcee', res)

    res = run_reddemcee(log_corr_gauss, ndim20, p0_20, 5000, ntemps=8)
    print_row('reddemcee', res)

    print("\n" + "=" * 75)
    print("Done.")
