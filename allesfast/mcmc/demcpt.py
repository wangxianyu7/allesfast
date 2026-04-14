"""
demcpt.py — Differential Evolution MCMC with Parallel Tempering

Python implementation of EXOFASTv2's exofast_demcpt_multi.pro
(Eastman et al. 2013, 2019; ter Braak 2006; Ford 2006).

Source: EXOZIPPy/exozippy/jointfit/demcpt.py (ported into allesfast).

Features
--------
  - Parallel Tempering (optional, ntemps > 1)
  - Multiprocessing for parallel log-posterior evaluation
  - HDF5 checkpoint save/load
  - Automatic burn-in via _getburnndx (EXOFASTv2 getburnndx algorithm)
  - Gelman-Rubin + independent-draws (Tz) convergence
  - emcee-compatible API: get_chain(), get_log_prob(), acceptance_fraction

Usage
-----
    from allesfast.demcpt import DEMCPTSampler

    sampler = DEMCPTSampler(log_posterior, ndim=5, nchains=20, ntemps=8)
    converged = sampler.run(p0=np.zeros(5), nsteps=50000, scale=np.ones(5)*0.1,
                            nworkers=4)
    sampler.save("demcpt_save.h5")

    flat = sampler.flatchain        # burn-in removed, bad chains discarded
    logp = sampler.flatlog_prob
"""

import numpy as np
from multiprocessing import Pool
import h5py
from time import time as _timer
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):          # no-op decorator
        def decorator(func):
            return func
        return decorator


# ---------------------------------------------------------------------------
#  JIT-compiled MCMC inner-loop kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _swap_step(pos, logp, betas, nchains, ntemps, rng_uniform, rng_log_uniform,
               bidirectional):
    """Parallel tempering swap step (all pairs, all chains). Returns (nswap, nattempt)."""
    nswap = 0
    nattempt = 0
    idx = 0
    for m in range(ntemps - 1):
        for j in range(nchains):
            if rng_uniform[idx] < 0.5:
                nattempt += 1
                log_alpha = ((betas[m] - betas[m + 1])
                             * (logp[j, m + 1] - logp[j, m]))
                if rng_log_uniform[idx] < log_alpha:
                    nswap += 1
                    if bidirectional:
                        for d in range(pos.shape[2]):
                            pos[j, m, d], pos[j, m + 1, d] = pos[j, m + 1, d], pos[j, m, d]
                        logp[j, m], logp[j, m + 1] = logp[j, m + 1], logp[j, m]
                    else:
                        for d in range(pos.shape[2]):
                            pos[j, m, d] = pos[j, m + 1, d]
                        logp[j, m] = logp[j, m + 1]
            idx += 1
    return nswap, nattempt

def _de_proposals(pos_m, nc, ndim, gamma, scale, rng_r1, rng_r2, rng_jitter):
    """Generate DE proposals for all chains at one temperature level (numpy vectorized)."""
    idx = np.arange(nc)
    r1 = rng_r1.copy()
    r1[r1 >= idx] += 1
    r2 = rng_r2.copy()
    r2[r2 >= np.minimum(idx, r1)] += 1
    r2[r2 >= np.maximum(idx, r1)] += 1
    jitter = (rng_jitter - 0.5) * scale / 10.0
    proposals = pos_m + gamma * (pos_m[r1] - pos_m[r2] + jitter)
    return proposals

@njit(cache=True)
def _snooker_proposals(pos_m, nc, ndim, gamma_snk, rng_iz, rng_i1, rng_i2):
    """Generate DE-Snooker proposals for all chains at one temperature level.

    Following ter Braak & Vrugt (2008): project the DE step along the
    direction from the current point to a random 'pivot' point z.
    """
    proposals = np.empty((nc, ndim))
    log_facs = np.empty(nc)
    for j in range(nc):
        # Pick 3 distinct others: z (pivot), z1, z2
        iz = int(rng_iz[j])
        if iz >= j: iz += 1
        i1 = int(rng_i1[j])
        if i1 >= min(j, iz): i1 += 1
        if i1 >= max(j, iz): i1 += 1
        i2 = int(rng_i2[j])
        if i2 >= min(j, min(iz, i1)): i2 += 1
        if i2 >= min(max(j, iz), max(j, i1), max(iz, i1)): i2 += 1
        if i2 >= max(j, max(iz, i1)): i2 += 1

        # Direction: current - pivot
        norm_sq = 0.0
        for d in range(ndim):
            norm_sq += (pos_m[j, d] - pos_m[iz, d])**2
        norm = np.sqrt(max(norm_sq, 1e-30))

        # Unit vector u = (x - z) / |x - z|
        # Project z1, z2 onto u, then step along u
        proj1 = 0.0
        proj2 = 0.0
        for d in range(ndim):
            u_d = (pos_m[j, d] - pos_m[iz, d]) / norm
            proj1 += u_d * pos_m[i1, d]
            proj2 += u_d * pos_m[i2, d]

        for d in range(ndim):
            u_d = (pos_m[j, d] - pos_m[iz, d]) / norm
            proposals[j, d] = pos_m[j, d] + u_d * gamma_snk * (proj1 - proj2)

        # Metropolis factor: (ndim-1)/2 * log(|q-z|/|x-z|)
        norm_q_sq = 0.0
        for d in range(ndim):
            norm_q_sq += (proposals[j, d] - pos_m[iz, d])**2
        norm_q = np.sqrt(max(norm_q_sq, 1e-30))
        log_facs[j] = 0.5 * (ndim - 1.0) * (np.log(norm_q) - np.log(norm))

    return proposals, log_facs

@njit(cache=True)
def _stretch_proposals(pos_m, nc, ndim, a, rng_r1, rng_z):
    """Generate stretch-move proposals for all chains at one temperature level."""
    proposals = np.empty((nc, ndim))
    log_facs = np.empty(nc)
    for j in range(nc):
        r1 = rng_r1[j]
        if r1 >= j: r1 += 1
        z = ((a - 1.0) * rng_z[j] + 1.0) ** 2 / a
        for d in range(ndim):
            proposals[j, d] = pos_m[r1, d] + z * (pos_m[j, d] - pos_m[r1, d])
        log_facs[j] = (ndim - 1) * np.log(z)
    return proposals, log_facs

@njit(cache=True)
def _stretch_proposals_split(pos_m, nc, ndim, a, rng_r, rng_z, first_half):
    """Ensemble-split stretch proposals: one half proposes from the other half.

    first_half=True: walkers 0..nc/2-1 propose using nc/2..nc-1 as complement
    first_half=False: walkers nc/2..nc-1 propose using 0..nc/2-1 as complement
    """
    half = nc // 2
    n_prop = half if first_half else (nc - half)
    proposals = np.empty((n_prop, ndim))
    log_facs = np.empty(n_prop)
    indices = np.empty(n_prop, dtype=np.int64)  # which walker each proposal is for

    for k in range(n_prop):
        if first_half:
            j = k                      # walker index in pos_m
            comp_start = half          # complement is second half
            comp_size = nc - half
        else:
            j = half + k
            comp_start = 0
            comp_size = half

        r = comp_start + (rng_r[k] % comp_size)
        z = ((a - 1.0) * rng_z[k] + 1.0) ** 2 / a
        for d in range(ndim):
            proposals[k, d] = pos_m[r, d] + z * (pos_m[j, d] - pos_m[r, d])
        log_facs[k] = (ndim - 1) * np.log(z)
        indices[k] = j
    return proposals, log_facs, indices

@njit(cache=True)
def _accept_split(pos_m, logp_m, proposals, new_lps, log_facs, beta,
                  rng_log_u, indices, n_prop):
    """Accept/reject for ensemble-split proposals. Returns naccept."""
    naccept = 0
    for k in range(n_prop):
        j = indices[k]
        if np.isfinite(new_lps[k]):
            log_alpha = beta * (new_lps[k] - logp_m[j]) + log_facs[k]
            if rng_log_u[k] < log_alpha:
                naccept += 1
                for d in range(pos_m.shape[1]):
                    pos_m[j, d] = proposals[k, d]
                logp_m[j] = new_lps[k]
    return naccept

@njit(cache=True)
def _accept_step(pos_m, logp_m, proposals, new_lps, log_facs, beta, rng_log_u, nc):
    """Vectorized acceptance for one temperature level. Returns naccept."""
    naccept = 0
    for j in range(nc):
        if np.isfinite(new_lps[j]):
            log_alpha = beta * (new_lps[j] - logp_m[j]) + log_facs[j]
            if rng_log_u[j] < log_alpha:
                naccept += 1
                for d in range(pos_m.shape[1]):
                    pos_m[j, d] = proposals[j, d]
                logp_m[j] = new_lps[j]
    return naccept

@njit(cache=True)
def _accept_all_temps(pos, logp, all_proposals, all_new_lps, all_log_facs,
                      betas, rng_log_u, nchains, ntemps):
    """Accept/reject for ALL temperatures in one JIT call. Returns total naccept."""
    naccept = 0
    for m in range(ntemps):
        off = m * nchains
        for j in range(nchains):
            idx = off + j
            if np.isfinite(all_new_lps[idx]):
                log_alpha = betas[m] * (all_new_lps[idx] - logp[j, m]) + all_log_facs[idx]
                if rng_log_u[idx] < log_alpha:
                    naccept += 1
                    for d in range(pos.shape[2]):
                        pos[j, m, d] = all_proposals[idx, d]
                    logp[j, m] = all_new_lps[idx]
    return naccept


# ---------------------------------------------------------------------------
#  JIT-compiled diagnostic functions
# ---------------------------------------------------------------------------

@njit(cache=True)
def _gelman_rubin(chains):
    """
    Gelman-Rubin statistic (Rhat) and independent draws (Tz)
    following Ford 2006, equations 21-26.

    Parameters
    ----------
    chains : ndarray, shape (nsteps, nchains, ndim)

    Returns
    -------
    Rhat : ndarray (ndim,)   — eq 25: sqrt(V̂⁺ / W)
    Tz   : ndarray (ndim,)   — eq 26: m*n * min(V̂⁺ / B, 1)
    """
    nsteps, nchains, ndim = chains.shape
    Rhat = np.empty(ndim)
    Tz = np.empty(ndim)

    for d in range(ndim):
        chain_means = np.empty(nchains)
        chain_vars = np.empty(nchains)
        for c in range(nchains):
            s = 0.0
            for t in range(nsteps):
                s += chains[t, c, d]
            mu = s / nsteps
            chain_means[c] = mu
            v = 0.0
            for t in range(nsteps):
                diff = chains[t, c, d] - mu
                v += diff * diff
            chain_vars[c] = v / (nsteps - 1)

        W = 0.0
        for c in range(nchains):
            W += chain_vars[c]
        W /= nchains

        grand = 0.0
        for c in range(nchains):
            grand += chain_means[c]
        grand /= nchains

        var_of_means = 0.0
        for c in range(nchains):
            diff = chain_means[c] - grand
            var_of_means += diff * diff
        var_of_means /= (nchains - 1)

        B = nsteps * var_of_means
        Vplus = (nsteps - 1.0) / nsteps * W + var_of_means

        Rhat[d] = np.sqrt(Vplus / W) if W > 0 else np.inf

        if B > 0:
            ratio = Vplus / B
            if ratio > 1.0:
                ratio = 1.0
            Tz[d] = nchains * nsteps * ratio
        else:
            Tz[d] = 0.0

    return Rhat, Tz


@njit(cache=True)
def _njit_median(arr):
    """Median of a 1-D array (numba-compatible)."""
    s = arr.copy()
    s.sort()
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2.0
    return float(s[n // 2])


@njit(cache=True)
def _getburnndx(neg2logp):
    """
    EXOFASTv2-style burn-in detection and good-chain identification.

    Algorithm (mirrors ``getburnndx.pro``):
      1. Find the best chain (lowest chi² anywhere) and its minimum index.
      2. Clamp the preliminary burn-in to [10%, 75%] of nsteps.
      3. Compute ``medchi2`` = min over chains of (median chi² from burnndx
         to end).  This is the threshold: a chain is "burned in" once it
         has at least one point below this value.
      4. For each chain, record the first step where chi² < medchi2,
         clamped to [10%, 90%] of nsteps.  Chains that never cross get
         burnndx = nsteps-1.
      5. Sort the per-chain burn-in indices and pick the cutoff that
         **maximises the total number of usable links** (= post-burn-in
         steps × number of good chains).
      6. If fewer than 3 chains are good, fall back to using all chains.

    Parameters
    ----------
    neg2logp : ndarray (nsteps, nchains)
        ``-2 * log_posterior``, i.e. chi²-like values (lower = better).

    Returns
    -------
    burnndx : int
        First usable step index (discard steps 0 .. burnndx-1).
    good : int64 array
        Indices of good chains.
    """
    nsteps, nchains = neg2logp.shape
    lo = int(round(0.1 * nsteps))   # minimum burn-in (10 %)
    hi = int(round(0.9 * nsteps))   # maximum burn-in (90 %)

    # --- Step 1: find the best chain and its global-minimum index ----------
    best_chi2 = np.inf
    best_chain = 0
    best_ndx = 0
    for c in range(nchains):
        for t in range(nsteps):
            if neg2logp[t, c] < best_chi2:
                best_chi2 = neg2logp[t, c]
                best_chain = c
                best_ndx = t

    # --- Step 2: preliminary burnndx clamped to [10 %, 75 %] --------------
    prelim = best_ndx
    if prelim < lo:
        prelim = lo
    hi75 = int(round(0.75 * nsteps))
    if prelim > hi75:
        prelim = hi75

    # --- Step 3: medchi2 = min-of-chain-medians from prelim to end ---------
    medchi2 = np.inf
    for c in range(nchains):
        seg = neg2logp[prelim:nsteps, c].copy()
        seg.sort()
        n = len(seg)
        if n == 0:
            continue
        med = (seg[n // 2 - 1] + seg[n // 2]) / 2.0 if n % 2 == 0 else float(seg[n // 2])
        if med < medchi2:
            medchi2 = med

    # --- Step 4: per-chain first-crossing, clamped to [10 %, 90 %] --------
    burnndxs = np.empty(nchains, dtype=np.int64)
    for c in range(nchains):
        found = -1
        for t in range(nsteps):
            if neg2logp[t, c] < medchi2:
                found = t
                break
        if found == -1:
            burnndxs[c] = nsteps - 1
        else:
            clamped = found
            if clamped < lo:
                clamped = lo
            if clamped > hi:
                clamped = hi
            burnndxs[c] = clamped

    # --- Step 5: maximise total usable links --------------------------------
    sorted_idx = np.argsort(burnndxs)
    best_nlinks = 0
    best_cut = 0
    for j in range(nchains):
        nlinks = (nsteps - burnndxs[sorted_idx[j]]) * (j + 1)
        if nlinks > best_nlinks:
            best_nlinks = nlinks
            best_cut = j

    burnndx = burnndxs[sorted_idx[best_cut]]
    # Final clamp: never use less than the last 10 %
    if burnndx > hi:
        burnndx = hi

    good_sorted = sorted_idx[:best_cut + 1]

    # --- Step 6: fallback if < 3 good chains --------------------------------
    if len(good_sorted) < 3:
        good_sorted = np.arange(nchains, dtype=np.int64)

    # Return sorted good-chain indices
    good = good_sorted.copy()
    good.sort()
    return burnndx, good


@njit(cache=True)
def _find_burnin(neg2logp):
    """Convenience wrapper: return only the burn-in index."""
    burnndx, _good = _getburnndx(neg2logp)
    return burnndx


@njit(cache=True)
def _identify_good_chains(neg2logp):
    """Convenience wrapper: return only the good-chain indices."""
    _burnndx, good = _getburnndx(neg2logp)
    return good


# ---------------------------------------------------------------------------
#  Sampler
# ---------------------------------------------------------------------------

class DEMCPTSampler:
    """
    Differential Evolution MCMC with optional Parallel Tempering.

    Parameters
    ----------
    log_posterior : callable
        Function  theta(ndim,) -> float  returning log-posterior.
    ndim : int
    nchains : int, optional
        Default max(2*ndim, 10).
    ntemps : int, optional
        Number of temperature rungs. 1 = no tempering.
    Tf : float, optional
        Temperature factor for the hottest rung (default 200).
    stretch : bool, optional
        Use stretch move instead of DE (default False).
    maxgr : float, optional
        Convergence threshold for Gelman-Rubin (default 1.01).
    mintz : float, optional
        Convergence threshold for independent draws (default 1000).
    swap_mode : str, optional
        'bidirectional' (default): standard PT swap — cold and hot
        exchange positions.  'unidirectional': EXOFASTv2 style — cold
        chain adopts hot position, hot chain keeps its own.
    seed : int or None
    """

    def __init__(self, log_posterior, ndim, nchains=None, ntemps=1, Tf=200.0,
                 stretch=False, stretch_fraction=0.5, snooker_fraction=0.0,
                 vectorize=False,
                 maxgr=1.01, mintz=1000, seed=None,
                 adapt_temps=False, adapt_halflife=1000,
                 swap_mode='bidirectional'):
        self.logpost_func = log_posterior
        # vectorize: if True, logpost accepts (nchains, ndim) and returns (nchains,)
        if vectorize:
            self._vectorized_logpost = log_posterior
            # Wrap for scalar calls during initialization
            def _scalar_wrap(theta):
                result = log_posterior(theta[np.newaxis, :])[0]
                return float(result)
            self.logpost_func = _scalar_wrap
        else:
            self._vectorized_logpost = None
        self.ndim = ndim
        self.nchains = nchains or max(2 * ndim, 3)
        self.ntemps = ntemps
        self.stretch = stretch
        # stretch_fraction: fraction of proposals that use stretch move (0-1)
        # snooker_fraction: fraction that use DE-Snooker (0-1)
        # remainder uses DE. stretch=True overrides stretch_fraction to 1.0.
        if stretch:
            self.stretch_fraction = 1.0
        else:
            self.stretch_fraction = float(stretch_fraction)
        self.snooker_fraction = float(snooker_fraction)
        self.gamma_snooker = 1.7  # default snooker gamma (ter Braak & Vrugt 2008)
        self.maxgr = maxgr
        self.mintz = mintz
        self.rng = np.random.default_rng(seed)

        # temperature ladder  (betas[0]=1 cold, betas[-1]=1/Tf hot)
        if ntemps > 1:
            self.betas = (1.0 / Tf) ** (np.arange(ntemps) / (ntemps - 1))
        else:
            self.betas = np.array([1.0])

        # adaptive temperature ladder (SAR method, cf. reddemcee)
        self.adapt_temps = adapt_temps and ntemps > 1
        self.adapt_halflife = adapt_halflife

        # swap mode: 'bidirectional' (standard PT) or 'unidirectional' (EXOFASTv2)
        if swap_mode not in ('bidirectional', 'unidirectional'):
            raise ValueError(f"swap_mode must be 'bidirectional' or 'unidirectional', got {swap_mode!r}")
        self.swap_mode = swap_mode

        self.gamma = 2.38 / np.sqrt(2.0 * ndim)
        self.a_stretch = 2.0

        # results (populated by run)
        self._chain = None
        self._log_prob = None
        self._pos_full = None     # (nchains, ntemps, ndim)
        self._logp_full = None    # (nchains, ntemps)
        self._acceptance_rate = np.nan
        self._betas_history = None  # (nsteps,ntemps) if adapt_temps

    # ----- adaptive temperature ladder (SAR) --------------------------------

    def _adapt_betas(self, swap_accept_rates, iteration):
        """Adjust betas to equalise swap acceptance rates between neighbours.

        Uses the SAR (Swap Acceptance Rate) method following reddemcee
        (Vousden et al. 2016; Peña Rojas 2024).  Works in temperature
        space (T = 1/beta): multiplicatively adjusts the temperature
        gaps so that pairs with lower swap rates get pushed closer
        together.  The adjustment decays as
        halflife / (iteration + halflife) so the ladder stabilises.

        Parameters
        ----------
        swap_accept_rates : ndarray (ntemps-1,)
            Running swap acceptance rate for each adjacent pair.
        iteration : int
            Current MCMC step (controls decay).
        """
        ntemps = self.ntemps
        if ntemps < 3:
            return  # need at least 3 temps to adjust interior betas

        kappa = self.adapt_halflife / (iteration + self.adapt_halflife)

        # SAR adjustment: equalise swap acceptance rates
        dS = -np.diff(swap_accept_rates)  # (ntemps-2,)
        dS[np.abs(dS) < 1e-8] = 0.0

        # Work in temperature space (T = 1/beta), following reddemcee
        betas = self.betas.copy()
        Ts = 1.0 / betas[:-1]          # exclude hottest (infinite T OK)
        deltaTs = np.diff(Ts)           # (ntemps-2,) gaps between temps

        # Multiplicative adjustment on gaps (same as reddemcee PTMove.adapt)
        deltaTs *= np.exp(kappa * dS)

        # Reconstruct interior betas from adjusted gaps
        new_Ts = np.cumsum(deltaTs) + Ts[0]  # Ts[0] = 1/betas[0] = 1
        betas[1:-1] = 1.0 / new_Ts

        # Safety: ensure monotone decreasing and positive
        for k in range(1, ntemps - 1):
            betas[k] = min(betas[k], betas[k - 1] * 0.999)
            betas[k] = max(betas[k], betas[k + 1] * 1.001)

        betas[0] = 1.0  # pin cold chain
        self.betas = betas

    # ----- proposal helpers --------------------------------------------------

    def _de_proposals_batch(self, m, pos, scale):
        """Generate DE proposals for ALL chains at temperature m (vectorized)."""
        nc = self.nchains
        ndim = self.ndim
        rng = self.rng
        # For each chain, pick two distinct others
        idx = np.arange(nc)
        r1 = rng.integers(0, nc - 1, size=nc)
        r1[r1 >= idx] += 1
        r2 = rng.integers(0, nc - 2, size=nc)
        r2[r2 >= np.minimum(idx, r1)] += 1
        r2[r2 >= np.maximum(idx, r1)] += 1
        jitter = (rng.random((nc, ndim)) - 0.5) * scale / 10.0
        proposals = (pos[:, m]
                     + self.gamma * (pos[r1, m] - pos[r2, m] + jitter))
        return proposals, np.zeros(nc)

    def _stretch_proposals_batch(self, m, pos):
        """Generate stretch-move proposals for ALL chains at temperature m (vectorized)."""
        nc = self.nchains
        ndim = self.ndim
        rng = self.rng
        a = self.a_stretch
        # Pick a random complement chain for each walker
        r1 = rng.integers(0, nc - 1, size=nc)
        idx = np.arange(nc)
        r1[r1 >= idx] += 1
        z = ((a - 1.0) * rng.random(nc) + 1.0) ** 2 / a
        proposals = pos[r1, m] + z[:, None] * (pos[:, m] - pos[r1, m])
        log_facs = (ndim - 1) * np.log(z)
        return proposals, log_facs

    # ----- main loop ---------------------------------------------------------

    def run(self, p0, nsteps, nthin=1, scale=None, population=None,
            progress=True, check_every=None, npass_required=6, nworkers=1,
            save_every=0, save_file=None, deadline=None):
        """
        Run the sampler.

        Parameters
        ----------
        p0 : ndarray (ndim,)
            Best-fit starting point.
        nsteps : int
            Number of stored steps per chain.
        nthin : int
            Keep every nthin-th sample (default 1).
        scale : ndarray (ndim,) or None
            Per-parameter step scale.  If None, uses 1% of |p0|.
        population : ndarray (npop, ndim) or None
            DE population to seed walkers from.  Cold-chain walkers are
            initialised at population positions (first min(npop, nchains)
            members); extra walkers and all hot chains are perturbed from
            a random population member.  If npop > nchains the population
            is sorted by lnprob and the best nchains are used.
        progress : bool
        check_every : int or None
            Steps between convergence checks.  Default nsteps//20.
        npass_required : int
            Consecutive passes needed (default 6).
        nworkers : int
            Number of worker processes. 1 = serial.
        save_every : int
            Checkpoint every N steps. 0 = disabled.
        save_file : str or None
            HDF5 path for checkpointing. Required if save_every > 0.

        Returns
        -------
        converged : bool
        """
        ndim = self.ndim
        nchains = self.nchains
        ntemps = self.ntemps
        betas = self.betas
        rng = self.rng
        logpost = self.logpost_func

        p0 = np.asarray(p0, dtype=np.float64)
        if scale is None:
            scale = np.abs(p0) * 0.01
            scale[scale == 0] = 1e-5

        if check_every is None:
            check_every = 10#max(100, nsteps // 20)

        if save_every > 0 and save_file is None:
            raise ValueError("save_file is required when save_every > 0")

        use_pool = nworkers > 1

        # ---- prepare population for seeding ----------------------------------
        have_pop = population is not None and len(population) > 0
        if have_pop:
            population = np.asarray(population, dtype=np.float64)
            npop = len(population)
            # If population is larger than nchains, keep the best members
            if npop > nchains:
                pop_lps = np.array([logpost(p) for p in population])
                best_idx = np.argsort(pop_lps)[::-1][:nchains]
                population = population[best_idx]
                npop = nchains
        else:
            npop = 0

        # ---- initialise walkers (nchains, ntemps, ndim) ---------------------
        pos = np.empty((nchains, ntemps, ndim))
        logp = np.full((nchains, ntemps), -np.inf)

        if progress:
            if have_pop:
                print(f"Initialising {nchains} chains x {ntemps} temps "
                      f"(seeding from DE population of {npop}) ...")
            else:
                print(f"Initialising {nchains} chains x {ntemps} temps ...")

        _n_total = nchains * ntemps
        _n_done = 0
        _n_retries = 0
        for j in range(nchains):
            for m in range(ntemps):
                niter = 0
                while True:
                    if have_pop and m == 0 and j < npop and niter == 0:
                        # Cold chain with population member → use directly
                        trial = population[j].copy()
                    elif have_pop and niter == 0:
                        # Hot chain or extra cold walker → perturb random member
                        src = population[rng.integers(npop)]
                        factor = min(np.sqrt(500.0 / ndim), 3.0)
                        trial = src + factor * scale * rng.standard_normal(ndim)
                    elif j == 0 and m == 0 and niter == 0:
                        # No population: first cold walker at p0
                        trial = p0.copy()
                    else:
                        # No population: perturb from p0 (EXOFASTv2 style)
                        factor = min(np.sqrt(500.0 / ndim), 3.0)
                        trial = p0 + (factor / np.exp(niter / 1000.0)
                                      * scale * rng.standard_normal(ndim))
                    lp = logpost(trial)
                    if np.isfinite(lp):
                        pos[j, m] = trial
                        logp[j, m] = lp
                        break
                    niter += 1
                    if niter > 10000:
                        raise RuntimeError(
                            f"Cannot find finite logpost near p0 "
                            f"(chain {j}, temp {m})")
                _n_done += 1
                _n_retries += niter
                if progress:
                    print(f"\r  Init: {_n_done}/{_n_total} "
                          f"(retries={_n_retries})   ",
                          end="", flush=True)
        if progress:
            print()

        # ---- storage (cold chain only) -------------------------------------
        chain = np.empty((nsteps, nchains, ndim))
        log_prob = np.empty((nsteps, nchains))
        chain[0] = pos[:, 0]
        log_prob[0] = logp[:, 0]

        naccept = 0
        nattempt = 0
        nswap = 0
        nswap_attempt = 0
        npass = 0
        converged = False
        final_step = nsteps
        last_saved = None

        # per-pair swap tracking for adaptive temps
        if self.adapt_temps:
            pair_swap = np.zeros(ntemps - 1)
            pair_attempt = np.zeros(ntemps - 1)
            betas_history = np.empty((nsteps, ntemps))
            betas_history[0] = self.betas.copy()

        if progress:
            workers_s = f" ({nworkers} workers)" if use_pool else ""
            print(f"Running MCMC{workers_s} ...")

        pool = Pool(nworkers) if use_pool else None
        pbar = tqdm(total=nsteps - 1, desc='MCMC', disable=not (progress and tqdm is not None),
                    mininterval=0.5, dynamic_ncols=True) if tqdm is not None else None
        try:
            _bidir = (self.swap_mode == 'bidirectional')
            _sf = self.stretch_fraction
            _snf = self.snooker_fraction
            _a = self.a_stretch
            _gamma = self.gamma
            _gamma_snk = self.gamma_snooker
            _ntotal = nchains * ntemps  # total proposals per step

            for i in range(1, nsteps):
                for _thin in range(nthin):
                    # --- temperature swaps (JIT) ---
                    if ntemps > 1:
                        _n_rng = (ntemps - 1) * nchains
                        _ru = rng.random(_n_rng)
                        _rlu = np.log(rng.random(_n_rng))
                        _ns, _na = _swap_step(pos, logp, betas, nchains,
                                              ntemps, _ru, _rlu, _bidir)
                        nswap += _ns
                        nswap_attempt += _na

                    # --- generate ALL proposals across ALL temps (JIT) ---
                    all_proposals = np.empty((_ntotal, ndim))
                    all_log_facs = np.zeros(_ntotal)
                    for m in range(ntemps):
                        off = m * nchains
                        # Choose move: stretch / snooker / DE
                        _draw = rng.random()
                        if _sf >= 1.0 or (_sf > 0 and _draw < _sf):
                            p, lf = _stretch_proposals(
                                pos[:, m], nchains, ndim, _a,
                                rng.integers(0, nchains - 1, size=nchains),
                                rng.random(nchains))
                            all_proposals[off:off+nchains] = p
                            all_log_facs[off:off+nchains] = lf
                        elif _snf > 0 and _draw < _sf + _snf:
                            p, lf = _snooker_proposals(
                                pos[:, m], nchains, ndim, _gamma_snk,
                                rng.integers(0, nchains - 1, size=nchains),
                                rng.integers(0, nchains - 2, size=nchains),
                                rng.integers(0, nchains - 3, size=nchains))
                            all_proposals[off:off+nchains] = p
                            all_log_facs[off:off+nchains] = lf
                        else:
                            p = _de_proposals(
                                np.ascontiguousarray(pos[:, m]), nchains, ndim, _gamma, scale,
                                rng.integers(0, nchains - 1, size=nchains),
                                rng.integers(0, nchains - 2, size=nchains),
                                rng.random((nchains, ndim)))
                            all_proposals[off:off+nchains] = p

                    # --- logp evaluation: ONE batch for all temps ---
                    nattempt += _ntotal
                    if pool is not None:
                        _chunksize = max(1, _ntotal // (nworkers * 2))
                        all_new_lps = np.array(pool.map(
                            logpost, list(all_proposals),
                            chunksize=_chunksize))
                    elif self._vectorized_logpost is not None:
                        all_new_lps = self._vectorized_logpost(all_proposals)
                    else:
                        all_new_lps = np.array([logpost(all_proposals[j])
                                                for j in range(_ntotal)])

                    # --- acceptance for ALL temps (JIT) ---
                    _rlu = np.log(rng.random(_ntotal))
                    naccept += _accept_all_temps(
                        pos, logp, all_proposals, all_new_lps,
                        all_log_facs, betas, _rlu, nchains, ntemps)

                    chain[i] = pos[:, 0]
                    log_prob[i] = logp[:, 0]

                # --- adaptive temperature ladder ---
                if self.adapt_temps:
                    safe = pair_attempt > 0
                    rates = np.zeros(ntemps - 1)
                    rates[safe] = pair_swap[safe] / pair_attempt[safe]
                    self._adapt_betas(rates, i)
                    betas = self.betas  # update local ref
                    betas_history[i] = betas.copy()

                if pbar is not None:
                    acc = naccept / max(nattempt, 1) * 100
                    _post = {'acc': f'{acc:.1f}%'}
                    if ntemps > 1 and nswap_attempt > 0:
                        _post['swap'] = f'{nswap/nswap_attempt*100:.1f}%'
                    pbar.set_postfix(_post, refresh=False)
                    pbar.update(1)

                if save_every > 0 and (i + 1) % save_every == 0:
                    self._chain = chain[:i + 1]
                    self._log_prob = log_prob[:i + 1]
                    self._pos_full = pos
                    self._logp_full = logp
                    self.save(save_file)
                    last_saved = i + 1

                if (i + 1) % check_every == 0 and i > 2 * nchains:
                    chi2 = -2.0 * log_prob[:i + 1]
                    burnndx = _find_burnin(chi2)
                    post_burn = chain[burnndx:i + 1]

                    if post_burn.shape[0] > 10:
                        Rhat, Tz = _gelman_rubin(post_burn)
                        max_gr = float(np.max(Rhat))
                        min_tz = float(np.min(Tz))
                        acc = naccept / max(nattempt, 1) * 100
                        swap_s = (f"; swap={nswap/nswap_attempt*100:.1f}%"
                                  if ntemps > 1 and nswap_attempt > 0 else "")
                        if progress:
                            save_s = (f" | saved at step {last_saved}"
                                      if last_saved is not None else "")
                            _msg = (f"  {100*(i+1)/nsteps:5.1f}% | "
                                    f"accept={acc:.1f}%{swap_s} | "
                                    f"GR={max_gr:.4f} (<{self.maxgr}) | "
                                    f"Tz={min_tz:.0f} (>{self.mintz}){save_s}")
                            if pbar is not None:
                                pbar.write(_msg)
                            else:
                                print("\r" + _msg + "   ", end="", flush=True)

                        if max_gr < self.maxgr and min_tz > self.mintz:
                            npass += 1
                            if npass >= npass_required:
                                converged = True
                                final_step = i + 1
                                break
                        else:
                            npass = 0

                if deadline is not None and _timer() >= deadline:
                    final_step = i + 1
                    if progress:
                        print(f"\n  Time limit reached at step {final_step}/{nsteps}. Stopping.")
                    break
        finally:
            if pool is not None:
                pool.close()
                pool.join()
            if pbar is not None:
                pbar.close()

        chain = chain[:final_step]
        log_prob = log_prob[:final_step]

        if progress:
            if converged:
                print(f"\n  Converged at step {final_step}/{nsteps}.")
            elif deadline is not None and final_step < nsteps:
                pass  # already printed above
            else:
                print(f"\n  Reached max steps ({nsteps}). NOT converged.")

        self._chain = chain
        self._log_prob = log_prob
        self._pos_full = pos
        self._logp_full = logp
        self._acceptance_rate = naccept / max(nattempt, 1)
        if self.adapt_temps:
            self._betas_history = betas_history[:final_step]
        return converged

    def _run_continue(self, nsteps, nthin=1, scale=None, progress=True,
                      check_every=None, npass_required=6, nworkers=1,
                      save_every=0, save_file=None, deadline=None):
        """Continue sampling from an existing chain."""
        if self._chain is None or self._log_prob is None:
            raise RuntimeError("No existing chain to continue. Call run() or load() first.")
        if self._pos_full is None or self._logp_full is None:
            raise RuntimeError("Missing full temperature state; cannot resume reliably.")

        ndim = self.ndim
        nchains = self.nchains
        ntemps = self.ntemps
        betas = self.betas
        rng = self.rng
        logpost = self.logpost_func

        if scale is None:
            p0 = np.median(self._chain[-1], axis=0)
            scale = np.abs(p0) * 0.01
            scale[scale == 0] = 1e-5

        if check_every is None:
            check_every = max(100, nsteps // 20)

        if save_every > 0 and save_file is None:
            raise ValueError("save_file is required when save_every > 0")

        use_pool = nworkers > 1
        pos = self._pos_full
        logp = self._logp_full

        old_steps = self._chain.shape[0]
        total_steps = nsteps - old_steps
        if total_steps <= 0:
            if progress:
                print(f"  Already at {old_steps}/{nsteps} steps; nothing to do.")
            return False
        chain = np.empty((nsteps, nchains, ndim))
        log_prob = np.empty((nsteps, nchains))
        chain[:old_steps] = self._chain
        log_prob[:old_steps] = self._log_prob

        naccept = 0
        nattempt = 0
        nswap = 0
        nswap_attempt = 0
        npass = 0
        converged = False
        final_step = total_steps
        last_saved = None

        # per-pair swap tracking for adaptive temps
        if self.adapt_temps:
            pair_swap = np.zeros(ntemps - 1)
            pair_attempt = np.zeros(ntemps - 1)
            betas_history = np.empty((nsteps, ntemps))
            if self._betas_history is not None:
                betas_history[:old_steps] = self._betas_history[:old_steps]
            else:
                betas_history[:old_steps] = self.betas[None, :]

        if progress:
            workers_s = f" ({nworkers} workers)" if use_pool else ""
            print(f"Continuing MCMC{workers_s}: +{total_steps} thinned steps "
                  f"(resuming from {old_steps}, target {nsteps}) ...")

        pool = Pool(nworkers) if use_pool else None
        try:
            _bidir = (self.swap_mode == 'bidirectional')
            _sf = self.stretch_fraction
            _a = self.a_stretch
            _gamma = self.gamma

            for k in range(total_steps):
                for _thin in range(nthin):
                    # --- temperature swaps (JIT) ---
                    if ntemps > 1:
                        _n_rng = (ntemps - 1) * nchains
                        _ru = rng.random(_n_rng)
                        _rlu = np.log(rng.random(_n_rng))
                        _ns, _na = _swap_step(pos, logp, betas, nchains,
                                              ntemps, _ru, _rlu, _bidir)
                        nswap += _ns
                        nswap_attempt += _na

                    # --- proposals + acceptance per temperature ---
                    for m in range(ntemps):
                        use_stretch = (_sf >= 1.0 or
                                       (_sf > 0 and rng.random() < _sf))

                        if use_stretch:
                            proposals, log_facs = _stretch_proposals(
                                pos[:, m], nchains, ndim, _a,
                                rng.integers(0, nchains - 1, size=nchains),
                                rng.random(nchains))
                        else:
                            proposals = _de_proposals(
                                pos[:, m], nchains, ndim, _gamma, scale,
                                rng.integers(0, nchains - 1, size=nchains),
                                rng.integers(0, nchains - 2, size=nchains),
                                rng.random((nchains, ndim)))
                            log_facs = np.zeros(nchains)

                        # --- logp evaluation (Python callback) ---
                        nattempt += nchains
                        if pool is not None:
                            new_lps = np.array(pool.map(logpost, list(proposals)))
                        elif self._vectorized_logpost is not None:
                            new_lps = self._vectorized_logpost(proposals)
                        else:
                            new_lps = np.array([logpost(proposals[j])
                                                for j in range(nchains)])

                        # --- acceptance (JIT) ---
                        _rlu = np.log(rng.random(nchains))
                        naccept += _accept_step(
                            pos[:, m], logp[:, m], proposals, new_lps,
                            log_facs, betas[m], _rlu, nchains)

                i = old_steps + k
                chain[i] = pos[:, 0]
                log_prob[i] = logp[:, 0]

                # --- adaptive temperature ladder ---
                if self.adapt_temps:
                    safe = pair_attempt > 0
                    rates = np.zeros(ntemps - 1)
                    rates[safe] = pair_swap[safe] / pair_attempt[safe]
                    self._adapt_betas(rates, i)
                    betas = self.betas
                    betas_history[i] = betas.copy()

                if pbar is not None:
                    acc = naccept / max(nattempt, 1) * 100
                    _post = {'acc': f'{acc:.1f}%'}
                    if ntemps > 1 and nswap_attempt > 0:
                        _post['swap'] = f'{nswap/nswap_attempt*100:.1f}%'
                    pbar.set_postfix(_post, refresh=False)
                    pbar.update(1)

                if save_every > 0 and (i + 1) % save_every == 0:
                    self._chain = chain[:i + 1]
                    self._log_prob = log_prob[:i + 1]
                    self._pos_full = pos
                    self._logp_full = logp
                    self.save(save_file)
                    last_saved = i + 1

                if (i + 1) % check_every == 0 and i > 2 * nchains:
                    chi2 = -2.0 * log_prob[:i + 1]
                    burnndx = _find_burnin(chi2)
                    post_burn = chain[burnndx:i + 1]

                    if post_burn.shape[0] > 10:
                        Rhat, Tz = _gelman_rubin(post_burn)
                        max_gr = float(np.max(Rhat))
                        min_tz = float(np.min(Tz))
                        acc = naccept / max(nattempt, 1) * 100
                        swap_s = (f"; swap={nswap/nswap_attempt*100:.1f}%"
                                  if ntemps > 1 and nswap_attempt > 0 else "")
                        if progress:
                            save_s = (f" | saved at step {last_saved}"
                                      if last_saved is not None else "")
                            _msg = (f"  {100*(i+1)/nsteps:5.1f}% | "
                                    f"accept={acc:.1f}%{swap_s} | "
                                    f"GR={max_gr:.4f} (<{self.maxgr}) | "
                                    f"Tz={min_tz:.0f} (>{self.mintz}){save_s}")
                            if pbar is not None:
                                pbar.write(_msg)
                            else:
                                print("\r" + _msg + "   ", end="", flush=True)

                        if max_gr < self.maxgr and min_tz > self.mintz:
                            npass += 1
                            if npass >= npass_required:
                                converged = True
                                final_step = i + 1
                                break
                        else:
                            npass = 0
                if converged:
                    break
                if deadline is not None and _timer() >= deadline:
                    final_step = i + 1
                    if progress:
                        print(f"\n  Time limit reached at step {final_step}/{nsteps}. Stopping.")
                    break
        finally:
            if pool is not None:
                pool.close()
                pool.join()
            if pbar is not None:
                pbar.close()

        chain = chain[:final_step]
        log_prob = log_prob[:final_step]

        if progress:
            if converged:
                print(f"\n  Converged at step {final_step}/{nsteps}.")
            elif deadline is not None and final_step < nsteps:
                pass  # already printed above
            else:
                print(f"\n  Reached max steps ({nsteps}). NOT converged.")

        self._chain = chain
        self._log_prob = log_prob
        self._pos_full = pos
        self._logp_full = logp
        self._acceptance_rate = naccept / max(nattempt, 1)
        if self.adapt_temps:
            self._betas_history = betas_history[:final_step]
        return converged

    # ----- emcee-compatible API ----------------------------------------------

    def get_chain(self, flat=False, discard=0, thin=1):
        """
        Return the chain array, compatible with emcee's API.

        Parameters
        ----------
        flat : bool
            If True, flatten steps and chains into one axis.
        discard : int
            Number of leading steps to discard (burn-in).
        thin : int
            Keep every thin-th step.

        Returns
        -------
        ndarray : (nsteps, nchains, ndim)  or  (nsteps*nchains, ndim) if flat
        """
        if self._chain is None:
            raise RuntimeError("No chain data. Call run() first.")
        chain = self._chain[discard::thin]   # (nsteps, nchains, ndim)
        if flat:
            return chain.reshape(-1, self.ndim)
        return chain

    def get_log_prob(self, flat=False, discard=0, thin=1):
        """
        Return the log-probability array, compatible with emcee's API.

        Returns
        -------
        ndarray : (nsteps, nchains)  or  (nsteps*nchains,) if flat
        """
        if self._log_prob is None:
            raise RuntimeError("No chain data. Call run() first.")
        lp = self._log_prob[discard::thin]   # (nsteps, nchains)
        if flat:
            return lp.ravel()
        return lp

    def get_autocorr_time(self, discard=0, c=5, tol=10, quiet=False):
        """
        Return NaN array — DEMCPT uses GR/Tz for convergence, not autocorr time.
        Emcee-compatible signature; mcmc_output.py handles NaN gracefully.
        """
        return np.full(self.ndim, np.nan)

    @property
    def acceptance_fraction(self):
        """Overall acceptance rate replicated per chain (emcee-compatible)."""
        return np.full(self.nchains, self._acceptance_rate)

    # ----- convergence properties --------------------------------------------

    @property
    def flatchain(self):
        """Flat chain with burn-in removed and bad chains discarded."""
        if self._chain is None:
            return None
        chi2 = -2.0 * self._log_prob
        burnndx, good = _getburnndx(chi2)
        return self._chain[burnndx:, good].reshape(-1, self.ndim)

    @property
    def flatlog_prob(self):
        """Flat log-posterior with burn-in removed and bad chains discarded."""
        if self._log_prob is None:
            return None
        chi2 = -2.0 * self._log_prob
        burnndx, good = _getburnndx(chi2)
        return self._log_prob[burnndx:, good].ravel()

    # ----- diagnostics -------------------------------------------------------

    def summary(self, param_names=None, logger=None):
        """Print convergence summary and return diagnostics dict.

        Parameters
        ----------
        param_names : list of str, optional
            Human-readable parameter names.
        logger : callable, optional
            Logging function (e.g. ``logprint``).  Defaults to ``print``.
        """
        _log = logger if logger is not None else print

        if self._chain is None:
            _log("No chain. Call run() first.")
            return None

        chi2 = -2.0 * self._log_prob
        burnndx, good = _getburnndx(chi2)
        n_bad = self.nchains - len(good)

        post_burn = self._chain[burnndx:, good]
        Rhat, Tz = _gelman_rubin(post_burn)

        nsteps = self._chain.shape[0]
        _log(f"\nConvergence check (DEMCPT)")
        _log(f"----------------------------")
        _log(f"Steps: {nsteps}  Burn-in: {burnndx}  "
             f"Good chains: {len(good)}/{self.nchains}")
        if n_bad:
            _log(f"  WARNING: {n_bad} bad chain(s) discarded")
        _log("")
        _log(f"{'#':>4s}  {'Parameter':<16s} {'Rhat':>8s} {'Tz':>10s}  Status")
        _log("-" * 52)
        for d in range(self.ndim):
            name = param_names[d] if param_names is not None and d < len(param_names) else str(d)
            ok = Rhat[d] < self.maxgr and Tz[d] > self.mintz
            mark = "OK" if ok else "**BAD**"
            _log(f"{d:4d}  {name:<16s} {Rhat[d]:8.4f} {Tz[d]:10.1f}  {mark}")

        all_ok = all(
            Rhat[d] < self.maxgr and Tz[d] > self.mintz
            for d in range(self.ndim)
        )
        _log("")
        if all_ok:
            _log("All parameters converged (Rhat < %.2f, Tz > %d)." %
                 (self.maxgr, self.mintz))
        else:
            _log("WARNING: Some parameters have NOT converged!")

        return {"Rhat": Rhat, "Tz": Tz, "burnin": burnndx,
                "good_chains": good, "n_bad": n_bad}

    # ----- HDF5 I/O ----------------------------------------------------------

    def save(self, filename):
        """Save sampler state to HDF5 file."""
        if self._chain is None:
            raise RuntimeError("No chain data. Call run() first.")
        with h5py.File(filename, "w") as f:
            f.create_dataset("chain", data=self._chain, compression="gzip",
                             compression_opts=4)
            f.create_dataset("log_prob", data=self._log_prob, compression="gzip",
                             compression_opts=4)
            if self._pos_full is not None:
                f.create_dataset("pos", data=self._pos_full, compression="gzip",
                                 compression_opts=4)
            if self._logp_full is not None:
                f.create_dataset("logp_full", data=self._logp_full, compression="gzip",
                                 compression_opts=4)
            if self._betas_history is not None:
                f.create_dataset("betas_history", data=self._betas_history,
                                 compression="gzip", compression_opts=4)
            cfg = f.create_group("config")
            cfg.attrs["ndim"] = self.ndim
            cfg.attrs["nchains"] = self.nchains
            cfg.attrs["ntemps"] = self.ntemps
            cfg.attrs["maxgr"] = self.maxgr
            cfg.attrs["mintz"] = self.mintz
            cfg.attrs["stretch"] = self.stretch
            cfg.attrs["stretch_fraction"] = self.stretch_fraction
            cfg.attrs["adapt_temps"] = self.adapt_temps
            cfg.attrs["adapt_halflife"] = self.adapt_halflife
            cfg.create_dataset("betas", data=self.betas)

    @classmethod
    def load(cls, filename, log_posterior):
        """Load sampler state from HDF5 file."""
        with h5py.File(filename, "r") as f:
            chain = f["chain"][:]
            log_prob_data = f["log_prob"][:]
            pos_full = f["pos"][:] if "pos" in f else None
            logp_full = f["logp_full"][:] if "logp_full" in f else None
            cfg = f["config"]
            ndim = int(cfg.attrs["ndim"])
            nchains = int(cfg.attrs["nchains"])
            ntemps = int(cfg.attrs["ntemps"])
            maxgr = float(cfg.attrs["maxgr"])
            mintz = float(cfg.attrs["mintz"])
            stretch = bool(cfg.attrs["stretch"])
            stretch_fraction = float(cfg.attrs.get("stretch_fraction", 0.0))
            adapt_temps = bool(cfg.attrs.get("adapt_temps", False))
            adapt_halflife = float(cfg.attrs.get("adapt_halflife", 1000))
            betas = cfg["betas"][:]
            betas_history = f["betas_history"][:] if "betas_history" in f else None

        sampler = cls(log_posterior, ndim=ndim, nchains=nchains,
                      ntemps=ntemps, stretch=stretch, stretch_fraction=stretch_fraction,
                      maxgr=maxgr, mintz=mintz,
                      adapt_temps=adapt_temps, adapt_halflife=adapt_halflife)
        sampler.betas = betas
        sampler._chain = chain
        sampler._log_prob = log_prob_data
        sampler._pos_full = pos_full
        sampler._logp_full = logp_full
        sampler._betas_history = betas_history
        return sampler

    def save_as_emcee_backend(self, filename):
        """
        Write the cold chain as an emcee-compatible HDF5 file.

        The output file has the same structure as emcee's HDFBackend so that
        all downstream allesfast functions (mcmc_output, allesclass, etc.)
        work unchanged by reading mcmc_save.h5.

        Format (group /mcmc):
            attrs : version, nwalkers, ndim, has_blobs, iteration
            datasets : chain (nsteps, nwalkers, ndim)
                       log_prob (nsteps, nwalkers)
                       accepted (nwalkers,)
        """
        if self._chain is None:
            raise RuntimeError("No chain data. Call run() first.")
        nsteps, nchains, ndim = self._chain.shape
        with h5py.File(filename, "w") as f:
            g = f.create_group("mcmc")
            g.attrs["version"] = "1.0.0_demcpt"
            g.attrs["nwalkers"] = nchains
            g.attrs["ndim"] = ndim
            g.attrs["has_blobs"] = False
            g.attrs["iteration"] = nsteps
            g.create_dataset("chain", data=self._chain, dtype="float64",
                             compression="gzip", compression_opts=4)
            g.create_dataset("log_prob", data=self._log_prob, dtype="float64",
                             compression="gzip", compression_opts=4)
            accepted = g.create_dataset("accepted", shape=(nchains,), dtype="float64")
            accepted[:] = self._acceptance_rate * nsteps
