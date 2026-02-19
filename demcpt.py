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
  - Automatic burn-in via _find_burnin (median chi² crossing)
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
def _find_burnin(neg2logp):
    """
    Burn-in index: the latest step at which any chain first crosses
    below the median -2*logpost.

    Parameters
    ----------
    neg2logp : ndarray (nsteps, nchains)

    Returns
    -------
    burnin : int
    """
    nsteps, nchains = neg2logp.shape

    flat = neg2logp.ravel().copy()
    flat.sort()
    n = len(flat)
    median_val = (flat[n // 2 - 1] + flat[n // 2]) / 2.0 if n % 2 == 0 else flat[n // 2]

    burn = 0
    for c in range(nchains):
        for t in range(nsteps):
            if neg2logp[t, c] <= median_val:
                if t > burn:
                    burn = t
                break
    return burn


@njit(cache=True)
def _identify_good_chains(neg2logp):
    """
    Discard chains stuck in local minima.
    A chain is "bad" if its median -2*logpost exceeds
    (overall median of chain medians) + 5 * 1.4826 * MAD.

    Parameters
    ----------
    neg2logp : ndarray (nsteps, nchains)

    Returns
    -------
    good : int64 array of good chain indices
    """
    nsteps, nchains = neg2logp.shape

    medians = np.empty(nchains)
    for c in range(nchains):
        col = neg2logp[:, c].copy()
        col.sort()
        n = len(col)
        medians[c] = (col[n // 2 - 1] + col[n // 2]) / 2.0 if n % 2 == 0 else col[n // 2]

    sm = medians.copy()
    sm.sort()
    n = len(sm)
    overall = (sm[n // 2 - 1] + sm[n // 2]) / 2.0 if n % 2 == 0 else sm[n // 2]

    devs = np.abs(medians - overall)
    ds = devs.copy()
    ds.sort()
    mad = (ds[n // 2 - 1] + ds[n // 2]) / 2.0 if n % 2 == 0 else ds[n // 2]

    threshold = overall + 5.0 * 1.4826 * mad

    good = []
    for c in range(nchains):
        if medians[c] <= threshold:
            good.append(c)
    return np.array(good, dtype=np.int64)


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
    seed : int or None
    """

    def __init__(self, log_posterior, ndim, nchains=None, ntemps=1, Tf=200.0,
                 stretch=False, maxgr=1.01, mintz=1000, seed=None):
        self.logpost_func = log_posterior
        self.ndim = ndim
        self.nchains = nchains or max(2 * ndim, 3)
        self.ntemps = ntemps
        self.stretch = stretch
        self.maxgr = maxgr
        self.mintz = mintz
        self.rng = np.random.default_rng(seed)

        # temperature ladder  (betas[0]=1 cold, betas[-1]=1/Tf hot)
        if ntemps > 1:
            self.betas = (1.0 / Tf) ** (np.arange(ntemps) / (ntemps - 1))
        else:
            self.betas = np.array([1.0])

        self.gamma = 2.38 / np.sqrt(2.0 * ndim)
        self.a_stretch = 2.0

        # results (populated by run)
        self._chain = None
        self._log_prob = None
        self._pos_full = None     # (nchains, ntemps, ndim)
        self._logp_full = None    # (nchains, ntemps)
        self._acceptance_rate = np.nan

    # ----- proposal helpers --------------------------------------------------

    def _de_proposals_batch(self, m, pos, scale):
        """Generate DE proposals for ALL chains at temperature m."""
        nc = self.nchains
        ndim = self.ndim
        rng = self.rng
        proposals = np.empty((nc, ndim))
        for j in range(nc):
            pool = np.delete(np.arange(nc), j)
            r1, r2 = rng.choice(pool, 2, replace=False)
            jitter = (rng.random(ndim) - 0.5) * scale / 10.0
            proposals[j] = (pos[j, m]
                            + self.gamma * (pos[r1, m] - pos[r2, m] + jitter))
        return proposals, np.zeros(nc)

    def _stretch_proposals_batch(self, m, pos):
        """Generate stretch-move proposals for ALL chains at temperature m."""
        nc = self.nchains
        ndim = self.ndim
        rng = self.rng
        a = self.a_stretch
        proposals = np.empty((nc, ndim))
        log_facs = np.empty(nc)
        for j in range(nc):
            r1 = rng.integers(0, nc - 1)
            if r1 >= j:
                r1 += 1
            z = ((a - 1.0) * rng.random() + 1.0) ** 2 / a
            proposals[j] = pos[r1, m] + z * (pos[j, m] - pos[r1, m])
            log_facs[j] = (ndim - 1) * np.log(z)
        return proposals, log_facs

    # ----- main loop ---------------------------------------------------------

    def run(self, p0, nsteps, nthin=1, scale=None, progress=True,
            check_every=None, npass_required=6, nworkers=1,
            save_every=0, save_file=None):
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

        # ---- initialise walkers (nchains, ntemps, ndim) ---------------------
        pos = np.empty((nchains, ntemps, ndim))
        logp = np.full((nchains, ntemps), -np.inf)

        if progress:
            print(f"Initialising {nchains} chains x {ntemps} temps ...")

        for j in range(nchains):
            for m in range(ntemps):
                niter = 0
                while True:
                    if j == 0 and niter == 0:
                        trial = p0.copy()
                    else:
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

        if progress:
            workers_s = f" ({nworkers} workers)" if use_pool else ""
            print(f"Running MCMC{workers_s} ...")

        pool = Pool(nworkers) if use_pool else None
        try:
            for i in range(1, nsteps):
                for _thin in range(nthin):
                    for m in range(ntemps):

                        if m < ntemps - 1:
                            for j in range(nchains):
                                if rng.random() < 0.5:
                                    nswap_attempt += 1
                                    log_alpha = ((betas[m] - betas[m + 1])
                                                 * (logp[j, m + 1] - logp[j, m]))
                                    if np.log(rng.random()) < log_alpha:
                                        nswap += 1
                                        pos[j, m], pos[j, m + 1] = (
                                            pos[j, m + 1].copy(), pos[j, m].copy())
                                        logp[j, m], logp[j, m + 1] = (
                                            logp[j, m + 1], logp[j, m])

                        if self.stretch:
                            proposals, log_facs = self._stretch_proposals_batch(
                                m, pos)
                        else:
                            proposals, log_facs = self._de_proposals_batch(
                                m, pos, scale)

                        nattempt += nchains
                        if pool is not None:
                            new_lps = np.array(pool.map(logpost, list(proposals)))
                        else:
                            new_lps = np.array([logpost(proposals[j])
                                                for j in range(nchains)])

                        for j in range(nchains):
                            if np.isfinite(new_lps[j]):
                                log_alpha = (betas[m] * (new_lps[j] - logp[j, m])
                                             + log_facs[j])
                                if np.log(rng.random()) < log_alpha:
                                    naccept += 1
                                    pos[j, m] = proposals[j]
                                    logp[j, m] = new_lps[j]

                    chain[i] = pos[:, 0]
                    log_prob[i] = logp[:, 0]

                if save_every > 0 and (i + 1) % save_every == 0:
                    self._chain = chain[:i + 1]
                    self._log_prob = log_prob[:i + 1]
                    self._pos_full = pos
                    self._logp_full = logp
                    self.save(save_file)
                    last_saved = i + 1
                    if progress:
                        pct = 100 * (i + 1) / nsteps
                        acc = naccept / max(nattempt, 1) * 100
                        swap_s = (f"; swap={nswap/nswap_attempt*100:.1f}%"
                                  if ntemps > 1 and nswap_attempt > 0 else "")
                        print(f"\r  {pct:5.1f}% | accept={acc:.1f}%{swap_s}   ",
                              end="", flush=True)

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
                            print(
                                f"\r  {100*(i+1)/nsteps:5.1f}% | "
                                f"accept={acc:.1f}%{swap_s} | "
                                f"GR={max_gr:.4f} (<{self.maxgr}) | "
                                f"Tz={min_tz:.0f} (>{self.mintz}){save_s}   ",
                                end="", flush=True)

                        if max_gr < self.maxgr and min_tz > self.mintz:
                            npass += 1
                            if npass >= npass_required:
                                converged = True
                                final_step = i + 1
                                break
                        else:
                            npass = 0
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        chain = chain[:final_step]
        log_prob = log_prob[:final_step]

        if progress:
            if converged:
                print(f"\n  Converged at step {final_step}/{nsteps}.")
            else:
                print(f"\n  Reached max steps ({nsteps}). NOT converged.")

        self._chain = chain
        self._log_prob = log_prob
        self._pos_full = pos
        self._logp_full = logp
        self._acceptance_rate = naccept / max(nattempt, 1)
        return converged

    def _run_continue(self, nsteps, nthin=1, scale=None, progress=True,
                      check_every=None, npass_required=6, nworkers=1,
                      save_every=0, save_file=None):
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
        total_steps = old_steps + nsteps
        chain = np.empty((total_steps, nchains, ndim))
        log_prob = np.empty((total_steps, nchains))
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

        if progress:
            workers_s = f" ({nworkers} workers)" if use_pool else ""
            print(f"Continuing MCMC{workers_s} for +{nsteps} steps ...")

        pool = Pool(nworkers) if use_pool else None
        try:
            for k in range(nsteps):
                for _thin in range(nthin):
                    for m in range(ntemps):
                        if m < ntemps - 1:
                            for j in range(nchains):
                                if rng.random() < 0.5:
                                    nswap_attempt += 1
                                    log_alpha = ((betas[m] - betas[m + 1])
                                                 * (logp[j, m + 1] - logp[j, m]))
                                    if np.log(rng.random()) < log_alpha:
                                        nswap += 1
                                        pos[j, m], pos[j, m + 1] = (
                                            pos[j, m + 1].copy(), pos[j, m].copy())
                                        logp[j, m], logp[j, m + 1] = (
                                            logp[j, m + 1], logp[j, m])

                        if self.stretch:
                            proposals, log_facs = self._stretch_proposals_batch(m, pos)
                        else:
                            proposals, log_facs = self._de_proposals_batch(m, pos, scale)

                        nattempt += nchains
                        if pool is not None:
                            new_lps = np.array(pool.map(logpost, list(proposals)))
                        else:
                            new_lps = np.array([logpost(proposals[j])
                                                for j in range(nchains)])

                        for j in range(nchains):
                            if np.isfinite(new_lps[j]):
                                log_alpha = (betas[m] * (new_lps[j] - logp[j, m])
                                             + log_facs[j])
                                if np.log(rng.random()) < log_alpha:
                                    naccept += 1
                                    pos[j, m] = proposals[j]
                                    logp[j, m] = new_lps[j]

                i = old_steps + k
                chain[i] = pos[:, 0]
                log_prob[i] = logp[:, 0]

                if save_every > 0 and (i + 1) % save_every == 0:
                    self._chain = chain[:i + 1]
                    self._log_prob = log_prob[:i + 1]
                    self._pos_full = pos
                    self._logp_full = logp
                    self.save(save_file)
                    last_saved = i + 1
                    if progress:
                        pct = 100 * (i + 1) / total_steps
                        acc = naccept / max(nattempt, 1) * 100
                        swap_s = (f"; swap={nswap/nswap_attempt*100:.1f}%"
                                  if ntemps > 1 and nswap_attempt > 0 else "")
                        print(f"\r  {pct:5.1f}% | accept={acc:.1f}%{swap_s}   ",
                              end="", flush=True)

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
                            print(
                                f"\r  {100*(i+1)/total_steps:5.1f}% | "
                                f"accept={acc:.1f}%{swap_s} | "
                                f"GR={max_gr:.4f} (<{self.maxgr}) | "
                                f"Tz={min_tz:.0f} (>{self.mintz}){save_s}   ",
                                end="", flush=True)

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
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        chain = chain[:final_step]
        log_prob = log_prob[:final_step]

        if progress:
            if converged:
                print(f"\n  Converged at step {final_step}/{total_steps}.")
            else:
                print(f"\n  Reached max steps ({total_steps}). NOT converged.")

        self._chain = chain
        self._log_prob = log_prob
        self._pos_full = pos
        self._logp_full = logp
        self._acceptance_rate = naccept / max(nattempt, 1)
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
        burnndx = _find_burnin(chi2)
        good = _identify_good_chains(chi2[burnndx:])
        return self._chain[burnndx:, good].reshape(-1, self.ndim)

    @property
    def flatlog_prob(self):
        """Flat log-posterior with burn-in removed and bad chains discarded."""
        if self._log_prob is None:
            return None
        chi2 = -2.0 * self._log_prob
        burnndx = _find_burnin(chi2)
        good = _identify_good_chains(chi2[burnndx:])
        return self._log_prob[burnndx:, good].ravel()

    # ----- diagnostics -------------------------------------------------------

    def summary(self, param_names=None):
        """Print convergence summary and return diagnostics dict."""
        if self._chain is None:
            print("No chain. Call run() first.")
            return None

        chi2 = -2.0 * self._log_prob
        burnndx = _find_burnin(chi2)
        good = _identify_good_chains(chi2[burnndx:])
        n_bad = self.nchains - len(good)

        post_burn = self._chain[burnndx:, good]
        Rhat, Tz = _gelman_rubin(post_burn)

        nsteps = self._chain.shape[0]
        print(f"Steps: {nsteps}  Burn-in: {burnndx}  "
              f"Good chains: {len(good)}/{self.nchains}")
        if n_bad:
            print(f"  WARNING: {n_bad} bad chain(s) discarded")
        print()
        print(f"{'#':>4s}  {'Parameter':<16s} {'Rhat':>8s} {'Tz':>10s}  Status")
        print("-" * 52)
        for d in range(self.ndim):
            name = param_names[d] if param_names is not None and d < len(param_names) else str(d)
            ok = Rhat[d] < self.maxgr and Tz[d] > self.mintz
            mark = "OK" if ok else "**BAD**"
            print(f"{d:4d}  {name:<16s} {Rhat[d]:8.4f} {Tz[d]:10.1f}  {mark}")

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
            cfg = f.create_group("config")
            cfg.attrs["ndim"] = self.ndim
            cfg.attrs["nchains"] = self.nchains
            cfg.attrs["ntemps"] = self.ntemps
            cfg.attrs["maxgr"] = self.maxgr
            cfg.attrs["mintz"] = self.mintz
            cfg.attrs["stretch"] = self.stretch
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
            betas = cfg["betas"][:]

        sampler = cls(log_posterior, ndim=ndim, nchains=nchains,
                      ntemps=ntemps, stretch=stretch, maxgr=maxgr, mintz=mintz)
        sampler.betas = betas
        sampler._chain = chain
        sampler._log_prob = log_prob_data
        sampler._pos_full = pos_full
        sampler._logp_full = logp_full
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
