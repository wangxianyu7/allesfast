# Benchmark: PT Swap Mode (bidirectional vs unidirectional) vs emcee

**Date**: 2026-03-18
**Machine**: macOS Darwin 24.6.0 (Apple Silicon), single-threaded
**Script**: `bench_swap_mode.py`

## Background

DEMCPT supports parallel tempering with two swap strategies:

- **bidirectional** (default): Standard PT — cold and hot chains exchange
  positions upon acceptance. Satisfies detailed balance.
- **unidirectional** (EXOFASTv2 style): Cold chain adopts hot chain's
  position; hot chain keeps its own. Hot chains are never "polluted" by
  cold-chain positions.

We compare both against **emcee** with four different move strategies:

- **stretch**: Default affine-invariant stretch move (Goodman & Weare 2010)
- **DE**: Differential evolution move
- **DESnooker**: Snooker move (ter Braak & Vrugt 2008)
- **DE+Snooker**: 80% DE + 20% Snooker mix

## Setup

| Parameter | Value |
|-----------|-------|
| DEMCPT nchains | max(2*ndim, 10) |
| DEMCPT ntemps | 8 |
| DEMCPT Tf | 100 |
| emcee nwalkers | max(2*ndim, 10), even |
| Burn-in discard | 30% |
| ESS method | Batch means (20 batches) |
| Seed | 42 |

## Test 1: Bimodal Gaussian (5D, separation = 6sigma)

Two equal-weight Gaussians separated by 6sigma in each dimension.
Tests ability to hop between well-separated modes.

| Sampler | Time (s) | min ESS | ESS/s | Both modes? | Frac mode1 / mode2 |
|---------|----------|---------|-------|-------------|---------------------|
| demcpt-bidir | 3.42 | 535 | 156.6 | **YES** | 49% / 51% |
| demcpt-unidi | 2.68 | 408 | 152.0 | **YES** | 50% / 50% |
| emcee-stretch | 0.78 | 440 | 566.6 | NO | 100% / 0% |
| emcee-DE | 0.81 | 1020 | 1259.0 | NO | 100% / 0% |
| emcee-DESnooker | 1.31 | 120 | 91.3 | YES* | 91% / 9% |
| emcee-DE+Snooker | 0.92 | 899 | 973.9 | NO | 100% / 0% |

*DESnooker technically finds both modes but with severely unequal sampling
(91%/9%) and very low ESS.

**Key findings**:
- emcee with stretch, DE, and DE+Snooker moves **cannot escape the initial
  mode**. All walkers start near mode 1 and the proposals (based on
  differences between walkers) never span the inter-mode valley.
- DESnooker's long-range snooker jumps occasionally cross the valley, but
  the sampling is heavily biased and ESS/s is the lowest of all methods.
- Both DEMCPT variants reliably find both modes with near-equal weight
  (49-51%). This is the fundamental advantage of parallel tempering: hot
  chains flatten the valley and pass information to cold chains via swaps.

## Test 2: Rosenbrock banana (2D)

Curved, banana-shaped posterior. Tests exploration of nonlinear degeneracies.

| Sampler | Time (s) | min ESS | ESS/s | Mean (x, y) |
|---------|----------|---------|-------|-------------|
| demcpt-bidir | 4.20 | 791 | 188.4 | (0.79, 11.5) |
| demcpt-unidi | 4.20 | 256 | 60.8 | (1.41, 44.5) |
| emcee-stretch | 0.39 | 717 | 1858.0 | (1.16, 5.9) |
| emcee-DE | 0.42 | 292 | 698.4 | (0.91, 5.9) |
| emcee-DESnooker | 1.01 | 95 | 94.7 | (0.49, 3.7) |
| emcee-DE+Snooker | 0.54 | 1166 | 2145.0 | (0.60, 7.9) |

**Key findings**:
- Unidirectional DEMCPT is **3x worse** in ESS/s than bidirectional and
  produces a biased mean (y=44.5 vs ~6-12 for others).
- emcee moves work well here (unimodal, no valley to cross). The
  DE+Snooker mix achieves the highest ESS/s overall.
- DESnooker alone has poor ESS/s everywhere — it seems to be a niche move.

## Test 3: Correlated Gaussian (20D)

20-dimensional Gaussian with random covariance matrix. Tests high-dimensional
efficiency.

| Sampler | Time (s) | min ESS | ESS/s |
|---------|----------|---------|-------|
| demcpt-bidir | 12.79 | 1328 | 103.9 |
| demcpt-unidi | 15.19 | 1481 | 97.5 |
| emcee-stretch | 0.76 | 729 | 958.4 |
| emcee-DE | 0.79 | 1354 | 1704.5 |
| emcee-DESnooker | 2.91 | 407 | 139.6 |
| emcee-DE+Snooker | 1.20 | 862 | 720.4 |

**Key findings**:
- emcee-DE is the clear winner on this simple unimodal problem (highest
  ESS/s by far), as expected — no tempering overhead needed.
- Both DEMCPT variants are comparable (~100 ESS/s). The multi-temperature
  overhead dominates when tempering provides no benefit.

## Conclusions

1. **Multimodal distributions**: DEMCPT is the only reliable sampler.
   No emcee move variant can consistently discover and uniformly sample
   multiple modes. DESnooker occasionally crosses valleys but with
   heavily biased sampling and low efficiency.

2. **Bidirectional vs unidirectional PT swap**: Bidirectional (standard PT)
   is equal or better in all tests. Unidirectional severely underperforms
   on curved degeneracies (Rosenbrock). **Keep bidirectional as default.**

3. **Unimodal distributions**: emcee (especially with DE move) is
   significantly faster per ESS when tempering is unnecessary. This
   motivates using emcee for simple problems and DEMCPT when multimodality
   is suspected.

4. **emcee move choice**: DE move gives the best ESS/s on unimodal targets.
   The default stretch move is decent but not optimal. DESnooker is
   consistently the weakest in ESS/s.
