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

We compare both against plain **emcee** (affine-invariant ensemble sampler,
no tempering) as a baseline.

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
| demcpt-bidir | 3.48 | 535 | 153.5 | YES | 49% / 51% |
| demcpt-unidi | 2.72 | 408 | 150.0 | YES | 50% / 50% |
| emcee | 0.79 | 440 | 554.7 | **NO** | 100% / 0% |

**Key finding**: emcee cannot escape the initial mode. Both DEMCPT variants
find both modes with nearly equal weight. Bidirectional has slightly higher
ESS but both are comparable.

## Test 2: Rosenbrock banana (2D)

Curved, banana-shaped posterior. Tests exploration of nonlinear degeneracies.
True mean is approximately (1, 1) for the unscaled version; our scaled
version (divided by 10) has a broader distribution.

| Sampler | Time (s) | min ESS | ESS/s | Mean (x, y) |
|---------|----------|---------|-------|-------------|
| demcpt-bidir | 4.28 | 791 | 184.7 | (0.79, 11.5) |
| demcpt-unidi | 4.22 | 256 | **60.6** | (1.41, 44.5) |
| emcee | 0.39 | 717 | 1847.5 | (1.16, 5.9) |

**Key finding**: Unidirectional is **3x worse** in ESS/s than bidirectional
on this problem and produces a biased mean. The hot chains, never receiving
cold-chain positions, don't help the cold chain track the curved degeneracy
efficiently.

## Test 3: Correlated Gaussian (20D)

20-dimensional Gaussian with random covariance matrix. Tests high-dimensional
efficiency.

| Sampler | Time (s) | min ESS | ESS/s |
|---------|----------|---------|-------|
| demcpt-bidir | 13.07 | 1328 | 101.6 |
| demcpt-unidi | 15.65 | 1481 | 94.6 |
| emcee | 0.78 | 729 | 938.5 |

**Key finding**: Both DEMCPT variants are comparable. emcee is faster per
ESS on this simple unimodal problem (no need for tempering overhead).

## Conclusions

1. **Mode hopping**: DEMCPT's core advantage — emcee fails entirely on
   multimodal distributions. Both swap modes succeed equally.

2. **Curved degeneracies**: Bidirectional swap is significantly better than
   unidirectional (3x ESS/s). Unidirectional hot chains lose information
   about the cold chain's location, reducing their ability to propose useful
   positions along curved ridges.

3. **Simple distributions**: emcee wins on raw ESS/s when tempering is
   unnecessary (unimodal targets), due to no multi-temperature overhead.

4. **Recommendation**: Keep **bidirectional as the default**. The
   unidirectional mode is available via `swap_mode='unidirectional'` for
   users who want to experiment, but bidirectional is equal or better in
   all tests conducted here.
