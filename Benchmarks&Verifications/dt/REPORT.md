# Doppler Tomography (DT) Likelihood — Verification Report

**Date**: 2026-05-13
**Sub-module**: `allesfast/dt/`
**Reference**: EXOFASTv2 `dopptom_chi2.pro` (Beatty 2015, Eastman 2017)
**Test dataset**: KELT-17b TRES R=44000 spectrum (Zhou+ 2016)

---

## 1. Goal

Validate that `allesfast.dt.dopptom_chi2` reproduces the EXOFASTv2 IDL
implementation to numerical precision, across the relevant parameter space
that an MCMC fit will explore.

## 2. Method

We anchor at the EXOFASTv2 best-fit point for KELT-17b
(`kelt17.priors2`):

```
tc        = 2457287.7456104          # primary transit BJD
period    = 3.0801747 d              # 10**logp
e         = 0.0823                   # secosw² + sesinw²
omega     = -31.97°                  # arctan2(sesinw, secosw)
cosi      = 0.0817                   # → i = 85.31°
k = Rp/Rs = 0.0926
a/Rs      = 6.611                    # G·Mstar·P²/(4π²Rstar³)
lambda    = -115.54°                 # spin-orbit angle
u1, u2    = 0.2788, 0.3468           # ATLAS V band, log g = 4.27, Teff = 7454
vsini     = 44.2 km/s
vline     = 5.49 km/s
errscale  = 3.39
R_spec    = 44000
```

We sweep four parameters one at a time, holding the others at the anchor:

| Axis | Range | N points |
|------|-------|----------|
| `lambda` | −π … +π (rad) | 11 |
| `vsini`  | 35 … 55 km/s | 9 |
| `vline`  | 2 … 12 km/s | 11 |
| `cosi`   | 0.02 … 0.18 | 9 |

Total: **40 grid points**.  At each point we compute χ² in both Python
and IDL and compare.

### Convention adjustment

EXOFASTv2's reported "chi2" via `exofast_like(..., /chi2)` is actually
`-2·loglike = chisq + Σ log(2π σ²)`, divided by `IndepVels`.  To make a
direct comparison we subtract the constant log-normalisation term in IDL:

```
chi2_raw_idl = (idl_returned * IndepVels - Σlog(2π σ²)) / IndepVels
```

This gives the same "raw χ² normalised by `IndepVels`" quantity that the
Python port returns directly.

## 3. Results

```
Per-axis statistics
-------------------------------------------------------------------------
  lambda    N=11  max|Δχ²|=4.93e-04  mean|Δχ²|=3.16e-04  max rel=9.2e-07
  vsini     N= 9  max|Δχ²|=2.81e-04  mean|Δχ²|=1.18e-04  max rel=5.4e-07
  vline     N=11  max|Δχ²|=9.88e-05  mean|Δχ²|=5.73e-05  max rel=1.9e-07
  cosi      N= 9  max|Δχ²|=1.55e-03  mean|Δχ²|=3.52e-04  max rel=2.8e-06

Overall: max|Δχ²|=1.55e-03  max rel=2.76e-06  mean rel=3.87e-07
```

**Verdict: PASS** (threshold: relative diff < 10⁻³)

The two implementations agree to **~6 significant figures** of χ²
everywhere across the parameter sweep.  Residual differences (≤3×10⁻⁶
relative) are dominated by:

- floating-point ordering in the per-velocity Gaussian convolution
- minor differences in Mandel-Agol numerics (PyTransit's `eval_quad_z_v`
  vs IDL's `exofast_occultquad_cel`)

Both are well below MCMC noise (typical chain σ ≳ 1 in χ²).

## 4. Reproducing

```bash
cd allesfast/Benchmarks&Verifications/dt/

# Python first: writes the parameter grid and computes its own χ²
python bench_dt.py

# IDL: reads the grid, computes EXOFASTv2 χ² at the same points
idl -e ".r bench_dt_idl"

# Python again: now loads the IDL results and emits the comparison report
python bench_dt.py
```

Output files:

- `bench_grid_input.csv` — the parameter grid (Python → IDL)
- `bench_grid_idl_chi2.csv` — IDL χ² per grid point
- `bench_dt_results.csv` — joined table (axis, value, χ²_py, χ²_idl, |Δ|, rel)
- `bench_dt_report.txt` — text summary

## 5. Why this matters

- DT data is the **strongest direct probe of stellar obliquity** for
  fast-rotating hosts; bias in χ² at the 10⁻³ level would translate to
  ~σ/3 bias in `λ` posteriors for typical DT datasets.
- A passing benchmark here is a prerequisite for trusting allesfast DT
  fits for publication — and for joint RM+DT analyses.

## 6. Limitations & follow-up

The benchmark covers a 1-D sweep in 4 parameters.  Things **not yet** tested:

- 2-D and 3-D parameter cross-checks (off-anchor)
- Behaviour very close to limit (`vsini→0`, `vline→0`, `cosi→1`, `e→1`)
- Multi-DT-dataset fits (only TRES used here)
- Edge cases in the Gaussian convolution (small Rspec, large vline)

These would be useful in Phase 7 if higher precision becomes important.

## 7. Files

```
Benchmarks&Verifications/dt/
├── PLAN.md                       implementation plan
├── REPORT.md                     this file
├── bench_dt.py                   Python sweep + comparison
├── bench_dt_idl.pro              IDL counterpart
├── bench_dt_report.txt           generated text summary
├── bench_dt_results.csv          joined per-point comparison
├── bench_grid_input.csv          parameter grid (Python → IDL)
└── bench_grid_idl_chi2.csv       IDL χ² (one per point)
```
