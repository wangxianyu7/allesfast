# Doppler Tomography (DT) Port to allesfast — Implementation Plan

**Source**: `EXOFASTv2/dopptom_chi2.pro` (Beatty 2015, Eastman 2017)
**Target**: `allesfast/dt/` module + likelihood hook in `computer.py`
**Estimated effort**: 5–7 person-days
**Validation goal**: per-observation χ² match EXOFASTv2 IDL to < 1% RMS

---

## 1. Physics summary

For each observation in a DT dataset, we model the velocity-space **shadow**
(deficit in CCF residuals) caused by the planet occulting part of the rotating
stellar disk.

**Per-step ingredients** (one MCMC step):

1. **Transit geometry** at each observation time `bjd_i`:
   - Sky-projected positions `(x_p, y_p, z)` from Kepler's eq (analytic up to e)
   - LOS velocity of occulted region: `up = x_p cos λ − y_p sin λ`
   - Transit "depth weight": `β = 1 − F_MA(z, k, u1, u2)` (Mandel-Agol)
2. **Planet shadow profile** at each `(v, t)`:
   - Elliptical: `S(v, t) = c1·sqrt(1 − ((v − up_t)/k)²)` for `|v−up_t| < k vsini`
   - `c1 = 2/(π·k)` is normalisation
3. **Broadening kernel**:
   - Gaussian σ = `sqrt(vline² + (c/R_spec)²) / (2·sqrt(2·ln2))`
   - Convolve `S(v, t) ⊛ G(v)`
4. **Combine**: `model[v, t] = median(CCF) + β_t · normalized(conv)`
5. **χ²**: `Σ ((CCF_obs − model) / (RMS · errscale))² / IndepVels`
   - `IndepVels = R_spec/(FWHM2σ · meanstep · vsini)` accounts for oversampling

All velocities normalised by `vsini` internally; the final model is in absolute
velocity units.

---

## 2. Helper functions to port

| IDL | Allesfast already has? | Action |
|-----|-----------------------|--------|
| `quadld(logg, teff, feh, 'V')` | ✅ `allesfast/limb_darkening/LDC3.py` | Wrap to return (u1, u2) for V band |
| `exofast_getphase(e, ω, /pri)` | ❌ | Port: ~10 lines. Phase of primary transit relative to periastron |
| `exofast_getb2(bjd, ...)` | Partial — `kepler.py` solves M→E | Port: extend to return (x, y, z) sky-plane positions |
| `exofast_occultquad_cel` | ✅ via PyTransit RoadRunnerModel | Wrap to give a single-time evaluator |
| `gaus_convol(x, y, σ)` | scipy | Replace with `scipy.signal.fftconvolve` |
| `exofast_like(resid, 0, σ, /chi2)` | ✅ standard Gaussian likelihood | Inline as `np.sum((resid/σ)²)` |

**Total new code in `allesfast/dt/core.py`**: ~150 lines.

---

## 3. Data format

EXOFASTv2 uses an IDL fits structure with fields:
- `bjd` (ntimes,) — observation BJDs
- `vel` (nvels,) — velocity grid in km/s
- `stepsize` (nvels,) — Δv per pixel (NOT constant if grid is logarithmic)
- `ccf2d` (nvels, ntimes) — observed CCF residuals (median-subtracted)
- `rms` scalar or (nvels,) — pixel RMS estimate
- `rspec` scalar — spectrograph resolving power R

**Allesfast format** (npz, easier than fits):
```python
np.savez('inst.dt.npz',
    bjd=bjd,           # (ntimes,)
    vel=vel,           # (nvels,) km/s
    stepsize=stepsize, # (nvels,) km/s
    ccf2d=ccf2d,       # (nvels, ntimes)
    rms=rms,           # scalar or (nvels,)
    rspec=rspec,       # scalar
)
```

Reader in `allesfast/dt/io.py`: ~30 lines.

---

## 4. Parameter registration

### settings.csv additions

```csv
inst_dt,instA instB             # space-separated DT instrument labels
dt_file_instA,instA.dt.npz
dt_file_instB,instB.dt.npz
```

### params.csv additions

| Param | Type | Purpose |
|-------|------|---------|
| `A_vsini` | already exists | star's projected rotation |
| `A_vline_<INST>` | NEW | intrinsic line width (km/s) for inst `INST` |
| `A_dt_errscale_<INST>` | NEW | DT-specific error scaling (analog to `A_sed_errscale`) |
| `b_lambda` | already exists | spin-orbit angle |
| RM-related (vxi/vzeta) | already exists | available if needed |

Note: `vline` is allowed to be per-instrument because different
spectrographs sample different sets of lines.

---

## 5. Module layout

```
allesfast/dt/
├── __init__.py
├── core.py          # dopptom_chi2_single (main computation)
├── io.py            # read_dt_file, validate
├── geometry.py      # phase/getb2/orbit helpers (or extend kepler.py)
└── plotting.py      # shadow 2D image, observed/model/residual panels
```

---

## 6. Integration with allesfast core

### `basement.py`

```python
# in __init__:
self.settings['inst_dt'] = parse_inst_list('inst_dt')   # default: []
if self.settings['inst_dt']:
    from .dt.io import read_dt_file
    self.dt_data = {}
    for inst in self.settings['inst_dt']:
        path = self.settings[f'dt_file_{inst}']
        self.dt_data[inst] = read_dt_file(path)
```

### `computer.py` `calculate_lnlike_total`

After RV/flux blocks, add:
```python
for inst in config.BASEMENT.settings.get('inst_dt', []):
    chi2 = dopptom_chi2_single(
        dt_data=config.BASEMENT.dt_data[inst],
        params=params, companion=companion, inst=inst,
    )
    if not np.isfinite(chi2):
        return -np.inf
    lnlike_total += -0.5 * chi2
```

### `general_output.py`

After per-transit plots, call `plot_dt(inst)` for each DT instrument
to produce `mcmc_dt_<INST>.pdf` with the three-panel display
(observed shadow / model / residual).

---

## 7. Implementation phases

### Phase 1 — Core math (1.5 days)
- [ ] Port `exofast_getphase` → `dt/geometry.py:phase_of_primary_transit`
- [ ] Port `exofast_getb2` → `dt/geometry.py:sky_positions`
  - Solve Kepler's eq (allesfast already has this), return `x, y, z` in stellar radii
- [ ] Implement `gaussian_conv_fft(v, profile, sigma)` using `scipy.signal.fftconvolve`
- [ ] Write `dopptom_chi2_single(dt_data, k, t0, period, e, ω, cosi, ar, λ,
  u1, u2, vsini, vline, errscale, c=299792.458) → chi2`

### Phase 2 — I/O + integration (1 day)
- [ ] `read_dt_file(path)` → dict with required keys + sanity checks
- [ ] `basement.py` parse `inst_dt`, load dt_data into BASEMENT
- [ ] Register `A_vline_<INST>`, `A_dt_errscale_<INST>` parameter prefixes
- [ ] Hook into `calculate_lnlike_total` (one new for-loop)

### Phase 3 — Numerical validation (1 day)
- [ ] Pick one DT-bearing system you've already fit with EXOFASTv2 IDL
  (e.g. K2-140, WASP-33-like)
- [ ] Save the same `doptom` struct to npz
- [ ] Compute chi2 in both EXOFASTv2 and allesfast at identical parameters
- [ ] Acceptance: per-step chi2 agreement < 1% (allowance for numerical
  precision in gaussian convolution edge effects)

### Phase 4 — Plotting (1 day)
- [ ] 2D shadow figure: observed, model, residual side-by-side
- [ ] Time-series of individual line profiles (waterfall)
- [ ] χ² history vs time
- [ ] Save as `results/mcmc_dt_<INST>.pdf`

### Phase 5 — End-to-end MCMC run (0.5 day)
- [ ] Set up a small DT-bearing fit (e.g. existing K2-140 + a synthetic DT)
- [ ] Run DE → amoeba → MCMC, verify λ recovery
- [ ] Confirm runtime hit < 50% per logpost (compared to without DT)

### Phase 6 — Benchmark + report (0.5–1 day)
- [ ] `Benchmarks&Verifications/dt/`:
  - `bench_dt.py` (or whatever name)
  - `dt_comparison_report.txt`: chi² match with EXOFASTv2 over a parameter grid
  - `REPORT.md`: methodology, results, conclusions

---

## 8. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| `gaus_convol` IDL vs scipy edge effects | Use `mode='same'` with manual zero-padding wider than 5σ; validate on a known-input case |
| Sub-sample independent vels factor wrong | Cross-check `IndepVels` formula with EXOFASTv2 exactly |
| Per-instrument vline cross-talk with vsini | RM data + DT data co-fitted should constrain both; benchmark on synthetic |
| Performance: 100ms/step DT × 1000s of steps | Vectorize the time-loop with broadcasting once per-step works; consider numba `@njit` |
| LD coefficients for V band may not match observation bandpass | Allow `dt_ld_band_<INST>` setting (default 'V') |

---

## 9. Out-of-scope

- Multi-planet DT (rare; would need separate `up_t` for each planet)
- Joint CCF stack + line profile fit (tracit-style ring grid)
- Time-variable line shape (stellar pulsations, spots)

These can be Phase 7+ if the standard implementation works.
