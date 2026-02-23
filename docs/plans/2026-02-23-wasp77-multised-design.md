# WASP-77 Multi-SED Fit — Design Document

**Date:** 2026-02-23

---

## 1. System Overview

WASP-77 is a visual binary (separation 3.3″, unresolved by TESS) hosting the hot Jupiter WASP-77Ab (P = 1.360 d).

| Component | Teff (K) | Rstar (Rsun) | Mstar (Msun) | EEP |
|-----------|----------|--------------|--------------|-----|
| Star A (G-type) | 5500 | 0.9471 | 0.9031 | 395.2 |
| Star B (K-type) | 4622 | 0.7004 | 0.7322 | 342.5 |

Both stars share: initfeh = 0.1477, Av = 0.0575, distance = 104.94 pc.
Age difference constraint: σ = 0.1 Gyr.

**Instruments:**

| Name | File | Type | RM? |
|------|------|------|-----|
| TESS120 | n20240518.TESS.120.flux | phot (120 s) | — |
| TESS600 | n20240518.TESS.600.flux | phot (600 s) | — |
| CORALIE | n20240817.CORALIE0RV.rv | RV only | no |
| HARPS | n20240817.HARPS0RV.rv | RV only | no |
| HARPSjiang0 | n20240817.HARPJiang0RM.rv | RM night 0 | V-band |
| HARPSjiang1 | n20240817.HARPJiang10RM.rv | RM night 1 | V-band |
| HARPSjiang2 | n20240817.HARPJiang20RM.rv | RM night 2 | V-band |
| HARPSW77 | n20240817.HARPSW770RM.rv | RM | V-band |

RV files are in **m/s** and must be converted to km/s (÷ 1000).

---

## 2. Code Changes

### 2a. `coupled_tolerance` column in params.csv

Extend the existing `coupled_with` mechanism with an 8th column.

**`basement.py`:**
- Parse column 8 (`coupled_tolerance`) as float array, default 0.
- Store as `self.coupled_tolerance`.
- Existing exact-link logic (`coupled_with` + tolerance = 0) unchanged.

**`computer.py` — `lnprob`:**
After all `update_params` calls, apply soft-link Gaussian penalties:
```python
for i, key in enumerate(config.BASEMENT.allkeys):
    tol = config.BASEMENT.coupled_tolerance[i]
    ref_key = config.BASEMENT.coupled_with[i]
    if ref_key and tol > 0:
        val = params.get(key)
        ref = params.get(ref_key)
        if val is not None and ref is not None:
            lnp -= 0.5 * ((val - ref) / tol)**2
```

### 2b. `massradius_mist.py` — expose mistage

Add helper (reuses existing bilinear interpolation):
```python
def get_mistage(eep, mstar, feh, vvcrit=None, alpha=None) -> float:
    """Return derived age (Gyr) from EEP track interpolation."""
```

Modify `mist_chi2` in `star/mist_sed.py` to also store `mistage` into
`params['A_age']` / `params['B_age']` so the `coupled_tolerance` mechanism
can act on them.

### 2c. Multi-star SED support

**`basement.py`:**
```python
# detect nstars from presence of B_* fitted params
nstars = 2 if any(k.startswith('B_') for k in self.fitkeys) else 1
self.sed_data = read_sed_file(_sed_path, nstars=nstars)
```

**`star/mist_sed.py` — `sed_chi2` and `mist_chi2`:**
Accept `StellarInputs | list[StellarInputs]`. Single-star path unchanged.

**`computer.py` — `lnprob`:**
```python
stars = [star_A]
if params.get('B_rstar') is not None:
    star_B = StellarInputs(
        teff=params['B_teff'], rstar=params['B_rstar'],
        mstar=params['B_mstar'], eep=params['B_eep'],
        feh=params['B_feh'],        # coupled → A_feh
        av=params['A_av'],
        distance=A_distance,
    )
    stars.append(star_B)
lnp -= 0.5 * mist_chi2(stars)
lnp -= 0.5 * sed_chi2(stars, ...)
```

---

## 3. WASP-77 params.csv (key entries)

Column format: `name,value,fit,bounds,label,unit,coupled_with,coupled_tolerance`

### Planet b
```
b_rr,0.1290,1,uniform 0 1,$R_b/R_\star$,,
b_rsuma,0.214012,1,uniform 0 1,$(R_\star+R_b)/a_b$,,
b_cosi,0.04198,1,uniform 0 1,$\cos i_b$,,
b_epoch,2457420.8836,1,uniform 2457419.88 2457421.88,$t_0$,BJD_TDB,
b_period,1.360029,1,uniform 1.26 1.46,$P_b$,d,
b_K,0.3264,1,uniform 0 2,$K_b$,km/s,
b_f_c,-0.0646,1,uniform -1 1,$\sqrt{e}\cos\omega$,,
b_f_s,0.0625,1,uniform -1 1,$\sqrt{e}\sin\omega$,,
b_dilution_TESS120,0.214,1,normal 0.214 0.05,$\delta_{\rm TESS120}$,,
b_dilution_TESS600,0.214,1,normal 0.214 0.05,$\delta_{\rm TESS600}$,,
```

### Star A
```
A_rstar,0.9471,1,normal 0.9471 0.05,$R_A$,Rsun,
A_mstar,0.9031,1,normal 0.9031 0.05,$M_A$,Msun,
A_teff,5500,1,normal 5500 132,$T_{\rm eff,A}$,K,
A_feh,0.1477,1,normal 0.1477 0.08,$[\rm Fe/H]_A$,dex,
A_eep,395.2,1,uniform 200 500,$\rm EEP_A$,,
A_av,0.0575,1,uniform 0 1,$A_v$,mag,
A_parallax,9.512,1,normal 9.512 0.118,$\varpi_A$,mas,
A_lambda,2.62,1,uniform -180 180,$\lambda$,deg,
A_vsini,2.600,1,normal 2.600 0.500,$v\sin i$,km/s,
A_xi,1.6188,1,normal 1.619 0.5,$\xi$,km/s,
A_zeta,2.7299,1,normal 2.730 0.5,$\zeta$,km/s,
A_age,10.709,0,fixed,$\rm age_A$,Gyr,   ← derived; set by mist_chi2
```

### Star B (free params + coupled params)
```
B_rstar,0.7004,1,normal 0.7004 0.05,$R_B$,Rsun,
B_mstar,0.7322,1,normal 0.7322 0.05,$M_B$,Msun,
B_teff,4622,1,normal 4622 200,$T_{\rm eff,B}$,K,
B_eep,342.5,1,uniform 200 500,$\rm EEP_B$,,
B_feh,0.1477,0,fixed,$[\rm Fe/H]_B$,dex,A_feh,0    ← exact link
B_av,0.0575,0,fixed,$A_{v,B}$,mag,A_av,0            ← exact link
B_parallax,9.512,0,fixed,$\varpi_B$,mas,A_parallax,0 ← exact link
B_age,10.709,0,fixed,$\rm age_B$,Gyr,A_age,0.1       ← soft link σ=0.1 Gyr
```

---

## 4. settings.csv (key entries)

```
companions_phot,b
companions_rv,b
inst_phot,TESS120 TESS600
inst_rv,CORALIE HARPS HARPSjiang0 HARPSjiang1 HARPSjiang2 HARPSW77
mcmc_sampler,demcpt
mcmc_total_steps,20000
mcmc_nwalkers,64
mcmc_ntemps,8
mcmc_thin_by,10
mcmc_maxgr,1.01
mcmc_mintz,1000
de_ngen,10000
use_mist_prior,True
use_sed_prior,True
sed_file,1129033.sed
t_exp_TESS120,0.001389     # 120s in days
t_exp_n_int_TESS120,1
t_exp_TESS600,0.006944     # 600s in days
t_exp_n_int_TESS600,5
A_ld_law_TESS120,quad
A_ld_law_TESS600,quad
A_ld_law_HARPSjiang0,quad
A_ld_law_HARPSjiang1,quad
A_ld_law_HARPSjiang2,quad
A_ld_law_HARPSW77,quad
b_flux_weighted_HARPSjiang0,True
b_flux_weighted_HARPSjiang1,True
b_flux_weighted_HARPSjiang2,True
b_flux_weighted_HARPSW77,True
cornerplot,False
```

---

## 5. Data file preparation

- Rename `*.flux` → `*.Tran.csv` (comma-delimited, no header)
- Convert all `*.rv` from m/s → km/s (divide col 2 and col 3 by 1000)
- Rename `*.rv` → `*.RV.csv`

---

## 6. Implementation order

1. Add `coupled_tolerance` to `basement.py` parser and `computer.py` lnprob loop
2. Add `get_mistage()` helper to `massradius_mist.py`; modify `mist_chi2` to write `A_age`/`B_age` into params
3. Extend `sed_chi2` / `mist_chi2` to accept `list[StellarInputs]`
4. Extend `computer.py` lnprob to build `star_B` when `B_rstar` present
5. Extend `basement.py` to detect `nstars` and pass to `read_sed_file`
6. Prepare WASP-77 data files (convert RV units, rename)
7. Write `params.csv` and `settings.csv`
8. Smoke test with `show_initial_guess`
