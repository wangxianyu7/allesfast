# KELT-17b — full joint fit example (transit + RV + RM + DT + SED)

End-to-end demonstration of allesfast for a hot, fast-rotating system.
Originally from the [EXOFASTv2 KELT-17 example](https://github.com/jdeast/EXOFASTv2/tree/master/examples/kelt17/DT).

## What's included

| Data type | Source / instruments | Count |
|-----------|---------------------|-------|
| **Transit photometry** | 6 ground-based telescopes (CROW, KUO, Whitin, Pvdk, WCO, MVRC, PEST, ZRO) × 5 bands (B, V, Sloan g/r/i/z, I) | **12 light curves** |
| **Radial velocity** | TRES (out-of-transit) | **1 file** |
| **Rossiter-McLaughlin RV** | TRES in-transit | **1 file** |
| **Doppler tomography** | TRES R = 44000 CCF residual stack | **1 dataset** (5273 vels × 33 frames) |
| **SED** | Tycho BT/VT + 2MASS J/H/K + AllWISE W1-W4 + Gaia G/Bp/Rp | **13 bands** |

The DT dataset is the same one we use for validating
`allesfast.dt.dopptom_chi2` against EXOFASTv2 to ~10⁻⁶ relative
precision (see `Benchmarks&Verifications/dt/REPORT.md`).

## File naming

```text
<DATE>.<TELESCOPE>.<BAND>.Tran.csv      # transit photometry
<DATE>.<INSTRUMENT>.RV.csv              # radial velocity
<INSTRUMENT>.csv                        # short alias used by basement
kelt17.sed                              # SED bandpasses + magnitudes
n20160226.KELT-17b.TRES.44000.fits      # DT CCF residual cube
```

Each instrument label is `<TELESCOPE>_<BAND>` for photometry,
`<INSTRUMENT>` for RV/RM, and `TRES_DT` for the DT dataset.

## Layout in `settings.csv`

```csv
companions_phot,b
companions_rv,b
inst_phot,CROW_I KUO_I KUO_V Pvdk_Sloanz Whitin_Sloang Whitin_Sloani \
          WCO_Sloanz MVRC_Sloang MVRC_Sloani PEST_B ZRO_I MVRC_Sloanr
inst_rv,TRES TRESRM
inst_dt,TRES_DT
dt_file_TRES_DT,n20160226.KELT-17b.TRES.44000.fits
dt_ld_band_TRES_DT,V
sed_file,kelt17.sed
use_mist_prior,True
use_sed_prior,True
```

## DT validation against EXOFASTv2 IDL

Using best-fit parameters from EXOFASTv2's `kelt17.priors2`:

| Quantity | IDL `dopptom_chi2.pro` | allesfast `dopptom_chi2` |
|----------|------------------------|--------------------------|
| χ² (raw / IndepVels) | 511.10 | **511.10** |
| Model pixel max | 0.039730 | 0.039730 |
| Pixel-level max diff | — | 1.2 × 10⁻⁶ |
| Pixel-level mean abs diff | — | 6 × 10⁻⁹ |

Full 40-point parameter sweep (lambda, vsini, vline, cosi):
**max relative diff = 2.76 × 10⁻⁶** — see
`Benchmarks&Verifications/dt/REPORT.md`.

## Quick start

```python
import allesfast
allesfast.show_initial_guess('examples/KELT-17_DT')   # plot at initial guess
allesfast.mcmc_fit('examples/KELT-17_DT')             # DE → Amoeba → MCMC
allesfast.mcmc_output('examples/KELT-17_DT')          # tables + diagnostic figs
```

The MCMC stage writes a 3-panel DT shadow plot
(`mcmc_dt_TRES_DT.pdf`) and the usual joint posterior tables / corner
plots / per-transit fits.

## Quick test (DT χ² only)

```bash
python test_dt_port.py   # Python: should print chi2 ≈ 511.10
idl -e ".r test_dt_idl"  # IDL    : should print chi2_raw ≈ 511.10
```
