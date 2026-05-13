# KELT-17b Doppler Tomography port verification

This directory contains the TRES R≈44000 DT observation of KELT-17b from
[Zhou+ 2016](https://arxiv.org/abs/1607.03512), used as the reference
dataset for validating allesfast's `allesfast.dt.dopptom_chi2` against
EXOFASTv2's `dopptom_chi2.pro` (Beatty 2015, Eastman 2017).

## Files

| File | Description |
|------|-------------|
| `n20160226.KELT-17b.TRES.44000.fits` | DT residual CCFs (5273 vels × 33 times), R = 44000 |
| `test_dt_port.py` | Python χ² evaluation at best-fit |
| `test_dt_idl.pro` | IDL counterpart |
| `compare_models.py` | pixel-level model comparison |

## Validation result

Using best-fit parameters from EXOFASTv2's `kelt17.priors2`:

| Quantity | IDL `dopptom_chi2.pro` | allesfast `dopptom_chi2` |
|----------|------------------------|--------------------------|
| χ² (raw / IndepVels) | **511.10** | **511.102** |
| Model max value | 0.039730 | 0.039730 |
| Pixel-level max diff | — | 1.2 × 10⁻⁶ |
| Pixel-level mean abs diff | — | 6 × 10⁻⁹ |

The two implementations agree to **6 significant figures** in χ² and
~10⁻⁵ relative precision per pixel — well below MCMC noise.

## Best-fit parameter set used

```python
tc       = 2457287.7456103642        # primary transit BJD
period   = 10**0.4885753488 = 3.0801747 d
e        = 0.0823                    # = secosw² + sesinw²
omega    = arctan2(sesinw, secosw)  # = -31.97°
cosi     = 0.0817318222
k = Rp/Rs= 0.0926333230
ar = a/Rs= 6.6110                    # K3L: G·M*·P²/(4π²·R*³)
lambda   = -2.0164720094 rad         # = -115.54°
u1, u2   = 0.2788, 0.3468            # quadld('V', logg=4.27, Teff=7454, [Fe/H]=0)
vsini    = 44.2 km/s
vline    = 5.49 km/s
errscale = 3.39
R_spec   = 44000
```
