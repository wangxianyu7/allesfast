# TTV Prior Design

**Date:** 2026-02-24
**Feature:** Linear-ephemeris chi2 penalty from observed transit midtimes

## Problem

Observed transit midtimes (e.g. from WASP-77 literature) constrain the orbital period `P` and epoch `tc` independently of the photometric data. Currently there is no mechanism to include these as a likelihood term without entering full TTV-fitting mode.

## File Format

```
# BJD   sigma(days)
2455870.4505400  0.0010417
2456271.6588800  0.0009722
...
```

Two-column whitespace-separated. Auto-discovered in `datadir` by pattern `*.{companion}.ttv` (e.g. `n20240817.b.ttv` → companion `b`).

## Mathematical Form

For each observed midtime `T_obs_i` with uncertainty `σ_i`:

```
N_i     = round((T_obs_i − tc) / P)
T_pred_i = tc + N_i × P
χ²      = Σ_i (T_obs_i − T_pred_i)² / σ_i²
lnp    += −0.5 × χ²
```

No extra jitter parameter; fixed σ from file.

## Activation

In `settings.csv`:
```
use_ttv_prior,True
```

No other configuration needed — file discovery is automatic.

## Architecture

### 1. `basement.py` — `__init__` after `load_data()`

Scan `datadir` for `*.{companion}.ttv` files and load into:
```python
self.ttv_data = {
    'b': (T_obs_array, sigma_array),   # shape (N,)
    ...
}
```

### 2. `computer.py` — `calculate_external_priors(params)`

New block (after soft-link penalties, before eccentricity constraints):
```python
if config.BASEMENT.settings.get('use_ttv_prior', False):
    for companion, (T_obs, sigma) in config.BASEMENT.ttv_data.items():
        tc = params[companion + '_epoch']
        P  = params[companion + '_period']
        if P <= 0:
            return -np.inf
        N      = np.round((T_obs - tc) / P)
        T_pred = tc + N * P
        lnp   += -0.5 * np.sum((T_obs - T_pred)**2 / sigma**2)
```

## Testing

- Unit test: penalty is zero when params match the linear ephemeris exactly
- Unit test: penalty grows correctly when tc or P are offset
- Unit test: file discovery finds `*.b.ttv` → companion `b`
- Unit test: `use_ttv_prior,False` → no effect even if `.ttv` files exist

## Non-Goals

- No jitter/scale parameter on σ
- Not related to `fit_ttvs` (which samples individual transit times as free parameters)
- No σ inflation for correlated noise
