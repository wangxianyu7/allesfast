# Multi-star (triple/quadruple) extensibility plan

## Background

Current codebase supports binary (A+B) systems. This document records what needs
to change to support triple (A+B+C) or quadruple (A+B+C+D) systems.

## Status of each component

### Already N-star ready (no changes needed)

| File | Why it's ready |
|------|---------------|
| `star/mist_sed.py::mist_chi2` | Uses `labels = ['A','B','C','D']` and iterates over `stars` list |
| `star/mist_sed.py::sed_chi2` | Derives `nstars = len(stars)` dynamically |
| `star/sed_utils.py::mistmultised` | Fully vectorised over `nstars` axis |
| `star/massradius_mist.py::massradius_mist` | Single-star function, called per star |
| `star/models.py::StellarInputs` | Generic dataclass |

### Needs changes

#### 1. `computer.py` — star construction (hardcoded A/B)

Current pattern:
```python
star_A = StellarInputs(teff=..., rstar=..., mstar=..., eep=..., age=..., feh=..., av=..., distance=...)
stars = [star_A]
if params.get('B_rstar') is not None:
    star_B = StellarInputs(teff=..., rstar=..., mstar=..., eep=..., age=..., feh=..., av=..., distance=...)
    stars.append(star_B)
```

Proposed refactor — replace with loop:
```python
stars = []
for label in ['A', 'B', 'C', 'D']:
    if label == 'A' or params.get(f'{label}_rstar') is not None:
        stars.append(StellarInputs(
            teff     = params.get(f'{label}_teff'),
            rstar    = params.get(f'{label}_rstar'),
            mstar    = params.get(f'{label}_mstar'),
            eep      = params.get(f'{label}_eep'),
            age      = params.get(f'{label}_age'),
            feh      = params.get(f'{label}_feh'),   # coupled_with A_feh via params.csv
            av       = params.get(f'{label}_av', params.get('A_av')),   # fallback to A_av
            distance = _distance,
        ))
```

**Tricky detail**: B/C/D typically inherit `feh`, `av`, `parallax`, `age` from A via
`coupled_with` in params.csv. The `params` dict will already contain the resolved
(coupled) values at call time, so the loop above works as long as params.csv is set
up correctly.

#### 2. `basement.py` — `nstars` detection (line ~972)

Current (only detects B_):
```python
self.nstars = 2 if any(k.startswith('B_') for k in self.allkeys) else 1
```

Proposed fix:
```python
self.nstars = 1 + sum(
    1 for label in ['B', 'C', 'D']
    if any(k.startswith(f'{label}_') for k in self.allkeys)
)
```

#### 3. `deriver.py` / `general_output.py` / `mcmc_output.py`

Need to check for any hardcoded `B_` prefix loops in output / derived-parameter
tables. Expected to be minor.

## params.csv conventions for triple system

```
# Star C — feh, av, parallax, age all coupled to A
C_rstar,0.7,1,uniform 0.1 2.0,...
C_mstar,0.7,1,uniform 0.1 2.0,...
C_teff,4500,1,normal 4500 200,...
C_eep,300,1,uniform 200 500,...
C_feh,0.1,0,fixed,...,A_feh,0
C_av,0.05,0,fixed,...,A_av,0
C_parallax,9.5,0,fixed,...,A_parallax,0
C_age,10.7,0,fixed,...,A_age,0.1
```

## Effort estimate

- `computer.py` refactor: ~30 lines changed
- `basement.py` nstars fix: 1 line
- output files audit: ~1 hour
- Total: small, safe to do when a real triple-star system is needed

## Related bugs fixed (2026-02-24)

### EEP divergence in binary (B_eep 342→254)

**Root cause**: EXOFASTv2's `massradius_mist.pro` lines 265-268 always apply an
age chi2 penalty (the `if keyword_set(fitage)` guard was commented out).
allesfast was missing this term, so B_eep was only constrained by Teff/Rstar/[Fe/H]
from the SED, not by the system age.

Without age chi2:  B_eep=254 → derived age 0.5 Gyr (wrong!)
With age chi2:     B_eep=342 → derived age 9.7 Gyr ✓ (matches fixed age 10.7 Gyr)

**Fix applied**:
1. `star/massradius_mist.py` — added `age=None` param; when provided, appends
   `((mistage - age) / (percenterror * mistage))^2` to chi2
2. `star/mist_sed.py` — passes `s.age` into `massradius_mist()`
3. `computer.py` — adds `age=params.get('B_age', None)` to star_B construction

This fix is automatically correct for C/D stars once the loop refactor is done.
