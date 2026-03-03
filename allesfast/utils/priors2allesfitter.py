"""
priors2allesfitter.py

Convert an EXOFASTv2-style .priors file to allesfitter params.csv entries.

EXOFASTv2 prior syntax (per line):
    param value                       # starting value only
    param value sigma                 # Gaussian(value, sigma)
    param value -1 lo hi              # Uniform(lo, hi)
    param value sigma lo hi           # TruncNormal(value, sigma, lo, hi)

Mapping to allesfitter params.csv:
    mstar  val            -> A_mstar   uniform 0.1 10.0
    rstar  val            -> A_rstar   uniform 0.1 20.0
    teff   val            -> A_teff    uniform 3500 50000
    feh    val sig        -> A_feh     normal val sig
    parallax val sig      -> A_parallax normal val sig
    tc     val [sig]      -> b_epoch      uniform val-1 val+1
    period val [sig]      -> b_period     uniform val-1 val+1
    vsini  val -1 lo hi   -> A_vsini   uniform lo hi
    vsini  val sig        -> A_vsini   normal val sig
    # av / appks are SED-only, skipped

Usage:
    from allesfast.utils.priors2allesfitter import priors_to_params

    lines = priors_to_params('380255458.priors', companion='b')
    for l in lines:
        print(l)
"""

import os
import re
import numpy as np


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_priors(filepath):
    """
    Parse an EXOFASTv2 .priors file.

    Returns a dict:
        { param_name: {'value': float, 'sigma': float or None,
                       'lo': float or None, 'hi': float or None} }
    sigma == -1.0 means uniform prior (lo, hi set).
    """
    parsed = {}
    with open(filepath) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            # strip inline comments
            line = line.split('#')[0].strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 2:
                continue

            name = tokens[0]
            try:
                vals = [float(t) for t in tokens[1:]]
            except ValueError:
                continue

            entry = {'value': vals[0], 'sigma': None, 'lo': None, 'hi': None}

            if len(vals) == 2:
                # param value sigma
                entry['sigma'] = vals[1]
            elif len(vals) == 4:
                # param value sigma lo hi  (sigma=-1 → uniform)
                entry['sigma'] = vals[1]
                entry['lo']    = vals[2]
                entry['hi']    = vals[3]

            parsed[name] = entry
    return parsed


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

# Parameters that are SED photometric data (go into sed file, not params.csv) → skip
_SKIP = {'appks', 's_Av', 'appks_err'}

# Default wide uniform bounds for stellar params (when no sigma given)
_STELLAR_UNIFORM = {
    'mstar':    (0.1,   10.0),
    'rstar':    (0.1,   20.0),
    'teff':     (3500., 50000.),
}

# allesfitter param names
_ALLES_NAME = {
    'mstar':    'A_mstar',
    'rstar':    'A_rstar',
    'teff':     'A_teff',
    'feh':      'A_feh',
    'parallax': 'A_parallax',
    'vsini':    'A_vsini',
    'av':       'A_av',
    'age':      'A_age',
}

# Labels for params.csv
_LABEL = {
    'A_mstar':    r'$M_\star$',
    'A_rstar':    r'$R_\star$',
    'A_teff':     r'$T_{\rm eff}$',
    'A_feh':      r'$[\rm Fe/H]$',
    'A_parallax': r'$\varpi$',
    'A_vsini':    r'$v \sin i_\star$',
    'A_av':       r'$A_V$',
    'A_age':      r'$\rm Age$',
}

_UNIT = {
    'A_mstar':    r'$M_\odot$',
    'A_rstar':    r'$R_\odot$',
    'A_teff':     'K',
    'A_feh':      'dex',
    'A_parallax': 'mas',
    'A_vsini':    'km/s',
    'A_av':       'mag',
    'A_age':      'Gyr',
}


def _make_row(name, value, bounds_str, label='', unit=''):
    """Return a formatted params.csv line (no trailing newline)."""
    # Use fixed notation for large values (e.g. BJD epochs)
    val_str = f'{value:.6f}' if abs(value) >= 1000 else f'{value:.6g}'
    return f'{name},{val_str},1,{bounds_str},{label},{unit}'


def priors_to_params(priorfile, companion='b',
                     tc_window=1.0, period_window=1.0):
    """
    Convert a .priors file to allesfitter params.csv rows.

    Parameters
    ----------
    priorfile : str
        Path to the EXOFASTv2-style .priors file.
    companion : str
        Planet label used in allesfitter (e.g. 'b').
    tc_window : float
        Half-width of the uniform prior on epoch [days].
    period_window : float
        Half-width of the uniform prior on period [days].

    Returns
    -------
    list of str
        Lines ready to be appended to params.csv.
        Each line follows the format:  name,value,fit,bounds,label,unit
    """
    parsed = _parse_priors(priorfile)
    rows = []

    # -----------------------------------------------------------------------
    # Stellar params: mstar, rstar, teff
    # -----------------------------------------------------------------------
    for exo_name in ('mstar', 'rstar', 'teff'):
        if exo_name not in parsed:
            continue
        e = parsed[exo_name]
        val = e['value']
        alles_name = _ALLES_NAME[exo_name]
        lo, hi = _STELLAR_UNIFORM[exo_name]
        bounds = f'uniform {lo:.6g} {hi:.6g}'
        rows.append(_make_row(alles_name, val, bounds,
                              _LABEL[alles_name], _UNIT[alles_name]))

    # -----------------------------------------------------------------------
    # feh, parallax: Gaussian if sigma given, else wide uniform
    # -----------------------------------------------------------------------
    for exo_name in ('feh', 'parallax'):
        if exo_name not in parsed:
            continue
        e = parsed[exo_name]
        val   = e['value']
        sigma = e['sigma']
        lo    = e['lo']
        hi    = e['hi']
        alles_name = _ALLES_NAME[exo_name]

        if sigma is not None and sigma > 0:
            if lo is not None and hi is not None:
                bounds = f'trunc_normal {val:.6g} {sigma:.6g} {lo:.6g} {hi:.6g}'
            else:
                bounds = f'normal {val:.6g} {sigma:.6g}'
        elif sigma is not None and sigma < 0 and lo is not None and hi is not None:
            bounds = f'uniform {lo:.6g} {hi:.6g}'
        else:
            # fallback wide uniform
            defaults = {'feh': (-5.0, 1.0), 'parallax': (0.0, 100.0)}
            lo_d, hi_d = defaults.get(exo_name, (0.0, 1.0))
            bounds = f'uniform {lo_d:.6g} {hi_d:.6g}'

        rows.append(_make_row(alles_name, val, bounds,
                              _LABEL[alles_name], _UNIT[alles_name]))

    # -----------------------------------------------------------------------
    # av: extinction — uniform prior (lo, hi from dust maps)
    # -----------------------------------------------------------------------
    if 'av' in parsed:
        e = parsed['av']
        val   = e['value']
        sigma = e['sigma']
        lo    = e['lo']
        hi    = e['hi']
        if sigma is not None and sigma < 0 and lo is not None and hi is not None:
            bounds = f'uniform {lo:.6g} {hi:.6g}'
        elif sigma is not None and sigma > 0:
            if lo is not None and hi is not None:
                bounds = f'trunc_normal {val:.6g} {sigma:.6g} {lo:.6g} {hi:.6g}'
            else:
                bounds = f'normal {val:.6g} {sigma:.6g}'
        else:
            bounds = 'uniform 0 1'
        rows.append(_make_row('A_av', val, bounds, _LABEL['A_av'], _UNIT['A_av']))

    # -----------------------------------------------------------------------
    # age: stellar age in Gyr — free parameter, wide uniform if not in file
    # -----------------------------------------------------------------------
    if 'age' in parsed:
        e = parsed['age']
        val   = e['value']
        sigma = e['sigma']
        lo    = e['lo']
        hi    = e['hi']
        if sigma is not None and sigma < 0 and lo is not None and hi is not None:
            bounds = f'uniform {lo:.6g} {hi:.6g}'
        elif sigma is not None and sigma > 0:
            if lo is not None and hi is not None:
                bounds = f'trunc_normal {val:.6g} {sigma:.6g} {lo:.6g} {hi:.6g}'
            else:
                bounds = f'normal {val:.6g} {sigma:.6g}'
        else:
            bounds = 'uniform 0 13.8'
        rows.append(_make_row('A_age', val, bounds, _LABEL['A_age'], _UNIT['A_age']))
    else:
        # age not in priors file → add as free parameter with wide uniform prior
        rows.append(_make_row('A_age', 5.0, 'uniform 0 13.8',
                              _LABEL['A_age'], _UNIT['A_age']))

    # -----------------------------------------------------------------------
    # vsini
    # -----------------------------------------------------------------------
    if 'vsini' in parsed:
        e = parsed['vsini']
        val   = e['value']
        sigma = e['sigma']
        lo    = e['lo']
        hi    = e['hi']

        if sigma is not None and sigma < 0 and lo is not None and hi is not None:
            bounds = f'uniform {lo:.6g} {hi:.6g}'
        elif sigma is not None and sigma > 0:
            bounds = f'normal {val:.6g} {sigma:.6g}'
        else:
            bounds = 'uniform 0 100'

        rows.append(_make_row('A_vsini', val, bounds,
                              _LABEL['A_vsini'], _UNIT['A_vsini']))

    # -----------------------------------------------------------------------
    # tc → b_epoch (or <companion>_epoch)
    # -----------------------------------------------------------------------
    for tc_key in ('tc', 't0', 'T0'):
        if tc_key not in parsed:
            continue
        e = parsed[tc_key]
        val   = e['value']
        sigma = e['sigma']
        alles_name = f'{companion}_epoch'

        if sigma is not None and sigma > 0:
            bounds = f'normal {val:.6f} {sigma:.6f}'
        else:
            bounds = f'uniform {val - tc_window:.6f} {val + tc_window:.6f}'

        rows.append(_make_row(alles_name, val, bounds,
                              rf'$T_{{0;{companion}}}$', r'$\mathrm{BJD}$'))
        break

    # -----------------------------------------------------------------------
    # period → b_period (or <companion>_period)
    # -----------------------------------------------------------------------
    for p_key in ('period', 'per', 'P'):
        if p_key not in parsed:
            continue
        e = parsed[p_key]
        val   = e['value']
        sigma = e['sigma']
        alles_name = f'{companion}_period'

        if sigma is not None and sigma > 0:
            bounds = f'normal {val:.8f} {sigma:.8f}'
        else:
            bounds = f'uniform {val - period_window:.8f} {val + period_window:.8f}'

        rows.append(_make_row(alles_name, val, bounds,
                              rf'$P_{{{companion}}}$', r'$\mathrm{d}$'))
        break

    return rows


def update_params_csv(priorfile, params_csv, companion='b',
                      tc_window=1.0, period_window=1.0):
    """
    Write priors_to_params() output into params_csv, updating existing
    rows where the parameter name already exists, appending new ones.

    Parameters
    ----------
    priorfile : str
        Path to .priors file.
    params_csv : str
        Path to params.csv to update.
    companion, tc_window, period_window : see priors_to_params().
    """
    new_rows = priors_to_params(priorfile, companion=companion,
                                tc_window=tc_window,
                                period_window=period_window)
    # Build lookup: name → full row string
    new_lookup = {}
    for row in new_rows:
        name = row.split(',')[0]
        new_lookup[name] = row

    # Read existing params.csv
    with open(params_csv) as f:
        existing = f.readlines()

    updated = []
    seen = set()
    for line in existing:
        stripped = line.strip()
        if stripped.startswith('#') or not stripped:
            updated.append(line)
            continue
        name = stripped.split(',')[0]
        if name in new_lookup:
            updated.append(new_lookup[name] + '\n')
            seen.add(name)
        else:
            updated.append(line)

    # Append any rows not already present
    appended = [n for n in new_lookup if n not in seen]
    if appended:
        updated.append('#stellar and orbital priors from .priors file\n')
        for name in appended:
            updated.append(new_lookup[name] + '\n')

    with open(params_csv, 'w') as f:
        f.writelines(updated)

    print(f'Updated {params_csv}:')
    for row in new_rows:
        name = row.split(',')[0]
        status = 'updated' if name in seen else 'appended'
        print(f'  [{status}] {row}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert EXOFASTv2 .priors to allesfitter params.csv entries')
    parser.add_argument('priorfile', help='.priors file path')
    parser.add_argument('--params-csv', help='params.csv to update in-place')
    parser.add_argument('--companion', default='b', help='Planet label (default: b)')
    parser.add_argument('--tc-window',     type=float, default=1.0)
    parser.add_argument('--period-window', type=float, default=1.0)
    args = parser.parse_args()

    if args.params_csv:
        update_params_csv(args.priorfile, args.params_csv,
                          companion=args.companion,
                          tc_window=args.tc_window,
                          period_window=args.period_window)
    else:
        rows = priors_to_params(args.priorfile, companion=args.companion,
                                tc_window=args.tc_window,
                                period_window=args.period_window)
        print('#name,value,fit,bounds,label,unit')
        for r in rows:
            print(r)
