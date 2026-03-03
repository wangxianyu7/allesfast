"""
mkprior2.py

Update an EXOFASTv2-style .priors file with best-fit median values from an
allesfast MCMC run.

Mirrors EXOZIPPy's mkprior2.pro / EXOFASTv2's mkprior2.pro: for each
parameter line in the original .priors file, the starting value is replaced
with the MCMC median while Gaussian widths, bounds, and comments are
preserved.

The output filename increments a numeric suffix:
    wasp18.priors   → wasp18.priors.2
    wasp18.priors.2 → wasp18.priors.3

Usage:
    from allesfast.utils.mkprior2 import mkprior2
    mkprior2('path/to/run', 'wasp18.priors')
"""

import os
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# EXOFASTv2 name → allesfast fitted param name
# ---------------------------------------------------------------------------
_EXOFAST_TO_ALLES = {
    'mstar':    'A_mstar',
    'rstar':    'A_rstar',
    'teff':     'A_teff',
    'feh':      'A_feh',
    'parallax': 'A_parallax',
    'vsini':    'A_vsini',
    'av':       'A_av',
    'eep':      'A_eep',
    'age':      'A_age',
}

# Keys that map to companion epoch / period
_EPOCH_KEYS  = {'tc', 't0', 'T0'}
_PERIOD_KEYS = {'period', 'per', 'P'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_priorfile(parfile):
    """Compute the next prior filename by incrementing a numeric suffix.

    wasp18.priors     → wasp18.priors.2
    wasp18.priors.2   → wasp18.priors.3
    """
    p = Path(parfile)
    suffix = p.suffix.lstrip('.')
    if suffix.isdigit():
        return str(p.with_suffix(f'.{int(suffix) + 1}'))
    return str(p) + '.2'


def _read_mcmc_table(results_dir):
    """Read mcmc_table.csv and return {param_name: median_value}.

    Skips comment lines, fixed parameters, and non-numeric medians.
    """
    table_file = os.path.join(results_dir, 'mcmc_table.csv')
    if not os.path.exists(table_file):
        raise FileNotFoundError(f'mcmc_table.csv not found in {results_dir}')

    medians = {}
    with open(table_file) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            parts = stripped.split(',')
            if len(parts) < 2:
                continue
            name = parts[0]
            try:
                medians[name] = float(parts[1])
            except ValueError:
                pass  # '(fixed)' or non-numeric
    return medians


def _format_val(val, name):
    """Format a numeric value with appropriate precision."""
    if name.startswith('tc') or name.startswith('period'):
        return f'{val:.10f}'
    elif name.startswith('gamma') or name.startswith('jittervar'):
        return f'{val:.8f}'
    elif name.startswith('parallax'):
        return f'{val:.5f}'
    elif abs(val) > 100:
        return f'{val:.5f}'
    elif abs(val) > 1:
        return f'{val:.6f}'
    else:
        return f'{val:.8f}'


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def mkprior2(datadir, priorfile, companions=None, outfile=None, verbose=True):
    """Write an updated .priors file using MCMC median values from allesfast.

    Parameters
    ----------
    datadir : str
        Path to the allesfast run directory (must contain results/mcmc_table.csv).
    priorfile : str or Path
        Path to the original EXOFASTv2-style .priors file.
    companions : list of str, optional
        Companion labels (e.g. ['b', 'c']). Default: ['b'].
    outfile : str or Path, optional
        Output path. If None, auto-increments the filename suffix.
    verbose : bool
        Print the output path when done.

    Returns
    -------
    str
        Path to the written prior file.
    """
    if companions is None:
        companions = ['b']

    results_dir = os.path.join(str(datadir), 'results')
    medians = _read_mcmc_table(results_dir)

    priorfile = str(priorfile)
    if outfile is None:
        outfile = _next_priorfile(priorfile)

    lines_out = []
    with open(priorfile) as f:
        for line in f:
            raw = line.rstrip('\n')

            # Preserve blank lines and pure-comment lines
            stripped = raw.lstrip()
            if not stripped or stripped.startswith('#'):
                lines_out.append(raw)
                continue

            # Split off inline comment
            if '#' in raw:
                code_part, comment_part = raw.split('#', 1)
                comment_part = '  #' + comment_part
            else:
                code_part = raw
                comment_part = ''

            parts = code_part.split()
            if len(parts) < 2:
                lines_out.append(raw)
                continue

            name = parts[0]
            vals = []
            for v in parts[1:]:
                try:
                    vals.append(float(v))
                except ValueError:
                    break

            if not vals:
                lines_out.append(raw)
                continue

            # Resolve the allesfast best-fit value
            bestval = None

            if name in _EXOFAST_TO_ALLES:
                bestval = medians.get(_EXOFAST_TO_ALLES[name])

            elif name in _EPOCH_KEYS:
                for companion in companions:
                    v = medians.get(f'{companion}_epoch')
                    if v is not None:
                        bestval = v
                        break

            elif name in _PERIOD_KEYS:
                for companion in companions:
                    v = medians.get(f'{companion}_period')
                    if v is not None:
                        bestval = v
                        break

            if bestval is not None and np.isfinite(bestval):
                new_parts = [name, _format_val(bestval, name)]
                if len(vals) >= 2:
                    new_parts.append(f'{vals[1]}')   # preserve sigma
                if len(vals) >= 3:
                    new_parts.append(f'{vals[2]}')   # preserve lo
                if len(vals) >= 4:
                    new_parts.append(f'{vals[3]}')   # preserve hi
                lines_out.append(' '.join(new_parts) + comment_part)
            else:
                lines_out.append(raw)

    with open(outfile, 'w') as f:
        f.write('\n'.join(lines_out) + '\n')

    if verbose:
        print(f'Updated priors: {outfile}')

    return outfile
