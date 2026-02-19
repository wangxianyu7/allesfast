"""
settings2allesfitter.py

Convert an EXOFASTv2-style settings.txt to an allesfitter settings.csv.

EXOFASTv2 settings.txt syntax (key=value; arrays as [a,b,...]):
    tranpath  = n*Tran.csv          # glob pattern for transit data files
    rvpath    = n*RV.csv            # glob pattern for RV data files
    nplanets  = 1                   # number of planets (companions b, c, ...)
    maxsteps  = 10000               # total MCMC steps
    nthin     = 100                 # MCMC thinning
    rossiter  = [1,0]               # per RV instrument: 1 = flux-weighted (RM)
    rmbands   = ["V","notrm",...]   # RM photometry band per instrument (informational)
    exptime   = [2.0,3.3,1]        # exposure time in minutes, phot then RV
    ninterp   = [1,2,1]            # interpolation points per instrument
    fittran   = [1,0]              # per phot instrument: 1 = include in fit
    fitrv     = [1,1]              # per RV instrument: 1 = include in fit

Instrument discovery (from file_naming.py conventions):
    Transit : nYYYYMMDD.band.Instru.Tran.csv → instrument = field[2], band = field[1]
    RV      : nYYYYMMDD.Instru.RV.csv        → instrument = field[1]
    Files sorted by date (field[0]).

Usage:
    from allesfast.utils.settings2allesfitter import settings_to_csv

    settings_to_csv('settings.txt')                       # writes settings.csv alongside
    settings_to_csv('settings.txt', 'my_settings.csv')   # custom output path
"""

import os
import re
import glob as _glob
import json
import numpy as np


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_settings(filepath):
    """
    Parse an EXOFASTv2-style settings.txt.

    Returns a dict with string values for scalars and lists for arrays.
    Array values are returned as lists of strings.
    """
    result = {}
    with open(filepath) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            line = line.split('#')[0].strip()
            if '=' not in line:
                continue
            key, _, val = line.partition('=')
            key = key.strip()
            val = val.strip()
            if val.startswith('['):
                # Parse as JSON-style array (handles both numeric and string arrays)
                try:
                    parsed = json.loads(val)
                    result[key] = parsed
                except json.JSONDecodeError:
                    # Fallback: strip brackets and split
                    inner = val.strip('[]')
                    result[key] = [x.strip().strip('"').strip("'")
                                   for x in inner.split(',')]
            else:
                result[key] = val
    return result


# ---------------------------------------------------------------------------
# Instrument discovery
# ---------------------------------------------------------------------------

def _discover_instruments(directory, pattern, ftype):
    """
    Glob for data files matching *pattern* in *directory*.

    ftype: 'phot' or 'rv'

    Returns a list of dicts sorted by date:
        phot: {'file': ..., 'instru': ..., 'band': ...}
        rv  : {'file': ..., 'instru': ...}
    """
    hits = sorted(_glob.glob(os.path.join(directory, pattern)))
    instruments = []
    for path in hits:
        base = os.path.basename(path)
        stem = base.rsplit('.', 1)[0]   # drop .csv
        fields = stem.split('.')
        # Expected: nYYYYMMDD[.band].Instru.{Tran|RV}
        # Tran: fields = [nYYYYMMDD, band, Instru, Tran]
        # RV  : fields = [nYYYYMMDD, Instru, RV]
        if ftype == 'phot' and len(fields) >= 4:
            instruments.append({
                'file':   path,
                'instru': fields[-2],
                'band':   fields[-3],
            })
        elif ftype == 'rv' and len(fields) >= 3:
            instruments.append({
                'file':   path,
                'instru': fields[-2],
            })
    return instruments


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

COMPANIONS = list('bcdefghij')   # up to 9 planets


def _ensure_ldc_params(params_csv, ld_insts):
    """
    Ensure params.csv contains host_ldc_q1_{inst} and host_ldc_q2_{inst}
    rows (with uniform 0-1 prior) for every instrument in *ld_insts*.
    Skips instruments already present.  Does nothing if params.csv missing.
    """
    if not os.path.isfile(params_csv):
        return
    with open(params_csv) as f:
        existing_names = set()
        lines = f.readlines()
        for line in lines:
            s = line.strip()
            if s and not s.startswith('#'):
                existing_names.add(s.split(',')[0])

    to_add = []
    for inst in ld_insts:
        for qi, label in [('q1', r'$q_{1; %s}$' % inst),
                          ('q2', r'$q_{2; %s}$' % inst)]:
            name = f'host_ldc_{qi}_{inst}'
            if name not in existing_names:
                to_add.append(f'{name},0.5,1,uniform 0.0 1.0,{label},\n')

    if to_add:
        with open(params_csv, 'a') as f:
            f.write('#limb darkening (auto-added by settings2allesfitter)\n')
            f.writelines(to_add)
        print(f'  Added LDC params for: {[i for i in ld_insts]}')


def _count_free_params(params_csv):
    """
    Count the number of fitted parameters (fit column == '1') in params.csv.
    Returns None if the file does not exist.
    """
    if not os.path.isfile(params_csv):
        return None
    count = 0
    with open(params_csv) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            cols = line.split(',')
            if len(cols) >= 3 and cols[2].strip() == '1':
                count += 1
    return count if count > 0 else None


def settings_to_csv(settings_path, output_path=None):
    """
    Convert a settings.txt to allesfitter settings.csv.

    Parameters
    ----------
    settings_path : str
        Path to the EXOFASTv2-style settings.txt.
    output_path : str, optional
        Where to write the output.  Defaults to settings.csv in the same
        directory as settings.txt.

    Returns
    -------
    str
        The generated settings.csv content.
    """
    directory   = os.path.dirname(os.path.abspath(settings_path))
    s           = _parse_settings(settings_path)
    if output_path is None:
        output_path = os.path.join(directory, 'settings.csv')

    # --- Count free parameters from params.csv for nwalkers -----------------
    params_csv = os.path.join(directory, 'params.csv')
    n_free = _count_free_params(params_csv)

    # --- Companions ---------------------------------------------------------
    nplanets   = int(s.get('nplanets', 1))
    companions = COMPANIONS[:nplanets]       # ['b'] or ['b','c'] etc.

    # --- Discover instruments -----------------------------------------------
    tranpath = s.get('tranpath', 'n*Tran.csv')
    rvpath   = s.get('rvpath',   'n*RV.csv')

    phot_all = _discover_instruments(directory, tranpath, 'phot')
    rv_all   = _discover_instruments(directory, rvpath,   'rv')

    # Filter by fittran / fitrv flags
    fittran = [int(x) for x in s.get('fittran', [1] * len(phot_all))]
    fitrv   = [int(x) for x in s.get('fitrv',   [1] * len(rv_all))]

    phot_insts = [p['instru'] for i, p in enumerate(phot_all)
                  if i < len(fittran) and fittran[i]]
    rv_insts   = [r['instru'] for i, r in enumerate(rv_all)
                  if i < len(fitrv) and fitrv[i]]

    all_insts_ordered = phot_all + rv_all   # for exptime/ninterp indexing

    # --- Exposure times & ninterp -------------------------------------------
    raw_exptime = s.get('exptime', [])
    raw_ninterp = s.get('ninterp', [])

    exptime_map = {}   # instru → days
    ninterp_map = {}   # instru → int

    for i, info in enumerate(all_insts_ordered):
        inst = info['instru']

        # Exposure time: use provided value if available, else auto-detect from data file
        if i < len(raw_exptime):
            try:
                minutes = float(raw_exptime[i])
                exptime_map[inst] = minutes / 60.0 / 24.0
            except (ValueError, TypeError):
                pass
        else:
            try:
                time = np.genfromtxt(info['file'], delimiter=',', usecols=0)
                minutes = np.median(np.diff(time)) * 24.0 * 60.0
                exptime_map[inst] = minutes / 60.0 / 24.0
            except Exception:
                pass

        # ninterp
        if i < len(raw_ninterp):
            try:
                ninterp_map[inst] = int(np.ceil(minutes / 2))
            except (ValueError, TypeError):
                pass
        else:
            ninterp_map[inst] = int(np.ceil(minutes / 2))

    # --- Rossiter / flux-weighted -------------------------------------------
    # rmbands takes precedence: any value other than "notrm" means RM is modelled.
    # Fall back to rossiter=[1,0,...] if rmbands is absent.
    raw_rmbands  = s.get('rmbands', [])
    raw_rossiter = s.get('rossiter', [])
    rossiter_map = {}   # instru → bool
    for i, r in enumerate(rv_all):
        inst = r['instru']
        if i < len(raw_rmbands):
            band = str(raw_rmbands[i]).strip().strip('"').strip("'")
            rossiter_map[inst] = (band.lower() != 'notrm')
        elif i < len(raw_rossiter):
            try:
                rossiter_map[inst] = bool(int(float(raw_rossiter[i])))
            except (ValueError, TypeError):
                pass

    # --- MCMC / DEMCPT settings ---------------------------------------------
    mcmc_total  = int(s.get('maxsteps', 10000))
    mcmc_thin   = int(s.get('nthin', 10))
    mcmc_ntemps = int(s.get('ntemps', 1))
    mcmc_maxgr  = float(s.get('maxgr', 1.01))
    mcmc_mintz  = int(s.get('mintz', 1000))

    # -----------------------------------------------------------------------
    # Build output
    # -----------------------------------------------------------------------
    lines = ['#name,value']

    def section(title):
        sep = '#' * 79
        lines.append(f'{sep},')
        lines.append(f'# {title},')
        lines.append(f'{sep},')

    # General
    section('General settings')
    lines.append(f'companions_phot,{" ".join(companions)}')
    lines.append(f'companions_rv,{" ".join(companions)}')
    if phot_insts:
        lines.append(f'inst_phot,{" ".join(phot_insts)}')
    if rv_insts:
        lines.append(f'inst_rv,{" ".join(rv_insts)}')

    # Fit performance
    section('Fit performance settings')
    lines.append('multiprocess,True')
    lines.append('multiprocess_cores,all')
    lines.append('fast_fit,True')
    lines.append('fast_fit_width,0.3333333333333333')
    lines.append('secondary_eclipse,False')
    lines.append('phase_curve,False')
    lines.append('shift_epoch,True')
    lines.append('inst_for_b_epoch,all')

    # MCMC
    section('MCMC settings')
    if n_free is not None:
        nwalkers = 2 * n_free
        lines.append(f'mcmc_nwalkers,{nwalkers}  # 2 x {n_free} free params')
    else:
        lines.append('mcmc_nwalkers,50  # params.csv not found; set to 2 x n_free')
    lines.append(f'mcmc_total_steps,{mcmc_total}')
    lines.append(f'mcmc_thin_by,{mcmc_thin}')
    if mcmc_ntemps > 1:
        lines.append(f'mcmc_ntemps,{mcmc_ntemps}')
    lines.append(f'mcmc_maxgr,{mcmc_maxgr}')
    lines.append(f'mcmc_mintz,{mcmc_mintz}')

    # Nested Sampling
    section('Nested Sampling settings')
    lines.append('ns_modus,dynamic')
    lines.append('ns_nlive,500')
    lines.append('ns_bound,single')
    lines.append('ns_sample,rwalk')
    lines.append('ns_tol,0.01')

    # Limb darkening — only needed for phot and RM (flux-weighted) RV instruments
    section('Limb darkening law per object and instrument')
    for inst in phot_insts:
        lines.append(f'host_ld_law_{inst},quad')
    for inst in rv_insts:
        if rossiter_map.get(inst, False):
            lines.append(f'host_ld_law_{inst},quad')

    # Baseline
    section('Baseline settings per instrument')
    for inst in phot_insts:
        lines.append(f'baseline_flux_{inst},hybrid_offset')
    for inst in rv_insts:
        lines.append(f'baseline_rv_{inst},hybrid_offset')

    # Errors
    section('Error settings per instrument')
    for inst in phot_insts:
        lines.append(f'error_flux_{inst},sample')
    for inst in rv_insts:
        lines.append(f'error_rv_{inst},sample')

    # Exposure times
    section('Exposure times for interpolation (days)')
    for inst in phot_insts:
        if inst in exptime_map:
            lines.append(f't_exp_{inst},{exptime_map[inst]:.10f}')
        else:
            lines.append(f't_exp_{inst},')

    # Ninterp
    section('Number of points for exposure interpolation')
    for inst in phot_insts:
        if inst in ninterp_map and ninterp_map[inst] > 1:
            lines.append(f't_exp_n_int_{inst},{ninterp_map[inst]}')
        else:
            lines.append(f't_exp_n_int_{inst},')
            


    # MIST + SED stellar model priors (auto-enabled when *.sed file found)
    section('Stellar model priors')
    lines.append('use_mist_prior,True')
    sed_files = sorted(_glob.glob(os.path.join(directory, '*.sed')))
    if sed_files:
        lines.append('use_sed_prior,True')
        lines.append(f'sed_file,{os.path.basename(sed_files[0])}')
    else:
        lines.append('use_sed_prior,False')

    # Flux-weighted RVs (Rossiter-McLaughlin)
    section('Flux weighted RVs per object and instrument')
    for comp in companions:
        for inst in rv_insts:
            if rossiter_map.get(inst, False):
                print(f'  Marking {inst} as flux-weighted (Rossiter-McLaughlin) in settings.csv')
                lines.append(f'{comp}_flux_weighted_{inst},True')
                lines.append(f't_exp_n_int_{inst},{ninterp_map[inst]}')
                lines.append(f't_exp_{inst},{exptime_map[inst]:.10f}')


    content = '\n'.join(lines) + '\n'

    with open(output_path, 'w') as f:
        f.write(content)

    print(f'Written: {output_path}')
    _report(phot_insts, rv_insts, companions, exptime_map, ninterp_map,
            rossiter_map, mcmc_total, mcmc_thin, n_free)

    # Ensure params.csv has LDC rows for all instruments that need LD
    rm_rv_insts = [inst for inst in rv_insts if rossiter_map.get(inst, False)]
    ld_insts = phot_insts + rm_rv_insts
    _ensure_ldc_params(params_csv, ld_insts)

    return content


def _report(phot_insts, rv_insts, companions, exptime_map, ninterp_map,
            rossiter_map, mcmc_total, mcmc_thin, n_free):
    nwalkers = 2 * n_free if n_free else '?'
    print(f'  companions   : {companions}')
    print(f'  phot insts   : {phot_insts}')
    print(f'  RV insts     : {rv_insts}')
    print(f'  free params  : {n_free}  →  mcmc_nwalkers={nwalkers}')
    print(f'  MCMC steps   : {mcmc_total}  thin={mcmc_thin}')
    for inst in phot_insts + rv_insts:
        exp = exptime_map.get(inst)
        ni  = ninterp_map.get(inst)
        rm  = rossiter_map.get(inst, False)
        exp_str = f'{exp*24*60:.2f} min → {exp:.8f} d' if exp else '(not set)'
        print(f'  {inst:15s}  t_exp={exp_str}  n_int={ni}  flux_weighted={rm}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert EXOFASTv2 settings.txt to allesfitter settings.csv')
    parser.add_argument('settings_txt', help='Path to settings.txt')
    parser.add_argument('--output', '-o', default=None,
                        help='Output path (default: settings.csv beside input)')
    args = parser.parse_args()

    settings_to_csv(args.settings_txt, args.output)
