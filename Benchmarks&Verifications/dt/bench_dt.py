"""Doppler Tomography χ² parameter-grid benchmark: allesfast vs EXOFASTv2 IDL.

Sweep across lambda, vsini, and vline; compute χ² at each grid point in
Python.  The companion IDL script writes a CSV with χ² at the same grid
points.  This script then loads the IDL CSV and compares per-point.

Output: bench_dt_results.csv + bench_dt_report.txt

Dataset: KELT-17b TRES R=44000 (Zhou+ 2016).
"""
import sys
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import json
from allesfast.dt.io import read_dt_fits
from allesfast.dt.core import dopptom_chi2


HERE = Path(__file__).parent
DATA = REPO / 'examples' / 'KELT-17_DT' / 'n20160226.KELT-17b.TRES.44000.fits'


# Best-fit parameters (from EXOFASTv2 kelt17.priors2)
BASE = dict(
    tc       = 2457287.7456103642,
    period   = 10 ** 0.4885753488,
    k        = 0.0926333230,
    cosi     = 0.0817318222,
    secosw   = 0.2433198217,
    sesinw   = -0.1518641162,
    lam      = -2.0164720094,
    u1       = 0.2788,
    u2       = 0.3468,
    vsini    = 44.2,
    vline    = 5.49,
    errscale = 3.39,
    logg     = 4.265827,  # for IDL quadld
    teff     = 7454.0,
    feh      = 0.0,
)
G_unit = 2942.71377
MSTAR  = 10 ** 0.2604162879
RSTAR  = 1.6458230572
BASE['ar'] = (G_unit * MSTAR * BASE['period']**2 / (4 * np.pi**2))**(1/3) / RSTAR
BASE['e'] = BASE['secosw']**2 + BASE['sesinw']**2
BASE['omega'] = float(np.arctan2(BASE['sesinw'], BASE['secosw']))


def run_python_grid():
    dat = read_dt_fits(str(DATA))
    rows = []
    # Sweep lambda (radians)
    for lam in np.linspace(-np.pi, np.pi, 11):
        p = dict(BASE)
        p['lam'] = float(lam)
        chi2 = dopptom_chi2(
            dat, p['tc'], p['period'], p['e'], p['omega'], p['cosi'],
            p['k'], p['ar'], p['lam'],
            p['u1'], p['u2'], p['vsini'], p['vline'], p['errscale'],
        )
        rows.append(dict(scan='lambda', value=float(lam), chi2_py=float(chi2)))

    # Sweep vsini
    for vsini in np.linspace(35.0, 55.0, 9):
        p = dict(BASE)
        p['vsini'] = float(vsini)
        chi2 = dopptom_chi2(
            dat, p['tc'], p['period'], p['e'], p['omega'], p['cosi'],
            p['k'], p['ar'], p['lam'],
            p['u1'], p['u2'], p['vsini'], p['vline'], p['errscale'],
        )
        rows.append(dict(scan='vsini', value=float(vsini), chi2_py=float(chi2)))

    # Sweep vline
    for vline in np.linspace(2.0, 12.0, 11):
        p = dict(BASE)
        p['vline'] = float(vline)
        chi2 = dopptom_chi2(
            dat, p['tc'], p['period'], p['e'], p['omega'], p['cosi'],
            p['k'], p['ar'], p['lam'],
            p['u1'], p['u2'], p['vsini'], p['vline'], p['errscale'],
        )
        rows.append(dict(scan='vline', value=float(vline), chi2_py=float(chi2)))

    # Sweep cosi (impact parameter)
    for cosi in np.linspace(0.02, 0.18, 9):
        p = dict(BASE)
        p['cosi'] = float(cosi)
        chi2 = dopptom_chi2(
            dat, p['tc'], p['period'], p['e'], p['omega'], p['cosi'],
            p['k'], p['ar'], p['lam'],
            p['u1'], p['u2'], p['vsini'], p['vline'], p['errscale'],
        )
        rows.append(dict(scan='cosi', value=float(cosi), chi2_py=float(chi2)))

    return rows


def write_grid_for_idl(rows, out_csv):
    """Write a simple CSV listing (scan, value) pairs so IDL can iterate."""
    with open(out_csv, 'w') as f:
        f.write('scan,value\n')
        for r in rows:
            f.write(f"{r['scan']},{r['value']:.10f}\n")


def load_idl_chi2(in_csv):
    """Load IDL chi2 outputs."""
    if not os.path.exists(in_csv):
        return None
    chi2s = []
    with open(in_csv) as f:
        next(f)  # header
        for line in f:
            chi2s.append(float(line.strip().split(',')[-1]))
    return np.array(chi2s)


if __name__ == '__main__':
    print(f'BASE params: tc={BASE["tc"]}, P={BASE["period"]:.6f}, '
          f'a/Rs={BASE["ar"]:.4f}, k={BASE["k"]:.4f}, '
          f'e={BASE["e"]:.4f}, omega={np.degrees(BASE["omega"]):.2f}°')
    print()

    rows = run_python_grid()
    n_total = len(rows)
    print(f'Computed Python χ² at {n_total} grid points across 4 axes')

    grid_csv = HERE / 'bench_grid_input.csv'
    write_grid_for_idl(rows, str(grid_csv))
    print(f'Wrote grid to {grid_csv.name}')

    idl_chi2 = load_idl_chi2(str(HERE / 'bench_grid_idl_chi2.csv'))
    out_path = HERE / 'bench_dt_results.csv'
    report_path = HERE / 'bench_dt_report.txt'

    with open(out_path, 'w') as f:
        if idl_chi2 is not None:
            f.write('scan,value,chi2_py,chi2_idl,abs_diff,rel_diff\n')
        else:
            f.write('scan,value,chi2_py\n')
        for i, r in enumerate(rows):
            if idl_chi2 is not None and i < len(idl_chi2):
                d = float(r['chi2_py'] - idl_chi2[i])
                rel = float(abs(d) / max(abs(idl_chi2[i]), 1e-30))
                f.write(f"{r['scan']},{r['value']:.10f},"
                        f"{r['chi2_py']:.6f},{idl_chi2[i]:.6f},"
                        f"{d:+.4e},{rel:.4e}\n")
            else:
                f.write(f"{r['scan']},{r['value']:.10f},{r['chi2_py']:.6f}\n")

    # Summary
    with open(report_path, 'w') as f:
        f.write('Doppler Tomography χ² benchmark: allesfast vs EXOFASTv2 IDL\n')
        f.write('=' * 64 + '\n\n')
        f.write(f'Dataset: KELT-17b TRES R=44000\n')
        f.write(f'         5273 vels × 33 times = 174,009 pixels\n')
        f.write(f'Anchor:  EXOFASTv2 best-fit (kelt17.priors2)\n\n')
        f.write(f'Grid: {n_total} points across (lambda, vsini, vline, cosi)\n\n')
        if idl_chi2 is None:
            f.write('IDL results NOT YET COMPUTED. Run bench_dt_idl.pro then re-run.\n')
        else:
            n = min(len(rows), len(idl_chi2))
            diffs = np.array([rows[i]['chi2_py'] - idl_chi2[i] for i in range(n)])
            rels  = np.array([
                abs(diffs[i]) / max(abs(idl_chi2[i]), 1e-30)
                for i in range(n)
            ])
            f.write('Per-axis statistics:\n')
            f.write('-' * 64 + '\n')
            for axis in ['lambda', 'vsini', 'vline', 'cosi']:
                idxs = [i for i, r in enumerate(rows[:n]) if r['scan'] == axis]
                if not idxs: continue
                ad = diffs[idxs]
                rd = rels[idxs]
                f.write(f'  {axis:8s}  N={len(idxs):2d}  '
                        f'max|Δχ²|={np.max(np.abs(ad)):.4e}  '
                        f'mean|Δχ²|={np.mean(np.abs(ad)):.4e}  '
                        f'max rel={np.max(rd):.4e}\n')
            f.write('\n')
            f.write(f'Overall:  max|Δχ²|={np.max(np.abs(diffs)):.4e}  '
                    f'max rel={np.max(rels):.4e}  '
                    f'mean rel={np.mean(rels):.4e}\n')
            ok = np.max(rels) < 1e-3
            f.write('\nVerdict: ' + ('PASS' if ok else 'FAIL') +
                    f' (threshold rel < 1e-3)\n')

    with open(report_path) as fh:
        print(fh.read())
