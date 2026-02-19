"""
file_naming.py

Utilities for generating and applying the standard allesfast file naming convention:

    RV:      nYYYYMMDD.Instru.RV.csv
    Transit: nYYYYMMDD.band.Instru.Tran.csv

The date (YYYYMMDD) is derived from the first BJD value in the data file.

Usage (command line):
    # Rename a single file
    python file_naming.py CORALIE.csv --type rv --instru CORALIE
    python file_naming.py K2.csv --type tran --instru K2 --band Kepler

    # Dry run (print new name without renaming)
    python file_naming.py K2.csv --type tran --instru K2 --band Kepler --dry-run

    # Batch rename from a settings.csv
    python file_naming.py --from-settings settings.csv
"""

import os
import argparse
import numpy as np
from astropy.time import Time


def _read_first_bjd(filepath):
    """Read the first BJD value (first column, first data row) from a CSV."""
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            first_val = line.split(',')[0].strip()
            return float(first_val)
    raise ValueError(f'No data rows found in {filepath}')


def bjd_to_datestr(bjd):
    """Convert a BJD (JD scale, TDB) to a YYYYMMDD string."""
    t = Time(bjd, format='jd', scale='tdb')
    return t.to_datetime().strftime('%Y%m%d')


def make_rv_name(instru, datestr):
    """Return standard RV filename: nYYYYMMDD.Instru.RV.csv"""
    return f'n{datestr}.{instru}.RV.csv'


def make_tran_name(band, instru, datestr):
    """Return standard Transit filename: nYYYYMMDD.band.Instru.Tran.csv"""
    return f'n{datestr}.{band}.{instru}.Tran.csv'


def rename_file(filepath, ftype, instru, band=None, dry_run=False):
    """
    Rename a data file to the standard allesfast naming convention.

    Parameters
    ----------
    filepath : str
        Path to the existing CSV file.
    ftype : str
        'rv' or 'tran'.
    instru : str
        Instrument name (e.g. 'CORALIE', 'K2').
    band : str, optional
        Filter/band name (required for ftype='tran', e.g. 'Kepler', 'r', 'V').
    dry_run : bool
        If True, print the new name but do not rename.

    Returns
    -------
    str
        The new filename (basename only).
    """
    ftype = ftype.lower()
    if ftype not in ('rv', 'tran'):
        raise ValueError(f"ftype must be 'rv' or 'tran', got '{ftype}'")
    if ftype == 'tran' and band is None:
        raise ValueError("band is required for ftype='tran'")

    bjd = _read_first_bjd(filepath)
    datestr = bjd_to_datestr(bjd)

    if ftype == 'rv':
        new_name = make_rv_name(instru, datestr)
    else:
        new_name = make_tran_name(band, instru, datestr)

    dirpath = os.path.dirname(os.path.abspath(filepath))
    new_path = os.path.join(dirpath, new_name)

    if dry_run:
        print(f'[dry-run] {os.path.basename(filepath)} -> {new_name}')
    else:
        os.rename(filepath, new_path)
        print(f'{os.path.basename(filepath)} -> {new_name}')

    return new_name


def rename_from_settings(settings_path, band_map, dry_run=False):
    """
    Rename all instrument files in a project directory based on settings.csv.

    Parameters
    ----------
    settings_path : str
        Path to settings.csv.
    band_map : dict
        Mapping from photometry instrument name to filter name.
        e.g. {'K2': 'Kepler', 'TESS': 'TESS', 'EulerCam': 'r'}
    dry_run : bool
        If True, print new names but do not rename.
    """
    dirpath = os.path.dirname(os.path.abspath(settings_path))

    # Parse settings.csv for inst_phot and inst_rv
    inst_phot = []
    inst_rv = []
    with open(settings_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            key, val = parts[0].strip(), parts[1].strip()
            if key == 'inst_phot':
                inst_phot = val.split()
            elif key == 'inst_rv':
                inst_rv = val.split()

    for instru in inst_rv:
        fpath = os.path.join(dirpath, f'{instru}.csv')
        if os.path.exists(fpath):
            rename_file(fpath, 'rv', instru, dry_run=dry_run)
        else:
            print(f'[skip] {instru}.csv not found')

    for instru in inst_phot:
        fpath = os.path.join(dirpath, f'{instru}.csv')
        if os.path.exists(fpath):
            band = band_map.get(instru)
            if band is None:
                print(f'[skip] no band specified for {instru}; add to band_map')
                continue
            rename_file(fpath, 'tran', instru, band=band, dry_run=dry_run)
        else:
            print(f'[skip] {instru}.csv not found')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Rename allesfast data files to standard naming convention')
    parser.add_argument('file', nargs='?', help='CSV file to rename')
    parser.add_argument('--type', choices=['rv', 'tran'], help="'rv' or 'tran'")
    parser.add_argument('--instru', help='Instrument name')
    parser.add_argument('--band', help='Filter/band name (required for tran)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print new name without renaming')
    parser.add_argument('--from-settings', metavar='SETTINGS_CSV',
                        help='Batch rename from settings.csv')
    parser.add_argument('--band-map', nargs='+', metavar='INSTRU=band',
                        help='Band map for --from-settings, e.g. K2=Kepler TESS=TESS')
    args = parser.parse_args()

    if args.from_settings:
        band_map = {}
        if args.band_map:
            for item in args.band_map:
                k, v = item.split('=')
                band_map[k.strip()] = v.strip()
        rename_from_settings(args.from_settings, band_map, dry_run=args.dry_run)
    elif args.file:
        if not args.type or not args.instru:
            parser.error('--type and --instru are required when renaming a single file')
        rename_file(args.file, args.type, args.instru,
                    band=args.band, dry_run=args.dry_run)
    else:
        parser.print_help()
