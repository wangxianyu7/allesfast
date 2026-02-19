import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import allesfast
import os
import matplotlib as mpl

mpl.use("Agg")
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

from allesfast.utils.settings2allesfitter import settings_to_csv
from allesfast.utils.priors2allesfitter import update_params_csv


def prepare(path):
    """Generate settings.csv and params.csv from EXOFASTv2 config files."""
    settings_txt = os.path.join(path, 'settings.txt')
    params_csv   = os.path.join(path, 'params.csv')

    # Find the .priors file automatically
    priors_files = [f for f in os.listdir(path) if f.endswith('.priors')]
    if not priors_files:
        raise FileNotFoundError(f'No .priors file found in {path}')
    priorfile = os.path.join(path, priors_files[0])

    settings_to_csv(settings_txt)
    update_params_csv(priorfile, params_csv, companion='b')


def run_allesfast(path):
    allesfast.show_initial_guess(path, do_logprint=False)
    allesfast.demcpt_fit(path)
    allesfast.mcmc_output(path)


path = '.'

# prepare(path)
run_allesfast(path)
