import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import allesfast
import os
import matplotlib as mpl

mpl.use("Agg")
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

path = '.'

allesfast.show_initial_guess(path, do_logprint=False)
allesfast.mcmc_fit(path)
allesfast.mcmc_output(path)
