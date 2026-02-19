#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

import os
import numpy as np

from ..priors.simulate_PDF import simulate_PDF


def _to_float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def _get_value(params, keys):
    if params is None:
        return None, None
    for key in keys:
        if key in params:
            value = _to_float(params[key])
            if value is not None:
                return value, key
    return None, None


def _get_err(params, key, alt_keys=()):
    if params is None:
        return None, None

    lerr = _to_float(params.get(key + "_lerr", None))
    uerr = _to_float(params.get(key + "_uerr", None))
    if (lerr is not None) and (uerr is not None):
        return lerr, uerr

    err = _to_float(params.get(key + "_err", None))
    if err is not None:
        return err, err

    for alt in alt_keys:
        lerr = _to_float(params.get(alt + "_lerr", None))
        uerr = _to_float(params.get(alt + "_uerr", None))
        if (lerr is not None) and (uerr is not None):
            return lerr, uerr
        err = _to_float(params.get(alt + "_err", None))
        if err is not None:
            return err, err

    return None, None


def get_stellar_row_from_params(params):
    row = {
        "R_star": None,
        "R_star_lerr": None,
        "R_star_uerr": None,
        "M_star": None,
        "M_star_lerr": None,
        "M_star_uerr": None,
        "Teff_star": None,
        "Teff_star_lerr": None,
        "Teff_star_uerr": None,
    }

    rstar, rkey = _get_value(params, ("host_rstar", "R_star"))
    mstar, mkey = _get_value(params, ("host_mstar", "M_star"))
    teff, tkey = _get_value(params, ("host_teff", "Teff_star"))

    if rstar is not None:
        row["R_star"] = rstar
        row["R_star_lerr"], row["R_star_uerr"] = _get_err(params, rkey, alt_keys=("host_rstar", "R_star"))
    if mstar is not None:
        row["M_star"] = mstar
        row["M_star_lerr"], row["M_star_uerr"] = _get_err(params, mkey, alt_keys=("host_mstar", "M_star"))
    if teff is not None:
        row["Teff_star"] = teff
        row["Teff_star_lerr"], row["Teff_star_uerr"] = _get_err(params, tkey, alt_keys=("host_teff", "Teff_star"))

    return row


def get_stellar_row_from_file(datadir):
    fname = os.path.join(datadir, "params_star.csv")
    if not os.path.exists(fname):
        return None

    buf = np.genfromtxt(fname, delimiter=",", names=True, dtype=None, encoding="utf-8", comments="#")
    if buf is None:
        return None

    row = {
        "R_star": _to_float(buf["R_star"]),
        "R_star_lerr": _to_float(buf["R_star_lerr"]),
        "R_star_uerr": _to_float(buf["R_star_uerr"]),
        "M_star": _to_float(buf["M_star"]),
        "M_star_lerr": _to_float(buf["M_star_lerr"]),
        "M_star_uerr": _to_float(buf["M_star_uerr"]),
        "Teff_star": _to_float(buf["Teff_star"]),
        "Teff_star_lerr": _to_float(buf["Teff_star_lerr"]),
        "Teff_star_uerr": _to_float(buf["Teff_star_uerr"]),
    }
    return row


def get_stellar_row(datadir, params=None):
    row = get_stellar_row_from_params(params)
    if has_stellar_info(row):
        return row
    row_from_file = get_stellar_row_from_file(datadir)
    if row_from_file is not None:
        return row_from_file
    return row


def has_stellar_info(row, require=("R_star", "M_star", "Teff_star")):
    if row is None:
        return False
    for key in require:
        value = row.get(key, None)
        if value is None:
            return False
        if not np.isfinite(value):
            return False
    return True


def sample_stellar(row, size):
    out = {}
    for key in ("R_star", "M_star", "Teff_star"):
        value = row.get(key, None)
        lerr = row.get(key + "_lerr", None)
        uerr = row.get(key + "_uerr", None)
        if value is None:
            out[key] = np.full(size, np.nan)
            continue
        if (lerr is not None) and (uerr is not None) and (lerr > 0) and (uerr > 0):
            out[key] = simulate_PDF(value, lerr, uerr, size=size, plot=False)
        else:
            out[key] = np.full(size, value)
    return out


def summary_dict(row):
    return {
        "R_star_median": row.get("R_star", np.nan),
        "R_star_lerr": row.get("R_star_lerr", np.nan),
        "R_star_uerr": row.get("R_star_uerr", np.nan),
        "M_star_median": row.get("M_star", np.nan),
        "M_star_lerr": row.get("M_star_lerr", np.nan),
        "M_star_uerr": row.get("M_star_uerr", np.nan),
    }


def plot_params_star(row):
    return {"R_star": row.get("R_star", np.nan)}
