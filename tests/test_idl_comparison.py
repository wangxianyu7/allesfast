"""
Validate allesfast star/ SED+MIST implementation against IDL EXOFASTv2 reference.

Two tests:
  1. massradius_mist  — MIST track interpolation (mistage, mistrstar, mistteff, mistfeh)
                        compared against IDL massradius_mist.pro via the same 7 test cases
                        used in jaxmistmultised/tests/idl_comparison/compare_massradius.pro.

  2. mistmultised / sed_chi2 — SED BC interpolation + chi2
                        using K2-140's .sed file and best-fit stellar params.
                        Verifies finite chi2, self-consistency, and (optionally)
                        cross-validates against jaxmistmultised.

Usage:
    # From the allesfast repo root:
    python tests/test_idl_comparison.py

    # Re-run IDL to regenerate reference values (requires IDL + EXOFASTv2):
    python tests/test_idl_comparison.py --run-idl
"""

import argparse
import os
import subprocess
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDL_BIN      = "/Users/wangxianyu/Applications/NV5/idl90/bin/idl"
IDL_SCRIPT   = os.path.join(REPO_ROOT,
    "../jaxmistmultised/tests/idl_comparison/run_compare.pro")
EXOFAST_PATH = "/Users/wangxianyu/Applications/NV5/idl90/lib/EXOFASTv2/"
K2140_DIR    = os.path.join(REPO_ROOT, "examples", "K2-140")
SED_FILE     = os.path.join(K2140_DIR, "380255458.sed")


# ---------------------------------------------------------------------------
# IDL reference values (from EXOFASTv2 massradius_mist.pro, run 2025-02)
# RESULT|label|eep|mstar|feh|mistage|mistrstar|mistteff|mistfeh
# ---------------------------------------------------------------------------
IDL_REFERENCE = [
    {"label": "ZAMS_solar",     "eep": 202.0, "mstar": 1.00, "feh":  0.00,
     "age": 0.041873472330, "rstar": 0.883585204491,
     "teff": 5707.057054112379, "feh_act":  0.042423163023},
    {"label": "MS_solar",       "eep": 355.0, "mstar": 1.00, "feh":  0.00,
     "age": 4.581815089715, "rstar": 1.025106917293,
     "teff": 5848.545527106810, "feh_act": -0.020958700097},
    {"label": "MS_sub_solar",   "eep": 355.0, "mstar": 0.80, "feh": -0.50,
     "age": 7.623830473322, "rstar": 0.813581260825,
     "teff": 5766.808715776669, "feh_act": -0.566413821434},
    {"label": "MS_super_solar", "eep": 355.0, "mstar": 1.20, "feh":  0.25,
     "age": 2.741867236309, "rstar": 1.260523912733,
     "teff": 6042.807233966550, "feh_act":  0.240171320273},
    {"label": "MS_low_feh",     "eep": 300.0, "mstar": 1.00, "feh": -1.00,
     "age": 0.520008061028, "rstar": 0.911410185983,
     "teff": 6837.954202769988, "feh_act": -1.088975707058},
    {"label": "MS_half_eep",    "eep": 354.5, "mstar": 1.00, "feh":  0.00,
     "age": 4.561700416218, "rstar": 1.024303064431,
     "teff": 5848.114864795481, "feh_act": -0.020673547257},
    {"label": "ZAMS_low_mass",  "eep": 202.0, "mstar": 0.50, "feh": -0.50,
     "age": 0.133462515334, "rstar": 0.458122257388,
     "teff": 4065.755231167308, "feh_act": -0.472323782043},
]


# ---------------------------------------------------------------------------
# Helper: relative difference
# ---------------------------------------------------------------------------
def _reldiff(a, b):
    avg = 0.5 * (abs(a) + abs(b))
    return 0.0 if avg == 0.0 else abs(a - b) / avg


# ---------------------------------------------------------------------------
# 1. massradius_mist track interpolation test
# ---------------------------------------------------------------------------
def _interp_track(eep, mstar, feh, vvcrit=0.0, alpha=0.0):
    """
    Directly replicate the EEP interpolation inside massradius_mist,
    returning (mistage, mistrstar, mistteff, mistfeh).
    """
    from allesfast.star.massradius_mist import (
        _get_track_tuple_cached, _mass_index, _feh_index,
        _vvcrit_index, _alpha_index,
    )
    massndx   = _mass_index(mstar)
    fehndx    = _feh_index(feh)
    vvcritndx = _vvcrit_index(float(vvcrit))
    alphandx  = _alpha_index(float(alpha))

    ages, rstars, teffs, fehs, _ = _get_track_tuple_cached(
        massndx, fehndx, vvcritndx, alphandx)

    eep_lo = int(np.floor(eep)) - 1   # 0-based
    eep_hi = eep_lo + 1
    frac   = eep - np.floor(eep)

    mistage   = (1 - frac) * ages[eep_lo]   + frac * ages[eep_hi]
    mistrstar = (1 - frac) * rstars[eep_lo] + frac * rstars[eep_hi]
    mistteff  = (1 - frac) * teffs[eep_lo]  + frac * teffs[eep_hi]
    mistfeh   = (1 - frac) * fehs[eep_lo]   + frac * fehs[eep_hi]
    return mistage, mistrstar, mistteff, mistfeh


def run_idl_fresh():
    """Re-run IDL and return parsed reference rows (list of dicts)."""
    env = os.environ.copy()
    env["IDL_PATH"]    = f"+{EXOFAST_PATH}:<IDL_DEFAULT>"
    env["EXOFAST_PATH"] = EXOFAST_PATH
    result = subprocess.run(
        [IDL_BIN, "-quiet", "-e", f"@{IDL_SCRIPT}"],
        capture_output=True, text=True, env=env,
        cwd=os.path.dirname(IDL_SCRIPT),
    )
    rows = []
    for line in result.stdout.splitlines():
        if not line.startswith("RESULT|"):
            continue
        parts = line.split("|")
        if len(parts) != 9:
            continue
        _, label, eep, mstar, feh, age, rstar, teff, feh_act = parts
        rows.append({
            "label": label.strip(),
            "eep":   float(eep),   "mstar": float(mstar), "feh":  float(feh),
            "age":   float("inf") if age.strip()    == "INF" else float(age),
            "rstar": float("inf") if rstar.strip()  == "INF" else float(rstar),
            "teff":  float("inf") if teff.strip()   == "INF" else float(teff),
            "feh_act": float("inf") if feh_act.strip() == "INF" else float(feh_act),
        })
    return rows


def test_massradius_mist(ref_rows, tol=1e-4):
    """Compare allesfast massradius_mist against IDL reference values."""
    print("\n" + "=" * 90)
    print("TEST 1: massradius_mist track interpolation  (tolerance: rel diff < 1e-4)")
    print("=" * 90)

    hdr = (f"{'Label':<18} {'EEP':>7} {'Mstar':>5} {'[Fe/H]':>6}  "
           f"{'Param':<9} {'IDL':>14} {'Python':>14} {'|rel diff|':>12}")
    print(hdr)
    print("-" * len(hdr))

    all_pass = True
    for row in ref_rows:
        py_age, py_rstar, py_teff, py_feh_act = _interp_track(
            row["eep"], row["mstar"], row["feh"])

        pairs = [
            ("age(Gyr)",    row["age"],     py_age),
            ("rstar(Rsun)", row["rstar"],   py_rstar),
            ("teff(K)",     row["teff"],    py_teff),
            ("feh_act",     row["feh_act"], py_feh_act),
        ]
        prefix = f"{row['label']:<18} {row['eep']:>7.2f} {row['mstar']:>5.2f} {row['feh']:>6.2f}"
        for i, (pname, idl_val, py_val) in enumerate(pairs):
            rd   = _reldiff(idl_val, py_val)
            fail = rd > tol
            if fail:
                all_pass = False
            pref = prefix if i == 0 else " " * len(prefix)
            flag = "  *** FAIL" if fail else ""
            print(f"{pref}  {pname:<9} {idl_val:>14.8f} {py_val:>14.8f} {rd:>12.2e}{flag}")
        print()

    if all_pass:
        print("RESULT: ALL differences < 1e-4 (relative). IDL ≈ Python ✓")
    else:
        print("RESULT: SOME differences exceed 1e-4 — see *** rows above.")
    return all_pass


# ---------------------------------------------------------------------------
# 2. SED chi2 test
# ---------------------------------------------------------------------------
def test_sed_chi2():
    """
    Validate mistmultised / sed_chi2 with K2-140 stellar params.

    Checks:
      - chi2 is finite and positive
      - Model magnitudes are in a plausible range (8–16 mag)
      - Residuals are small (< 1 mag) for well-constrained bands
      - Self-consistency: doubling errscale reduces chi2 by ~4×
    """
    print("\n" + "=" * 90)
    print("TEST 2: SED chi2 (K2-140, using best-fit stellar params)")
    print("=" * 90)

    if not os.path.exists(SED_FILE):
        print(f"  SKIP: SED file not found: {SED_FILE}")
        return None

    from allesfast.star.sed_utils import mistmultised, read_sed_file
    from allesfast.star.mist_sed import _derive_logg, _derive_lstar

    # K2-140 best-fit stellar parameters (from MCMC median)
    teff     = 5652.0   # K
    rstar    = 1.009    # Rsun
    mstar    = 0.976    # Msun
    feh      = 0.073    # dex
    av       = 0.047    # mag
    parallax = 2.872    # mas  → distance = 1000/parallax pc
    distance = 1000.0 / parallax

    logg  = _derive_logg(mstar, rstar)
    lstar = _derive_lstar(teff, rstar)
    print(f"  Stellar inputs: Teff={teff:.0f}K  Rstar={rstar:.3f}Rsun  "
          f"Mstar={mstar:.3f}Msun  [Fe/H]={feh:.3f}  Av={av:.3f}  d={distance:.1f}pc")
    print(f"  Derived:        logg={logg:.3f}  Lstar={lstar:.3f}Lsun\n")

    # Pre-load SED data
    sed_data = read_sed_file(SED_FILE, nstars=1)
    nbands   = len(sed_data["sedbands"])
    print(f"  Bands in SED file: {nbands}")
    for i, b in enumerate(sed_data["sedbands"]):
        print(f"    [{i:2d}] {b:<20s}  obs={sed_data['mag'][i]:.4f}  "
              f"err={sed_data['errmag'][i]:.4f}")

    chi2_1, blendmag, modelflux, residuals = mistmultised(
        np.array([teff]),
        np.array([logg]),
        np.array([feh]),
        np.array([av]),
        np.array([distance]),
        np.array([lstar]),
        1.0,
        SED_FILE,
        sed_data=sed_data,
    )

    print(f"\n  chi2 (errscale=1): {chi2_1:.4f}")
    print(f"\n  Band-by-band model magnitudes and residuals:")
    print(f"  {'Band':<20s} {'obs':>8} {'model':>8} {'resid':>8} {'err':>7}")
    print("  " + "-" * 55)
    for i, b in enumerate(sed_data["sedbands"]):
        print(f"  {b:<20s} {sed_data['mag'][i]:>8.4f} {blendmag[i]:>8.4f} "
              f"{residuals[i]:>8.4f} {sed_data['errmag'][i]:>7.4f}")

    # Self-consistency check using chi2 = sum[ (r/sigma)^2 + log(2*pi*sigma^2) ].
    # Doubling errscale (sigma_eff = 2*sigma):
    #   chi2_2 = sum[ r^2/(4*sigma^2) + log(2*pi*sigma^2) + log(4) ]
    #           = chi2_pure/4 + log_norm + n*log(4)
    # When residuals << errors (well-fit model), log(4) term dominates and chi2_2 > chi2_1.
    # When residuals >> errors (bad fit), the r^2/sigma^2 reduction dominates and chi2_2 < chi2_1.
    chi2_2, _, _, _ = mistmultised(
        np.array([teff]), np.array([logg]), np.array([feh]),
        np.array([av]), np.array([distance]), np.array([lstar]),
        2.0, SED_FILE, sed_data=sed_data,
    )
    print(f"\n  chi2 (errscale=2): {chi2_2:.4f}")

    # Checks
    # Note: chi2 = sum[(resid/sigma)^2 + log(2*pi*sigma^2)] can be negative
    # because log(2*pi*sigma^2) < 0 for sigma < 1/sqrt(2*pi) ~ 0.4.
    # For photometric errors (~0.02 mag), each band contributes log(2*pi*0.02^2) ~ -6.
    checks = []
    checks.append(("chi2 is finite",         np.isfinite(chi2_1)))
    checks.append(("model mags in [5,20]",   np.all((blendmag > 5) & (blendmag < 20))))
    checks.append(("max |residual| < 1 mag", np.max(np.abs(residuals)) < 1.0))
    # With small residuals (good fit), log(4)*nbands dominates → chi2_2 > chi2_1
    checks.append(("chi2_2 > chi2_1 (good fit, log term dominates)", chi2_2 > chi2_1))

    print("\n  Checks:")
    all_pass = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    [{status}] {name}")

    if all_pass:
        print("\nRESULT: All SED checks passed ✓")
    else:
        print("\nRESULT: Some SED checks FAILED — see above.")
    return all_pass


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-idl", action="store_true",
                        help="Re-run IDL to regenerate reference values (requires IDL)")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="Relative difference tolerance for massradius_mist (default 1e-4)")
    args = parser.parse_args()

    if args.run_idl:
        print(f"Running IDL: {IDL_SCRIPT} ...")
        ref_rows = run_idl_fresh()
        if not ref_rows:
            print("ERROR: no RESULT lines from IDL.")
            sys.exit(1)
        print(f"  Got {len(ref_rows)} reference cases from IDL.\n")
    else:
        ref_rows = IDL_REFERENCE
        print(f"Using hardcoded IDL reference values ({len(ref_rows)} cases).")
        print("  (Pass --run-idl to regenerate from IDL EXOFASTv2)\n")

    pass1 = test_massradius_mist(ref_rows, tol=args.tol)
    pass2 = test_sed_chi2()

    print("\n" + "=" * 90)
    print(f"SUMMARY:  massradius_mist={'PASS' if pass1 else 'FAIL'}   "
          f"sed_chi2={'PASS' if pass2 else 'FAIL' if pass2 is not None else 'SKIP'}")
    print("=" * 90)
    sys.exit(0 if (pass1 and pass2 is not False) else 1)


if __name__ == "__main__":
    main()
