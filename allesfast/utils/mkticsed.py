"""
mkticsed.py

Python port of EXOFASTv2's mkticsed.pro (Eastman et al.)

Create the SED input file and a minimal prior file for a given TIC ID,
querying TICv8.2 and several photometric catalogs via Vizier.

Usage (command line):
    python mkticsed.py 402026209
    python mkticsed.py 402026209 --sedfile wasp4.sed --priorfile wasp4.priors
    python mkticsed.py --ra 30.5631 --dec -42.0671

Dependencies:
    astroquery, astropy, numpy
Optional:
    dustmaps          (for SFD Av upper limit)
    gaiadr3-zeropoint (for Lindegren+2021 Gaia DR3 parallax correction)
"""

import argparse
import numpy as np
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

Vizier.ROW_LIMIT = -1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(table, col, idx=0, default=np.nan):
    """Return table[col][idx] as float, or default if missing/masked."""
    if col not in table.colnames:
        return default
    try:
        val = table[col][idx]
        f = float(val)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _safe_str(table, col, idx=0, default=''):
    if col not in table.colnames:
        return default
    try:
        return str(table[col][idx]).strip()
    except Exception:
        return default


def _flux_to_magerr(e_flux, flux):
    """Convert flux error to magnitude error."""
    if flux <= 0 or not np.isfinite(e_flux) or not np.isfinite(flux):
        return np.nan
    return 2.5 / np.log(10) * abs(e_flux / flux)


def strom_conv(V, sigV, by, sigby, m1, sigm1, c1, sigc1):
    """
    Translate Stromgren colour combinations to individual uvby magnitudes.
    Returns (u, sigu, v, sigv, b, sigb, y, sigy).
    """
    y = V
    b = V + by
    u = V + 3*by + 2*m1 + c1
    v = V + 2*by + m1
    sigy = sigV
    sigb = np.sqrt(sigV**2 + sigby**2)
    sigu = np.sqrt(sigV**2 + (3*sigby)**2 + (2*sigm1)**2 + sigc1**2)
    sigv = np.sqrt(sigV**2 + (2*sigby)**2 + sigm1**2)
    return u, sigu, v, sigv, b, sigb, y, sigy


def _get_av_upper_limit(ra, dec):
    """Return total line-of-sight Av from the SFD dust map as an upper limit."""
    try:
        from dustmaps.sfd import SFDQuery
        coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
        sfd = SFDQuery()
        ebv = float(sfd(coords))
        av = ebv * 3.1
        return f'av 0 -1 0 {av:.9f}'
    except ImportError:
        return '# dustmaps not installed; set Av upper limit manually\n# av 0 -1 0 <value>'
    except Exception as e:
        return f'# Could not query SFD dust map ({e}); set Av upper limit manually\n# av 0 -1 0 <value>'


def _query_region(catalog, coord, radius_arcmin):
    """Query Vizier by sky position; return first table or None."""
    v = Vizier(columns=['*'], row_limit=-1)
    try:
        result = v.query_region(coord, radius=radius_arcmin * u.arcmin,
                                catalog=catalog)
        if result and len(result) > 0:
            return result[0]
    except Exception as e:
        print(f'  [warn] Vizier query failed for {catalog}: {e}')
    return None


def _query_tic(ticid):
    """Query TICv8.2 by TIC ID using constraints; return table row or None."""
    v = Vizier(columns=['*'], row_limit=-1)
    try:
        result = v.query_constraints(catalog='IV/39/tic82', TIC=str(ticid))
        if result and len(result) > 0:
            t = result[0]
            mask = np.array([str(row) == str(ticid) for row in t['TIC']])
            if np.any(mask):
                return t[mask]
    except Exception as e:
        print(f'  [warn] TIC query failed: {e}')
    return None


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def mkticsed(ticid=None, ra=None, dec=None,
             priorfile=None, sedfile=None,
             use_galex=False, use_tycho=False, use_ucac=False,
             use_merm=False, use_stromgren=False, use_kepler=False):
    """
    Generate SED and prior files from TICv8.2.

    Parameters
    ----------
    ticid : str
        TIC ID (numeric string, e.g. '402026209').
    ra, dec : float
        RA/Dec in decimal degrees (alternative to ticid; less robust).
    priorfile : str
        Output prior filename. Defaults to '<ticid>.priors'.
    sedfile : str
        Output SED filename. Defaults to '<ticid>.sed'.
    use_galex, use_tycho, use_ucac, use_merm, use_stromgren, use_kepler : bool
        Include these photometry sources (commented out by default).
    """

    # -----------------------------------------------------------------------
    # 1. Query TICv8.2
    # -----------------------------------------------------------------------
    if ra is not None and dec is not None:
        print('WARNING: querying by RA/Dec is less robust than querying by TIC ID')
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
        v = Vizier(columns=['*'], row_limit=-1)
        result = v.query_region(coord, radius=2 * u.arcmin, catalog='IV/39/tic82')
        if not result or len(result) == 0:
            print(f'No TIC match at RA={ra}, Dec={dec}')
            return
        qtic = result[0]
        idx = int(np.argmin(qtic['_r']))
        ticid = str(qtic['TIC'][idx])
        print(f'Matching TIC ID is {ticid}')
        qtic = qtic[idx:idx+1]
    else:
        if ticid is None:
            raise ValueError('ticid is required')
        qtic = _query_tic(ticid)
        if qtic is None or len(qtic) == 0:
            print(f'No match for TIC {ticid}')
            return

    if priorfile is None:
        priorfile = f'{ticid}.priors'
    if sedfile is None:
        sedfile = f'{ticid}.sed'

    star_ra  = _safe(qtic, 'RAJ2000')
    star_dec = _safe(qtic, 'DEJ2000')
    star_coord = SkyCoord(ra=star_ra * u.deg, dec=star_dec * u.deg, frame='icrs')
    search_r = 2.0  # arcmin

    # -----------------------------------------------------------------------
    # 2. Stellar parameters from TICv8.2
    # -----------------------------------------------------------------------
    # Column names in IV/39/tic82 as returned by Vizier:
    #   Mass, Rad, Teff, [M/H], e_[M/H], 2MASS, GAIA, UCAC4, TYC
    mass = _safe(qtic, 'Mass')
    rad  = _safe(qtic, 'Rad')
    teff = _safe(qtic, 'Teff')
    feh  = _safe(qtic, '[M/H]')
    ufeh = _safe(qtic, 'e_[M/H]')
    if np.isfinite(ufeh):
        ufeh = max(0.08, ufeh)

    disp    = _safe_str(qtic, 'Disp')
    gaiaid  = _safe_str(qtic, 'GAIA')
    tmassid = _safe_str(qtic, '2MASS')
    tycid   = _safe_str(qtic, 'TYC')

    # -----------------------------------------------------------------------
    # 3. Write files
    # -----------------------------------------------------------------------
    sed_fmt = '{:<13s} {:9.6f} {:8.6f} {:8.6f}\n'

    with open(priorfile, 'w') as pf, open(sedfile, 'w') as sf:

        # --- Prior file header ---
        pf.write('#### TICv8.2 ####\n')
        if disp in ('SPLIT', 'DUPLICATE'):
            pf.write(f'# WARNING: disposition in TICv8.2 is {disp}\n')

        if np.isfinite(mass) and np.isfinite(rad) and np.isfinite(teff):
            pf.write(f'mstar {mass:.2f}\n')
            pf.write(f'rstar {rad:.2f}\n')
            pf.write(f'teff {int(teff)}\n')

        if np.isfinite(feh) and np.isfinite(ufeh):
            pf.write(f'feh {feh:.5f} {ufeh:.5f}\n')

        pf.write('##############\n')
        pf.write(_get_av_upper_limit(star_ra, star_dec) + '\n')

        # --- SED file header ---
        sf.write('# bandname magnitude used_errors catalog_errors\n')
        if disp in ('SPLIT', 'DUPLICATE'):
            sf.write(f'# WARNING: disposition in TICv8.2 is {disp}\n')

        # -------------------------------------------------------------------
        # 4. Gaia DR2 (I/345/gaia2) - commented out for reference
        # -------------------------------------------------------------------
        if gaiaid:
            qg2 = _query_region('I/345/gaia2', star_coord, search_r)
            if qg2 is not None:
                mask = np.array([str(s) == gaiaid for s in qg2['Source']])
                if np.any(mask):
                    row = qg2[mask][0:1]
                    gmag    = _safe(row, 'Gmag')
                    e_gmag  = _safe(row, 'e_Gmag')
                    bpmag   = _safe(row, 'BPmag')
                    e_bpmag = _safe(row, 'e_BPmag')
                    rpmag   = _safe(row, 'RPmag')
                    e_rpmag = _safe(row, 'e_RPmag')
                    plx     = _safe(row, 'Plx')
                    e_plx   = _safe(row, 'e_Plx')

                    if gmag > -9 and np.isfinite(e_gmag) and e_gmag < 1.0:
                        sf.write('#' + sed_fmt.format('Gaia', gmag, max(0.02, e_gmag), e_gmag))
                    if bpmag > -9 and np.isfinite(e_bpmag) and e_bpmag < 1.0:
                        sf.write('#' + sed_fmt.format('GaiaBP', bpmag, max(0.02, e_bpmag), e_bpmag))
                    if rpmag > -9 and np.isfinite(e_rpmag) and e_rpmag < 1.0:
                        sf.write('#' + sed_fmt.format('GaiaRP', rpmag, max(0.02, e_rpmag), e_rpmag))

                    # DR2 parallax (commented; fallback if DR3 unavailable)
                    if np.isfinite(plx) and np.isfinite(e_plx) and plx > 0:
                        k = 1.08
                        sigma_s = 0.021 if gmag <= 13 else 0.043
                        corr_plx  = plx + 0.030
                        corr_uplx = np.sqrt((k * e_plx)**2 + sigma_s**2)
                        pf.write(f'# NOTE: Gaia DR2 parallax ({plx:.5f}) corrected per Lindegren+ 2018\n')
                        pf.write(f'# parallax {corr_plx:.5f} {corr_uplx:.5f}\n')

        # -------------------------------------------------------------------
        # 5. Gaia DR3 (I/355/gaiadr3)
        #    Magnitude errors computed from flux errors: e_mag = 2.5/ln(10) * e_F/F
        #    AllWISE ID also retrieved here for WISE matching
        # -------------------------------------------------------------------
        wiseid = ''
        if gaiaid:
            qg3 = _query_region('I/355/gaiadr3', star_coord, search_r)
            if qg3 is not None:
                mask = np.array([str(s) == gaiaid for s in qg3['Source']])
                if np.any(mask):
                    row = qg3[mask][0:1]

                    # WISE ID from Gaia DR3 cross-match
                    wiseid = _safe_str(row, 'AllWISE')

                    ruwe = _safe(row, 'RUWE')
                    sf.write(f'# Gaia DR3 RUWE = {ruwe:.4f}\n')
                    sf.write('# RUWE is the renormalized sqrt(chi^2/dof) of the astrometric fit.\n')
                    sf.write('# A value above 1.4 is a strong indication of stellar multiplicity\n')

                    plx   = _safe(row, 'Plx')
                    e_plx = _safe(row, 'e_Plx')
                    gmag  = _safe(row, 'Gmag')

                    if np.isfinite(plx) and np.isfinite(e_plx) and plx > 0:
                        uplx = np.sqrt(e_plx**2 + 0.01**2)

                        # Lindegren+2021 zero-point correction (requires gaiadr3-zeropoint)
                        corrected = False
                        try:
                            from zero_point import zpt as gaia_zpt
                            nueff  = _safe(row, 'nueff')
                            pscol  = _safe(row, 'pscol')
                            elat   = _safe(row, 'ELAT')
                            try:
                                solved = int(_safe_str(row, 'Solved'))
                            except Exception:
                                solved = 0

                            in_range = (
                                (solved == 31 and np.isfinite(nueff) and 1.1 <= nueff <= 1.9) or
                                (solved == 95 and np.isfinite(pscol) and 1.24 <= pscol <= 1.72)
                            ) and 6 <= gmag <= 21
                            if in_range:
                                zp = gaia_zpt.get_zpt(gmag, nueff, pscol, elat, solved)
                                pf.write(f'# NOTE: Gaia DR3 parallax ({plx:.5f}) corrected by -{zp:.4f} mas (Lindegren+ 2021)\n')
                                pf.write(f'# NOTE: Parallax uncertainty added in quadrature with 0.01 mas\n')
                                pf.write(f'parallax {plx - zp:.5f} {uplx:.5f}\n')
                                corrected = True
                        except ImportError:
                            pass

                        if not corrected:
                            pf.write(f'# NOTE: Gaia DR3 parallax could not be corrected; using raw value\n')
                            pf.write(f'parallax {plx:.5f} {uplx:.5f}\n')

                    # Gaia DR3 photometry: errors from flux errors
                    fg   = _safe(row, 'FG');   e_fg  = _safe(row, 'e_FG')
                    fbp  = _safe(row, 'FBP');  e_fbp = _safe(row, 'e_FBP')
                    frp  = _safe(row, 'FRP');  e_frp = _safe(row, 'e_FRP')
                    gmag   = _safe(row, 'Gmag')
                    bpmag  = _safe(row, 'BPmag')
                    rpmag  = _safe(row, 'RPmag')
                    e_gmag  = _flux_to_magerr(e_fg,  fg)
                    e_bpmag = _flux_to_magerr(e_fbp, fbp)
                    e_rpmag = _flux_to_magerr(e_frp, frp)

                    if gmag > -9 and np.isfinite(e_gmag) and e_gmag < 1.0:
                        sf.write(sed_fmt.format('Gaia_G_EDR3', gmag, max(0.02, e_gmag), e_gmag))
                    if bpmag > -9 and np.isfinite(e_bpmag) and e_bpmag < 1.0:
                        sf.write(sed_fmt.format('Gaia_BP_EDR3', bpmag, max(0.02, e_bpmag), e_bpmag))
                    if rpmag > -9 and np.isfinite(e_rpmag) and e_rpmag < 1.0:
                        sf.write(sed_fmt.format('Gaia_RP_EDR3', rpmag, max(0.02, e_rpmag), e_rpmag))

        # -------------------------------------------------------------------
        # 6. 2MASS (II/246/out)
        #    Column is '2MASS' (not '_2MASS')
        # -------------------------------------------------------------------
        q2mass = _query_region('II/246/out', star_coord, search_r)
        if q2mass is not None and tmassid:
            mask = np.array([str(s).strip() == tmassid.strip() for s in q2mass['2MASS']])
            if np.any(mask):
                row = q2mass[mask][0:1]
                jmag  = _safe(row, 'Jmag'); e_jmag = _safe(row, 'e_Jmag')
                hmag  = _safe(row, 'Hmag'); e_hmag = _safe(row, 'e_Hmag')
                kmag  = _safe(row, 'Kmag'); e_kmag = _safe(row, 'e_Kmag')

                if jmag > -9 and e_jmag < 1.0:
                    sf.write(sed_fmt.format('J2M', jmag, max(0.02, e_jmag), e_jmag))
                if hmag > -9 and e_hmag < 1.0:
                    sf.write(sed_fmt.format('H2M', hmag, max(0.02, e_hmag), e_hmag))
                if kmag > -9 and e_kmag < 1.0:
                    sf.write(sed_fmt.format('K2M', kmag, max(0.02, e_kmag), e_kmag))
                    pf.write('# Apparent 2MASS K magnitude for the Mann relation\n')
                    pf.write(f'appks {kmag:.6f} {max(0.02, e_kmag):.6f}\n')

        # -------------------------------------------------------------------
        # 7. AllWISE (II/328/allwise)
        #    WISE ID comes from Gaia DR3 AllWISE column
        # -------------------------------------------------------------------
        qwise = _query_region('II/328/allwise', star_coord, search_r)
        if qwise is not None:
            wise_row = None
            if wiseid:
                # Normalize ID: 'J123232.95-093627.3' style
                mask = np.array([str(s).strip() == wiseid.strip() for s in qwise['AllWISE']])
                if np.any(mask):
                    wise_row = qwise[mask][0:1]
                else:
                    print(f'No exact WISE ID match ({wiseid}); skipping WISE')
            if wise_row is not None:
                w1 = _safe(wise_row, 'W1mag'); e_w1 = _safe(wise_row, 'e_W1mag')
                w2 = _safe(wise_row, 'W2mag'); e_w2 = _safe(wise_row, 'e_W2mag')
                w3 = _safe(wise_row, 'W3mag'); e_w3 = _safe(wise_row, 'e_W3mag')
                w4 = _safe(wise_row, 'W4mag'); e_w4 = _safe(wise_row, 'e_W4mag')

                if w1 > -9 and np.isfinite(e_w1) and e_w1 < 1.0:
                    sf.write(sed_fmt.format('WISE1', w1, max(0.03, e_w1), e_w1))
                if w2 > -9 and np.isfinite(e_w2) and e_w2 < 1.0:
                    sf.write(sed_fmt.format('WISE2', w2, max(0.03, e_w2), e_w2))
                if w3 > -9 and np.isfinite(e_w3) and e_w3 < 1.0:
                    sf.write(sed_fmt.format('WISE3', w3, max(0.03, e_w3), e_w3))
                if w4 > -9 and np.isfinite(e_w4) and e_w4 < 1.0:
                    sf.write(sed_fmt.format('WISE4', w4, max(0.10, e_w4), e_w4))

        # -------------------------------------------------------------------
        # 8. Metallicity fallback: Stromgren via Paunzen 2015 + Cassegrande+2011
        # -------------------------------------------------------------------
        if not (np.isfinite(feh) and np.isfinite(ufeh)):
            qp = _query_region('J/A+A/580/A23/catalog', star_coord, search_r)
            if qp is not None and tycid:
                try:
                    tyc_ids = [
                        f'{int(row["TYC1"]):04d}-{int(row["TYC2"]):05d}-{int(row["TYC3"]):01d}'
                        for row in qp
                    ]
                    match_idx = next((i for i, t in enumerate(tyc_ids) if t == tycid), None)
                    if match_idx is not None:
                        row = qp[match_idx:match_idx+1]
                        by = _safe(row, 'b-y')
                        m1 = _safe(row, 'm1')
                        c1 = _safe(row, 'c1')

                        if 0.23 < by < 0.63 and 0.05 < m1 <= 0.68 and 0.13 < c1 <= 0.60:
                            feh = (3.927 * np.log10(m1) - 14.459 * m1**3
                                   - 5.394 * by * np.log10(m1) + 36.069 * by * m1**3
                                   + 3.537 * c1 * np.log10(m1) - 3.500 * m1**3 * c1
                                   + 11.034 * by - 22.780 * by**2
                                   + 10.684 * c1 - 6.759 * c1**2 - 1.548)
                            ufeh = 0.10
                            pf.write('# Cassegrande+ 2011, eq 2\n')
                            pf.write(f'feh {feh:.5f} {ufeh:.5f}\n')
                        elif 0.43 < by < 0.63 and 0.07 < m1 <= 0.68 and 0.16 < c1 <= 0.49:
                            feh = (-0.116 * c1 - 1.624 * c1**2 + 8.955 * c1 * by
                                   + 42.008 * by - 99.596 * by**2 + 64.245 * by**3
                                   + 8.928 * c1 * m1 + 17.275 * m1 - 48.106 * m1**2
                                   + 45.802 * m1**3 - 8.467)
                            ufeh = 0.12
                            pf.write('# Cassegrande+ 2011, eq 3\n')
                            pf.write(f'feh {feh:.5f} {ufeh:.5f}\n')
                except Exception as e:
                    print(f'  [warn] Stromgren metallicity lookup failed: {e}')

            if not (np.isfinite(feh) and np.isfinite(ufeh)):
                pf.write('# wide Gaussian prior\n')
                pf.write('feh 0.00000 1.00000\n')

        # -------------------------------------------------------------------
        # 9. GALEX (II/312/ais) - commented out by default
        # -------------------------------------------------------------------
        qgalex = _query_region('II/312/ais', star_coord, search_r)
        if qgalex is not None and len(qgalex) > 0:
            sf.write('# Galex DR5, Bianchi+ 2011\n')
            sf.write('# http://adsabs.harvard.edu/abs/2011Ap%26SS.335..161B\n')
            sf.write('# Atmospheric models are generally untrustworthy here; may bias the fit\n')
            if len(qgalex) > 1:
                print('Warning: More than 1 GALEX source found; using nearest one only.')
                sf.write('# Warning: More than 1 GALEX source found; using nearest one only.\n')
                qgalex = qgalex[0:1]
            fuv  = _safe(qgalex, 'FUV'); e_fuv = _safe(qgalex, 'e_FUV')
            nuv  = _safe(qgalex, 'NUV'); e_nuv = _safe(qgalex, 'e_NUV')
            c = '' if use_galex else '#'
            if fuv > -99 and np.isfinite(e_fuv):
                sf.write(c + sed_fmt.format('galFUV', fuv, max(0.1, e_fuv), e_fuv))
            if nuv > -99 and np.isfinite(e_nuv):
                sf.write(c + sed_fmt.format('galNUV', nuv, max(0.1, e_nuv), e_nuv))

        # -------------------------------------------------------------------
        # 10. Tycho-2 (I/259/tyc2) - commented out by default
        # -------------------------------------------------------------------
        qtyc2 = _query_region('I/259/tyc2', star_coord, search_r)
        if qtyc2 is not None and len(qtyc2) > 0:
            sf.write('# Tycho catalog, Hoeg+ 2000\n')
            sf.write('# http://adsabs.harvard.edu/abs/2000A%26A...355L..27H\n')
            if len(qtyc2) > 1:
                print('Warning: More than 1 Tycho-2 source found; using nearest one only.')
                sf.write('# Warning: More than 1 Tycho-2 source found; using nearest one only.\n')
                qtyc2 = qtyc2[0:1]
            btmag = _safe(qtyc2, 'BTmag'); e_btmag = _safe(qtyc2, 'e_BTmag')
            vtmag = _safe(qtyc2, 'VTmag'); e_vtmag = _safe(qtyc2, 'e_VTmag')
            c = '' if use_tycho else '#'
            if btmag > -9 and np.isfinite(e_btmag):
                sf.write(c + sed_fmt.format('BT', btmag, max(0.02, e_btmag), e_btmag))
            if vtmag > -9 and np.isfinite(e_vtmag):
                sf.write(c + sed_fmt.format('VT', vtmag, max(0.02, e_vtmag), e_vtmag))

        # -------------------------------------------------------------------
        # 11. UCAC4 / APASS (I/322A/out) - commented out by default
        #     UCAC4 stores errors as integer centimag; convert *0.01 to mag
        # -------------------------------------------------------------------
        qucac4 = _query_region('I/322A/out', star_coord, search_r)
        if qucac4 is not None and len(qucac4) > 0:
            sf.write('# APASS DR6 (via UCAC4), Henden+ 2016\n')
            sf.write('# http://adsabs.harvard.edu/abs/2016yCat.2336....0H\n')
            if len(qucac4) > 1:
                print('Warning: More than 1 UCAC-4 source found; using nearest one only.')
                sf.write('# Warning: More than 1 UCAC-4 source found; using nearest one only.\n')
                qucac4 = qucac4[0:1]
            bmag    = _safe(qucac4, 'Bmag');  e_bmag   = _safe(qucac4, 'e_Bmag')
            vmag    = _safe(qucac4, 'Vmag');  e_vmag   = _safe(qucac4, 'e_Vmag')
            gmag_a  = _safe(qucac4, 'gmag');  e_gmag_a = _safe(qucac4, 'e_gmag')
            rmag    = _safe(qucac4, 'rmag');  e_rmag   = _safe(qucac4, 'e_rmag')
            imag    = _safe(qucac4, 'imag');  e_imag   = _safe(qucac4, 'e_imag')
            c = '' if use_ucac else '#'
            if bmag > -9 and np.isfinite(e_bmag) and e_bmag != 99:
                sf.write(c + sed_fmt.format('B', bmag, max(0.02, e_bmag * 0.01), e_bmag * 0.01))
            if vmag > -9 and np.isfinite(e_vmag) and e_vmag != 99:
                sf.write(c + sed_fmt.format('V', vmag, max(0.02, e_vmag * 0.01), e_vmag * 0.01))
            if gmag_a > -9 and np.isfinite(e_gmag_a):
                sf.write(c + sed_fmt.format('gSDSS', gmag_a, max(0.02, e_gmag_a * 0.01), e_gmag_a * 0.01))
            if rmag > -9 and np.isfinite(e_rmag):
                sf.write(c + sed_fmt.format('rSDSS', rmag, max(0.02, e_rmag * 0.01), e_rmag * 0.01))
            if imag > -9 and np.isfinite(e_imag):
                sf.write(c + sed_fmt.format('iSDSS', imag, max(0.02, e_imag * 0.01), e_imag * 0.01))

        # -------------------------------------------------------------------
        # 12. Stromgren (J/A+A/580/A23/catalog) - commented out by default
        # -------------------------------------------------------------------
        qp = _query_region('J/A+A/580/A23/catalog', star_coord, search_r)
        if qp is not None and len(qp) > 0:
            if len(qp) > 1:
                print('Warning: More than 1 Paunzen source found; using nearest one only.')
                sf.write('# Warning: More than 1 Paunzen source found; using nearest one only.\n')
                qp = qp[0:1]
            vmag_s  = _safe(qp, 'Vmag');  e_vmag_s = _safe(qp, 'e_Vmag')
            by      = _safe(qp, 'b-y');   e_by     = _safe(qp, 'e_b-y')
            m1      = _safe(qp, 'm1');    e_m1     = _safe(qp, 'e_m1')
            c1_val  = _safe(qp, 'c1');    e_c1     = _safe(qp, 'e_c1')
            if all(np.isfinite(v) for v in [vmag_s, by, m1, c1_val]):
                us, sus, vs, svs, bs, sbs, ys, sys_ = strom_conv(
                    vmag_s, max(0.01, e_vmag_s),
                    by, max(0.02, e_by),
                    m1, max(0.02, e_m1),
                    c1_val, max(0.02, e_c1)
                )
                sf.write('# Stromgren photometry, Paunzen, 2015\n')
                sf.write('# http://adsabs.harvard.edu/abs/2015A%26A...580A..23P\n')
                sf.write('# Matching is done by nearest neighbor with ~10% failure rate\n')
                c = '' if use_stromgren else '#'
                if us > -9: sf.write(c + sed_fmt.format('uStr', us, max(0.02, sus), sus))
                if vs > -9: sf.write(c + sed_fmt.format('vStr', vs, max(0.02, svs), svs))
                if bs > -9: sf.write(c + sed_fmt.format('bStr', bs, max(0.02, sbs), sbs))
                if ys > -9: sf.write(c + sed_fmt.format('yStr', ys, max(0.02, sys_), sys_))

        # -------------------------------------------------------------------
        # 13. Mermilliod 1994 UBV (II/168/ubvmeans) - commented out by default
        # -------------------------------------------------------------------
        qmerm = _query_region('II/168/ubvmeans', star_coord, search_r)
        if qmerm is not None and len(qmerm) > 0:
            if len(qmerm) > 1:
                print('Warning: More than 1 Mermilliod source found; using nearest one only.')
                sf.write('# Warning: More than 1 Mermilliod source found; using nearest one only.\n')
                qmerm = qmerm[0:1]
            sf.write('# Mermilliod, 1994\n')
            sf.write('# http://adsabs.harvard.edu/abs/1994yCat.2193....0M\n')
            sf.write('# Matching is done by nearest neighbor with ~10% failure rate\n')
            vmag  = _safe(qmerm, 'Vmag'); e_vmag = _safe(qmerm, 'e_Vmag')
            bv    = _safe(qmerm, 'B-V');  e_bv   = _safe(qmerm, 'e_B-V')
            ub    = _safe(qmerm, 'U-B');  e_ub   = _safe(qmerm, 'e_U-B')
            c = '' if use_merm else '#'
            if all(np.isfinite(v) for v in [ub, bv, vmag, e_ub, e_bv, e_vmag]):
                U   = ub + bv + vmag
                e_U = np.sqrt(e_ub**2 + e_bv**2 + e_vmag**2)
                sf.write(c + sed_fmt.format('U', U, max(0.02, e_U), e_U))
            if all(np.isfinite(v) for v in [bv, vmag, e_bv, e_vmag]):
                B   = bv + vmag
                e_B = np.sqrt(e_bv**2 + e_vmag**2)
                sf.write(c + sed_fmt.format('B', B, max(0.02, e_B), e_B))
            if np.isfinite(vmag) and np.isfinite(e_vmag):
                sf.write(c + sed_fmt.format('V', vmag, max(0.02, e_vmag), e_vmag))

    print(f'Done. Priors: {priorfile}   SED: {sedfile}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate SED and priors files from TICv8.2 (port of EXOFASTv2 mkticsed.pro)')
    parser.add_argument('ticid', nargs='?', help='TIC ID number')
    parser.add_argument('--ra',  type=float, help='RA in decimal degrees')
    parser.add_argument('--dec', type=float, help='Dec in decimal degrees')
    parser.add_argument('--priorfile', help='Output prior filename (default: <ticid>.priors)')
    parser.add_argument('--sedfile',   help='Output SED filename (default: <ticid>.sed)')
    parser.add_argument('--galex',     action='store_true', help='Enable GALEX photometry')
    parser.add_argument('--tycho',     action='store_true', help='Enable Tycho-2 photometry')
    parser.add_argument('--ucac',      action='store_true', help='Enable UCAC4/APASS photometry')
    parser.add_argument('--merm',      action='store_true', help='Enable Mermilliod UBV photometry')
    parser.add_argument('--stromgren', action='store_true', help='Enable Stromgren photometry')
    parser.add_argument('--kepler',    action='store_true', help='Enable KIS/Kepler photometry')
    args = parser.parse_args()

    mkticsed(
        ticid=args.ticid,
        ra=args.ra,
        dec=args.dec,
        priorfile=args.priorfile,
        sedfile=args.sedfile,
        use_galex=args.galex,
        use_tycho=args.tycho,
        use_ucac=args.ucac,
        use_merm=args.merm,
        use_stromgren=args.stromgren,
        use_kepler=args.kepler,
    )
