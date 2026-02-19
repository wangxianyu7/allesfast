import numpy as np
import requests
import re
import os
from scipy.io import readsav
from numba import njit
import functools
import os
import pathlib

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

def printandlog(msg, logname=None):
    print(msg)
    if logname:
        try:
            with open(logname, 'a') as f:
                f.write(str(msg) + "\n")
        except OSError:
            pass

def filepath(filename, root_dir, subdir):
    return os.path.join(root_dir, *subdir, filename)

def file_lines(filename):
    with open(filename) as f:
        return len(f.readlines())



# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
# def ninterpolate(data, point):
#     """
#     Perform N-dimensional linear interpolation at a single point.

#     Parameters
#     ----------
#     data : ndarray
#         The N-dimensional data array to interpolate.
#     point : array-like
#         A sequence of length N giving the coordinates at which to interpolate.

#     Returns
#     -------
#     float
#         Interpolated value.
        
#     ; MODIFICATION HISTORY:
#     ;
#     ;    Mon Jul 21 12:33:30 2003, J.D. Smith <jdsmith@as.arizona.edu>
#     ;		Written.
#     ;==========================================================================
#     ; Copyright (C) 2003, J.D. Smith
#     """
#     point = np.asarray(point)
#     ndim = point.size

#     if data.ndim != ndim:
#         raise ValueError("Point must specify 1 coordinate for each dimension of data")

#     if ndim == 1:
#         # 1D special case: np.interp
#         x = np.arange(data.shape[0])
#         return np.interp(point[0], x, data)

#     base = np.floor(point).astype(int)
#     frac = point - base
#     result = 0.0

#     for i in range(2 ** ndim):
#         # Get corner offset in binary
#         offset = [(i >> k) & 1 for k in range(ndim)]
#         idx = tuple(base[k] + offset[k] for k in range(ndim))

#         # Check bounds
#         if any(j < 0 or j >= data.shape[k] for k, j in enumerate(idx)):
#             continue  # skip out-of-bounds

#         weight = 1.0
#         for k in range(ndim):
#             weight *= (1 - frac[k]) if offset[k] == 0 else frac[k]

#         result += weight * data[idx]

#     return result

from scipy.ndimage import map_coordinates
# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
import numpy as np


def ninterpolate(data, point):
    """
    Multilinear interpolation of an n‑dimensional array.

    Parameters
    ----------
    data : ndarray
        n‑D array of values.
    point : 1‑D sequence (length n)
        Real‑valued coordinates at which to interpolate.

    Returns
    -------
    float
        Interpolated value at `point`.
    """
    data = np.asarray(data, dtype=float)
    point = np.asarray(point, dtype=float)

    # -- dimension checks -------------------------------------------------
    n = point.size
    if n != data.ndim:
        raise ValueError("`point` must supply one coordinate for each dimension")

    # -- trivial 1‑D case -------------------------------------------------
    if n == 1:
        x = point[0]
        x0 = int(np.floor(x))
        x1 = x0 + 1
        f = x - x0                        # fractional distance
        x0 = np.clip(x0, 0, data.shape[0] - 1)
        x1 = np.clip(x1, 0, data.shape[0] - 1)
        return (1.0 - f) * data[x0] + f * data[x1]

    # -- general n‑D multilinear case ------------------------------------
    base = np.floor(point).astype(int)    # “lower‑left” corner of the cell
    f = point - base                      # fractional part in each dimension

    value = 0.0
    two_n = 1 << n                        # 2**n corner combinations

    for corner in range(two_n):
        idx = base.copy()
        weight = 1.0
        for dim in range(n):
            if corner & (1 << dim):       # high corner along this axis
                idx[dim] += 1
                weight *= f[dim]
            else:                         # low corner along this axis
                weight *= (1.0 - f[dim])

            # stay inside array bounds
            idx[dim] = np.clip(idx[dim], 0, data.shape[dim] - 1)

        value += weight * data[tuple(idx)]

    return value


@njit(cache=True)
def _ninterpolate4d(data, x, y, z, w):
    """
    Specialized 4-D multilinear interpolation (faster than generic loop).
    data shape: (nx, ny, nz, nw)
    """
    x0 = int(np.floor(x)); y0 = int(np.floor(y)); z0 = int(np.floor(z)); w0 = int(np.floor(w))
    fx = x - x0; fy = y - y0; fz = z - z0; fw = w - w0

    nx, ny, nz, nw = data.shape
    x1 = x0 + 1; y1 = y0 + 1; z1 = z0 + 1; w1 = w0 + 1

    if x0 < 0: x0 = 0
    if y0 < 0: y0 = 0
    if z0 < 0: z0 = 0
    if w0 < 0: w0 = 0
    if x1 >= nx: x1 = nx - 1
    if y1 >= ny: y1 = ny - 1
    if z1 >= nz: z1 = nz - 1
    if w1 >= nw: w1 = nw - 1

    v0000 = data[x0, y0, z0, w0]
    v1000 = data[x1, y0, z0, w0]
    v0100 = data[x0, y1, z0, w0]
    v1100 = data[x1, y1, z0, w0]
    v0010 = data[x0, y0, z1, w0]
    v1010 = data[x1, y0, z1, w0]
    v0110 = data[x0, y1, z1, w0]
    v1110 = data[x1, y1, z1, w0]
    v0001 = data[x0, y0, z0, w1]
    v1001 = data[x1, y0, z0, w1]
    v0101 = data[x0, y1, z0, w1]
    v1101 = data[x1, y1, z0, w1]
    v0011 = data[x0, y0, z1, w1]
    v1011 = data[x1, y0, z1, w1]
    v0111 = data[x0, y1, z1, w1]
    v1111 = data[x1, y1, z1, w1]

    c000 = v0000 * (1.0 - fx) + v1000 * fx
    c100 = v0100 * (1.0 - fx) + v1100 * fx
    c010 = v0010 * (1.0 - fx) + v1010 * fx
    c110 = v0110 * (1.0 - fx) + v1110 * fx
    c001 = v0001 * (1.0 - fx) + v1001 * fx
    c101 = v0101 * (1.0 - fx) + v1101 * fx
    c011 = v0011 * (1.0 - fx) + v1011 * fx
    c111 = v0111 * (1.0 - fx) + v1111 * fx

    c00 = c000 * (1.0 - fy) + c100 * fy
    c10 = c010 * (1.0 - fy) + c110 * fy
    c01 = c001 * (1.0 - fy) + c101 * fy
    c11 = c011 * (1.0 - fy) + c111 * fy

    c0 = c00 * (1.0 - fz) + c10 * fz
    c1 = c01 * (1.0 - fz) + c11 * fz

    return c0 * (1.0 - fw) + c1 * fw


@functools.lru_cache(maxsize=8)
def _load_mist_grid(grid_path: str):
    """Read and cache mist.sed.grid.idl (grid definitions for the BC cubes)."""
    g = readsav(grid_path, python_dict=True)
    return g['teffgrid'], g['logggrid'], g['fehgrid'], g['avgrid']


@functools.lru_cache(maxsize=32)
def _load_bc_cube(bc_path: str):
    """Read and cache a single band’s 4‑D bolometric‑correction cube."""
    s = readsav(bc_path, python_dict=True)
    return s['bcarray'], s['filterproperties']

@functools.lru_cache(maxsize=256)
def _load_filter_curve(idl_path: str):
    """Read and cache a filter transmission curve and metadata.

    Accepts both IDL save (.idl) and numpy archive (.npz) formats.
    The .npz format stores the same fields as the IDL struct:
      weff, widtheff, zero_point, transmission  (all scalar/1-D float64).
    """
    if idl_path.endswith('.npz'):
        d = np.load(idl_path)
        transmission = d['transmission'].astype(np.float64)
        weff       = float(d['weff'])
        widtheff   = float(d['widtheff'])
        zero_point = float(d['zero_point'])
    else:
        filt = readsav(idl_path, python_dict=True)['filter']
        transmission = filt['transmission'][0]
        weff       = filt['weff'][0]
        widtheff   = filt['widtheff'][0]
        zero_point = filt['zero_point'][0]
    curve_sum = np.sum(transmission)
    return transmission, weff, widtheff, zero_point, curve_sum


def getfilter(filterid, root_dir=None, redo=False):
    """Download a filter transmission curve from the SVO and save as .npz.

    Replicates EXOFASTv2's getfilter.pro logic.
    The file is saved as <filterid with / -> _>.npz in the filtercurves directory.

    Parameters
    ----------
    filterid : str
        SVO filter ID, e.g. 'Tycho/Tycho.B'.
    root_dir : str or None
        EXOZIPPy module root.  Defaults to MODULE_PATH.
    redo : bool
        Re-download even if file exists.
    """
    import urllib.request, xml.etree.ElementTree as ET
    if root_dir is None:
        root_dir = MODULE_PATH
    outdir = pathlib.Path(root_dir) / 'sed' / 'filtercurves'
    npzname = filterid.replace('/', '_') + '.npz'
    npzpath = outdir / npzname
    if npzpath.exists() and not redo:
        return str(npzpath)

    # --- download SVO XML ---
    url = f'http://svo2.cab.inta-csic.es/theory/fps3/fps.php?ID={filterid}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'EXOZIPPy/1.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            xml_bytes = r.read()
    except Exception as exc:
        print(f'getfilter: download failed for {filterid}: {exc}')
        return None

    # --- parse WavelengthEff, WidthEff, ZeroPoint and transmission table ---
    C_MICRON_S = 2.99792458e14   # speed of light in µm/s
    weff_ang = widtheff_ang = zp_jy = None
    wavelengths, transmissions = [], []

    root = ET.fromstring(xml_bytes)
    ns = {'v': root.tag.split('}')[0].lstrip('{') if '}' in root.tag else ''}
    # walk all elements
    in_table = False
    td_vals = []
    for elem in root.iter():
        tag = elem.tag.split('}')[-1]
        name = elem.get('name', '')
        if tag == 'PARAM':
            if name == 'WavelengthEff':
                weff_ang = float(elem.get('value'))
            elif name == 'WidthEff':
                widtheff_ang = float(elem.get('value'))
            elif name == 'ZeroPoint':
                zp_jy = float(elem.get('value'))
        elif tag == 'TD':
            if elem.text:
                td_vals.append(float(elem.text))

    # TD values come in pairs: wavelength(Å), transmission
    for k in range(0, len(td_vals) - 1, 2):
        wavelengths.append(td_vals[k])
        transmissions.append(td_vals[k+1])

    if weff_ang is None or zp_jy is None or len(wavelengths) < 2:
        print(f'getfilter: could not parse SVO data for {filterid}')
        return None

    # --- convert to microns and interpolate onto 24000-pt grid ---
    wave_um  = np.array(wavelengths)  / 1e4   # Å → µm
    trans    = np.array(transmissions)
    intwave  = np.arange(24000) / 1000.0 + 0.1  # 0.100 … 24.099 µm

    # pad outside measured range with zeros
    wave_full = np.concatenate([intwave[intwave < wave_um[0]],
                                wave_um,
                                intwave[intwave > wave_um[-1]]])
    tran_full = np.concatenate([np.zeros(np.sum(intwave < wave_um[0])),
                                trans,
                                np.zeros(np.sum(intwave > wave_um[-1]))])
    inttran = np.interp(intwave, wave_full, tran_full)

    weff_um     = weff_ang     / 1e4
    widtheff_um = widtheff_ang / 1e4
    # zero_point: Jy → erg/s/cm²  (matching EXOFASTv2: 1e-23 * zp_jy * c / weff)
    zero_point  = 1e-23 * zp_jy * C_MICRON_S / weff_um

    np.savez(npzpath,
             weff=weff_um, widtheff=widtheff_um,
             zero_point=zero_point, transmission=inttran)
    print(f'getfilter: saved {npzpath}')
    return str(npzpath)

@functools.lru_cache(maxsize=4)
def _load_filter_names(root_dir: str):
    """Load and cache filter name mappings."""
    filter_file = filepath('filternames2.txt', root_dir, ['sed', 'mist'])
    return np.loadtxt(filter_file, dtype=str, comments="#", unpack=True)

@functools.lru_cache(maxsize=32)
def _load_bcarrays(bands_tuple, root_dir: str):
    """Load and cache stacked BC arrays for a given band set."""
    kname, mname, cname, svoname = _load_filter_names(root_dir)
    root = pathlib.Path(root_dir) / 'sed' / 'mist'
    bc_cubes, filterprops = [], []
    for band in bands_tuple:
        candidates = [band]
        if band in kname:
            candidates.append(mname[np.where(kname == band)[0][0]])
        if band in svoname:
            candidates.append(mname[np.where(svoname == band)[0][0]])

        for cand in candidates:
            bc_path = root / f"{cand}.idl"
            if bc_path.exists():
                bc, props = _load_bc_cube(str(bc_path))
                bc = np.transpose(bc, (3, 2, 1, 0))
                bc_cubes.append(bc)
                filterprops.append(props)
                break
        else:
            raise FileNotFoundError(f"{band} not supported – remove it from sed file")

    bcarrays = np.stack(bc_cubes, axis=-1)  # (nteff, nlogg, nfeh, nav, nbands)
    return bcarrays, filterprops

def mistmultised(teff, logg, feh, av, distance, lstar, errscale, sedfile,
                  *,
                  sed_data=None,
                  redo=False,
                  psname=None, debug=False, atmospheres=None,
                  wavelength=None, logname=None, xyrange=None,
                  blend0=None):
    """
    Parameters
    ----------
    teff, logg, feh, av, distance, lstar : array‑like, shape (nstars,)
    errscale   : scalar or (≥1,) array — identical to the IDL behaviour
    sedfile    : str  — path to the observed SED definition file

    Returns
    -------
    sedchi2      : float
    blendmag     : (nbands,) ndarray
    modelflux    : (nbands, nstars) ndarray
    magresiduals : (nbands,) ndarray
    """

    # ---------- 1. Input validation ---------------------------------------
    teff   = np.atleast_1d(teff).astype(float)
    nstars = teff.size

    def _check(name, arr):
        a = np.atleast_1d(arr).astype(float)
        if a.size != nstars:
            raise ValueError(f"{name} must have same length as teff")
        return a

    logg     = _check('logg',     logg)
    feh      = _check('feh',      feh)
    av       = _check('av',       av)
    distance = _check('distance', distance)
    lstar    = _check('lstar',    lstar)

    err0 = float(np.atleast_1d(errscale)[0])

    # ---------- 2. Read observed SED file ---------------------------------
    if sed_data is None:
        sed_data = read_sed_file(sedfile, nstars, logname=logname)
    sedbands   = sed_data['sedbands']
    mags       = sed_data['mag']
    errs       = sed_data['errmag']
    blend      = sed_data['blend']   # already (nbands, nstars) int array
    nbands     = len(sedbands)

    # ---------- 3. Load / cache the MIST grid & BC cubes ------------------
    root = pathlib.Path(MODULE_PATH) / 'sed' / 'mist'
    gridfile = root / 'mist.sed.grid.idl'
    teffgrid, logggrid, fehgrid, avgrid = _load_mist_grid(str(gridfile))

    bands_tuple = tuple(str(b) for b in sedbands)
    bcarrays, filterprops = _load_bcarrays(bands_tuple, str(MODULE_PATH))

    if blend0 is not None:
        blend0[:] = blend.copy()

    # ---------- 4. Interpolate bolometric corrections ---------------------
    bcs = np.empty((nbands, nstars))
    for j in range(nstars):
        coord = [get_grid_point(g, v) for g, v in
                 ((teffgrid, teff[j]),
                  (logggrid, logg[j]),
                  (fehgrid,  feh[j]),
                  (avgrid,   av[j]))]
        for i in range(nbands):
            if bcarrays[..., i].ndim == 4:
                bcs[i, j] = _ninterpolate4d(
                    bcarrays[..., i], coord[0], coord[1], coord[2], coord[3]
                )
            else:
                bcs[i, j] = ninterpolate(bcarrays[..., i], coord)
    # ---------- 5. Model magnitudes / fluxes ------------------------------
    mu         = 5.0 * np.log10(distance) - 5.0         # (nstars,)
    logL_term  = -2.5 * np.log10(lstar)                 # (nstars,)
    modelmag   = (logL_term[None, :] + 4.74             # (nbands, nstars)
                  - bcs + mu[None, :])

    modelflux  = 10.0 ** (-0.4 * modelmag)              # (nbands, nstars)

    if nstars == 1:
        blendmag       = modelmag[:, 0]                 # (nbands,)
        blendflux      = modelflux[:, 0]
        magresiduals   = mags - blendmag
    else:
        pos_flux  = (modelflux * (blend > 0)).sum(axis=1)
        neg_flux  = (modelflux * (blend < 0)).sum(axis=1)
        neg_flux[neg_flux == 0.0] = 1.0
        blendmag       = -2.5 * np.log10(pos_flux / neg_flux)
        magresiduals   = mags - blendmag
        blendflux      = (modelflux * blend).sum(axis=1)

    # ---------- 6. χ² likelihood (matches IDL exofast_like /chi2) ----------
    sigma = errs * err0
    sedchi2 = np.sum(magresiduals**2 / sigma**2 + np.log(2.0 * np.pi * sigma**2))
    return sedchi2, blendmag, modelflux, magresiduals


# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def get_grid_point(grid, value):
    """
    Linearly interpolate or extrapolate the index corresponding to a value in a 1D grid.

    Parameters:
    grid (np.ndarray): 1D sorted array of grid points.
    value (float): The value to locate within the grid.

    Returns:
    float: The fractional index position of the value.
    """
    grid = np.asarray(grid)
    ngrid = grid.size

    # Find index of the closest grid point
    match = np.argmin(np.abs(grid - value))

    if match == ngrid - 1:
        # Extrapolate beyond last grid point
        ndx = match + (value - grid[-1]) / (grid[-1] - grid[-2])
    elif match == 0:
        # Extrapolate before first grid point
        ndx = match + (value - grid[0]) / (grid[1] - grid[0])
    else:
        # Interpolate between two nearest grid points
        if value > grid[match]:
            ndx = match + (value - grid[match]) / (grid[match + 1] - grid[match])
        else:
            ndx = match + (value - grid[match]) / (grid[match] - grid[match - 1])

    return ndx


# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def read_coeffs(filename, delimiter=','):
    """
    Reads Gaia EDR3 zero point coefficient files.
    Returns:
        j, k : index arrays for basis functions
        g    : g_mag bins
        q_jk : coefficient matrix of shape (len(g), m)
        n, m : dimensions
    """
    with open(filename, 'r') as f:
        j_line = f.readline()
        k_line = f.readline()

    j = np.array(j_line.strip().split(delimiter)[1:], dtype=int)
    k = np.array(k_line.strip().split(delimiter)[1:], dtype=int)

    data = np.genfromtxt(filename, delimiter=delimiter, skip_header=2)

    g = data[:, 0]
    q_jk = data[:, 1:]

    if q_jk.shape[1] != len(j):
        raise ValueError("Mismatch in number of coefficients")

    n, m = q_jk.shape
    return j, k, g, q_jk, n, m

# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def get_zpt(phot_g_mean_mag, nu_eff_used_in_astrometry, pseudocolor,
            ecl_lat, astrometric_params_solved, exofast_path='./'):
    """
    Gaia EDR3 parallax zero-point correction based on Lindegren+ 2021 prescription.
    """
    # Determine file based on solution type
    if astrometric_params_solved == 31:
        color = nu_eff_used_in_astrometry
        filename = os.path.join(exofast_path, 'sed', 'z5_200720.txt')
        j, k, g, q_jk, n, m = read_coeffs(filename, delimiter=' ')
    elif astrometric_params_solved == 95:
        color = pseudocolor
        filename = os.path.join(exofast_path, 'sed', 'z6_200720.txt')
        j, k, g, q_jk, n, m = read_coeffs(filename, delimiter=',')
    else:
        raise ValueError("Unknown astrometric_params_solved value (expected 31 or 95)")

    sinbeta = np.sin(np.radians(ecl_lat))

    # Color basis functions
    c = np.array([
        1.0,
        np.clip(color - 1.48, -0.24, 0.24),
        min(0.24, max(0.0, 1.48 - color))**3,
        min(0.0, color - 1.24),
        max(0.0, color - 1.72)
    ])

    # Latitude basis functions
    b = np.array([
        1.0,
        sinbeta,
        sinbeta**2 - 1.0 / 3.0
    ])

    # Determine g-mag bin
    if phot_g_mean_mag <= 6.0:
        ig = 0
    elif phot_g_mean_mag > 20.0:
        ig = len(g) - 2
    else:
        ig = np.where(phot_g_mean_mag >= g)[0][-1]

    h = np.clip((phot_g_mean_mag - g[ig]) / (g[ig + 1] - g[ig]), 0.0, 1.0)

    # Interpolate and compute zpt
    zpt = 0.0
    for i in range(m):
        coeff = (1.0 - h) * q_jk[ig, i] + h * q_jk[ig + 1, i]
        zpt += coeff * c[j[i]] * b[k[i]]

    return zpt / 1e3  # Return in arcseconds


# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def read_sed_file(
    sedfile,
    nstars,
    sedbands=None,
    mag=None,
    errmag=None,
    blend=None,
    flux=None,
    errflux=None,
    filter_curves=None,
    weff=None,
    widtheff=None,
    zero_point=None,
    download_new=False,
    filter_curve_sum=None,
    logname=None,
):


    nlines = file_lines(sedfile)

    sedbands = np.empty(nlines, dtype=object)
    mag = np.empty(nlines)
    errmag = np.full(nlines, 99.0)
    flux = np.empty(nlines)
    errflux = np.full(nlines, 99.0)
    blend = np.zeros((nlines, nstars), dtype=int)
    filter_curves = np.empty((nlines, 24000))
    weff = np.empty(nlines)
    widtheff = np.empty(nlines)
    zero_point = np.empty(nlines)
    filter_curve_sum = np.empty(nlines)

    with open(sedfile, 'r') as f:
        lines = f.readlines()

    # Load filter name mapping (cached)
    root_dir = MODULE_PATH
    keivanname, mistname, claretname, svoname = _load_filter_names(root_dir)

    for i, line in enumerate(lines):
        line = line.strip()
        line = line.split('#')[0].strip()  # Remove comments
        if not line:
            continue

        entries = line.split()
        if len(entries) < 3:
            printandlog(f'Line {i+1} in SED file not a legal line: {lines[i]}', logname)
            continue

        sedbands[i] = entries[0]
        mag[i] = float(entries[1])
        errmag[i] = float(entries[2])

        # Attempt to load filter curve (.idl or .npz)
        def _find_filter(name):
            """Return path to existing .idl or .npz file, or None."""
            for ext in ('.idl', '.npz'):
                p = filepath(name + ext, root_dir, ['sed', 'filtercurves'])
                if os.path.isfile(p):
                    return p
            return None

        idlfile = _find_filter(sedbands[i])

        if idlfile is None:
            # Try mapping via filternames2.txt
            svo_id = None
            match = np.where(keivanname == sedbands[i])[0]
            if match.size == 1:
                svo_id = svoname[match[0]]
            else:
                match = np.where(mistname == sedbands[i])[0]
                if match.size == 1:
                    svo_id = svoname[match[0]]

            if svo_id and svo_id != 'Unsupported':
                idlfile = _find_filter(svo_id.replace('/', '_'))

            if idlfile is None and svo_id and svo_id != 'Unsupported':
                # Try to download from SVO
                result = getfilter(svo_id, root_dir=root_dir)
                if result:
                    idlfile = result

        if idlfile is None:
            printandlog(f'band="{sedbands[i]}" in SED file not recognized; skipping', logname)
            errmag[i] = 99.0
            continue

        transmission, w_eff, width_eff, zp, curve_sum = _load_filter_curve(str(idlfile))
        filter_curves[i, :] = transmission
        weff[i] = w_eff
        widtheff[i] = width_eff
        zero_point[i] = zp
        filter_curve_sum[i] = curve_sum

        flux[i] = zero_point[i] * 10 ** (-0.4 * mag[i])
        errflux[i] = flux[i] * np.log(10) / 2.5 * errmag[i]

        if len(entries) == 5:
            if '-' in entries[4]:
                pos_part, neg_part = entries[4].split('-')
                posndx = np.array([int(x) for x in pos_part.split(',')])
                good = posndx[(posndx < nstars) & (posndx >= 0)]
                bad = np.setdiff1d(posndx, good)
                if bad.size > 0:
                    printandlog(f'WARNING: STARNDX ({bad}) does not correspond to a star', logname)
                if good.size == 0:
                    continue
                blend[i, good] = 1

                negndx = np.array([int(x) for x in neg_part.split(',')])
                good = negndx[(negndx < nstars) & (negndx >= 0)]
                bad = np.setdiff1d(negndx, good)
                if bad.size > 0:
                    printandlog(f'WARNING: STARNDX ({bad}) does not correspond to a star', logname)
                if good.size == 0:
                    continue
                blend[i, good] = -1
            else:
                starndx = np.array([int(x) for x in entries[4].split(',')])
                good = starndx[(starndx < nstars) & (starndx >= 0)]
                bad = np.setdiff1d(starndx, good)
                if bad.size > 0:
                    printandlog(f'WARNING: STARNDX ({bad}) does not correspond to a star', logname)
                if good.size == 0:
                    continue
                blend[i, good] = 1
        else:
            blend[i, :] = 1  # Assume blended by default

    good = np.where(errmag < 1.0)[0]
    if good.size > 1:
        sedbands = sedbands[good]
        mag = mag[good]
        errmag = errmag[good]
        flux = flux[good]
        errflux = errflux[good]
        blend = blend[good, :]
        filter_curves = filter_curves[good, :]
        weff = weff[good]
        widtheff = widtheff[good]
        zero_point = zero_point[good]
        filter_curve_sum = filter_curve_sum[good]
    else:
        printandlog("Bands must have errors less than 1 mag; no good bands", logname)
        raise ValueError("No valid SED bands")

    return {
        'sedbands': sedbands,
        'mag': mag,
        'errmag': errmag,
        'flux': flux,
        'errflux': errflux,
        'blend': blend,
        'filter_curves': filter_curves,
        'weff': weff,
        'widtheff': widtheff,
        'zero_point': zero_point,
        'filter_curve_sum': filter_curve_sum
    }
