"""Doppler-Tomography chi-square — line-by-line port of
EXOFASTv2 dopptom_chi2.pro (Beatty 2015, Eastman 2017).

The model is the velocity-space shadow caused by a planet occulting
part of a rotating stellar disk:

    shadow(v, t) = β(t) · normalized( S(v - up(t)) ⊛ G(v) )

where
    β(t)   = 1 − F_MA(z, k, u1, u2)        Mandel-Agol transit depth
    up(t)  = x_p(t)·cos λ − y_p(t)·sin λ   LOS velocity of occulted region
    S(v)   = (2/π k) · sqrt(1 − (v/k)²)    elliptical "planet line"
    G(v)   = Gaussian with σ = sqrt(vline² + (c/R)²) / (2√(2 ln 2))

Final χ² = Σ ((CCF_obs − model) / (RMS · errscale))² / IndepVels,
where IndepVels = (c/R) / (FWHM2σ · meanstep · vsini) corrects for
oversampling within the spectrograph resolution element.
"""
import numpy as np
from pytransit.models.numba.ma_quadratic_nb import eval_quad_z_v

from .geometry import primary_transit_phase, sky_positions


_FWHM2SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
_C_KMS = 299792.458


def _gaus_convol(x, y, sig, nsigma=2.5):
    """Port of IDL gaus_convol.pro (1D direct convolution).

    Build a Gaussian kernel with sigma `sig` (in same units as `x`),
    normalised so its sum = 1, then convolve with `y` using mode='same'
    (matches IDL convol(/center)).
    """
    n = len(x)
    if n < 2:
        return y.copy()
    nsigma2 = 2.0 * nsigma
    conv = (np.max(x) - np.min(x)) / (n - 1)   # units per pixel
    n_pts = int(np.ceil(nsigma2 * sig / conv))
    n_pts = max(2, min(n_pts, n - 2))
    if n_pts % 2 == 0:
        n_pts += 1
    xvar = (np.arange(n_pts, dtype=float) / (n_pts - 1) - 0.5) * n_pts
    gaus = np.exp(-0.5 * (xvar / (sig / conv)) ** 2)
    gaus /= gaus.sum()
    return np.convolve(y, gaus, mode='same')


def dopptom_chi2(
    dt_data,
    tc, period, e, omega, cosi, k, ar, lam,
    u1, u2, vsini, vline, errscale,
    c_kms=_C_KMS,
    return_model=False,
):
    """Compute χ² of a DT dataset given physical parameters.

    Parameters
    ----------
    dt_data : dict
        Output of :func:`allesfast.dt.io.read_dt_fits`.
    tc : float          time of primary transit (BJD)
    period : float      orbital period (days)
    e : float           eccentricity
    omega : float       argument of periastron of the star (radians)
    cosi : float        cos(inclination)
    k : float           Rp/Rs
    ar : float          a/Rs
    lam : float         spin-orbit angle (radians)
    u1, u2 : float      quadratic limb darkening coefficients (V band)
    vsini : float       km/s
    vline : float       intrinsic line width (km/s)
    errscale : float    DT-specific error scaling
    c_kms : float       speed of light in km/s
    return_model : bool, optional
        Return ``(chi2, model)`` if True.  Default False (just chi2).

    Returns
    -------
    chi2 : float
    model : ndarray (nvels, ntimes), optional
    """
    # ---- validity checks (match IDL early-returns) ----
    if vline <= 0 or vsini <= 0 or errscale <= 0:
        return (np.inf, None) if return_model else np.inf
    if not (np.isfinite(u1) and np.isfinite(u2)):
        return (np.inf, None) if return_model else np.inf

    ccf2d    = dt_data['ccf2d']      # (nvels, ntimes)
    bjd      = dt_data['bjd']
    vel      = dt_data['vel']
    stepsize = dt_data['stepsize']
    rms      = dt_data['rms']
    rspec    = dt_data['rspec']
    median_ccf = dt_data['median_ccf']

    inc = np.arccos(cosi)
    velsini = vel / vsini
    stepsize_sini = stepsize / vsini
    meanstepsize = float(np.mean(stepsize_sini))

    rvel = c_kms / rspec   # spectrograph resolution in velocity (km/s)

    # IndepVels: chi² normalisation for oversampling
    indep_vels = (rvel / _FWHM2SIGMA) / (meanstepsize * vsini)

    # Gaussian broadening σ (in km/s, then to vsini units)
    gauss_term = np.sqrt(vline ** 2 + rvel ** 2) / _FWHM2SIGMA
    gauss_rel  = gauss_term / vsini   # in vsini units

    # Relevant velocity window (IDL caps at |v/vsini| ≤ 2000)
    relevant = np.where(np.abs(velsini) <= 2000.0)[0]
    if relevant.size == 0:
        return (np.inf, None) if return_model else np.inf

    velsini_rel = velsini[relevant]
    nrelvel = velsini_rel.size

    # Planet shadow profile
    velwidth = k    # in vsini units (planet has half-width k about its centre)
    c1 = 2.0 / (np.pi * velwidth)

    # ---- transit geometry ----
    phase = primary_transit_phase(e, omega)
    tp = tc - period * phase   # time of periastron

    x, y, z, _ = sky_positions(bjd, inc, ar, tp, period, e, omega)
    up = x * np.cos(lam) - y * np.sin(lam)   # vsini units

    # ---- transit depth (Mandel-Agol) ----
    # Primary transit: in EXOFASTv2 sign convention z < 0 (planet on
    # observer side; r2 = -a*(1-e²)/(1+e cos ν) in exofast_getb2.pro).
    # b = sqrt(x²+y²) is the sky-plane impact parameter (R*).
    # F = M-A(b, k, u1, u2); β = 1 − F.
    b = np.sqrt(x ** 2 + y ** 2)
    primary_mask = z < 0
    beta = np.zeros_like(bjd)
    if np.any(primary_mask):
        # PyTransit's eval_quad_z_v expects u as 2D (npb, 2).
        # It returns a tuple (flux_2d, lambda_e, lambda_d, eta_d); only the
        # first element is the limb-darkened flux at each z.
        u = np.array([[u1, u2]], dtype=np.float64)
        result = eval_quad_z_v(b[primary_mask], float(k), u)
        F = np.asarray(result[0]).ravel()
        beta[primary_mask] = 1.0 - F

    # ---- build shadow model in CCF residual space ----
    nvels, ntimes = ccf2d.shape
    model = np.full((nvels, ntimes), median_ccf, dtype=np.float64)

    for i in range(ntimes):
        if beta[i] <= 0:
            continue
        c2 = ((velsini_rel - up[i]) / velwidth) ** 2
        valid = c2 < 1.0
        if not np.any(valid):
            continue
        rotprofile = np.zeros(nrelvel)
        rotprofile[valid] = c1 * np.sqrt(1.0 - c2[valid])
        # Gaussian convolution (in vsini units)
        unnormalized = _gaus_convol(velsini_rel, rotprofile, gauss_rel)
        # Normalisation: integrate unnormalised over stepsize
        norm_int = float(np.sum(unnormalized * stepsize_sini[relevant]))
        if norm_int == 0.0 or not np.isfinite(norm_int):
            continue
        model[relevant, i] += beta[i] * (1.0 / norm_int) * unnormalized

    # ---- χ² with IndepVels correction ----
    resid = ccf2d - model
    chi2 = float(np.sum((resid / (rms * errscale)) ** 2) / indep_vels)

    if not np.isfinite(chi2):
        return (np.inf, model) if return_model else np.inf

    return (chi2, model) if return_model else chi2
