!PATH = '+/Users/wangxianyu/Applications/NV5/idl90/lib/EXOFASTv2:' + !PATH
setenv, 'EXOFAST_PATH=/Users/wangxianyu/Applications/NV5/idl90/lib/EXOFASTv2'

filename = './n20160226.KELT-17b.TRES.44000.fits'
doptom = exofast_readdt(filename, 0)

tc       = 2457287.7456103642d0
period   = 10d0^0.4885753488d0
p        = 0.0926333230d0
cosi     = 0.0817318222d0
secosw   = 0.2433198217d0
sesinw   = -0.1518641162d0
e        = secosw^2 + sesinw^2
omega    = atan(sesinw, secosw)
lambda   = -2.0164720094d0
G_si     = 6.674d-11
Msun     = 1.989d30
Rsun     = 6.957d8
mstar    = 10d0^0.2604162879d0
rstar    = 1.6458230572d0
G_unit   = 2942.71377d0
ar       = (G_unit*mstar*period^2/(4d0*!dpi^2))^(1d0/3d0) / rstar
g_cgs    = G_si * mstar * Msun / (rstar*Rsun)^2 * 100d0
logg     = alog10(g_cgs)
teff     = 7454d0
feh      = 0d0
vsini    = 44.2d0
vline    = 5.49d0
errscale = 3.39d0

chi2 = dopptom_chi2(doptom, tc, period, e, omega, cosi, p, ar, lambda, $
                    logg, teff, feh, vsini, vline, errscale)
;; Now doptom.model is filled in — dump to a npy-compatible file
print, 'After call, model min/max:', min(doptom.model), max(doptom.model)
print, 'model median:', median(doptom.model)
print, 'chi2 (returned):', chi2

; Compute true chi^2 (sum((resid/sigma)^2) without exofast_like normalisation)
resid = doptom.ccf2d - doptom.model
true_chi2_raw = total((resid/(doptom.rms*errscale))^2)
print, 'true raw chi2 sum:', true_chi2_raw

; Compute IndepVels
fwhm2sigma = 2d0*sqrt(2d0*alog(2d0))
c_kms      = 299792.458d0
rvel       = c_kms/doptom.rspec
meanstep   = mean(doptom.stepsize/vsini)
indep_vels = (rvel/fwhm2sigma) / (meanstep*vsini)
print, 'IndepVels:', indep_vels
print, 'true chi2 / IndepVels:', true_chi2_raw / indep_vels

; Save model to a flat binary file for comparison with Python
sz = size(doptom.model)
openw, lun, './idl_dt_model.bin', /get_lun
writeu, lun, doptom.model
close, lun
free_lun, lun
print, 'Saved model shape:', sz[1], sz[2]

end
