;; IDL companion to bench_dt.py
;; Reads bench_grid_input.csv (scan,value), computes EXOFASTv2 chi^2
;; at each grid point, writes bench_grid_idl_chi2.csv with one column.

!PATH = '+/Users/wangxianyu/Applications/NV5/idl90/lib/EXOFASTv2:' + !PATH
setenv, 'EXOFAST_PATH=/Users/wangxianyu/Applications/NV5/idl90/lib/EXOFASTv2'

here = file_dirname(routine_filepath('bench_dt_idl'))
filename = here + '/../../examples/KELT-17_DT/n20160226.KELT-17b.TRES.44000.fits'
doptom = exofast_readdt(filename, 0)

;; Anchor parameters (from EXOFASTv2 kelt17.priors2)
tc_base       = 2457287.7456103642d0
period_base   = 10d0^0.4885753488d0
k_base        = 0.0926333230d0
cosi_base     = 0.0817318222d0
secosw        = 0.2433198217d0
sesinw        = -0.1518641162d0
e_base        = secosw^2 + sesinw^2
omega_base    = atan(sesinw, secosw)
lam_base      = -2.0164720094d0
G_unit        = 2942.71377d0
mstar         = 10d0^0.2604162879d0
rstar         = 1.6458230572d0
ar_base       = (G_unit*mstar*period_base^2/(4d0*!dpi^2))^(1d0/3d0) / rstar
logg_base     = 4.265827d0
teff_base     = 7454d0
feh_base      = 0d0
vsini_base    = 44.2d0
vline_base    = 5.49d0
errscale_base = 3.39d0

;; Read grid from Python
openr, lun, here + '/bench_grid_input.csv', /get_lun
header = ''
readf, lun, header
scans = list()
values = list()
while ~eof(lun) do begin
  line = ''
  readf, lun, line
  parts = strsplit(line, ',', /extract)
  scans.add, parts[0]
  values.add, double(parts[1])
endwhile
free_lun, lun

;; Compute chi2 at each grid point
ngrid = scans.count()
chi2s = dblarr(ngrid)
print, 'Computing IDL chi2 at ', ngrid, ' grid points...'
for i = 0, ngrid - 1 do begin
  lam   = lam_base
  vsini = vsini_base
  vline = vline_base
  cosi  = cosi_base
  case scans[i] of
    'lambda': lam   = values[i]
    'vsini':  vsini = values[i]
    'vline':  vline = values[i]
    'cosi':   cosi  = values[i]
    else:
  endcase
  chi2s[i] = dopptom_chi2(doptom, tc_base, period_base, e_base, omega_base, $
                          cosi, k_base, ar_base, lam, $
                          logg_base, teff_base, feh_base, vsini, vline, $
                          errscale_base)
  if (i mod 10) eq 0 then print, '  i=', i, '  chi2=', chi2s[i]
endfor

;; chi^2 above is exofast_like(/chi2), which equals -2*loglike.
;; That includes a constant log-normalization term and is divided by IndepVels.
;; For comparison with Python's pure "raw chi^2 / IndepVels", subtract the
;; constant term.
fwhm2sigma = 2d0*sqrt(2d0*alog(2d0))
c_kms      = 299792.458d0
rvel       = c_kms / doptom.rspec
meanstep   = mean(doptom.stepsize/vsini_base)
indep_vels = (rvel/fwhm2sigma) / (meanstep*vsini_base)
sigma_base = doptom.rms * errscale_base
nelm       = double(n_elements(doptom.ccf2d))
const_term = nelm * alog(2d0 * !dpi * sigma_base^2)
print, 'IndepVels =', indep_vels
print, 'const_term (= sum log(2 pi sigma^2)) =', const_term
print, '  (this is added inside exofast_like(/chi2) and we subtract it)'

chi2s_raw = (chi2s * indep_vels - const_term) / indep_vels

;; Save
openw, lun, here + '/bench_grid_idl_chi2.csv', /get_lun
printf, lun, 'chi2_idl'
for i = 0, ngrid - 1 do printf, lun, chi2s_raw[i], format='(F0.10)'
free_lun, lun
print, 'Wrote bench_grid_idl_chi2.csv'

end
