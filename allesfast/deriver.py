#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 15:19:30 2018

@author:
Dr. Maximilian N. Günther
European Space Agency (ESA)
European Space Research and Technology Centre (ESTEC)
Keplerlaan 1, 2201 AZ Noordwijk, The Netherlands
Email: maximilian.guenther@esa.int
GitHub: mnguenther
Twitter: m_n_guenther
Web: www.mnguenther.com
"""

from __future__ import print_function, division, absolute_import

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: modules
import os
#import collections
import numpy as np
import matplotlib.pyplot as plt
import pickle
import arviz as az
from tqdm import tqdm 
from astropy.constants import M_earth, M_jup, M_sun, R_earth, R_jup, R_sun, au
import copy
from multiprocessing import Pool
from contextlib import closing

#::: allesfast modules
from . import config
from .utils.latex_printer import round_tex
from .general_output import logprint
from .limb_darkening import LDC3
from .computer import update_params, calculate_model, flux_fct, flux_subfct_ellc, flux_subfct_sinusoidal_phase_curves
from .exoworlds_rdx.lightcurves.index_transits import index_transits
from .lightcurves import get_epoch_occ
from .star import get_stellar_row, has_stellar_info, sample_stellar




###############################################################################
#::: constants (replaced with astropy.constants)
###############################################################################
#M_earth = 5.9742e+24 	#kg 	Earth mass
#M_jup   = 1.8987e+27 	#kg 	Jupiter mass
#M_sun   = 1.9891e+30 	#kg 	Solar mass
#R_earth = 6378136      #m 	Earth equatorial radius
#R_jup   = 71492000 	#m 	Jupiter equatorial radius
#R_sun   = 695508000 	#m 	Solar radius



###############################################################################
#::: globals
#::: sorry for that... it's multiprocessing, not me, I swear!
###############################################################################
# companion = None
# inst = None
# samples2 = None
# derived_samples = None


###############################################################################
#::: calculate values from model curves
###############################################################################
def calculate_values_from_model_curves(p, inst, companion):
    '''
    Parameters
    ----------
    p : dict
        parameters corresponding to one single sample
    inst : str
        instrument name
    companion : str
        companion name

    Returns
    -------
    list
        list containing the transit depth, occultation depth, and nightside flux
    '''
    
    #==========================================================================
    #::: init
    #==========================================================================
    depth_tr = np.nan
    depth_occ = np.nan
    nightside_flux = np.nan
    epoch_occ = get_epoch_occ(p[companion+'_epoch'], p[companion+'_period'], p[companion+'_f_s'], p[companion+'_f_c'])


    #==========================================================================
    #:: calculating
    #==========================================================================
    
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #::: without phase curve
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    if (config.BASEMENT.settings['phase_curve'] is False):
        #::: compute transit / primary eclipse depth
        depth_tr = 1e3 * (1. - flux_subfct_ellc(p, inst, companion, xx=[p[companion+'_epoch']])[0])
        
        #::: compute occultation / secondary eclipse depth (if wished)
        if (config.BASEMENT.settings['secondary_eclipse'] is True): 
            depth_occ = 1e3 * (1. - flux_subfct_ellc(p, inst, companion, xx=[epoch_occ])[0])


    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #::: with phase curve sine_series or sine_physical
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    elif (config.BASEMENT.settings['phase_curve'] is True) and (config.BASEMENT.settings['phase_curve_style'] in ['sine_series','sine_physical']):
        
        #::: 0: epoch, 1: epoch_occ
        xx0 = [p[companion+'_epoch'], epoch_occ]
        
        #::: for debugging
        # xx0 = np.linspace(p[companion+'_epoch']-0.25*p[companion+'_period'], p[companion+'_epoch']+0.75*p[companion+'_period'], 1001)
        
        #::: the full model flux with phase curve and dips
        phase_curve_dips = flux_fct(p, inst, companion, xx=xx0)

        #::: the phase curve without any dips
        ellc_flux, ellc_flux1, ellc_flux2 = flux_subfct_ellc(p, inst, companion, xx=xx0, return_fluxes=True)
        phase_curve_no_dips = flux_subfct_sinusoidal_phase_curves(p, inst, companion, np.ones_like(xx0), xx=xx0)
        
        #::: the phase curve with atmopsheric dips, but without nightside flux (sbratio=1e-12)
        p2 = copy.deepcopy(p)
        p2[companion+'_sbratio_'+inst] = 1e-12
        ellc_flux, ellc_flux1, ellc_flux2 = flux_subfct_ellc(p2, inst, companion, xx=xx0, return_fluxes=True)
        phase_curve_atmo_dips = flux_subfct_sinusoidal_phase_curves(p2, inst, companion, ellc_flux2, xx=xx0)

        #::: for debugging
        # fig = plt.figure()
        # plt.plot(xx0, phase_curve_dips, label='phase_curve_dips')
        # plt.plot(xx0, phase_curve_no_dips, label='phase_curve_no_dips')
        # plt.plot(xx0, phase_curve_atmo_dips, label='phase_curve_atmo_dips')
        # plt.legend()
        # plt.ylim([0.999,1.001])
        # plt.axhline(1,c='grey',ls='--')
        # fig.savefig(os.path.join(config.BASEMENT.outdir,'phase_curve_depths.pdf'), bbox_inches='tight')

        #::: compute transit / primary eclipse depth
        depth_tr = 1e3 * (phase_curve_no_dips[0] - phase_curve_dips[0]) #in ppt; 0: epoch
        
        #::: compute
        if (config.BASEMENT.settings['secondary_eclipse'] is True): 
            
            #::: compute occultation / secondary eclipse depth
            depth_occ = 1e3 * (phase_curve_no_dips[1] - phase_curve_dips[1]) #in ppt; 1: epoch_occ
                
            #::: compute nightside flux
            nightside_flux = 1e3 * (phase_curve_atmo_dips[1] - phase_curve_dips[1]) #in ppt; 1: epoch_occ
            

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #::: with phase curve ellc_physical
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    elif (config.BASEMENT.settings['phase_curve'] is True) and (config.BASEMENT.settings['phase_curve_style'] in ['ellc_physical']):
        pass #TODO: not yet implemented
        
    
    #==========================================================================
    #::: return
    #==========================================================================
    return [depth_tr, depth_occ, nightside_flux]




###############################################################################
#::: the main derive function
###############################################################################
def derive(samples, mode):
    '''
    Derives parameter of the system using Winn 2010
    
    Input:
    ------
    samples : array
        samples from the mcmc or nested sampling
    mode : str
        'mcmc' or 'ns'
        
    Returns:
    --------
    derived_samples : dict 
        with keys 'i', 'R1a', 'R2a', 'k', 'depth_undiluted', 'b_tra', 'b_occ', 'Ttot', 'Tfull'
        each key contains all the samples derived from the MCMC samples 
        (not mean values, but pure samples!)
        i = inclination 
        R1a = R1/a, radius companion over semiamplitude
        R2a = R2/a, radius star over semiamplitude
        Ttot = T_{1-4}, total transit width 
        Tfull = T_{2-3}, full-transit width
        
    Output:
    -------
    latex table of results
    corner plot of derived values posteriors
    '''
    
    #::: using a global keyword 
    #::: sorry for that... it's multiprocessing, not me, I swear!
    global companion
    global inst
    global samples2
    global derived_samples
    
    samples2 = samples #global variable
    N_samples = samples.shape[0]
    

    #==========================================================================
    #::: stellar 'posteriors'
    #==========================================================================
    row = get_stellar_row(config.BASEMENT.datadir, config.BASEMENT.params)
    if has_stellar_info(row):
        star = sample_stellar(row, size=N_samples)
    else:
        star = {'R_star':np.nan, 'M_star':np.nan, 'Teff_star':np.nan}
    
    
    #==========================================================================
    #::: derive all the params
    #==========================================================================
    companions = config.BASEMENT.settings['companions_all']
    
    def get_params(key):
        ind = np.where(config.BASEMENT.fitkeys==key)[0]
        if len(ind)==1: 
            return samples[:,ind].flatten() #if it was fitted for
        else: 
            try:
                if config.BASEMENT.params[key] is None:
                    return np.nan #if None, retun nan instead
                else:
                    return config.BASEMENT.params[key] #else take the input value
            except KeyError:
                return np.nan #if all fails, return nan
        
    def sin_d(alpha): return np.sin(np.deg2rad(alpha))
    def cos_d(alpha): return np.cos(np.deg2rad(alpha))
    def arcsin_d(x): return np.rad2deg(np.arcsin(x))
    def arccos_d(x): return np.rad2deg(np.arccos(x))

    #==========================================================================
    #::: derive rsuma from Kepler's 3rd law for each companion
    #::: (rsuma is never a free parameter; must be computed here for deriver)
    #==========================================================================
    _G_SI    = 6.674e-11
    _Msun_kg = 1.989e30
    _Rsun_m  = 6.957e8
    _day_s   = 86400.0
    derived_rsuma = {}
    for cc in companions:
        try:
            mstar_samples = get_params('A_mstar')
            if np.all(np.isnan(mstar_samples)):
                logmstar = get_params('A_logmstar')
                mstar_samples = 10.0 ** logmstar
            rstar_samples = get_params('A_rstar')
            period_samples = get_params(cc+'_period')
            rr_samples = get_params(cc+'_rr')
            M_kg = mstar_samples * _Msun_kg
            R_m  = rstar_samples * _Rsun_m
            P_s  = period_samples * _day_s
            a_m  = (_G_SI * M_kg * P_s**2 / (4. * np.pi**2))**(1./3.)
            derived_rsuma[cc] = R_m * (1. + rr_samples) / a_m
        except Exception:
            derived_rsuma[cc] = np.nan

    #==========================================================================
    #::: MIST-derived age posteriors
    #==========================================================================
    derived_age = {}
    if config.BASEMENT.settings.get('use_mist_prior', False):
        from .star.massradius_mist import get_mistage
        _vvcrit = config.BASEMENT.settings.get('mist_vvcrit', 0.0)
        _alpha  = config.BASEMENT.settings.get('mist_alpha', 0.0)
        for _ltr in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            _eep_key    = f'{_ltr}_eep'
            _age_key    = f'{_ltr}_age'
            _eep_samples = get_params(_eep_key)
            if np.isscalar(_eep_samples) and np.isnan(_eep_samples):
                continue
            # mstar: from logmstar (fitted) or mstar (fixed)
            _logm_key = f'{_ltr}_logmstar'
            _logm_samples = get_params(_logm_key)
            if np.isscalar(_logm_samples) and np.isnan(_logm_samples):
                _mstar_samples = get_params(f'{_ltr}_mstar')
                if np.isscalar(_mstar_samples) and np.isnan(_mstar_samples):
                    continue
            else:
                _mstar_samples = 10.0 ** np.atleast_1d(_logm_samples)
            # initfeh or feh
            _initfeh_samples = get_params(f'{_ltr}_initfeh')
            if np.isscalar(_initfeh_samples) and np.isnan(_initfeh_samples):
                _initfeh_samples = get_params(f'{_ltr}_feh')
                if np.isscalar(_initfeh_samples) and np.isnan(_initfeh_samples):
                    continue
            _eep_arr     = np.atleast_1d(_eep_samples)
            _mstar_arr   = np.atleast_1d(_mstar_samples)
            _initfeh_arr = np.atleast_1d(_initfeh_samples)
            _age_arr = np.array([
                get_mistage(float(_eep_arr[j % len(_eep_arr)]),
                            float(_mstar_arr[j % len(_mstar_arr)]),
                            float(_initfeh_arr[j % len(_initfeh_arr)]),
                            vvcrit=_vvcrit, alpha=_alpha)
                for j in range(N_samples)
            ])
            derived_age[_age_key] = _age_arr

    derived_samples = {}
    derived_samples.update(derived_age)
    for cc in companions:
        companion = cc

        #----------------------------------------------------------------------
        #::: radii
        #----------------------------------------------------------------------
        rsuma = derived_rsuma[companion]
        derived_samples[companion+'_R_star/a'] = rsuma / (1. + get_params(companion+'_rr'))
        derived_samples[companion+'_a/R_star'] = (1. + get_params(companion+'_rr')) / rsuma
        derived_samples[companion+'_R_companion/a'] = rsuma * get_params(companion+'_rr') / (1. + get_params(companion+'_rr'))
        derived_samples[companion+'_R_companion_(R_earth)'] = star['R_star'] * get_params(companion+'_rr') * R_sun.value / R_earth.value #in R_earth
        derived_samples[companion+'_R_companion_(R_jup)'] = star['R_star'] * get_params(companion+'_rr') * R_sun.value / R_jup.value #in R_jup

    
        #----------------------------------------------------------------------
        #::: orbit
        #----------------------------------------------------------------------
        derived_samples[companion+'_a_(R_sun)'] = star['R_star'] / derived_samples[companion+'_R_star/a']   
        derived_samples[companion+'_a_(AU)'] = derived_samples[companion+'_a_(R_sun)'] * R_sun.value/au.value
        derived_samples[companion+'_i'] = arccos_d(get_params(companion+'_cosi')) #in deg
        derived_samples[companion+'_e'] = get_params(companion+'_f_s')**2 + get_params(companion+'_f_c')**2
        derived_samples[companion+'_e_sinw'] = get_params(companion+'_f_s') * np.sqrt(derived_samples[companion+'_e'])
        derived_samples[companion+'_e_cosw'] = get_params(companion+'_f_c') * np.sqrt(derived_samples[companion+'_e'])
        derived_samples[companion+'_w'] = np.rad2deg(np.mod( np.arctan2(get_params(companion+'_f_s'), get_params(companion+'_f_c')), 2*np.pi) ) #in deg, from 0 to 360
        if np.isnan(derived_samples[companion+'_w']).all():
            derived_samples[companion+'_w'] = 0.
        
        
        #----------------------------------------------------------------------
        #::: masses
        #----------------------------------------------------------------------
        #::: for detached binaries, where K and q were fitted:
        if (companion+'_K' in config.BASEMENT.params) and len(config.BASEMENT.settings['inst_rv2'])>0:
            derived_samples[companion+'_M_companion_(M_earth)'] = get_params(companion+'_q') * star['M_star'] * M_sun.value / M_earth.value #in M_earth
            derived_samples[companion+'_M_companion_(M_jup)'] = get_params(companion+'_q') * star['M_star'] * M_sun.value / M_jup.value #in M_jup
            derived_samples[companion+'_M_companion_(M_sun)'] = get_params(companion+'_q') * star['M_star'] #in M_sun

        #::: for exoplanets or single-lined binaries, where only K was fitted, approximate/best-guess q form K:
        elif companion+'_K' in config.BASEMENT.params:
            a_1 = 0.019771142 * get_params(companion+'_K') * get_params(companion+'_period') * np.sqrt(1. - derived_samples[companion+'_e']**2)/sin_d(derived_samples[companion+'_i'])
    #        derived_samples[companion+'_a_rv'] = (1.+1./ellc_params[companion+'_q'])*a_1
            derived_samples[companion+'_q'] = 1./(( derived_samples[companion+'_a_(R_sun)'] / a_1 ) - 1.)
            derived_samples[companion+'_M_companion_(M_earth)'] = derived_samples[companion+'_q'] * star['M_star'] * M_sun.value / M_earth.value #in M_earth
            derived_samples[companion+'_M_companion_(M_jup)'] = derived_samples[companion+'_q'] * star['M_star'] * M_sun.value / M_jup.value #in M_jup
            derived_samples[companion+'_M_companion_(M_sun)'] = derived_samples[companion+'_q'] * star['M_star'] #in M_sun

            
        #----------------------------------------------------------------------
        #::: time of secondary eclipse   
        #---------------------------------------------------------------------- 
        if config.BASEMENT.settings['secondary_eclipse'] is True:
            derived_samples[companion+'_epoch_occ'] = get_params(companion+'_epoch') + get_params(companion+'_period')/2. * (1. + 4./np.pi * derived_samples[companion+'_e'] * cos_d(derived_samples[companion+'_w'])  ) #approximation from Winn2010
        
        
        #----------------------------------------------------------------------
        #::: impact params of primary eclipse with eccentricity corrections (from Winn 2010) 
        #----------------------------------------------------------------------
        eccentricity_correction_b_tra = ( (1. - derived_samples[companion+'_e']**2) / ( 1. + derived_samples[companion+'_e']*sin_d(derived_samples[companion+'_w']) ) )
        
        derived_samples[companion+'_b_tra'] = (1./derived_samples[companion+'_R_star/a']) * get_params(companion+'_cosi') * eccentricity_correction_b_tra
        
        
        #----------------------------------------------------------------------
        #::: impact params of secondary eclipse with eccentricity corrections (from Winn 2010) 
        #----------------------------------------------------------------------        
        eccentricity_correction_b_occ = ( (1. - derived_samples[companion+'_e']**2) / ( 1. - derived_samples[companion+'_e']*sin_d(derived_samples[companion+'_w']) ) )
        
        if config.BASEMENT.settings['secondary_eclipse'] is True:
            derived_samples[companion+'_b_occ'] = (1./derived_samples[companion+'_R_star/a']) * get_params(companion+'_cosi') * eccentricity_correction_b_occ
        
        
        #----------------------------------------------------------------------
        #::: transit duration (in hours) with eccentricity corrections (from Winn 2010) 
        #----------------------------------------------------------------------
        eccentricity_correction_T_tra = ( np.sqrt(1. - derived_samples[companion+'_e']**2) / ( 1. + derived_samples[companion+'_e']*sin_d(derived_samples[companion+'_w']) ) )
        
        derived_samples[companion+'_T_tra_tot'] = get_params(companion+'_period')/np.pi *24.  \
                                                  * np.arcsin( derived_samples[companion+'_R_star/a'] \
                                                               * np.sqrt( (1. + get_params(companion+'_rr'))**2 - derived_samples[companion+'_b_tra']**2 ) \
                                                               / sin_d(derived_samples[companion+'_i']) ) \
                                                  * eccentricity_correction_T_tra    #in h
        derived_samples[companion+'_T_tra_full'] = get_params(companion+'_period')/np.pi *24.  \
                                                   * np.arcsin( derived_samples[companion+'_R_star/a'] \
                                                                * np.sqrt( (1. - get_params(companion+'_rr'))**2 - derived_samples[companion+'_b_tra']**2  )\
                                                                / sin_d(derived_samples[companion+'_i']) ) \
                                                   * eccentricity_correction_T_tra    #in h
                                  
        
        #----------------------------------------------------------------------
        #::: primary and secondary eclipse depths (per inst) 
        #::: / transit and occultation depths (per inst)
        #----------------------------------------------------------------------
        for ii in config.BASEMENT.settings['inst_phot']:
            
            
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            #::: setup
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            inst = ii
            N_less_samples = 1000
            derived_samples[companion+'_depth_tr_dil_'+inst] = np.nan*np.empty(N_less_samples)
            derived_samples[companion+'_depth_occ_dil_'+inst] = np.nan*np.empty(N_less_samples)
            derived_samples[companion+'_nightside_flux_dil_'+inst] = np.nan*np.empty(N_less_samples)

            
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            #::: iterate through all samples, draw different models and measure the depths
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            print('Deriving eclipse depths (and more) from the model curves for companion', companion, 'and instrument', inst+'...')
            for i in range(N_less_samples):
                s = samples[ np.random.randint(low=0,high=samples2.shape[0]) , : ]
                p = update_params(s)
                r = calculate_values_from_model_curves(p, inst, companion)
                derived_samples[companion+'_depth_tr_dil_'+inst][i] = r[0]
                derived_samples[companion+'_depth_occ_dil_'+inst][i] = r[1]
                derived_samples[companion+'_nightside_flux_dil_'+inst][i] = r[2]
                
            
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            #resize the arrays to match the true N_samples (by redrawing the 1000 values)
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            derived_samples[companion+'_depth_tr_dil_'+inst] = np.resize(derived_samples[companion+'_depth_tr_dil_'+inst], N_samples)
            derived_samples[companion+'_depth_occ_dil_'+inst] = np.resize(derived_samples[companion+'_depth_occ_dil_'+inst], N_samples)
            derived_samples[companion+'_nightside_flux_dil_'+inst] = np.resize(derived_samples[companion+'_nightside_flux_dil_'+inst], N_samples)
        
    
        #----------------------------------------------------------------------
        #::: undiluted (per companion; per inst)
        #----------------------------------------------------------------------
        for inst in config.BASEMENT.settings['inst_phot']:
            dil = get_params('dil_'+inst)
            if all(np.atleast_1d(np.isnan(dil))): dil = 0
            derived_samples[companion+'_depth_tr_undil_'+inst] = derived_samples[companion+'_depth_tr_dil_'+inst] / (1. - dil) #in ppt
            derived_samples[companion+'_depth_occ_undil_'+inst] = derived_samples[companion+'_depth_occ_dil_'+inst] / (1. - dil) #in ppt
            derived_samples[companion+'_nightside_flux_undil_'+inst] = derived_samples[companion+'_nightside_flux_dil_'+inst] / (1. - dil) #in ppt
        
        
        #----------------------------------------------------------------------
        #::: equilibirum temperature
        #::: currently assumes Albedo of 0.3 and Emissivity of 1
        #----------------------------------------------------------------------
        albedo = 0.3
        emissivity = 1.
        derived_samples[companion+'_Teq'] = star['Teff_star']  * ( (1.-albedo)/emissivity )**0.25 * np.sqrt(derived_samples[companion+'_R_star/a'] / 2.)
        
        
        #----------------------------------------------------------------------
        #::: stellar density from orbit
        #----------------------------------------------------------------------
        if companion in config.BASEMENT.settings['companions_phot']:
            if all(np.atleast_1d(get_params(companion+'_rr'))<0.215443469): #see computer.py; get_params could return np.nan (float) or array; all(np.atleast_1d(...)) takes care of that
                derived_samples[companion+'_A_density'] = 3. * np.pi * (1./derived_samples[companion+'_R_star/a'])**3. / (get_params(companion+'_period')*86400.)**2 / 6.67408e-8 #in cgs
  
    
        #----------------------------------------------------------------------
        #::: companion densities
        #----------------------------------------------------------------------
        derived_samples[companion+'_density'] = ( (derived_samples[companion+'_M_companion_(M_earth)'] * M_earth) / (4./3. * np.pi * (derived_samples[companion+'_R_companion_(R_earth)'] * R_earth)**3 ) ).cgs.value #in cgs
        
        
        #----------------------------------------------------------------------
        #::: the companion's surface gravity (individual posterior distribution for each companion; via Southworth+ 2007)
        #----------------------------------------------------------------------
        try:
            derived_samples[companion+'_surface_gravity'] = 2. * np.pi / (get_params(companion+'_period')*86400.) * np.sqrt((1.-derived_samples[companion+'_e']**2)) * (get_params(companion+'_K')*1e5) / (derived_samples[companion+'_R_companion/a'])**2 / sin_d(derived_samples[companion+'_i'])
        except:
            pass
        
        
        #----------------------------------------------------------------------
        #::: period ratios (for ressonance studies)
        #----------------------------------------------------------------------
        if len(companions)>1:
            for other_companion in companions:
                if other_companion is not companion:
                    derived_samples[companion+'_period/'+other_companion+'_period'] = get_params(companion+'_period') / get_params(other_companion+'_period')
                        
                    
        #----------------------------------------------------------------------
        #::: limb darkening
        #----------------------------------------------------------------------
        for inst in config.BASEMENT.settings['inst_all']:
            
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            #::: host
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            if config.BASEMENT.settings['A_ld_law_'+inst] is None:
                pass
                
            elif config.BASEMENT.settings['A_ld_law_'+inst] == 'lin':
                derived_samples['A_ldc_u1_'+inst] = get_params('A_ldc_q1_'+inst)
                
            elif config.BASEMENT.settings['A_ld_law_'+inst] == 'quad':
                derived_samples['A_ldc_u1_'+inst] = 2 * np.sqrt(get_params('A_ldc_q1_'+inst)) * get_params('A_ldc_q2_'+inst)
                derived_samples['A_ldc_u2_'+inst] = np.sqrt(get_params('A_ldc_q1_'+inst)) * (1. - 2. * get_params('A_ldc_q2_'+inst))
                
            elif config.BASEMENT.settings['A_ld_law_'+inst] == 'sing':
                derived_samples['A_ldc_u1_'+inst] = np.nan*np.empty(N_samples)
                derived_samples['A_ldc_u2_'+inst] = np.nan*np.empty(N_samples)
                derived_samples['A_ldc_u3_'+inst] = np.nan*np.empty(N_samples)
                for i in range(N_samples):
                    u1, u2, u3 = LDC3.forward([get_params('A_ldc_q1_'+inst)[i], get_params('A_ldc_q2_'+inst)[i], get_params('A_ldc_q3_'+inst)[i]])
                    derived_samples['A_ldc_u1_'+inst][i] = u1
                    derived_samples['A_ldc_u2_'+inst][i] = u2
                    derived_samples['A_ldc_u3_'+inst][i] = u3
                
            else:
                raise ValueError("Currently only 'none', 'lin', 'quad' and 'sing' limb darkening are supported.")
            
        
    #==========================================================================
    #::: median stellar density
    #==========================================================================
    derived_samples['combined_A_density'] = []
    for companion in config.BASEMENT.settings['companions_phot']:
        try: derived_samples['combined_A_density'] = np.append(derived_samples['combined_A_density'], derived_samples[companion+'_A_density'])
        except: pass

    #==========================================================================
    #::: sv-parameterization: derive vsini and lambda per sample
    #::: Supports per-companion (e.g. b_svsinicoslambda) with fallback to global A_svsinicoslambda
    #==========================================================================
    # Global (legacy) sv-parameterization
    if ('A_svsinicoslambda' in config.BASEMENT.fitkeys) and ('A_svsinisinlambda' in config.BASEMENT.fitkeys):
        _sc = get_params('A_svsinicoslambda')
        _ss = get_params('A_svsinisinlambda')
        derived_samples['A_vsini']  = _sc**2 + _ss**2
        _lambda_raw = np.degrees(np.arctan2(_ss, _sc))
        # Unwrap lambda around circular median to avoid ±180° wrapping artifacts
        _lambda_rad = np.radians(_lambda_raw)
        _circ_median = np.degrees(np.arctan2(np.median(np.sin(_lambda_rad)),
                                              np.median(np.cos(_lambda_rad))))
        derived_samples['A_lambda'] = ((_lambda_raw - _circ_median + 180.) % 360. - 180.) + _circ_median
    # Per-companion sv-parameterization (e.g. b_svsinicoslambda → A_vsini + b_lambda)
    for companion in config.BASEMENT.settings['companions_all']:
        _comp_sc_key = companion + '_svsinicoslambda'
        _comp_ss_key = companion + '_svsinisinlambda'
        if (_comp_sc_key in config.BASEMENT.fitkeys) and (_comp_ss_key in config.BASEMENT.fitkeys):
            _sc = get_params(_comp_sc_key)
            _ss = get_params(_comp_ss_key)
            derived_samples['A_vsini']  = _sc**2 + _ss**2
            _lambda_raw = np.degrees(np.arctan2(_ss, _sc))
            _lambda_rad = np.radians(_lambda_raw)
            _circ_median = np.degrees(np.arctan2(np.median(np.sin(_lambda_rad)),
                                                  np.median(np.cos(_lambda_rad))))
            derived_samples[companion + '_lambda'] = ((_lambda_raw - _circ_median + 180.) % 360. - 180.) + _circ_median


    #==========================================================================
    #::: stellar derived parameters
    #==========================================================================
    # A_mstar from A_logmstar (EXOFASTv2 style)
    if 'A_logmstar' in config.BASEMENT.fitkeys:
        derived_samples['A_mstar'] = 10.0 ** get_params('A_logmstar')

    # A_logg from A_rstar + A_mstar (cgs: log10(G*M/R^2))
    _rstar_s = get_params('A_rstar')
    if 'A_mstar' in derived_samples:
        _mstar_s = derived_samples['A_mstar']
    elif 'A_mstar' in config.BASEMENT.fitkeys:
        _mstar_s = get_params('A_mstar')
    else:
        _mstar_s = None
    if _mstar_s is not None and not np.all(np.isnan(np.atleast_1d(_rstar_s))):
        _G_cgs    = 6.674e-8   # cm^3 g^-1 s^-2
        _Msun_cgs = 1.989e33   # g
        _Rsun_cgs = 6.957e10   # cm
        _g_cgs = _G_cgs * _mstar_s * _Msun_cgs / (_rstar_s * _Rsun_cgs)**2
        derived_samples['A_logg'] = np.log10(_g_cgs)

    # A_distance from A_parallax (mas → pc)
    if 'A_parallax' in config.BASEMENT.fitkeys:
        _par_s = get_params('A_parallax')
        derived_samples['A_distance'] = 1000.0 / _par_s


    ###############################################################################
    #::: write keys for output
    ###############################################################################
    names = []
    labels = []
    for companion in companions:
            
        names.append( companion+'_R_star/a' )
        labels.append( 'Host radius over semi-major axis '+companion+'; $R_\star/a_\mathrm{'+companion+'}$' )
        
        names.append( companion+'_a/R_star' )
        labels.append( 'Semi-major axis '+companion+' over host radius; $a_\mathrm{'+companion+'}/R_\star$' )
        
        names.append( companion+'_R_companion/a'  )
        labels.append( 'Companion radius '+companion+' over semi-major axis '+companion+'; $R_\mathrm{'+companion+'}/a_\mathrm{'+companion+'}$' )
        
        names.append( companion+'_R_companion_(R_earth)' )
        labels.append( 'Companion radius '+companion+'; $R_\mathrm{'+companion+'}$ ($\mathrm{R_{\oplus}}$)' )
        
        names.append( companion+'_R_companion_(R_jup)' )
        labels.append( 'Companion radius '+companion+'; $R_\mathrm{'+companion+'}$ ($\mathrm{R_{jup}}$)' )
        
        names.append( companion+'_a_(R_sun)' )
        labels.append( 'Semi-major axis '+companion+'; $a_\mathrm{'+companion+'}$ ($\mathrm{R_{\odot}}$)' )
        
        names.append( companion+'_a_(AU)' )
        labels.append( 'Semi-major axis '+companion+'; $a_\mathrm{'+companion+'}$ (AU)' )
        
        names.append( companion+'_i' )
        labels.append( 'Inclination '+companion+'; $i_\mathrm{'+companion+'}$ (deg)' )
        
        names.append( companion+'_e' )
        labels.append( 'Eccentricity '+companion+'; $e_\mathrm{'+companion+'}$' )
        
        names.append( companion+'_w' )
        labels.append( 'Argument of periastron '+companion+'; $w_\mathrm{'+companion+'}$ (deg)' )
        
        names.append( companion+'_q' )
        labels.append( 'Mass ratio '+companion+'; $q_\mathrm{'+companion+'}$' )
        
        names.append( companion+'_M_companion_(M_earth)' )
        labels.append( 'Companion mass '+companion+'; $M_\mathrm{'+companion+'}$ ($\mathrm{M_{\oplus}}$)' )
        
        names.append( companion+'_M_companion_(M_jup)' )
        labels.append( 'Companion mass '+companion+'; $M_\mathrm{'+companion+'}$ ($\mathrm{M_{jup}}$)' )
        
        names.append( companion+'_M_companion_(M_sun)' )
        labels.append( 'Companion mass '+companion+'; $M_\mathrm{'+companion+'}$ ($\mathrm{M_{\odot}}$)' )
        
        names.append( companion+'_b_tra' )
        labels.append( 'Impact parameter '+companion+'; $b_\mathrm{tra;'+companion+'}$' )
        
        names.append( companion+'_T_tra_tot'  )
        labels.append( 'Total transit duration '+companion+'; $T_\mathrm{tot;'+companion+'}$ (h)' )
        
        names.append( companion+'_T_tra_full' )
        labels.append( 'Full-transit duration '+companion+'; $T_\mathrm{full;'+companion+'}$ (h)' )
        
        names.append( companion+'_epoch_occ'  )
        labels.append( 'Epoch occultation '+companion+'; $T_\mathrm{0;occ;'+companion+'}$' )
        
        names.append( companion+'_b_occ'  )
        labels.append( 'Impact parameter occultation '+companion+'; $b_\mathrm{occ;'+companion+'}$' )
        
        names.append( companion+'_A_density' )
        labels.append( 'Host density from orbit '+companion+'; $\\rho_\mathrm{\star;'+companion+'}$ (cgs)' )
    
        names.append( companion+'_density' )
        labels.append( 'Companion density '+companion+'; $\\rho_\mathrm{'+companion+'}$ (cgs)' )
        
        names.append( companion+'_surface_gravity')
        labels.append( 'Companion surface gravity '+companion+'; $g_\mathrm{'+companion+'}$ (cgs)' )
        
        names.append( companion+'_Teq' )
        labels.append( 'Equilibrium temperature '+companion+'; $T_\mathrm{eq;'+companion+'}$ (K)' )
        
        for inst in config.BASEMENT.settings['inst_phot']:
            
            names.append( companion+'_depth_tr_undil_'+inst )
            labels.append( 'Transit depth (undil.) '+companion+'; $\delta_\mathrm{tr; undil; '+companion+'; '+inst+'}$ (ppt)' )
            
            names.append( companion+'_depth_tr_dil_'+inst )
            labels.append( 'Transit depth (dil.) '+companion+'; $\delta_\mathrm{tr; dil; '+companion+'; '+inst+'}$ (ppt)' )
        
            names.append( companion+'_depth_occ_undil_'+inst )
            labels.append( 'Occultation depth (undil.) '+companion+'; $\delta_\mathrm{occ; undil; '+companion+'; '+inst+'}$ (ppt)' )
            
            names.append( companion+'_depth_occ_dil_'+inst )
            labels.append( 'Occultation depth (dil.) '+companion+'; $\delta_\mathrm{occ; dil; '+companion+'; '+inst+'}$ (ppt)' )
            
            names.append( companion+'_nightside_flux_undil_'+inst )
            labels.append( 'Nightside flux (undil.)'+companion+'; $F_\mathrm{nightside; undil; '+companion+'; '+inst+'}$ (ppt)' )
            
            names.append( companion+'_nightside_flux_dil_'+inst )
            labels.append( 'Nightside flux (dil.)'+companion+'; $F_\mathrm{nightside; dil; '+companion+'; '+inst+'}$ (ppt)' )
            
            
            
        #::: period ratios (for ressonance studies)
        if len(companions)>1:
            for other_companion in companions:
                if other_companion is not companion:
                    names.append( companion+'_period/'+other_companion+'_period' )
                    labels.append( 'Period ratio; $P_\mathrm{'+companion+'} / P_\mathrm{'+other_companion+'}$' )
           
            
    #::: host
    for inst in config.BASEMENT.settings['inst_all']:    
        if config.BASEMENT.settings['A_ld_law_'+inst] is None:
            pass
            
        elif config.BASEMENT.settings['A_ld_law_'+inst] == 'lin':
            names.append( 'A_ldc_u1_'+inst )
            labels.append( 'Limb darkening; $u_\mathrm{1; '+inst+'}$' )
            
        elif config.BASEMENT.settings['A_ld_law_'+inst] == 'quad':
            names.append( 'A_ldc_u1_'+inst )
            labels.append( 'Limb darkening; $u_\mathrm{1; '+inst+'}$' )
            names.append( 'A_ldc_u2_'+inst )
            labels.append( 'Limb darkening; $u_\mathrm{2; '+inst+'}$' )
            
        elif config.BASEMENT.settings['A_ld_law_'+inst] == 'sing':
            names.append( 'A_ldc_u1_'+inst )
            labels.append( 'Limb darkening; $u_\mathrm{1; '+inst+'}$' )
            names.append( 'A_ldc_u2_'+inst )
            labels.append( 'Limb darkening; $u_\mathrm{2; '+inst+'}$' )
            names.append( 'A_ldc_u3_'+inst )
            labels.append( 'Limb darkening; $u_\mathrm{3; '+inst+'}$' )
            
        else:
            raise ValueError("Currently only 'none', 'lin', 'quad' and 'sing' limb darkening are supported.")
                
        
    names.append( 'combined_A_density' )
    labels.append( 'Combined host density from all orbits; $rho_\mathrm{\star; combined}$ (cgs)' )

    if 'A_vsini' in derived_samples:
        names.append( 'A_vsini' )
        labels.append( r'Projected stellar rotation; $v \sin i_\star$ (km/s)' )

    if 'A_lambda' in derived_samples:
        names.append( 'A_lambda' )
        labels.append( r'Spin-orbit angle; $\lambda$ (deg)' )
    for companion in config.BASEMENT.settings['companions_all']:
        _comp_lam_key = companion + '_lambda'
        if _comp_lam_key in derived_samples:
            names.append( _comp_lam_key )
            labels.append( r'Spin-orbit angle ' + companion + r'; $\lambda_\mathrm{' + companion + r'}$ (deg)' )

    if 'A_mstar' in derived_samples:
        names.append( 'A_mstar' )
        labels.append( r'Stellar mass; $M_\star$ ($M_\odot$)' )

    if 'A_logg' in derived_samples:
        names.append( 'A_logg' )
        labels.append( r'Stellar surface gravity; $\log g_\star$ (cgs)' )

    if 'A_distance' in derived_samples:
        names.append( 'A_distance' )
        labels.append( r'Distance; $d$ (pc)' )

    #::: MIST-derived age (one entry per star letter that has an age posterior)
    for _ltr in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        _age_key = f'{_ltr}_age'
        if _age_key in derived_samples:
            names.append(_age_key)
            labels.append(r'Stellar age; $\tau_{' + _ltr + r'}$ (Gyr)')


    ###############################################################################
    #::: delete pointless values
    ###############################################################################
    ind_good = []
    for i,name in enumerate(names):
        if (name in derived_samples) and isinstance(derived_samples[name], np.ndarray) and not all(np.isnan(derived_samples[name])) and not all(np.array(derived_samples[name])==0):
            ind_good.append(i)
            
    names = [ names[i] for i in ind_good ]
    labels = [ labels[i] for i in ind_good ]
    
    
    ###############################################################################
    #::: if any meaningful values are left, go output them
    ###############################################################################
    if len(names)>0:
            
        #=====================================================================
        #::: save all in pickle
        #=====================================================================
        pickle.dump(derived_samples, open(os.path.join(config.BASEMENT.outdir,mode+'_derived_samples.pickle'),'wb'))
        
        
        #=====================================================================
        #::: save txt & latex table & latex commands
        #=====================================================================
        with open(os.path.join(config.BASEMENT.outdir,mode+'_derived_table.csv'),'w') as outfile,\
             open(os.path.join(config.BASEMENT.outdir,mode+'_derived_latex_table.txt'),'w') as f,\
             open(os.path.join(config.BASEMENT.outdir,mode+'_derived_latex_cmd.txt'),'w') as f_cmd:
                 
            outfile.write('#property,value,lower_error,upper_error,source\n')
            
            f.write('Parameter & Value & Source \\\\ \n')
            f.write('\\hline \n')
            f.write('\\multicolumn{3}{c}{\\textit{Derived parameters}} \\\\ \n')
            f.write('\\hline \n')
            
            for name,label in zip(names, labels):
                ll, median, ul = np.nanpercentile(derived_samples[name], [15.865, 50., 84.135])
                outfile.write( str(label)+','+str(median)+','+str(median-ll)+','+str(ul-median)+',derived\n' )
                
                value = round_tex(median, median-ll, ul-median)
                f.write( label + ' & $' + value + '$ & derived \\\\ \n' )
                
                simplename = name.replace("_", "").replace("/", "over").replace("(", "").replace(")", "").replace("1", "one").replace("2", "two")
                f_cmd.write('\\newcommand{\\'+simplename+'}{$'+value+'$} %'+label+' = $'+value+'$\n')
                
        logprint('\nSaved '+mode+'_derived_results.csv, '+mode+'_derived_latex_table.txt, and '+mode+'_derived_latex_cmd.txt')
        
            
        #=====================================================================
        #::: plot corner
        #=====================================================================
        if 'combined_A_density' in names: names.remove('combined_A_density') #has (N_companions x N_dims) dimensions, thus does not match the rest
        
        #::: clean up any isolated NaN's before plotting
        for name in names:
            median = np.nanmedian(derived_samples[name])
            ind = np.where(np.isnan(derived_samples[name]))
            derived_samples[name][ind] = median

        #::: subsample for speed
        n_samples = len(derived_samples[names[0]])
        if n_samples > 10000:
            idx = np.random.choice(n_samples, 10000, replace=False)
        else:
            idx = np.arange(n_samples)

        #::: build arviz InferenceData
        var_dict = {name: derived_samples[name][idx][np.newaxis, :] for name in names}
        idata = az.from_dict(posterior=var_dict)

        #::: plot with arviz (hexbin is more robust than kde for narrow posteriors)
        az.rcParams['plot.max_subplots'] = max(len(names) ** 2 + 1, 40)
        axs = az.plot_pair(
            idata,
            kind='hexbin',
            marginals=True,
            gridsize=30,
            hexbin_kwargs={'cmap': 'Blues'},
            marginal_kwargs={'color': '#1f77b4'},
            point_estimate='median',
        )
        fig = axs.ravel()[0].get_figure() if hasattr(axs, 'ravel') else plt.gcf()

        #::: add titles on diagonal: name = median +/- errors
        n_derived = len(names)
        fs_title = max(5, 10 - n_derived // 8)
        for i, name in enumerate(names):
            lo, med, hi = np.nanpercentile(derived_samples[name], [15.865, 50., 84.135])
            value = round_tex(med, med - lo, hi - med)
            title = name + '\n' + r'$= ' + value + '$'
            axs[i, i].set_title(title, fontsize=fs_title, pad=3)

        corner_path = os.path.join(config.BASEMENT.outdir, mode+'_derived_corner.pdf')
        fig.savefig(corner_path, bbox_inches='tight')
        plt.close(fig)


        #=====================================================================
        #::: plot 1-D derived posteriors
        #=====================================================================
        from scipy.stats import gaussian_kde

        n_panels = len(names)
        ncols = 2
        nrows = int(np.ceil(n_panels / ncols))
        fig_post, axes_post = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
        axf = axes_post.flatten()

        for i, name in enumerate(names):
            ax = axf[i]
            vals = derived_samples[name][idx]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 10 or np.std(vals) == 0:
                ax.set(title=name, ylabel='Probability', yticks=[])
                continue
            xmin, xmax = np.percentile(vals, [0.5, 99.5])
            xgrid = np.linspace(xmin, xmax, 200)
            try:
                kde = gaussian_kde(vals)
                ax.fill_between(xgrid, kde(xgrid), alpha=0.3, color='#1f77b4')
                ax.plot(xgrid, kde(xgrid), color='#1f77b4', lw=2.0)
            except Exception:
                ax.hist(vals, bins=50, density=True, color='#1f77b4', alpha=0.5)
            lo, med, hi = np.nanpercentile(vals, [15.865, 50., 84.135])
            ax.axvline(med, color='red', ls='-', lw=1.5)
            ax.axvline(lo, color='red', ls='--', lw=1.0)
            ax.axvline(hi, color='red', ls='--', lw=1.0)
            value = round_tex(med, med - lo, hi - med)
            ax.set(title=name + r'  $= ' + value + '$', ylabel='Probability', yticks=[])

        for j in range(n_panels, len(axf)):
            axf[j].set_visible(False)

        plt.tight_layout()
        post_path = os.path.join(config.BASEMENT.outdir, mode+'_derived_posteriors.pdf')
        fig_post.savefig(post_path, bbox_inches='tight')
        plt.close(fig_post)


        #=====================================================================
        #::: finish
        #=====================================================================
        logprint('\nSaved '+corner_path)
        logprint('Saved '+post_path)
        
        
    else:
        logprint('\nNo values available to be derived.')
        
        
