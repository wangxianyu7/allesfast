#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:44:29 2018

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
import numpy as np
import matplotlib.pyplot as plt    
from matplotlib.ticker import FixedLocator
import os
from shutil import copyfile
import emcee
import arviz as az
import warnings

#::: allesfast modules
from .. import config
from .. import deriver
from ..computer import calculate_model, calculate_baseline, calculate_stellar_var
from ..general_output import afplot, afplot_per_transit, save_table, save_latex_table, logprint, get_params_from_samples, plot_ttv_results
from ..utils.latex_printer import round_tex
from ..plotting.plot_top_down_view import plot_top_down_view
from ..statistics import residual_stats
from ..star import make_sed_plot, make_mist_plot
from ..star import get_stellar_row, has_stellar_info, plot_params_star




###############################################################################
#::: draw samples from the MCMC save.5 (internally in the code)
###############################################################################
def draw_mcmc_posterior_samples(sampler, Nsamples=None, as_type='2d_array'):
    '''
    Default: return all possible sampels
    Set e.g. Nsamples=20 for plotting
    '''
#    global config.BASEMENT
    posterior_samples = sampler.get_chain(flat=True, discard=int(1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by']))
    if Nsamples:
        posterior_samples = posterior_samples[np.random.randint(len(posterior_samples), size=Nsamples)]

    if as_type=='2d_array':
        return posterior_samples
    
    elif as_type=='dic':
        posterior_samples_dic = {}
        for key in config.BASEMENT.fitkeys:
            ind = np.where(config.BASEMENT.fitkeys==key)[0]
            posterior_samples_dic[key] = posterior_samples[:,ind].flatten()
        return posterior_samples_dic



###############################################################################
#::: draw the maximum likelihood samples from the MCMC save.5 (internally in the code)
###############################################################################
def draw_mcmc_posterior_samples_at_maximum_likelihood(sampler, as_type='1d_array'):
    log_prob = sampler.get_log_prob(flat=True, discard=int(1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by']))
    posterior_samples = sampler.get_chain(flat=True, discard=int(1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by']))
    ind_max = np.argmax(log_prob)
    posterior_samples = posterior_samples[ind_max,:]
    
    if as_type=='1d_array':
        return posterior_samples

    elif as_type=='dic':
        posterior_samples_dic = {}
        for key in config.BASEMENT.fitkeys:
            ind = np.where(config.BASEMENT.fitkeys==key)[0]
            posterior_samples_dic[key] = posterior_samples[ind].flatten()
        return posterior_samples_dic



###############################################################################
#::: plot the MCMC chains
###############################################################################
def plot_MCMC_chains(sampler):

    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()

    #plot chains in 2 columns; emcee_3.0.0 format = (nsteps, nwalkers, nparameters)
    n_panels = config.BASEMENT.ndim + 1  # +1 for lnprob
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3*nrows))
    axf = axes.flatten()

    burn_idx = 1. * config.BASEMENT.settings['mcmc_burn_steps'] / config.BASEMENT.settings['mcmc_thin_by']

    #::: plot lnprob (first panel)
    axf[0].plot(log_prob, '-', rasterized=True)
    axf[0].axvline(burn_idx, color='k', linestyle='--')
    mini = np.min(log_prob[int(burn_idx):, :])
    maxi = np.max(log_prob[int(burn_idx):, :])
    axf[0].set(title='lnprob', xlabel='steps', rasterized=True, ylim=[mini, maxi])
    axf[0].xaxis.set_major_locator(FixedLocator(axf[0].get_xticks()))
    axf[0].set_xticklabels([int(label) for label in axf[0].get_xticks() * config.BASEMENT.settings['mcmc_thin_by']])

    #::: plot parameter chains
    for i in range(config.BASEMENT.ndim):
        ax = axf[i + 1]
        ax.set(title=config.BASEMENT.fitkeys[i], xlabel='steps')
        ax.plot(chain[:, :, i], '-', rasterized=True)
        ax.axvline(burn_idx, color='k', linestyle='--')
        ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))
        ax.set_xticklabels([int(label) for label in ax.get_xticks() * config.BASEMENT.settings['mcmc_thin_by']])

    #::: hide unused panels (when n_panels is odd)
    for j in range(n_panels, len(axf)):
        axf[j].set_visible(False)

    plt.tight_layout()
    return fig, axes


def plot_MCMC_posteriors(sampler):
    """Plot per-chain KDE posteriors with an overall average line.

    Each walker/chain is drawn as a thin coloured KDE curve so mixing
    quality is immediately visible.  A thick red line shows the combined
    posterior from all chains.
    """
    from scipy.stats import gaussian_kde

    discard = int(1. * config.BASEMENT.settings['mcmc_burn_steps']
                  / config.BASEMENT.settings['mcmc_thin_by'])
    # chain shape: (nsteps, nwalkers, ndim)
    chain = sampler.get_chain(discard=discard)
    nsteps, nwalkers, ndim = chain.shape

    n_panels = ndim
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axf = axes.flatten()

    # colour cycle for individual chains
    cmap = plt.cm.get_cmap('tab20', nwalkers)

    for i in range(ndim):
        ax = axf[i]
        all_vals = chain[:, :, i].flatten()
        xmin, xmax = np.percentile(all_vals, [0.5, 99.5])
        xgrid = np.linspace(xmin, xmax, 200)

        # per-chain KDE
        for w in range(nwalkers):
            vals_w = chain[:, w, i]
            if np.std(vals_w) == 0:
                continue
            try:
                kde_w = gaussian_kde(vals_w)
                ax.plot(xgrid, kde_w(xgrid), color=cmap(w), alpha=0.3, lw=0.5)
            except Exception:
                pass

        # combined KDE (thick red line)
        if np.std(all_vals) > 0:
            kde_all = gaussian_kde(all_vals)
            ax.plot(xgrid, kde_all(xgrid), color='red', lw=2.0)

        ax.set(title=config.BASEMENT.fitkeys[i], ylabel='Probability', yticks=[])

    for j in range(n_panels, len(axf)):
        axf[j].set_visible(False)

    plt.tight_layout()
    return fig, axes


# ##############################################################################
# ::: plot the MCMC corner plot (arviz-based, memory-efficient)
# ##############################################################################
def plot_MCMC_corner(sampler):
    samples = sampler.get_chain(flat=True, discard=int(1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by']))
    if samples.shape[0] > 10000:
        samples = samples[np.random.randint(samples.shape[0], size=10000), :]

    params_median, params_ll, params_ul = get_params_from_samples(samples)

    #::: build labels with units
    labels = []
    for i in range(len(config.BASEMENT.fitlabels)):
        lbl = str(config.BASEMENT.fitlabels[i])
        unit = str(config.BASEMENT.fitunits[i])
        if unit.strip():
            lbl = lbl + ' (' + unit + ')'
        labels.append(lbl)

    #::: shift epoch columns for readability
    ref_values = dict(zip(list(config.BASEMENT.fitkeys), list(config.BASEMENT.fittruths)))
    for companion in config.BASEMENT.settings['companions_all']:
        key = companion + '_epoch'
        if key in config.BASEMENT.fitkeys:
            ind = np.where(config.BASEMENT.fitkeys == key)[0][0]
            offset = int(params_median[key])
            samples[:, ind] -= offset
            labels[ind] += '-' + str(offset) + 'd'
            ref_values[key] -= offset

    #::: build arviz InferenceData from the flat samples
    #    Use fitkeys (unique) as dict keys; labels may have duplicates
    fitkeys_list = list(config.BASEMENT.fitkeys)
    var_dict = {}
    for i, key in enumerate(fitkeys_list):
        var_dict[key] = samples[np.newaxis, :, i]  # shape (1, n_samples) = 1 chain
    idata = az.from_dict(posterior=var_dict)

    #::: reference values for arviz
    ref_dict = {key: ref_values[key] for key in fitkeys_list}

    #::: build label mapping: fitkey -> display label
    labeller = az.labels.MapLabeller(var_name_map=dict(zip(fitkeys_list, labels)))

    #::: plot (hexbin is more robust than kde for narrow/bounded posteriors)
    ndim = len(fitkeys_list)
    az.rcParams['plot.max_subplots'] = max(ndim * ndim + 1, 40)
    axs = az.plot_pair(
        idata,
        kind='hexbin',
        marginals=True,
        labeller=labeller,
        reference_values=ref_dict,
        reference_values_kwargs={'color': 'red', 'linestyle': '--', 'lw': 1.5},
        gridsize=30,
        hexbin_kwargs={'cmap': 'Blues'},
        marginal_kwargs={'color': '#1f77b4'},
        point_estimate='median',
    )
    fig = axs.ravel()[0].get_figure() if hasattr(axs, 'ravel') else plt.gcf()

    #::: add titles on diagonal: label = median +/- errors
    fs_title = max(5, 10 - ndim // 8)
    for i, key in enumerate(fitkeys_list):
        med = np.median(samples[:, i])
        lo, hi = np.percentile(samples[:, i], [15.865, 84.135])
        value = round_tex(med, med - lo, hi - med)
        title = labels[i] + '\n' + r'$= ' + value + '$'
        axs[i, i].set_title(title, fontsize=fs_title, pad=3)

    return fig



###############################################################################
#::: print autocorr
###############################################################################
def print_autocorr(sampler):
    logprint('\nConvergence check')
    logprint('-------------------')
    
    logprint('{0: <20}'.format('Total steps:'),        '{0: <10}'.format(config.BASEMENT.settings['mcmc_total_steps']))
    logprint('{0: <20}'.format('Burn steps:'),         '{0: <10}'.format(config.BASEMENT.settings['mcmc_burn_steps']))
    logprint('{0: <20}'.format('Evaluation steps:'),   '{0: <20}'.format(config.BASEMENT.settings['mcmc_total_steps'] - config.BASEMENT.settings['mcmc_burn_steps']))
    
    N_evaluation_samples = int( 1. * config.BASEMENT.settings['mcmc_nwalkers'] * (config.BASEMENT.settings['mcmc_total_steps']-config.BASEMENT.settings['mcmc_burn_steps']) / config.BASEMENT.settings['mcmc_thin_by'] )
    logprint('{0: <20}'.format('Evaluation samples:'),   '{0: <20}'.format(N_evaluation_samples))
     
    # if N_evaluation_samples>200000:
    #     answer = input('It seems like you are asking for ' + str(N_evaluation_samples) + 'MCMC evaluation samples (calculated as mcmc_nwalkers * (mcmc_total_steps-mcmc_burn_steps) / mcmc_thin_by).'+\
    #                     'That is an aweful lot of samples.'+\
    #                     'What do you want to do?\n'+\
    #                     '1 : continue at any sacrifice\n'+\
    #                     '2 : abort and increase the mcmc_thin_by parameter in settings.csv (do not do this if you continued an old run!)\n')
    #     if answer==1: 
    #         pass
    #     else:
    #         raise ValueError('User aborted the run.')


    discard=int(1.*config.BASEMENT.settings['mcmc_burn_steps']/config.BASEMENT.settings['mcmc_thin_by'])
    tau = sampler.get_autocorr_time(discard=discard, c=5, tol=10, quiet=True)*config.BASEMENT.settings['mcmc_thin_by']
    logprint('Autocorrelation times:')
    logprint('\t', '{0: <30}'.format('parameter'), '{0: <20}'.format('tau (in steps)'), '{0: <20}'.format('Chain length (in multiples of tau)'))
    converged = True
    for i, key in enumerate(config.BASEMENT.fitkeys):
        chain_length = (1.*(config.BASEMENT.settings['mcmc_total_steps'] - config.BASEMENT.settings['mcmc_burn_steps']) / tau[i])
        logprint('\t', '{0: <30}'.format(key), '{0: <20}'.format(tau[i]), '{0: <20}'.format(chain_length))
        if (chain_length < 50) or np.isinf(chain_length) or np.isnan(chain_length):
            converged = False
            
    if converged:
        logprint('\nSuccesfully converged! All chains are at least 50x the autocorrelation length.\n')
    else:
        logprint('\nNot yet converged! Some chains are less than 50x the autocorrelation length. Please continue to run with longer chains, or start again with more walkers.\n')
        

###############################################################################
#::: analyse the output from save_mcmc.h5 file
###############################################################################
def mcmc_output(datadir, quiet=False):
    '''
    Inputs:
    -------
    datadir : str
        the working directory for allesfast
        must contain all the data files
        output directories and files will also be created inside datadir
            
    Outputs:
    --------
    This will output information into the console, and create a output files 
    into datadir/results/ (or datadir/QL/ if QL==True)    
    '''
    config.init(datadir, quiet=quiet)
    
    
    #::: security check
    if os.path.exists(os.path.join(config.BASEMENT.outdir,'mcmc_table.csv')):
        overwrite = '1'
    
    
    #::: load the mcmc_save.h5
    #::: copy over into tmp file (in case chain is still running and you want a quick look already)     
    copyfile(os.path.join(config.BASEMENT.outdir,'mcmc_save.h5'), os.path.join(config.BASEMENT.outdir,'mcmc_save_tmp.h5'))
    reader = emcee.backends.HDFBackend( os.path.join(config.BASEMENT.outdir,'mcmc_save_tmp.h5'), read_only=True )
    completed_steps = reader.get_chain().shape[0]*config.BASEMENT.settings['mcmc_thin_by']

    # Always recompute burn-in from the actual chain.
    # config.BASEMENT.settings['mcmc_burn_steps'] may be 0 (default) or stale
    # if mcmc_output() is called separately after the fit.
    from .mcmc import get_burnndx   # lazy import: avoids mcmc <-> mcmc_output circular dependency
    log_prob_raw = reader.get_log_prob()   # shape: (nsteps, nwalkers)
    burnndx = get_burnndx(log_prob_raw)
    config.BASEMENT.settings['mcmc_burn_steps'] = burnndx * config.BASEMENT.settings['mcmc_thin_by']
    logprint(f'\nAuto burn-in: stored step {burnndx} → '
             f'{config.BASEMENT.settings["mcmc_burn_steps"]} total steps '
             f'({100.*burnndx/log_prob_raw.shape[0]:.0f}% of chain)')

    if completed_steps < config.BASEMENT.settings['mcmc_total_steps']:
        #go into quick look mode
        #check how many total steps are actually done so far:
        config.BASEMENT.settings['mcmc_total_steps'] = config.BASEMENT.settings['mcmc_thin_by']*reader.get_chain().shape[0]
        #if this is at least twice the wished-for burn_steps, then let's keep those
        #otherwise, set burn_steps automatically to 75% of how many total steps are actually done so far
        if config.BASEMENT.settings['mcmc_total_steps'] > 2*config.BASEMENT.settings['mcmc_burn_steps']:
            pass
        else:
            config.BASEMENT.settings['mcmc_burn_steps'] = int(0.75*config.BASEMENT.settings['mcmc_total_steps'])
    
    
    #::: print autocorr
    if config.BASEMENT.settings['mcmc_sampler'] == 'emcee':
        print_autocorr(reader)


    #::: plot the fit
    posterior_samples = draw_mcmc_posterior_samples(reader, Nsamples=20) #only 20 samples for plotting
    
    for companion in config.BASEMENT.settings['companions_all']:
        fig, axes = afplot(posterior_samples, companion)
        if fig is not None:
            fig.savefig( os.path.join(config.BASEMENT.outdir,'mcmc_fit_'+companion+'.pdf'), bbox_inches='tight' )
            plt.close(fig)
    kwargs_dict = None
    if kwargs_dict is None:
        kwargs_dict = {}
    for companion in config.BASEMENT.settings['companions_phot']:
        for inst in config.BASEMENT.settings['inst_phot']:
            first_transit = 0
            while (first_transit >= 0):
                try:
                    kwargs_dict['first_transit'] = first_transit
                    fig, axes, last_transit, total_transits = afplot_per_transit(posterior_samples, inst, companion, kwargs_dict=kwargs_dict)
                    fig.savefig( os.path.join(config.BASEMENT.outdir,'mcmc_fit_per_transit_'+inst+'_'+companion+'_' + str(last_transit) + 'th.pdf'), bbox_inches='tight' )
                    plt.close(fig)
                    if total_transits > 0 and last_transit < total_transits - 1:
                        first_transit = last_transit
                    else:
                        first_transit = -1
                except Exception as e:
                    first_transit = -1
                    pass
    
    #::: plot the chains
    fig, axes = plot_MCMC_chains(reader)
    fig.savefig( os.path.join(config.BASEMENT.outdir,'mcmc_chains.pdf'), bbox_inches='tight' )
    plt.close(fig)

    #::: plot the 1-D posterior distributions
    fig, axes = plot_MCMC_posteriors(reader)
    fig.savefig( os.path.join(config.BASEMENT.outdir,'mcmc_posteriors.pdf'), bbox_inches='tight' )
    plt.close(fig)

    #::: plot the corner
    if config.BASEMENT.settings['cornerplot']:
        fig = plot_MCMC_corner(reader)
        fig.savefig( os.path.join(config.BASEMENT.outdir,'mcmc_corner.pdf'), bbox_inches='tight' )
        plt.close(fig)


    #::: save the tables
    posterior_samples = draw_mcmc_posterior_samples(reader) #all samples
    save_table(posterior_samples, 'mcmc')
    save_latex_table(posterior_samples, 'mcmc')

    #::: save mcmc_best.csv — best sample from chain, params.csv-compatible
    try:
        from .mcmc import _save_optimized_best
        _flat_lp    = reader.get_log_prob(flat=True, discard=burnndx)
        _flat_chain = reader.get_chain(flat=True, discard=burnndx)
        _best_theta = _flat_chain[int(np.argmax(_flat_lp))]
        _best_file  = os.path.join(config.BASEMENT.outdir, 'mcmc_best.csv')
        _save_optimized_best(dict(zip(config.BASEMENT.fitkeys, _best_theta)), _best_file)
        logprint(f'\nSaved {_best_file}')
    except Exception as e:
        logprint(f'\nWARNING: could not save mcmc_best.csv – {e}')
    
    
    #::: derive values (using stellar parameters from params.csv or params_star.csv)
    stellar_row = get_stellar_row(config.BASEMENT.datadir, config.BASEMENT.params)
    if has_stellar_info(stellar_row):
        deriver.derive(posterior_samples, 'mcmc')
    else:
        logprint('No complete stellar parameters found (R_star, M_star, Teff_star). Cannot derive final parameters.')
    
    
    #::: retrieve the median parameters and curves
    params_median, params_ll, params_ul = get_params_from_samples(posterior_samples)
    
    
    #::: check the residuals
    #for inst in config.BASEMENT.settings['inst_all']:
    #    if inst in config.BASEMENT.settings['inst_phot']: key='flux'
    #    elif inst in config.BASEMENT.settings['inst_rv']: key='rv'
    #    elif inst in config.BASEMENT.settings['inst_rv2']: key='rv2'
    #    model = calculate_model(params_median, inst, key)
    #    baseline = calculate_baseline(params_median, inst, key)
    #    stellar_var = calculate_stellar_var(params_median, inst, key)
    #    residuals = config.BASEMENT.data[inst][key] - model - baseline - stellar_var
    #    residual_stats(residuals)
    
    
    #::: make top-down orbit plot (using stellar parameters from params.csv or params_star.csv)
    try:
        params_star = plot_params_star(stellar_row)
        fig, ax = plot_top_down_view(params_median, params_star)
        fig.savefig( os.path.join(config.BASEMENT.outdir,'top_down_view.pdf'), bbox_inches='tight' )
        plt.close(fig)        
    except:
        logprint('\nOrbital plots could not be produced.')

    #::: stellar diagnostic plots (SED + MIST)
    try:
        _sed_file = config.BASEMENT.settings.get('sed_file', None)
        path = make_sed_plot(params_median, config.BASEMENT.datadir, config.BASEMENT.outdir,
                             outfile='mcmc_sed_fit.pdf', sed_file=_sed_file)
        if path is not None:
            logprint('\nSaved', path)
    except Exception:
        pass
    try:
        path = make_mist_plot(params_median, config.BASEMENT.outdir, outfile='mcmc_mist_track.pdf')
        if path is not None:
            logprint('\nSaved', path)
    except Exception:
        pass
        
        
    #::: plot TTV results (if wished for)
    if config.BASEMENT.settings['fit_ttvs'] == True:
        plot_ttv_results(params_median, params_ll, params_ul)
        
        
    #::: save params_best.csv — original params.csv with value column set to max-likelihood sample
    #    (copy to params.csv to restart MCMC from the best-fit point)
    try:
        _ml_dic = draw_mcmc_posterior_samples_at_maximum_likelihood(reader, as_type='dic')
        _params_ml = {k: float(v[0]) for k, v in _ml_dic.items()}
        _src = os.path.join(config.BASEMENT.datadir, 'params.csv')
        _dst = os.path.join(config.BASEMENT.outdir, 'params_best.csv')
        _out_lines = []
        with open(_src, 'r') as _f:
            for _line in _f:
                _stripped = _line.rstrip('\n')
                if _stripped.startswith('#') or _stripped.strip() == '':
                    _out_lines.append(_stripped)
                    continue
                _cols = _stripped.split(',')
                _name = _cols[0].strip()
                _fit_flag = _cols[2].strip() if len(_cols) > 2 else '0'
                if _fit_flag == '1' and _name in _params_ml:
                    _val = _params_ml[_name]
                    _cols[1] = '{:.10g}'.format(_val)
                    # for epoch params, tighten bounds to value ±0.1 so that
                    # load_params() passes the bounds check on next run
                    if _name.endswith('_epoch') and len(_cols) > 3:
                        _bnd = _cols[3].strip().split()
                        if _bnd[0] == 'uniform':
                            _cols[3] = 'uniform {:.10g} {:.10g}'.format(_val - 0.1, _val + 0.1)
                _out_lines.append(','.join(_cols))
        with open(_dst, 'w') as _f:
            _f.write('\n'.join(_out_lines) + '\n')
        logprint('\nSaved params_best.csv to', _dst)
    except Exception as _e:
        logprint('\nCould not save params_best.csv:', _e)


    #::: save model data files using posterior median
    try:
        from ..general_output import save_modelfiles
        save_modelfiles(posterior_samples, 'mcmc')
    except Exception as _e:
        logprint(f'\nWARNING: save_modelfiles failed – {_e}')

    #::: clean up and delete the tmp file
    os.remove(os.path.join(config.BASEMENT.outdir,'mcmc_save_tmp.h5'))

    logprint('\nDone. For all outputs, see', config.BASEMENT.outdir, '\n')
    
    
    
    

###############################################################################
#::: get MCMC samples (for top-level user)
###############################################################################
def get_mcmc_posterior_samples(datadir, Nsamples=None, as_type='dic'): #QL=False, 
    # config.init(datadir, QL=QL)
    config.init(datadir)
    reader = emcee.backends.HDFBackend( os.path.join(config.BASEMENT.outdir,'mcmc_save.h5'), read_only=True )
    return draw_mcmc_posterior_samples(reader, Nsamples=Nsamples, as_type=as_type) #only 20 samples for plotting
    
