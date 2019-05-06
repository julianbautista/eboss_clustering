from __future__ import print_function

import os, sys
import numpy as np
import pylab as plt
import healpy as hp
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import minimize
import argparse
import fitsio
import systematic_fitter as sf

plt.ion()

indir = os.environ['LRG_DIR']+'/cats'
cap = 'SGC'
target = 'LRG'
dat_file = indir+f'/eBOSS_{target}_full_{cap}_vtest.dat.fits'
ran_file = indir+f'/eBOSS_{target}_full_{cap}_vtest.ran.fits'
out_file = indir+f'/eBOSS_{target}_full_{cap}_vtest'
sample_name = f'eBOSS {target} {cap} test DR16'
sample_root = f'{target}_{cap}_vtest'

#-- Read data and randoms
print('Reading galaxies from ', dat_file)
dat = Table.read(dat_file)
print('Reading randoms  from ', ran_file)
ran = Table.read(ran_file)

#-- Cut for LRGs
zmin = 0.6
zmax = 1.0
random_fraction = 1.0
seed = 123
np.random.seed(seed)

#-- Cut the sample 
print('Cutting galaxies and randoms between zmin=%.3f and zmax=%.3f'%\
     (zmin, zmax))

#-- Select spectroscopic sample
wd = ((dat['IMATCH']==1)|(dat['IMATCH']==2))&\
     (dat['Z']>=zmin)&\
     (dat['Z']<=zmax)&\
     (dat['COMP_BOSS']>0.5)&\
     (dat['sector_SSR']>0.5) 
wr = (ran['Z']>=zmin)&\
     (ran['Z']<=zmax)&\
     (ran['COMP_BOSS']>0.5)&\
     (ran['sector_SSR']>0.5)&\
     (np.random.rand(len(ran))<random_fraction)


#-- Defining RA, DEC and weights
data_ra, data_dec = dat['RA'][wd], dat['DEC'][wd]
rand_ra, rand_dec = ran['RA'][wr], ran['DEC'][wr]
data_we = (dat['WEIGHT_CP']*dat['WEIGHT_FKP']/dat['sector_SSR'])[wd]
rand_we = (ran['COMP_BOSS'])[wr]*(ran['WEIGHT_FKP'])[wr]


#-- Read systematic values for data and randoms
data_syst, rand_syst = sf.read_systematic_maps(data_ra, data_dec, rand_ra, rand_dec)


#-- Create fitter object
s = sf.Syst(data_we, rand_we)

if target == 'LRG':
    use_maps = ['STAR_DENSITY', 'EBV', 'PSF_I', 'DEPTH_I_MINUS_EBV', 'AIRMASS']
    fit_maps = ['STAR_DENSITY', 'EBV']
if target == 'QSO':
    use_maps = ['STAR_DENTISY', 'EBV', 'PSF_G', 'SKY_G', 'DEPTH_G_MINUS_EBV', 'AIRMASS']
    fit_maps = ['STAR_DENSITY', 'DEPTH_G_MINUS_EBV']

#-- Add the systematic maps we want 
for syst_name in use_maps:
    s.add_syst(syst_name, data_syst[syst_name], rand_syst[syst_name])
s.cut_outliers(p=0.5, verbose=1)

#-- Perform global fit
nbins=20
s.prepare(nbins=nbins)
s.fit_minuit(fit_maps=fit_maps)
s.plot_overdensity(pars=[None, s.best_pars], ylim=[0.5, 1.5], 
    title=f'{sample_name}: global fit')

#-- Get weights for global fit
data_weightsys_global = 1/s.get_model(s.best_pars, data_syst)

#-- Export global weights
export_global=1
if export_global:
    dat_weight_systot = np.zeros(len(dat))
    ran_weight_systot = np.zeros(len(ran))
    dat_weight_systot[wd] = data_weightsys_global 
    ran_weight_systot[wr] = np.random.choice(data_weightsys_global, size=np.sum(wr), replace=True)
    dat['WEIGHT_SYSTOT'] = dat_weight_systot 
    ran['WEIGHT_SYSTOT'] = ran_weight_systot
    print('Exporting', out_file+'_syst_global')
    dat.write(out_file+'_syst_global.dat.fits', overwrite=True)
    ran.write(out_file+'_syst_global.ran.fits', overwrite=True)



#-- Divide the sample into subsamples based of the "x" value
#-- which can be the redshift Z or the FIBER2MAG
sub_samples=0
if sub_samples:

    #x_name = 'Redshift'
    #x_data = dat['Z'][wd]
    x_name = r'$i_{\rm fib2}$'
    x_data = sf.flux_to_mag(dat['FIBER2FLUX'][:, 3].data[wd], 3).data

    x_nbins=10
    x_pcut = 0.5 #-- exclude this fraction of extreme values of x
    chi2_thres=16.
    save_plots=0
   
    #-- Fits slopes for each bin in x, and returns a list of fitter objects slist 
    x_bins, chi2_list, s_list = \
        sf.fit_slopes_per_xbin(s, x_name, x_data, 
            x_nbins=x_nbins, p=p_zcut, fit_maps=fit_maps, 
            plot_delta=True, sample_name=sample_name, sample_root=sample_root)

    #-- Fit polynomials to dependency of slopes versus dependent variable 
    coeffs = sf.fit_smooth_slopes_vs_x(x_bins, s_list, chi2_thres = chi2_thres)

    #-- Plot it
    sf.plot_slopes_vs_x(x_bins, s_list, x_name=x_name, title=sample_name, 
                 global_pars=s.best_pars, global_errors=s.errors) 
    if save_plots:
        plt.savefig(f'plots/syst_{sample_root}_slopesperbin.pdf')


    #-- Get parameters for the smooth fit per bin
    pars_xbin_smooth = sf.get_pars_from_coeffs(coeffs, x_data[s.w_data])

    #-- Compare chi2 over the full sample
    print('\nFinal chi2 values over full sample')
    chi2_before = s.get_chi2()
    chi2_global = s.get_chi2(s.best_pars)
    chi2_smooth = s.get_chi2(pars_xbin_smooth)
    ndata = s.ndata
    print(f' - chi2 (no fit):          {chi2_before:.1f}/{ndata} = {chi2_before/ndata:.2f}')
    print(f' - chi2 (global fit):      {chi2_global:.1f}/{ndata} = {chi2_global/ndata:.2f}')
    print(f' - chi2 (smooth evol fit): {chi2_smooth:.1f}/{ndata} = {chi2_smooth/ndata:.2f}')

    #-- Make overdensity plots for full sample with new smooth per bin parameters
    s.plot_overdensity(pars=[None, s.best_pars, pars_xbin_smooth], 
        ylim=[0.75, 1.25], title=f'{sample_name}')
    if save_plots:
        plt.savefig(f'plots/syst_{sample_root}_global_binned_smooth.pdf')


    #-- Loop over bins again to obtain chi2 for the smooth model
    sf.get_chi2_xbin_smooth(s, x_bins, x_data, coeffs, chi2_list)

    #-- plot chi2 values for fits in individual bins
    sf.plot_chi2_vs_x(x_bins, chi2_list, x_name=x_name, title=sample_name)
    if save_plots:
        plt.savefig(f'plots/syst_{sample_root}_chi2perbin.pdf')

    #-- Compute final weights
    pars_smooth = sf.get_pars_from_coeffs(coeffs, x_data)
    data_weightsys_smooth = 1/s.get_model(pars_smooth, data_syst) 


    #-- Histograms of weights for different cases
    plt.figure()
    bins = np.linspace(np.min(data_weightsys_smooth), np.max(data_weightsys_smooth), 100)
    bins = np.linspace(0, 10, 200)
    _=plt.hist(data_weightsys_global, bins=bins, histtype='step', label='Global fit')
    _=plt.hist(data_weightsys_smooth, bins=bins, histtype='step', label='Bin fit smooth')
    wout = ((x_data<x_bins[0])|(x_data>x_bins[-1]))|(~s.w_data)
    _=plt.hist(data_weightsys_smooth[wout], bins=bins, histtype='step', label='Outside fitting range') 
    plt.legend(loc=0)
    plt.axvline(1, color='k', ls=':')
    plt.xlabel('WEIGHT_SYSTOT')
    plt.yscale('log')
    plt.title(sample_name)
    plt.savefig(f'plots/syst_{sample_root}_hist.pdf')


    #-- Plot extreme values of weights in the sky
    sf.plot_weights_sky(data_ra, data_dec, data_weightsys_global, title=f'{sample_name}: Fit global')
    plt.savefig(f'plots/syst_{sample_root}_radec_global.png')
    sf.plot_weights_sky(data_ra, data_dec, data_weightsys_smooth, title=f'{sample_name}: Fit per bin smooth')
    plt.savefig(f'plots/syst_{sample_root}_radec_binned.png')


    #-- Write systematic weights for the fit per bin
    export_binned=0
    if export_binned:
        wext = (data_weightsys_smooth < 0.1)|(data_weightsys_smooth > 2.5)
        data_weightsys_smooth[wext] = 0 
        dat_weight_systot = np.zeros(len(dat))
        ran_weight_systot = np.zeros(len(ran))
        dat_weight_systot[wd] = data_weightsys_smooth 
        ran_weight_systot[wr] = np.random.choice(data_weightsys_smooth[~wext], size=sum(wr), replace=True)
        dat['WEIGHT_SYSTOT'] = dat_weight_systot 
        ran['WEIGHT_SYSTOT'] = ran_weight_systot
        print('Exporting', out_file+'_syst_binned')
        dat.write(out_file+'_syst_binned.dat.fits', overwrite=True)
        ran.write(out_file+'_syst_binned.ran.fits', overwrite=True)




