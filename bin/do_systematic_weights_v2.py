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
import systematic_fitter


def get_pix(nside, ra, dec, nest=0):
    return hp.ang2pix(nside, np.radians(-dec+90), np.radians(ra), nest=nest)

def flux_to_mag(flux, band, ebv=None):
    ''' Converts SDSS fluxes to magnitudes, correcting for extinction optionally (EBV)'''
    #-- coefs to convert from flux to magnitudes
    b = np.array([1.4, 0.9, 1.2, 1.8, 7.4])[band]*1e-10
    mag = -2.5/np.log(10)*(np.arcsinh((flux/1e9)/(2*b)) + np.log(b))
    #-- extinction coefficients for SDSS u, g, r, i, and z bands
    ext_coeff = np.array([4.239, 3.303, 2.285, 1.698, 1.263])[band]
    if not ebv is None:
        mag -= ext_coeff*ebv
    return mag

def read_systematic_maps(data_ra, data_dec, rand_ra, rand_dec):
    
    #-- Dictionaries containing all different systematic values
    data_syst = {}
    rand_syst = {}

    #-- NHI map
    nhi_file = os.environ['EBOSS_CLUSTERING_DIR']+'/etc/NHI_HPX.fits.gz'
    nhi_file = 'toto'
    if os.path.exists(nhi_file):
        nhi = Table.read(nhi_file)['NHI'].data
        #-- rotate from galactic to equatorial coordinates
        R = hp.Rotator(coord=['G', 'C'], inv=True)
        theta, phi = hp.pix2ang(hp.get_nside(nhi), np.arange(nhi.size))
        mtheta, mphi = R(theta, phi)
        nhi_eq = hp.get_interp_val(nhi, mtheta, mphi)
        data_nhi = nhi_eq[get_pix(hp.get_nside(nhi_eq), data_ra, data_dec)]
        rand_nhi = nhi_eq[get_pix(hp.get_nside(nhi_eq), rand_ra, rand_dec)]
        data_syst['log10(NHI)'] = np.log10(data_nhi)
        rand_syst['log10(NHI)'] = np.log10(rand_nhi)

    #-- SDSS systematics
    sdss_syst = Table.read(os.environ['EBOSS_CLUSTERING_DIR']+
                           '/etc/SDSSimageprop_Nside512.fits')
    data_pix = get_pix(512, data_ra, data_dec) 
    rand_pix = get_pix(512, rand_ra, rand_dec)
    syst_names = ['EBV', 'SKY_I', 'SKY_Z', 'PSF_I', 'PSF_Z', 'DEPTH_I', 'DEPTH_Z', 'AIRMASS']
    for syst_name in syst_names:
        if 1==1 and syst_name.startswith('DEPTH'):
            if syst_name.endswith('R'):
                cam = 2
            if syst_name.endswith('I'):
                cam = 3
            if syst_name.endswith('Z'):
                cam = 4
            depth_minus_ebv = flux_to_mag(sdss_syst[syst_name], cam, ebv=sdss_syst['EBV']).data
            data_syst[syst_name+'_MINUS_EBV'] = depth_minus_ebv[data_pix]
            rand_syst[syst_name+'_MINUS_EBV'] = depth_minus_ebv[rand_pix]
        else:
            data_syst[syst_name] = sdss_syst[syst_name][data_pix].data
            rand_syst[syst_name] = sdss_syst[syst_name][rand_pix].data

    #-- Star density
    star_density = np.loadtxt(os.environ['EBOSS_CLUSTERING_DIR']+
                              '/etc/allstars17.519.9Healpixall256.dat')
    data_pix = get_pix(256, data_ra, data_dec, nest=1)
    rand_pix = get_pix(256, rand_ra, rand_dec, nest=1)
    data_syst['STAR_DENSITY'] = star_density[data_pix]
    rand_syst['STAR_DENSITY'] = star_density[rand_pix]

    return data_syst, rand_syst


def fit_per_bin(s, zname, data_z, nbinsz=10, p=1., fit_maps=None):
    ''' From a systematic_fitter.Syst object, divide in subsamples based 
        on the value of a variable called zname with values data_z 
    '''

    #-- define zbins excluding p% of extreme values
    zbins = np.array([ np.percentile(data_z, p/2+(i*(100-p)/(nbinsz))) 
                       for i in range(nbinsz+1)])
    zcen = np.zeros(nbinsz)
    chi2_before = np.zeros(nbinsz) 
    chi2_global = np.zeros(nbinsz)
    chi2_zbin = np.zeros(nbinsz)

    #-- List of best fit parameters
    fit_pars = []
    fit_errors = [] 
    chi2 = []

    #-- Loop over zbins
    for i in range(nbinsz):
        zmin = zbins[i]
        zmax = zbins[i+1]
        wdz = (data_z>=zmin) & (data_z<zmax)
        zcen[i] = np.median(data_z[wdz])

        print('===')
        print(f'=== Getting subsample {i+1} of {nbinsz}, {zmin:.3f} < {zname} < {zmax:.3f} ===')
        print('===')
        
        ss = s.get_subsample(wdz)
        ss.fit_minuit(fit_maps=fit_maps)
        ss.plot_overdensity(ylim=[0., 2.], 
             title=f'{sample_name}: {zmin:.3f} < {zname} < {zmax:.3f} fit in this bin #{i}')
        plt.savefig(f'plots/syst_{sample_root}_bin{i}.pdf')

        fit_pars.append(ss.best_pars)
        fit_errors.append(ss.errors)
        chi2.append({'before':ss.get_chi2(), 
                     'global':ss.get_chi2(s.best_pars), 
                     'bin':ss.get_chi2(ss.best_pars)})

    return fit_pars, fit_errors, chi2, zcen, zbins


def plot_pars_vs_z(zcen, pars, errors, zname='Z', ylim=None,
    global_pars=None, global_errors=None, title=None, chi2_thres=9, zmin=None, zmax=None):
    
    #-- convert dict_keys into list
    par_names = [*pars[0]]
    npar = len(par_names)
    
    coeffs = {}

    #-- some options for plotting
    figsize = (15, 3) if npar > 2 else (6,3)
    f, ax = plt.subplots(1, npar, sharey=False, figsize=figsize)
    if npar == 1:
        ax = [ax]
    if npar > 1:
        f.subplots_adjust(wspace=0.13, left=0.05, right=0.98,
                          top=0.98, bottom=0.15)
    if not ylim is None:
        ax[0].set_ylim(ylim)

    print('\nFitting polynomials over slopes per bin')
    for par_name, axx in zip(par_names, ax):
        y = np.array([par[par_name] for par in pars]) 
        dy = np.array([error[par_name] for error in errors])
        axx.errorbar(zcen, y, dy, fmt='o', ms=3)
        #-- Fitting slopes with polynomials with increasing order
        for order in range(3):
            coeff = np.polyfit(zcen, y, order, w=1/dy)
            
            #-- Compute model to get chi2
            ymodel = np.polyval(coeff, zcen)
            chi2 = np.sum((y-ymodel)**2/dy**2)

            #-- Plot the model using the full range of data_z
            x = np.linspace(zmin, zmax, 30)
            ymodel = np.polyval(coeff, x)
            axx.plot(x, ymodel, label=r'$n_{\rm poly}=%d, \chi^2 = %.1f$'%(order, chi2))
            
            if par_name in coeffs:
                #-- check if chis is smaller by at least chi2_thres units than lower order
                coeff_before = coeffs[par_name]
                ymodel_before = np.polyval(coeff_before, zcen)
                chi2_before = np.sum((y-ymodel_before)**2/dy**2)
                if (chi2 < chi2_before - chi2_thres):
                    print('  ', par_name, ' fit with ', order, 'order poly with chi2=', chi2)
                    coeffs[par_name] = coeff
            else:
                coeffs[par_name] = coeff
                
        if not global_pars is None:
            x = np.median(zcen)
            y = global_pars[par_name]
            dy = global_errors[par_name]
            axx.errorbar(x, y, dy, fmt='*', ms=8, label='Global fit') 
        axx.locator_params(axis='x', nbins=4, tight=True)
        axx.legend(loc=0, fontsize=8)
        axx.set_xlabel(zname)
        axx.set_title(par_name)

    if title:
        f.subplots_adjust(top=0.85)
        plt.suptitle(title)

    return coeffs

def get_pars_from_coeffs(coeffs, zvalues):

    pars_bin = {}
    for par_name in coeffs:
        pars_bin[par_name] = np.polyval(coeffs[par_name], zvalues)
    return pars_bin

def plot_chi2_vs_z(zcen, chi2s, zname='redshift', title=None):
    
    plt.figure()
    for chi2name in chi2s[0]:
        c2 = np.array([chi2[chi2name] for chi2 in chi2s])
        plt.plot(zcen, c2, 'o-', label=chi2name)
    plt.legend(loc=0)
    plt.ylabel(r'$\chi^2$ per %s bin'%zname)
    plt.xlabel(zname)
    if title:
        plt.title(title)

def get_chi2_bin_smooth(s, zbins, data_z, coeffs, chi2s):
    #-- Compute chi2 for each bin, this time with the smooth poly parameters
    for i in range(nbinsz):
        zmin = zbins[i]
        zmax = zbins[i+1]
        wdz = (data_z>=zmin) & (data_z<zmax) & (s.w_data)
        ss = s.get_subsample(wdz)
        pars_bin_zbin = get_pars_from_coeffs(coeffs, data_z[wdz])
        chi2s[i]['bin_smooth'] = ss.get_chi2(pars_bin_zbin)
        #ss.plot_overdensity(pars=[None, s.best_pars, pars_bin_zbin], ylim=[0, 2], 
        #    title=f'{sample_name}: {zmin:.3f} < {zname} < {zmax:.3f} fit in this bin')

def ra(x):
    return x-360*(x>300)





indir = os.environ['LRG_DIR']+'/cats'
cap = 'NGC'
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
data_syst, rand_syst = read_systematic_maps(data_ra, data_dec, rand_ra, rand_dec)


#-- Create fitter object
s = systematic_fitter.Syst(data_we, rand_we)

use_maps = ['STAR_DENSITY', 'EBV', 'PSF_Z', 'DEPTH_I_MINUS_EBV', 'AIRMASS']
#use_maps = ['EBV', 'PSF_I', 'SKY_I', 'DEPTH_I_MINUS_EBV', 'AIRMASS']

fit_maps = ['STAR_DENSITY', 'EBV']

#-- Add the systematic maps we want 
for syst_name in use_maps:
    s.add_syst(syst_name, data_syst[syst_name], rand_syst[syst_name])
s.cut_outliers(p=0.5, verbose=1)

#-- Perform global fit
nbins=15
s.prepare(nbins=nbins)
s.fit_minuit(fit_maps=fit_maps)
#s.plot_overdensity(pars=[None, s.best_pars], ylim=[0., 2.0], 
#    title=f'{sample_name}: global fit')


#-- Divide the sample into subsamples based of the "z" value
#-- which can be the redshift Z or the FIBER2MAG

#zname = 'Redshift'
#data_z = dat['Z'][wd]
zname = r'$g_{\rm fib2}$'
data_z = flux_to_mag(dat['FIBER2FLUX'][:, 1].data[wd], 1).data
zname = r'$i_{\rm fib2}$'
data_z = flux_to_mag(dat['FIBER2FLUX'][:, 3].data[wd], 3).data

nbinsz=10
chi2_thres=16.
p_zcut = 1.
fit_pars, fit_errors, chi2s, zcen, zbins = \
    fit_per_bin(s, zname, data_z, nbinsz=nbinsz, p=p_zcut, fit_maps=fit_maps)

#-- Plot slopes versus dependent variable at fit polynomials
coeffs = plot_pars_vs_z(zcen, fit_pars, fit_errors, zname=zname, 
             global_pars=s.best_pars, global_errors=s.errors, 
             title=sample_name, zmin=zbins[0], zmax=zbins[-1], 
             chi2_thres=chi2_thres)
plt.savefig(f'plots/syst_{sample_root}_slopesperbin.pdf')


#-- Get parameters for the smooth fit per bin
pars_bin = get_pars_from_coeffs(coeffs, data_z[s.w_data])

#-- Compare chi2 over the full sample
print('\nFinal chi2 values over full sample')
chi2_before = s.get_chi2()
chi2_global = s.get_chi2(s.best_pars)
chi2_smooth = s.get_chi2(pars_bin)
ndata = s.ndata
print(f' - chi2 (no fit):          {chi2_before:.1f}/{ndata} = {chi2_before/ndata:.2f}')
print(f' - chi2 (global fit):      {chi2_global:.1f}/{ndata} = {chi2_global/ndata:.2f}')
print(f' - chi2 (smooth evol fit): {chi2_smooth:.1f}/{ndata} = {chi2_smooth/ndata:.2f}')

#-- Make overdensity plots for full sample with new smooth per bin parameters
s.plot_overdensity(pars=[None, s.best_pars, pars_bin], 
    ylim=[0.75, 1.25], title=f'{sample_name}')
plt.savefig(f'plots/syst_{sample_root}_global_binned.pdf')

#-- Loop over bins again to obtain chi2 for the smooth model
get_chi2_bin_smooth(s, zbins, data_z, coeffs, chi2s)

#-- plot chi2 values for fits in individual bins
plot_chi2_vs_z(zcen, chi2s, zname=zname, title=sample_name)
plt.savefig(f'plots/syst_{sample_root}_chi2perbin.pdf')


#-- Compute final weights
pars_smooth = get_pars_from_coeffs(coeffs, data_z)
data_weightsys_global = 1/s.get_model(s.best_pars, data_syst)
data_weightsys_smooth = 1/s.get_model(pars_smooth, data_syst) 


#-- Histograms of weights for different cases
plt.figure()
bins = np.linspace(np.min(data_weightsys_smooth), np.max(data_weightsys_smooth), 100)
bins = np.linspace(0, 10, 200)
_=plt.hist(data_weightsys_global, bins=bins, histtype='step', label='Global fit')
_=plt.hist(data_weightsys_smooth, bins=bins, histtype='step', label='Bin fit smooth')
wout = ((data_z<zbins[0])|(data_z>zbins[-1]))|(~s.w_data)
_=plt.hist(data_weightsys_smooth[wout], bins=bins, histtype='step', label='Outside fitting range') 
plt.legend(loc=0)
plt.axvline(1, color='k', ls=':')
plt.xlabel('WEIGHT_SYSTOT')
plt.yscale('log')
plt.title(sample_name)
plt.savefig(f'plots/syst_{sample_root}_hist.pdf')

#-- Plot extreme values of weights in the sky
def plot_weights_sky(data_ra, data_dec, data_weightsys, title=None):
    plt.figure(figsize=(12, 7))
    plt.scatter(ra(data_ra), data_dec, c=data_weightsys, vmin=0.5, vmax=1.5, lw=0, s=2, cmap='jet', label=None)
    wext = (data_weightsys<0.5)|(data_weightsys > 2.)
    if sum(wext)>0:
        plt.plot(ra(data_ra[wext]), data_dec[wext], 'ko', ms=4, 
            label=r'$w_{\rm sys} < 0.5 \ {\rm or} \ w_{\rm sys} > 2.$ : %d galaxies'%sum(wext))
    plt.xlabel('RA [deg]')
    plt.ylabel('DEC [deg]')
    c = plt.colorbar()
    c.set_label('WEIGHT_SYSTOT')
    if title:
        plt.title(title)
    plt.legend(loc=0)
    plt.tight_layout()

plot_weights_sky(data_ra, data_dec, data_weightsys_global, title=f'{sample_name}: Fit global')
plt.savefig(f'plots/syst_{sample_root}_radec_global.png')
plot_weights_sky(data_ra, data_dec, data_weightsys_smooth, title=f'{sample_name}: Fit per bin smooth')
plt.savefig(f'plots/syst_{sample_root}_radec_binned.png')
 
#-- Write systematic weights for global fit
export=0
if export:
    dat_weight_systot = np.zeros(len(dat))
    ran_weight_systot = np.zeros(len(ran))
    dat_weight_systot[wd] = data_weightsys_global 
    ran_weight_systot[wr] = np.random.choice(data_weightsys_global, size=sum(wr), replace=True)
    dat['WEIGHT_SYSTOT'] = dat_weight_systot 
    ran['WEIGHT_SYSTOT'] = ran_weight_systot
    print('Exporting', out_file+'_syst_global')
    dat.write(out_file+'_syst_global.dat.fits', overwrite=True)
    ran.write(out_file+'_syst_global.ran.fits', overwrite=True)


    #-- Write systematic weights for the fit per bin
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




