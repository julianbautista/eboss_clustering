from __future__ import print_function

import os, sys
import numpy as np
import pylab as plt
import healpy as hp
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import minimize
import argparse
import systematic_fitter

indir = '/Users/julian/Work/lrgs/v4/cats'
dat_file = indir+'/eBOSS_LRG_full_SGC_v4.dat.fits'
ran_file = indir+'/eBOSS_LRG_full_SGC_v4.ran.fits'
zmin = 0.6
zmax = 1.0
random_fraction = 1.0

#-- Read data and randoms
print('Reading galaxies from ', dat_file)
dat = Table.read(dat_file)
print('Reading randoms  from ', ran_file)
ran = Table.read(ran_file)

#-- Cut the sample 
print('Cutting galaxies and randoms between zmin=%.3f and zmax=%.3f'%\
     (zmin, zmax))
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
rand_we = (ran['COMP_BOSS'])[wr]#*ran['WEIGHT_FKP'])[wr]


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

#-- NHI map
nhi = Table.read('/Users/julian/Work/software/eboss_clustering/etc/NHI_HPX.fits.gz')['NHI'].data

#-- rotate from galactic to equatorial coordinates
R = hp.Rotator(coord=['G', 'C'], inv=True)
theta, phi = hp.pix2ang(hp.get_nside(nhi), np.arange(nhi.size))
mtheta, mphi = R(theta, phi)
nhi_eq = hp.get_interp_val(nhi, mtheta, mphi)
data_nhi = nhi_eq[get_pix(hp.get_nside(nhi_eq), data_ra, data_dec)]
rand_nhi = nhi_eq[get_pix(hp.get_nside(nhi_eq), rand_ra, rand_dec)]

#-- SDSS systematics
sdss_syst = Table.read('/Users/julian/Work/software/eboss_clustering/etc/SDSS_WISE_imageprop_nside512.fits')
data_pix = get_pix(512, data_ra, data_dec) 
rand_pix = get_pix(512, rand_ra, rand_dec)


#-- Create fitter object
s = systematic_fitter.Syst(data_we, rand_we)
s.add_syst('log10(NHI)', np.log10(data_nhi), np.log10(rand_nhi))

#-- add SDSS systematics to fitter
syst_names = ['STAR_DENSITY', 'EBV', 'PSF_Z', 'DEPTH_I', 'AIRMASS', 'W1_MED', 'W1_COVMED']
for syst_name in syst_names:
    data_syst = sdss_syst[syst_name][data_pix]
    rand_syst = sdss_syst[syst_name][rand_pix]
    if syst_name.startswith('DEPTH'):
        data_syst = flux_to_mag(data_syst, 3, ebv=sdss_syst['EBV'][data_pix])
        rand_syst = flux_to_mag(rand_syst, 3, ebv=sdss_syst['EBV'][rand_pix])
        syst_name += '_MINUS_EBV'
    w = np.isnan(data_syst)|np.isinf(data_syst)
    data_syst[w] = hp.UNSEEN
    w = np.isnan(rand_syst)|np.isinf(rand_syst)
    rand_syst[w] = hp.UNSEEN
    s.add_syst(syst_name, data_syst, rand_syst)

s.cut_outliers()

nbins=20
s.prepare(nbins=nbins)
s.fit_minuit(fit_maps=['STAR_DENSITY', 'log10(NHI)'])
s.plot_overdensity(nbinsh=100, ylim=[0.5, 1.5], title='SGC: Fitting STAR_DENSITY + NHI + EBV')

#-- Perform the fit
#s.fit_minuit(fit_maps=['STAR_DENSITY'])
#s.plot_overdensity(nbinsh=100, ylim=[0.5, 1.5], title='SGC: Fitting STAR_DENSITY only')

#s.fit_minuit(fit_maps=['STAR_DENSITY', 'log10(NHI)'])
#s.plot_overdensity(nbinsh=100, ylim=[0.5, 1.5], title='SGC: Fitting STAR_DENSITY + NHI')

#s.fit_minuit(fit_maps=['STAR_DENSITY', 'EBV'])
#s.plot_overdensity(nbinsh=100, ylim=[0.5, 1.5], title='SGC: Fitting STAR_DENSITY + EBV')

plt.show()

plot_deltas=0
export=0
output=0

#-- Make plots
if plot_deltas:
    print('Plotting deltas versus systematics')
    s.plot_overdensity(ylim=[0.5, 1.5])
    plt.tight_layout()
    plt.show()

#-- Export to table
if export:
    print('Exporting to', export)
    s.export(export)
    
#-- Export catalogs
if output:
    print('Exporting catalogs to ', output)
    dat.write(output+'.dat.fits', overwrite=True)
    ran.write(output+'.ran.fits', overwrite=True)
