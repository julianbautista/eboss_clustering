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

indir = os.environ['LRG_DIR']+'/cats'
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
rand_we = (ran['COMP_BOSS'])[wr]*(ran['WEIGHT_FKP'])[wr]


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


#-- Dictionaries containing all different systematic values
data_syst = {}
rand_syst = {}



#-- NHI map
nhi_file = os.environ['EBOSS_CLUSTERING_DIR']+'/etc/NHI_HPX.fits.gz'
if os.path.exists(nhi_file):
    nhi = Table.read()['NHI'].data
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
                       '/etc/SDSS_WISE_imageprop_nside512.fits')
data_pix = get_pix(512, data_ra, data_dec) 
rand_pix = get_pix(512, rand_ra, rand_dec)
syst_names = ['STAR_DENSITY', 'EBV', 'PSF_Z', 'DEPTH_I', 'AIRMASS', 'W1_MED', 'W1_COVMED']
for syst_name in syst_names:
    if 1==0 and syst_name.startswith('DEPTH'):
        depth_minus_ebv = flux_to_mag(sdss_syst[syst_name], 3, ebv=sdss_syst['EBV']).data
        data_syst[syst_name+'_MINUS_EBV'] = depth_minus_ebv[data_pix]
        rand_syst[syst_name+'_MINUS_EBV'] = depth_minus_ebv[rand_pix]
    else:
        data_syst[syst_name] = sdss_syst[syst_name][data_pix].data
        rand_syst[syst_name] = sdss_syst[syst_name][rand_pix].data
    #s.add_syst(syst_name, data_syst, rand_syst)

#-- Create fitter object
s = systematic_fitter.Syst(data_we, rand_we)
for syst_name in data_syst.keys():
    s.add_syst(syst_name, data_syst[syst_name], rand_syst[syst_name])
s.cut_outliers()

#-- List of best fit parameters
fit_pars = []
fit_errors = []

#-- Perform global fit
nbins=20
s.prepare(nbins=nbins)
s.fit_minuit(fit_maps=['STAR_DENSITY', 'log10(NHI)', 'DEPTH_I'])
s.plot_overdensity(ylim=[0.5, 1.5], title='SGC: global fit')

fit_pars.append(s.best_pars)
fit_errors.append(s.errors)

#-- Divide the sample into subsamples based of the data_z value
#-- testing Z or FIBER2MAG
data_z = dat['Z'][wd]
rand_z = ran['Z'][wr]
data_z = flux_to_mag(dat['FIBER2FLUX'][:, 4].data[wd], 4).data
rand_z = np.zeros(rand_we.size)+np.median(data_z)
nbinsz = 3
zbins = np.array([ np.percentile(data_z, i*100/(nbinsz)) for i in range(nbinsz+1)])

for i in range(nbinsz):
    wdz = (data_z>=zbins[i])&(data_z<zbins[i+1])
    wrz = np.ones(rand_z.size)==1
    ss = s.get_subsample(wdz, wrz)
    ss.pars = s.pars
    ss.fit_index = s.fit_index
    #-- first plot results for global fit 
    ss.plot_overdensity(ylim=[0.5, 1.5], title='SGC: %.2f < z < %.2f global fit'%(zbins[i], zbins[i+1]))
    
    #-- then plot results for subsample fit 
    ss.fit_minuit(fit_maps=['STAR_DENSITY', 'log10(NHI)', 'DEPTH_I'])
    ss.plot_overdensity(ylim=[0.5, 1.5], title='SGC: %.2f < z < %.2f fit in this zbin'%(zbins[i], zbins[i+1]))
    print('chi2 (before) =', ss.get_chi2())
    print('chi2 (global fit) =', ss.get_chi2(*s.pars))
    print('chi2 (zbin fit) =', ss.get_chi2(*ss.best_pars.values()))

    fit_pars.append(ss.best_pars)
    fit_errors.append(ss.errors)


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
