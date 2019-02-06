import sys
from astropy.table import Table
import numpy as np
import pylab as plt
import healpy as hp
from imaging_systematics import MultiLinearFit
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

indir = '/Users/julian/Work/lrgs/v4/cats'
dat_file = indir+'/eBOSS_LRG_full_NGC_v4.dat.fits'
ran_file = indir+'/eBOSS_LRG_full_NGC_v4.ran.fits'
random_fraction = 1.0
zmin=0.6
zmax=1.0
title = 'eBOSS LRG NGC %.2f < z < %.2f'%(zmin, zmax)
save = 'plots/syst_LRG_full_NGC_v4_zmin%.2f_zmax%.2f_chi2.pdf'%(zmin, zmax)

#-- Read data and randoms
print('Reading galaxies from ', dat_file)
dat = Table.read(dat_file)
print('Reading randoms  from ', ran_file)
ran = Table.read(ran_file)

#-- Apply cuts in redshift and completeness (only for FULL catalogs)
wd = ((dat['IMATCH']==1)|(dat['IMATCH']==2))&\
     (dat['Z']>=zmin)&\
     (dat['Z']<=zmax)&\
     (dat['COMP_BOSS']>0.5)&\
     (dat['sector_SSR']>0.5)
wr = (ran['Z']>=zmin)&\
     (ran['Z']<=zmax)&\
     (ran['COMP_BOSS']>0.5)&\
     (ran['sector_SSR']>0.5)

#-- Get RA and DEC
data_ra, data_dec = dat['RA'][wd], dat['DEC'][wd]
rand_ra, rand_dec = ran['RA'][wr], ran['DEC'][wr]

#-- Define weights
data_we = (dat['WEIGHT_CP']*dat['WEIGHT_FKP']/dat['sector_SSR'])[wd]
rand_we = (ran['COMP_BOSS'])[wr]#*ran['WEIGHT_FKP'])[wr]



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
syst_names = ['STAR_DENSITY', 'SKY_Z', 'AIRMASS', 'EBV', 'DEPTH_Z', 'PSF_Z', 'W1_MED', 'W1_COVMED']
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
s.prepare(nbins=20)


fit_maps_name = ['NONE']
fit_maps_index = [-1]
fit_maps_chi2 = [s.get_chi2()]
for nmaps in range(s.nsyst):
    deltachi2 = np.zeros(s.nsyst)
    for i in range(s.nsyst):
        if i in fit_maps_index: continue
        fit_maps = fit_maps_name+[s.syst_names[i]]
        print(fit_maps)
        s.fit_minuit(fit_maps=fit_maps)
        deltachi2[i] = s.chi2min
    #-- get index where deltachi2 is positive (was computed) and minimum
    w = np.where(deltachi2==deltachi2[deltachi2>0].min())[0][0]
    fit_maps_index.append(w)
    fit_maps_name.append(s.syst_names[w])
    fit_maps_chi2.append(deltachi2[w])
    print('=========')
    print(fit_maps_name)
    print(fit_maps_chi2)
    print('=========')

fit_maps_chi2 = np.array(fit_maps_chi2)
fit_maps_name = np.array(fit_maps_name)
x = np.arange(fit_maps_chi2.size)

f = plt.figure(figsize=(6, 6))
axes = f.add_subplot(211)
axes.plot(x, fit_maps_chi2-fit_maps_chi2.min(), 'o-')
axes.set_xlim(-1, x[-1]+1)
axes.set_ylabel(r'$\chi^2-\chi^2_{\rm min}$')
axes.set_yscale('symlog')
axes.set_xticks(x)
axes.set_title(title)
axes = f.add_subplot(212)
axes.plot(x[1:], fit_maps_chi2[1:]-fit_maps_chi2[:-1], 'o-')
axes.axhline(0,  color='k', ls='--')
axes.axhline(-9, color='k', ls=':')
axes.set_xlim(-1, x[-1]+1)
axes.set_ylabel(r'$\frac{d \chi^2}{d \ \rm{map}}$')
axes.set_xticks(x)
axes.set_xticklabels(fit_maps_name, rotation=65) 
axes.set_yscale('symlog')
f.subplots_adjust(left=0.16, right=0.97, bottom=0.2, hspace=0.05)
#plt.savefig(save)
plt.show()


