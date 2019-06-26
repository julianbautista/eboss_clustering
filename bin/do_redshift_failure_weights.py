import numpy as np
import pylab as plt
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import minimize
import iminuit

def assign_specsn2(dat):
    ''' Assigns the correct value of spectrograph S/N for 
        the three cameras '''

    w1 = (dat['FIBERID']>=1)&(dat['FIBERID']<=500)
    w2 = (dat['FIBERID']>=501)&(dat['FIBERID']<=1000)
    for c in ['G', 'R', 'I']:
        dat['SPECSN2_%s'%c] = np.zeros(len(dat))
        dat['SPECSN2_%s'%c][w1] = dat['SPEC1_%s'%c][w1]
        dat['SPECSN2_%s'%c][w2] = dat['SPEC2_%s'%c][w2]

def where_all_spectra(dat):

    wall = (dat['IMATCH']==1)|\
           (dat['IMATCH']==4)|\
           (dat['IMATCH']==7)|\
           (dat['IMATCH']==9)
    return wall

def where_good_spectra(dat):

    wgood = (dat['IMATCH']==1)|\
            (dat['IMATCH']==4)|\
            (dat['IMATCH']==9)
    return wgood

def bin_redshift_failures(dat, field='FIBERID', nbins=500, weights=None,
                          wall=None, wgood=None):
    ''' Compute redshift efficiency = ngood/nall 
        as a function of a quantity (field) in the catalog
        
        Input
        dat: astropy.table.Table object with input catalog
        Output
        bins: array with binned quantity
        ngood: number of good redshifts per bin
        nall: number of spectra per bin

    '''
    if weights is None:
        weights = np.ones(len(dat))

    #-- select objects with observed spectra
    if wall is None:
        wall = where_all_spectra(dat) 

    #-- select objects with confident classification
    if wgood is None:
        wgood = where_good_spectra(dat)

    #-- count galaxies in bins of field
    x = dat[field]
    xmin = x[wall].min()
    xmax = x[wall].max()
    dx = (xmax-xmin)/nbins
    bins = np.linspace(xmin-dx/100, 
                       xmax+dx/100,
                       nbins)
    ngood, bins = np.histogram(x[wgood], bins=bins, weights=weights[wgood])
    nall,  bins = np.histogram(x[wall] , bins=bins) 
    return bins, ngood, nall

def error_of_ratio_poisson(a, b):

    return np.sqrt( a/b**2 + a**2/b**3)

def fit_specsn(dat, band='I', nbins=500):
    ''' Fits for model of efficiencies as a functino of SPECSN
        as in Eq. 8 of Bautista et al. 2018 
        http://adsabs.harvard.edu/abs/2018ApJ...863..110B

        Input
        -----
        dat: astropy.table.Table object with the catalog
        
        Output
        ------
        coeff: array with best-fit coefficients, to be used 
               with get_specsn_efficiency 

    '''
    bins, ngood, nall = bin_redshift_failures(dat, 
                            field='SPECSN2_%s'%band, nbins=nbins)
    x = 0.5*(bins[1:]+bins[:-1])

    w = (nall > 50)&(nall-ngood > 0)
    nall = nall[w]
    ngood = ngood[w]
    x = x[w]

    #-- we actually fit a linear function of ngood/nfail 
    #-- which is zero for sn2=0  
    y = ngood/(nall-ngood)
    dy = error_of_ratio_poisson(ngood, nall-ngood)
    
    coeff = np.polyfit(x, y, 1, w=1/dy) 
    ymodel = np.polyval(coeff, x)

    chi2 = np.sum( (y-ymodel)**2/dy**2 )
    ndata = y.size
    npars = coeff.size
    rchi2 = chi2/(ndata-npars)

    print('Fit of efficiency vs spectro S/N')
    print('coeff = ', coeff)
    print('chi2 =', chi2, 'ndata =', ndata, 'npars =', npars, 'rchi2 =',rchi2)
    return coeff


def fit_fiberid(dat, nbins=500, verbose=False):

    bins, ngood, nall = bin_redshift_failures(dat,
                            field='FIBERID', nbins=nbins)
    x = 0.5*(bins[1:]+bins[:-1])

    w = (nall > 50)&(nall-ngood > 0)
    nall = nall[w]
    ngood = ngood[w]
    x = x[w]
    y = ngood/(nall)
    dy = error_of_ratio_poisson(nall-ngood, nall)

    def chi2(p):
        model = get_fiberid_efficiency(p, x)
        residual = y-model
        chi2 = np.sum(residual**2/dy**2)
        return chi2

    par0 = np.array([0., 0., 0., 0., 0., 1.])

    par_names = ['C%d'%i for i in range(par0.size)]
    init_pars = {par_names[i]: par0[i] for i in range(par0.size)}
    for i in range(par0.size):
        init_pars['error_C%d'%i] = 10
    mig = iminuit.Minuit(chi2, throw_nan=False,
                 forced_parameters=par_names,
                 print_level=0, errordef=1,
                 use_array_call=True,
                  **init_pars)
    #mig.tol = 1.0
    imin = mig.migrad()
    is_valid = imin[0]['is_valid']
    coeff = mig.values.values()
    chi2min = mig.fval
    ndata = x.size
    npars = mig.narg
    rchi2 = chi2min/(ndata-npars)
    ymodel = get_fiberid_efficiency(coeff, x)

    if verbose:
        print('Minuit === Fit of efficiency vs FIBERID')
        print('chi2 =', chi2min, 'ndata =', ndata, 'npars =', npars, 'rchi2 =',rchi2)
    return coeff, x, y, dy, ymodel

def get_fiberid_efficiency(coeff, fiberid):
    #x = fiberid/np.mean(fiberid)-1
    x = fiberid/1000
    #return coeff[0] - coeff[1]*np.abs(x)**coeff[2]
    m = np.polyval(coeff, x)
    return m

def get_specsn_efficiency(coeff, specsn2):
    ''' Model used to fit efficiencies as function of SPECSN2'''
    return 1-1/(1+np.polyval(coeff, specsn2))
    
def get_specsn_weights(dat, band='I'):
    ''' Computes redshift failure weights from the spectrograph SN2'''

    #- fit model
    coeff = fit_specsn(dat, band=band)

    weight_noz = np.zeros(len(dat))

    w = dat['SPECSN2_%s'%band]>0
    specsn2 = dat['SPECSN2_%s'%band][w]

    #-- evaluate model
    weight_noz[w] = 1/get_specsn_efficiency(coeff, specsn2)

    return weight_noz, coeff

def get_fiberid_weights(dat, nbins=126, plotit=False, verbose=False):
    ''' Computes redshift failure weights from FIBERID '''  

    weight_noz = np.zeros(len(dat))
    fiberid_bins = [0, 250, 500, 750, 1000]

    if plotit:
        plt.figure(figsize=(14, 5))
    for i in range(4):
        fmin = fiberid_bins[i]
        fmax = fiberid_bins[i+1]
        w = (dat['FIBERID'] > fmin)&(dat['FIBERID'] <= fmax)
        coeff, x, y, dy, ymodel = fit_fiberid(dat[w], nbins=nbins, verbose=verbose)
        weight_noz[w] = 1/get_fiberid_efficiency(coeff, dat['FIBERID'][w])

        if plotit:
            plt.errorbar(x, y, dy, fmt='o', color='C%i'%i, ms=2, alpha=0.3)
            plt.plot(x, ymodel, 'C%d'%i)
    if plotit:
        plt.ylim(0.88, 1.01)
        plt.grid()
    
    
    return weight_noz 

def plot_failures(dat, weight_noz=None, coeff=None):
    ''' Plot redshift efficiencies as function of 
        several fields before/after corrections '''

    fields = ['FIBERID', 
              'XFOCAL', 
              'YFOCAL', 
              'SPECSN2_I']

    wall = where_all_spectra(dat)
    wgood = where_good_spectra(dat)

    weight_noz = dat['WEIGHT_NOZ']

    #-- plot before/after corrections 
    for field in fields: 

        plt.figure(figsize=(5,4))

        #-- plot before corrections
        if field == 'SPECSN2_I': 
            nbins = 500
        else: 
            nbins = 100
        bins, ng, na = bin_redshift_failures(dat, field=field, nbins=nbins,
                                             wgood=wgood, wall=wall)
        centers = 0.5*(bins[1:]+bins[:-1])
        w = (na>50)&(na-ng>0)
        x = centers[w]
        y = ng[w]/na[w]
        dy = error_of_ratio_poisson(na[w]-ng[w], na[w])
        plt.errorbar(x, y, dy, fmt='.', label=r'No $w_{noz}$')

        #-- plot after corrections
        bins, ng, na = bin_redshift_failures(dat, field=field, nbins=nbins, 
                                             wgood=wgood, wall=wall,
                                             weights=weight_noz )
        y = ng[w]/na[w]
        #-- purposefully do not recompute errors, should be similar as before
        #-- and the ratio formula doesn't work for corrected counts 
        plt.errorbar(x, y, dy, fmt='.', label=r'With $w_{noz}$')
        plt.xlabel(field)
        plt.ylabel('Redshift efficiency')
        plt.legend(loc=0, fontsize=10)
        plt.tight_layout()



def get_weights_noz(dat):
    '''  Computes redshit failure weights from catalog
         based on IMATCH == 7 (failures) 

         Currently correcting for FIBERID and SPECSN2_I.

         A new column "WEIGHT_NOZ" is created with the 
         weights. 

         Input
         -----
         dat: astropy.table.Table object with catalog

    '''
    if 'SPECSN2_I' not in dat.colnames:
        assign_specsn2(dat)
    
    #-- dependency with spectro S/N
    weight_noz1, coeff = get_specsn_weights(dat)

    #-- dependency with FIBERID
    weight_noz2 = get_fiberid_weights(dat, verbose=True)

    #-- final weight is simply the product 
    weight_noz = weight_noz1*weight_noz2

    wall = where_all_spectra(dat)
    wgood = where_good_spectra(dat)

    eff_before = np.sum(wgood)/np.sum(wall)
    eff_after = np.sum(wgood*weight_noz)/np.sum(wall)
    
    #-- normalize in order to get overall redshift efficiency = 1
    weight_noz *= 1./eff_after

    dat['WEIGHT_NOZ'] = weight_noz



dat = Table.read('/mnt/lustre/eboss/DR16_LRG_data/v5/eBOSS_LRG_full_SGC_v5.dat.fits')
get_weights_noz(dat)
plot_failures(dat)





