import numpy as np
import pylab as plt
from astropy.io import fits
from astropy.table import Table

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

def fit_specsn(dat, band='I', nbins=1000):
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

def get_fiberid_weights(dat, nbins=100):
    ''' Computes redshift failure weights from FIBERID '''  

    #-- get redshift weights from FIBERID dependency
    bins, ng, na = bin_redshift_failures(dat, field='FIBERID', nbins=nbins)
    efficiency = ng/na

    #-- Need to model this better
    #-- For now, just assigning the values on each bin
    index = np.floor( (dat['FIBERID']-bins[0]) / 
                      (bins[-1]      -bins[0]) * 
                      (bins.size-1)).astype(int)
    weight_noz = np.zeros(len(dat))
    w = (index>=0)&(index<efficiency.size)
    weight_noz[w] = 1/efficiency[index[w]]

    return weight_noz 

def plot_failures(dat):
    ''' Plot redshift efficiencies as function of 
        several fields before/after corrections '''

    fields = ['FIBERID', 
              'XFOCAL', 
              'YFOCAL', 
              'SPECSN2_I']

    weight_noz1, coeff = get_specsn_weights(dat)
    weight_noz2 = get_fiberid_weights(dat)
    weight_noz = weight_noz1*weight_noz2

    #-- normalize to overall redshift weight
    wall = where_all_spectra(dat)
    wgood = where_good_spectra(dat)
    eff_before = np.sum(wgood)/np.sum(wall)
    eff_after = np.sum(wgood*weight_noz)/np.sum(wall)
    weight_noz *= 1./eff_after 

    #-- plot before/after corrections 
    for field in fields: 

        plt.figure(figsize=(5,4))

        #-- plot before corrections
        bins, ng, na = bin_redshift_failures(dat, field=field, nbins=100,
                                             wgood=wgood, wall=wall)
        centers = 0.5*(bins[1:]+bins[:-1])
        w = na>50
        x = centers[w]
        y = ng[w]/na[w]
        dy = error_of_ratio_poisson(na[w]-ng[w], na[w])
        plt.errorbar(x, y, dy, fmt='.', label=r'No $w_{noz}$')

        #-- plot after corrections
        bins, ng, na = bin_redshift_failures(dat, field=field, nbins=100, 
                                             wgood=wgood, wall=wall,
                                             weights=weight_noz )
        y = ng[w]/na[w]
        #-- purposefully do not recompute errors, should be similar as before
        #-- and the ratio formula doesn't work for corrected counts 
        #dy = error_of_ratio_poisson(na[w]-ng[w], na[w])
        plt.errorbar(x, y, dy, fmt='.', label=r'With $w_{noz}$')

        #-- plot model for the SPECSN dependency
        if field=='SPECSN2_I':
            xmodel = np.linspace(x.min(), x.max(), 100)
            ymodel = get_specsn_efficiency(coeff, xmodel)
            plt.plot(xmodel, ymodel, label='Best-fit model')
        
        plt.xlabel(field)
        plt.ylabel('Redshift efficiency')
        plt.legend(loc=0, fontsize=10)
        plt.tight_layout()

    return weight_noz


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
        assign_specsn(dat)
    
    #-- fix dependency with spectro S/N
    weight_noz1, coeff = get_specsn_weights(dat)
    #-- fix dependency with FIBERID
    weight_noz2 = get_fiberid_weights(dat)

    #-- final weight is simply the product 
    weight_noz = weight_noz1*weight_noz2

    wall = where_all_spectra(dat)
    wgood = where_good_spectra(dat)

    eff_before = np.sum(wgood)/np.sum(wall)
    eff_after = np.sum(wgood*weight_noz)/np.sum(wall)
    
    #-- normalize in order to get overall redshift efficiency = 1
    weight_noz *= 1./eff_after

    dat['WEIGHT_NOZ'] = weight_noz







