from __future__ import print_function

import os
import numpy as N
import pylab as P
import healpy as hp
from astropy.io import fits


from scipy.optimize import minimize

class MultiFit:

    def __init__(self, data_ra=None, data_dec=None, data_we=None, 
                       rand_ra=None, rand_dec=None, rand_we=None,
                       nside=256, maps_index=N.arange(7), 
                       fit_index=N.arange(7)):
        ''' MultiFit initializer

            Read FITS file containing all healpix maps
            
            nside: integer
                 nside of the output (should be less or equal to 256)
            maps_index: integer array
                 index of maps wanted
                 Default: all maps
                 0 - stellar density
                 1 - i-band depth
                 2 - z sky flux
                 3 - z FWHM
                 4 - Extinction r-band
                 5 - Wise W1 CovMed
                 6 - Wise W1 Med
            fit_index: integer array
                 index of maps that you want to fit 
                 e.g.,  N.array([0, 4, 6]) fits simultaneously 
                 stellar density, extinction r-band and Wise W1 Med 
                 and not the others.
                 Default: fit all maps
            Returns
            -------
            syst_maps: dict with systematic maps and information about them
        ''' 

        self.syst_maps = self.read_systematic_maps_fits(nside=nside, 
                              maps_index=maps_index)

        self.maps_index = maps_index
        self.fit_index = fit_index
    
        self.ra_data = data_ra
        self.dec_data= data_dec
        self.weights_data = data_we

        self.ra_rand = rand_ra
        self.dec_rand = rand_dec
        self.weights_rand = rand_we

    def read_systematic_maps_fits(self, nside=256, maps_index=N.arange(7),
            infits=os.environ['EBOSS_CLUSTERING_DIR']+\
                    '/etc/systematic_maps_256nest.fits'):
        ''' read FITS file containing all healpix maps
                 
            nside: integer
                 nside of the output (should be less or equal to 256)
            maps_index: integer array
                 index of maps wanted
                 0 - stellar density
                 1 - i-band depth
                 2 - z sky flux
                 3 - z FWHM
                 4 - Extinction r-band
                 5 - Wise W1 CovMed
                 6 - Wise W1 Med
            infits: string
                path to fits file containing systematic maps

            Returns
            -------
            syst_maps: dict with systematic maps and information about them
        '''
        if nside>256:
            print('nside should be at most 256')
            return

        a = fits.open(infits)
        allsyst = a[0].data[maps_index]
        nsideh = a[0].header['NSIDE']
        names = a[1].data.NAMES[maps_index].astype(str)
        suffix = a[1].data.SUFFIX[maps_index].astype(str)
        nmaps = len(maps_index)

        #-- normalize star density into units per deg2 
        allsyst[0] *= (12*nside**2)/(4*N.pi*(180**2/N.pi**2))

        #-- modify nside of maps if needed
        if nside!=nsideh:
            nallsyst = N.zeros((nmaps, 12*nside**2))
            for i in range(nmaps):
                nallsyst[i] = hp.ud_grade(allsyst[i], nside, \
                                order_in='NESTED', order_out='NESTED')
            allsyst = nallsyst
           
        self.maps = allsyst
        self.maps_names = names
        self.maps_suffix = suffix
        self.nside = nside
        self.nmaps = nmaps


    def read_systematic_maps_ascii(nside=256, index=N.array([0, 6])):
        '''Similar to read_systematic_maps_fits() but reads from ASCII files. 
           It's longer! '''

        print('Reading healpix maps of systematics:')

        sysdir = os.environ['MKESAMPLE_DIR']+'/inputFiles/'

        healpixfiles = N.array([sysdir+'allstars17.519.9Healpixall256.dat', \
                        sysdir+'healdepthinm512.dat',               \
                        sysdir+'abhi_sdss_zsky_512nest.txt',        \
                        sysdir+'abhi_sdss_zfwhm_512nest.txt',       \
                        sysdir+'abhi_sdss_rext_512nest.txt',        \
                        sysdir+'abhi_wise_w1covmed_512nest.txt',    \
                        sysdir+'abhi_wise_w1med_512nest.txt'])
        #                sysdir+'abhi_wise_w1moon_512nest.txt' ]

        names = N.array(['stellar density', \
                       'i-band depth', \
                       'z sky flux', \
                       'z FWHM', \
                       'Extiction r-band', \
                       'Wise W1 Cov Med', \
                        'Wise W1 Med']) #, 'Wise W1 Moon' ]

        suffix = N.array([  '_stardensity', \
                    '_idepth', \
                    '_zsky', \
                    '_zfwhm', \
                    '_rext', \
                    '_w1covmed', \
                    '_w1med']) #, \
                    #'_w1moon']

        healpixfiles = healpixfiles[index]
        names = names[index]
        suffix = suffix[index]

        nmaps = len(healpixfiles)
        allsyst = N.zeros((nmaps, 12*nside**2))
        for i in range(nmaps):
            print('   %s'%os.path.basename(healpixfiles[i]))
            onemap = N.loadtxt(healpixfiles[i])
            onemap = hp.ud_grade(onemap, nside, order_in='NESTED', \
                        order_out='NESTED')
            allsyst[i] = onemap

        syst_maps = {'values':allsyst, 'names':names, 'suffix':suffix, \
                     'nmaps':nmaps, 'nside':nside}

        return syst_maps

    #-- create FITS file from ASCII tables
    def export_systematics(syst):
        '''Creates FITS file from systematic dict 
            
           syst: dict output from read_systematic_maps_ascii()
        '''

        hdulist = fits.HDUList()
        prihdu = fits.PrimaryHDU(syst['values'])
        h = prihdu.header
        h['NSYST'] = syst['nsyst']
        h['NSIDE'] = syst['nside']

        collist = list()
        collist.append(fits.Column(name='NAMES', format='20A', 
                                   array=syst['names']))
        collist.append(fits.Column(name='SUFFIX', format='20A', 
                                   array=syst['suffix']))
        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(collist))

        hdulist.append(prihdu)
        hdulist.append(tbhdu)
        hdulist.writeto(os.environ['MKESAMPLE_DIR']+\
                'inputFiles/systematic_maps_%dnest.fits'%syst['nside'], 
                clobber=True)


    def plot_systematic_map(self, index):

        maps = self.maps
        names = self.maps_names
        nmaps = names.size

        i = index
        w = (maps[i] != 0.)
        maps[i, ~w] = hp.UNSEEN
        hp.mollview(maps[i], nest=1, min=N.percentile(maps[i, w], 1), \
                        max=N.percentile(maps[i, w], 99.), title=names[i])

    def systematic_correlation_matrix(maps):
        '''Computes covariance between maps'''

        vsyst = N.var(maps, axis=1, ddof=1)
        return N.cov(maps)/N.sqrt(N.outer(vsyst, vsyst))

    def plot_systematic_correlation_matrix(syst):
        '''Plots the correlation matrix of systematic maps'''

        maps = syst['values']
        names = syst['names']

        corr = systematic_correlation_matrix(maps)
        fig = P.figure()
        P.pcolormesh(corr, vmin=-1, vmax=1)
        axes = fig.add_subplot(111)
        axes.set_xticks(N.arange(len(maps))+0.5)
        axes.set_yticks(N.arange(len(maps))+0.5)    
        axes.set_xticklabels(names, rotation=65)
        axes.set_yticklabels(names)
        P.colorbar()
        P.tight_layout()
  
     
    def get_map_values(self, index, ra, dec):

        pix = hp.ang2pix(self.nside, (-dec+90.)*N.pi/180, ra*N.pi/180, nest=1)
        return self.maps[index, pix] 

    def get_model_weights(self, pars, syst): 
        ''' Compute weights from parameters and systematic values
            Input
            ------
            pars : (fit_index.size+1) vector containing parameters of fit
            syst : (N_galaxies, N_syst) array containing systematic values
        '''
        edges = self.edges
        fit_index = self.fit_index
        #-- weights are a linear combination of maps
        #-- first parameters is a constant, others are slopes
        weights = pars[0]
        for i, imap in enumerate(fit_index):
            weights += pars[i+1]*(syst [:,  imap]-edges[imap, 0])/\
                                 (edges[imap, -1]-edges[imap, 0]) 
        return weights

    def prepare(self, nbins = 10):

        nmaps = self.nmaps

        n_data = self.ra_data.size
        n_rand = self.ra_rand.size
        
        we_data = self.weights_data
        we_rand = self.weights_rand

        #-- assign systematic values to all galaxies
        syst_data = N.zeros((n_data, nmaps))
        syst_rand = N.zeros((n_rand, nmaps))

        for i in range(nmaps):
            syst_data[:, i] = self.get_map_values(i, 
                                self.ra_data, self.dec_data)
            syst_rand[:, i] = self.get_map_values(i, 
                                self.ra_rand, self.dec_rand)

        #-- cut galaxies with extreme values of systematics
        w_data = N.ones(n_data) == 1
        w_rand = N.ones(n_rand) == 1
        for i in range(nmaps):
            syst_min = N.percentile(syst_rand[:, i], 0.2)
            syst_max = N.percentile(syst_rand[:, i], 99.8)
            w_data &= (syst_data[:, i] > syst_min) & \
                      (syst_data[:, i] < syst_max)
            w_rand &= (syst_rand[:, i] > syst_min) & \
                      (syst_rand[:, i] < syst_max)
            
        syst_data = syst_data[w_data, :]
        syst_rand = syst_rand[w_rand, :]
        we_data = we_data[w_data]
        we_rand = we_rand[w_rand]

        print('Data before/after cut: ', n_data, we_data.size)
        print('Rand before/after cut: ', n_rand, we_rand.size)

        #-- compute histograms        
        factor = N.sum(we_rand)/N.sum(we_data)
        edges      = N.zeros((nmaps, nbins+1))
        centers    = N.zeros((nmaps, nbins))
        h_data     = N.zeros((nmaps, nbins))
        h_rand     = N.zeros((nmaps, nbins))

        for i in range(nmaps):
            edges[i] = N.linspace(syst_rand[:, i].min(), \
                                  syst_rand[:, i].max(), nbins+1)
            centers[i] = 0.5*(edges[i][:-1]+edges[i][1:]) 
            h_data[i], _ = N.histogram(syst_data[:, i], bins=edges[i], \
                                       weights=we_data)
            h_rand[i], _ = N.histogram(syst_rand[:, i], bins=edges[i], \
                                       weights=we_rand)

        self.syst_data = syst_data
        self.syst_rand = syst_rand
        self.we_data = we_data
        self.we_rand = we_rand
        self.factor = factor
        self.edges = edges
        self.centers = centers
        self.h_data = h_data
        self.h_rand = h_rand
        self.dens = h_data/h_rand * factor
        self.edens = N.sqrt((h_data   /h_rand**2 + \
                        h_data**2/h_rand**3   )) * factor

    def get_histograms(self, pars=None):
        syst_data = self.syst_data
        syst_rand = self.syst_rand
        we_data = self.we_data
        we_rand = self.we_rand
        factor = self.factor
        edges = self.edges
        centers = self.centers
        h_data = self.h_data
        h_rand = self.h_rand
        
        if pars is None:
            pars = N.zeros(self.nmaps+1)
            pars[0] = 1.

        we_model = 1/self.get_model_weights(pars, syst_data)

        for i in range(self.nmaps):
            h_data[i], _ = N.histogram(syst_data[:, i], bins=edges[i], \
                                       weights=we_data*we_model)
        self.h_data = h_data
        if pars[0] != 1.:
            self.pars = pars
        self.dens = h_data/h_rand * factor
        self.edens = N.sqrt((h_data   /h_rand**2 + \
                        h_data**2/h_rand**3   )) * factor

    def get_chi2(self, pars=None):
        self.get_histograms(pars=pars)
        return N.sum((self.dens-1.)**2/self.edens**2)

    def plot_overdensity(self, ylim=[0.75, 1.25],\
                         nbinsh=50):

        centers = self.centers
        names = self.maps_names
        nmaps = self.nmaps
        nbins = centers[0].size
        syst_data = self.syst_data
   
        #-- if the fit has been done, plot both before and after fits 
        pars = [None, self.pars] if hasattr(self, 'pars') else [None]

        #-- setting up the windows
        figsize = (15, 3) if nmaps > 1 else (5,3)
        f, ax = P.subplots(1, nmaps, sharey=True, figsize=figsize)
        if nmaps == 1:
            ax = [ax] 
        if nmaps > 1: 
            f.subplots_adjust(wspace=0.05, left=0.05, right=0.98, 
                              top=0.98, bottom=0.15)
        ax[0].set_ylim(ylim)

        #-- compute histograms for before/after parameters
        for par in pars:
            self.get_histograms(pars=par)
            dens = self.dens
            edens = self.edens
            for i in range(nmaps):
             
                chi2 = N.sum( (dens[i]-1.)**2/edens[i]**2)
                label = r'$\chi^2_{r}  = %.1f/%d = %.2f$'%\
                     (chi2, nbins, chi2/nbins)

                ax[i].errorbar(centers[i], dens[i], edens[i], \
                                    fmt='.', label=label)
                ax[i].axhline( 1.0, color='k', ls='--')
                ax[i].locator_params(axis='x', nbins=5, tight=True)
                
                #-- add title and legend
                ax[i].legend(loc=0, numpoints=1, fontsize=10)
                ax[i].set_xlabel(names[i])

        #-- overplot histogram (normalizing to the 1/3 of the y-axis)
        for i in range(nmaps):
            h_syst, bins = N.histogram(syst_data[:, i], bins=nbinsh)

            x = 0.5*(bins[:-1]+bins[1:])
            y = h_syst/h_syst.max()*0.3*(ylim[1]-ylim[0])+ylim[0]
            ax[i].step(x, y, where='mid', color='g')

        ax[0].set_ylabel('Density fluctuations')

    def fit_pars(self):

        pars0 = N.zeros(self.fit_index.size+1)
        pars0[0] = 1.
        success = False
        ntries = 0
        while success is False and ntries < 3:
            ntries += 1
            print('Fitting parameters - trial #%d of 3'%ntries)
            pars_object = minimize(self.get_chi2, pars0, \
                                   method='Nelder-Mead')
            print(pars_object['message'])
            success = pars_object['success']
            pars0 = pars_object['x']

        self.pars = pars_object['x']
         
    def get_weights(self, ra, dec):
        nmaps = self.nmaps
        syst_values = N.zeros((ra.size, nmaps))

        for i in range(nmaps):
            syst_values[:, i] = self.get_map_values(i, ra, dec) 
       
        return 1/self.get_model_weights(self.pars, syst_values)
        


