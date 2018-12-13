from __future__ import print_function

import os, sys
import numpy as np
import pylab as plt
import healpy as hp
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import minimize
import argparse

class MultiLinearFit:

    def __init__(self, data_ra=None, data_dec=None, data_we=None, 
                       rand_ra=None, rand_dec=None, rand_we=None,
                       maps=['STAR_DENSITY',
                             'SKY_Z', 
                             'AIRMASS', 
                             'EBV', 
                             'DEPTH_Z', 
                             'PSF_Z', 
                             'W1_MED', 
                             'W1_COVMED'], 
                       fit_maps=None, nbins_per_syst=10,
                       infits=None, nest=False):
        ''' Systematic fits 

        ''' 

        print('Using the following maps:      ', maps)
        #print('Fitting for the following maps:', fit_maps)

        if infits is None:
            infits='/mnt/lustre/bautista/software/eboss_clustering/etc/'+\
                   'SDSS_WISE_imageprop_nside512.fits'
        if not self.read_systematics_maps(maps=maps, infits=infits): sys.exit()
        self.nest = nest


        self.data_ra = data_ra
        self.data_dec= data_dec
        self.data_we = data_we

        self.rand_ra = rand_ra
        self.rand_dec = rand_dec
        self.rand_we = rand_we
        
        if data_ra.size != data_dec.size != data_we.size:
            print('ERROR: data_ra, data_dec, data_we do not have same size')
            sys.exit()
        if rand_ra.size != rand_dec.size != rand_we.size:
            print('ERROR: data_ra, data_dec, data_we do not have same size')
            sys.exit()

        self.prepare(nbins=nbins_per_syst)

    def flux_to_mag(self, flux, band, ebv=None):

        #-- coefs to convert from flux to magnitudes
        b = np.array([1.4, 0.9, 1.2, 1.8, 7.4])[band]*1e-10
        mag = -2.5/np.log(10)*(np.arcsinh((flux/1e9)/(2*b)) + np.log(b))

        #-- extinction coefficients for SDSS u, g, r, i, and z bands
        ext_coeff = np.array([4.239, 3.303, 2.285, 1.698, 1.263])[band]
        if not ebv is None:
            mag -= ext_coeff*ebv

        return mag

    def read_systematics_maps(self, \
        infits='/mnt/lustre/bautista/software/eboss_clustering/etc/'+\
               'SDSS_WISE_imageprop_nside512.fits', \
        maps=None):
      
        print('Reading maps from:', infits)  
        a = fits.open(infits)[1].data
        print('Available maps:', a.names)

        #-- convert depth
        if 'DEPTH_G' in a.names:
            a.DEPTH_G = self.flux_to_mag(a.DEPTH_G, 1, ebv=a.EBV)
            a.DEPTH_R = self.flux_to_mag(a.DEPTH_R, 2, ebv=a.EBV)
            a.DEPTH_I = self.flux_to_mag(a.DEPTH_I, 3, ebv=a.EBV)
            a.DEPTH_Z = self.flux_to_mag(a.DEPTH_Z, 4, ebv=a.EBV)
 
        nmaps = len(maps)
        npix  = a.size
        syst = np.zeros((nmaps, npix)) 
        
        for i in range(len(maps)):
            try:
                syst[i] = a.field(maps[i])
            except:
                print('ERROR reading %s. Available choices:'%maps[i], a.names) 
                return False

        w = np.isnan(syst)|np.isinf(syst)
        syst[w] = hp.UNSEEN

        self.maps = syst 
        self.maps_names = maps
        self.maps_suffix = maps
        self.nside = hp.npix2nside(npix) 
        self.nmaps = nmaps
        return True

    def plot_systematic_map(self, index):

        syst = self.maps[index]
        name = self.maps_names[index]
        w = syst != hp.UNSEEN
        hp.mollview(syst, title=name, 
                    min=np.percentile(syst[w], 1), 
                    max=np.percentile(syst[w], 99.), 
                    nest=self.nest)

    def get_pix(self, ra, dec):
        return hp.ang2pix(self.nside, \
                          (-dec+90.)*np.pi/180, \
                          ra*np.pi/180,
                          nest=self.nest)

    def get_map_values(self, index, ra, dec):
        pix = self.get_pix(ra, dec)
        return self.maps[index, pix] 

    def get_model(self, pars, syst): 
        ''' Compute model from parameters and systematic values
            Input
            ------
            pars : (fit_index.size+1) vector containing parameters of fit
            syst : (N_galaxies, N_syst) array containing systematic values
        '''
        edges = self.edges
        fit_index = self.fit_index
        #-- model is a linear combination of maps
        #-- first parameters is a constant, others are slopes
        model = pars[0]
        model += np.sum(pars[1:]*(syst[:, fit_index]  -edges[fit_index, 0])/\
                                 (edges[fit_index, -1]-edges[fit_index, 0]), axis=1)
        return model

    def prepare(self, nbins = 10):
        ''' Prepare histograms and cut outlier values of systematics'''

        nmaps = self.nmaps

        data_we = self.data_we
        rand_we = self.rand_we
        n_data = data_we.size
        n_rand = rand_we.size

        #-- assign systematic values to all galaxies and randoms
        data_pix = self.get_pix(self.data_ra, self.data_dec) 
        rand_pix = self.get_pix(self.rand_ra, self.rand_dec) 
        data_syst = self.maps[:, data_pix].T
        rand_syst = self.maps[:, rand_pix].T 

        #-- cut galaxies and randoms with extreme values of systematics
        w_data = np.ones(n_data) == 1
        w_rand = np.ones(n_rand) == 1
        for i in range(nmaps):
            syst_min = np.percentile(rand_syst[:, i], 0.5) #22.4
            syst_max = np.percentile(rand_syst[:, i], 99.5) #23.8
            w_data &= (data_syst[:, i] > syst_min) & \
                      (data_syst[:, i] < syst_max)
            w_rand &= (rand_syst[:, i] > syst_min) & \
                      (rand_syst[:, i] < syst_max)
        data_syst = data_syst[w_data, :]
        rand_syst = rand_syst[w_rand, :]
        data_we = data_we[w_data]
        rand_we = rand_we[w_rand]
        self.data_ra = self.data_ra[w_data]
        self.data_dec = self.data_dec[w_data]

        print('Number of galaxies before/after cutting outliers: ', n_data, data_we.size)
        print('Number of randoms  before/after cutting outliers: ', n_rand, rand_we.size)

        #-- compute histograms        
        factor = np.sum(rand_we)/np.sum(data_we)
        edges      = np.zeros((nmaps, nbins+1))
        centers    = np.zeros((nmaps, nbins))
        h_data     = np.zeros((nmaps, nbins))
        h_rand     = np.zeros((nmaps, nbins))

        for i in range(nmaps):
            edges[i] = np.linspace(data_syst[:, i].min()-1e-7, \
                                   data_syst[:, i].max()+1e-7, \
                                   nbins+1)
            #edges[i] = np.linspace(22.4, 23.8, nbins+1)
            centers[i] = 0.5*(edges[i][:-1]+edges[i][1:]) 
            h_data[i], _ = np.histogram(data_syst[:, i], bins=edges[i], \
                                        weights=data_we)
            h_rand[i], _ = np.histogram(rand_syst[:, i], bins=edges[i], \
                                        weights=rand_we)

        h_index = np.floor((data_syst   -edges[:, 0])/\
                           (edges[:, -1]-edges[:, 0])*nbins).astype(int)

        self.data_syst = data_syst
        self.rand_syst = rand_syst
        self.data_we = data_we
        self.rand_we = rand_we
        self.factor = factor
        self.edges = edges
        self.centers = centers
        self.h_index = h_index
        self.h_data = h_data
        self.h_rand = h_rand
        
        #-- computing overdensity and error assuming poisson
        self.dens = h_data/h_rand * factor
        self.edens = np.sqrt((h_data   /h_rand**2 + \
                              h_data**2/h_rand**3   )) * factor
        self.chi2_before = np.sum((self.dens-1.)**2/self.edens**2)
        self.ndata = h_data.size
        self.rchi2_before = self.chi2_before/self.ndata

    def get_histograms(self, pars=None):
        data_syst = self.data_syst
        data_we = self.data_we
        h_data = self.h_data
        h_rand = self.h_rand
        h_index = self.h_index

        if pars is None:
            we_model = data_we*0+1
        else:
            we_model = 1/self.get_model(pars, data_syst)

        #-- doing histograms with np.bincount, it's faster
        for i in range(self.nmaps):
            h_data[i] = np.bincount(h_index[:, i], weights=data_we*we_model)

        if not pars is None:
            self.pars = pars
        self.h_data = h_data

        #-- computing overdensity and error assuming poisson
        self.dens = h_data/h_rand * self.factor
        self.edens = np.sqrt((h_data   /h_rand**2 + \
                              h_data**2/h_rand**3   )) * self.factor

    def get_chi2(self, pars=None):
        self.get_histograms(pars=pars)
        return np.sum((self.dens-1.)**2/self.edens**2)

    def plot_overdensity(self, ylim=[0.75, 1.25],\
                         nbinsh=50, title=None):

        centers = self.centers
        names = self.maps_names
        nmaps = self.nmaps
        nbins = centers[0].size
        data_syst = self.data_syst
   
        #-- if the fit has been done, plot both before and after fits 
        pars = [None, self.pars] if hasattr(self, 'pars') else [None]

        #-- setting up the windows
        figsize = (15, 3) if nmaps > 1 else (5,3)
        f, ax = plt.subplots(1, nmaps, sharey=True, figsize=figsize)
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
             
                chi2 = np.sum( (dens[i]-1.)**2/edens[i]**2)
                label = r'$\chi^2_{r}  = %.1f/%d = %.2f$'%\
                     (chi2, nbins, chi2/nbins)

                ax[i].errorbar(centers[i], dens[i], edens[i], \
                                    fmt='.', label=label)
                ax[i].axhline( 1.0, color='k', ls='--')
                ax[i].locator_params(axis='x', nbins=5, tight=True)
                
                #-- add title and legend
                ax[i].legend(loc=0, numpoints=1, fontsize=8)
                ax[i].set_xlabel(names[i])

        #-- overplot histogram (normalizing to the 1/3 of the y-axis)
        for i in range(nmaps):
            h_syst, bins = np.histogram(data_syst[:, i], bins=nbinsh)

            x = 0.5*(bins[:-1]+bins[1:])
            y = h_syst/h_syst.max()*0.3*(ylim[1]-ylim[0])+ylim[0]
            ax[i].step(x, y, where='mid', color='g')

        ax[0].set_ylabel('Density fluctuations')
        if title:
            f.subplots_adjust(top=0.9)
            plt.suptitle(title)

    def fit_pars(self, fit_maps=None):

        #-- If fit_maps is None, fit all maps 
        #-- Otherwise, define indices of maps to be fitted
        if fit_maps is None:
            fit_maps = self.maps_names
            fit_index = np.arange(len(fit_maps))
        else:
            maps = self.maps_names
            fit_index = []
            for i in range(len(maps)):
                if maps[i] in fit_maps:
                    fit_index.append(i)
            fit_index = np.array(fit_index)
        self.fit_index = fit_index
        self.fit_maps = fit_maps
 
        pars0 = np.zeros(self.fit_index.size+1)
        pars0[0] = 1.

        #-- try to coverge three times at most
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
        
        chi2_before = self.get_chi2()
        ndata = self.dens.size
        npars = pars0.size
        rchi2_before = chi2_before/(ndata-0)
        print('Before fit: chi2/(ndata-npars) = %.2f/(%d-%d) = %.3f'%\
               (chi2_before, ndata, 0, rchi2_before ))
        chi2_after = self.get_chi2(self.pars)
        rchi2_after = chi2_after/(ndata-npars)
        print('After fit:  chi2/(ndata-npars) = %.2f/(%d-%d) = %.3f'%\
               (chi2_after, ndata, npars, rchi2_after ))
        self.chi2_before = chi2_before
        self.rchi2_before = rchi2_before
        self.ndata = ndata
        self.npars = npars
        self.chi2_after = chi2_after
        self.rchi2_after = rchi2_after
         
    def get_weights(self, ra, dec):
        pix  = self.get_pix(ra, dec)
        syst = self.maps[:, pix].T 
        if not hasattr(self, 'pars'): 
            self.fit_pars()
        return 1/self.get_model(self.pars, syst)

    def export(self, fout):

        fout = open(fout, 'w')
        nbins = self.centers[0].size
       
        print('#- Photometric systematic fit output by Julian Bautista', file=fout) 
        print('#- Number of systematic maps read: %d'%self.nmaps, file=fout)
        print('#- Number of systematic maps fitted: %d'%len(self.fit_maps), file=fout)
        print('#- Number of bins per systematic: %d'%nbins, file=fout)
        print('#- Maps read:', file=fout)
        print('#- '+' '.join(self.maps_names), file=fout)
        print('#- Maps fitted:', file=fout)
        print('#- '+' '.join(self.fit_maps), file=fout)
        print('#- chi2 ndata npars rchi2 (before fit)', file=fout)
        print('#- %f %d %d %f'%(self.chi2_before, self.ndata, 0, self.rchi2_before), file=fout)
        print('#- chi2 ndata npars rchi2 (after fit)', file=fout)
        print('#- %f %d %d %f'%(self.chi2_after, self.ndata, self.npars, self.rchi2_after), file=fout)
        print('#-- Parameters (const + slope per fitted systematic)', file=fout)
        print('#--', self.pars, file=fout) 

        self.get_histograms()
        dens_before = self.dens*1
        edens_before = self.edens*1
        self.get_histograms(pars=self.pars)
        dens_after = self.dens
        edens_after = self.edens

        for j in range(self.nmaps):
            sname = self.maps_names[j]
            line = '#- %s_min  %s_cen  %s_max'%\
                   (sname, sname, sname)
            line += '  delta_before  edelta_before  delta_after  edelta_after \t'
            print(line, file=fout)

            for i in range(nbins):
                smin = self.edges[j, i]
                smax = self.edges[j, i+1]
                scen = self.centers[j, i]
                den0 =   dens_before[j, i]
                eden0 = edens_before[j, i]
                den1 =   dens_after[j, i]
                eden1 = edens_after[j, i]
                line = '%f \t %f \t %f \t %f \t %f \t %f \t %f'%\
                       (smin, scen, smax, den0, eden0, den1, eden1)
                print(line, file=fout)

        fout.close()


 
