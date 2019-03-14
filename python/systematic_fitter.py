from __future__ import print_function

import os, sys
import numpy as np
import pylab as plt
import healpy as hp
from astropy.io import fits
from astropy.table import Table
import iminuit
import iminuit.frontends
from scipy.optimize import minimize

class Syst:

    def __init__(self, data_we, rand_we):
        self.data_we = data_we 
        self.rand_we = rand_we
        self.data_syst = []
        self.rand_syst = []
        self.syst_names = []
        self.nsyst = 0
        self.ndata = data_we.size
        self.nrand = rand_we.size

    def add_syst(self, name, data_syst, rand_syst):
        assert( data_syst.size == self.ndata ) 
        assert( rand_syst.size == self.nrand )
        assert( name not in self.syst_names )
        self.syst_names.append(name)
        wd = np.isnan(data_syst)|np.isinf(data_syst)
        data_syst[wd] = hp.UNSEEN 
        self.data_syst.append(data_syst)
        wr = np.isnan(rand_syst)|np.isinf(rand_syst)
        rand_syst[wr] = hp.UNSEEN 
        self.rand_syst.append(rand_syst)
        self.nsyst += 1

    def cut_outliers(self, p=1.):
        ''' Cut galaxies and randoms with extreme values of systematics '''

        #-- make arrays
        self.data_syst = np.array(self.data_syst)
        self.rand_syst = np.array(self.rand_syst)

        w_data = np.ones(self.ndata) == 1
        w_rand = np.ones(self.nrand) == 1
        for i in range(self.nsyst):
            name = self.syst_names[i]
            data_syst = self.data_syst[i]
            rand_syst = self.rand_syst[i]
            w = data_syst!=hp.UNSEEN
            syst_min = np.percentile(data_syst[w], p/2) #22.4
            syst_max = np.percentile(data_syst[w], 100-p/2) #23.8
            w_data &= (data_syst >= syst_min) & \
                      (data_syst <= syst_max)
            w_rand &= (rand_syst >= syst_min) & \
                      (rand_syst <= syst_max)
            print(' cutting ', name, 'from', syst_min, 'to', syst_max)
            
        print('Number of galaxies before/after cutting outliers: ', w_data.size, np.sum(w_data))
        print('Number of randoms  before/after cutting outliers: ', w_rand.size, np.sum(w_rand))
        self.data_syst = self.data_syst[:, w_data]
        self.rand_syst = self.rand_syst[:, w_rand]
        self.data_we = self.data_we[w_data]
        self.rand_we = self.rand_we[w_rand]
        self.w_data = w_data
        self.w_rand = w_rand
        self.ndata = self.data_we.size
        self.nrand = self.rand_we.size

    def prepare(self, nbins=10):

        nsyst = self.nsyst
        data_syst = self.data_syst
        rand_syst = self.rand_syst

        #-- compute histograms        
        edges      = np.zeros((nsyst, nbins+1))
        centers    = np.zeros((nsyst, nbins))
        h_rand     = np.zeros((nsyst, nbins))

        for i in range(nsyst):
            edges[i] = np.linspace(data_syst[i].min()-1e-7, \
                                   data_syst[i].max()+1e-7, \
                                   nbins+1)
            centers[i] = 0.5*(edges[i][:-1]+edges[i][1:]) 
            h_rand[i], _ = np.histogram(rand_syst[i], bins=edges[i], 
                                        weights=self.rand_we)

        h_index = np.floor((data_syst.T -edges[:, 0])/\
                           (edges[:, -1]-edges[:, 0])*nbins).astype(int).T

        self.factor = np.sum(self.rand_we)/np.sum(self.data_we)
        self.edges = edges
        self.centers = centers
        self.h_index = h_index
        self.h_rand = h_rand
        self.nbins = nbins
        
    def get_subsample(self, wd, wr):
        wd = wd[self.w_data]
        wr = wr[self.w_rand]
        s = Syst(self.data_we[wd], self.rand_we[wr])
        for i in range(self.nsyst):
            s.add_syst(self.syst_names[i], self.data_syst[i, wd], self.rand_syst[i, wr])
        s.cut_outliers(p=0)
        s.prepare(nbins=self.nbins)
        s.edges = self.edges
        s.centers = self.centers

        for i in range(s.nsyst):
            s.h_rand[i], _ = np.histogram(s.rand_syst[i], bins=s.edges[i], 
                                          weights=s.rand_we)

        s.h_index = np.floor((s.data_syst.T -s.edges[:, 0])/\
                             (s.edges[:, -1]-s.edges[:, 0])*s.nbins).astype(int).T

        return s
        
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
            model += np.sum(pars[1:]*(syst[fit_index, :].T  -edges[fit_index, 0])/\
                                     (edges[fit_index, -1]  -edges[fit_index, 0]), axis=1)
            return model


    def get_histograms(self, pars=None):
        data_syst = self.data_syst
        data_we = self.data_we
        
        h_rand = self.h_rand
        h_index = self.h_index

        h_data = h_rand*0

        if pars is None:
            we_model = data_we*0+1
        else:
            we_model = 1/self.get_model(pars, data_syst)

        #-- doing histograms with np.bincount, it's faster
        for i in range(self.nsyst):
            h_data[i] = np.bincount(h_index[i], weights=data_we*we_model)

        if not pars is None:
            self.pars = pars

        #-- computing overdensity and error assuming poisson
        self.h_data = h_data
        self.dens = h_data/h_rand * self.factor
        self.edens = np.sqrt((h_data   /h_rand**2 + \
                              h_data**2/h_rand**3   )) * self.factor
        return self.dens, self.edens

    def plot_overdensity(self, ylim=[0.75, 1.25], 
        nbinsh=50, title=None):

        #-- if the fit has been done, plot both before and after fits 
        pars = [None, self.pars] if hasattr(self, 'pars') else [None]

        #-- setting up the windows
        nmaps = self.nsyst
        figsize = (15, 3) if nmaps > 1 else (5,3)
        f, ax = plt.subplots(1, nmaps, sharey=True, figsize=figsize)
        if nmaps == 1:
            ax = [ax] 
        if nmaps > 1: 
            f.subplots_adjust(wspace=0.05, left=0.05, right=0.98, 
                              top=0.98, bottom=0.15)
        ax[0].set_ylim(ylim)

        #-- compute histograms for before/after parameters
        nbins = self.nbins
        for par in pars:
            dens, edens = self.get_histograms(pars=par)
            for i in range(nmaps):
                chi2 = np.sum( (dens[i]-1.)**2/edens[i]**2)
                label = r'$\chi^2_{r}  = %.1f/%d = %.2f$'%\
                     (chi2, nbins, chi2/nbins)
                ax[i].errorbar(self.centers[i], dens[i], edens[i], fmt='.', label=label)
                ax[i].axhline( 1.0, color='k', ls='--')
                ax[i].locator_params(axis='x', nbins=5, tight=True)
                
                #-- add title and legend
                ax[i].legend(loc=0, numpoints=1, fontsize=8)
                ax[i].set_xlabel(self.syst_names[i])

        #-- overplot histogram (normalizing to the 1/3 of the y-axis)
        for i in range(nmaps):
            h_syst, bins = np.histogram(self.data_syst[i], bins=nbinsh)
            x = 0.5*(bins[:-1]+bins[1:])
            y = h_syst/h_syst.max()*0.3*(ylim[1]-ylim[0])+ylim[0]
            ax[i].step(x, y, where='mid', color='g')

        ax[0].set_ylabel('Density fluctuations')
        if title:
            f.subplots_adjust(top=0.9)
            plt.suptitle(title)

    def get_chi2(self, *pars):
        pars = None if len(pars) == 0 else pars
        dens, edens = self.get_histograms(pars=pars)
        return np.sum((dens-1.)**2/edens**2)

    def fit_pars(self, fit_maps=None):

        #-- If fit_maps is None, fit all maps 
        #-- Otherwise, define indices of maps to be fitted
        if fit_maps is None:
            fit_maps = self.syst_names
            fit_index = np.arange(len(fit_maps))
        else:
            maps = self.syst_names
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
        
        chi2_before = 0 #self.get_chi2(pars0)
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

    def fit_minuit(self, fit_maps=None, fixes=None, limits=None, priors=None):

        #-- If fit_maps is None, fit all maps 
        #-- Otherwise, define indices of maps to be fitted
        if fit_maps is None:
            fit_maps = self.syst_names
            fit_index = np.arange(len(fit_maps), dtype=int)
        else:
            maps = self.syst_names
            fit_index = []
            fit_maps_ordered = []
            for i in range(len(maps)):
                if maps[i] in fit_maps:
                    fit_index.append(i)
                    fit_maps_ordered.append(maps[i])
            fit_index = np.array(fit_index, dtype=int)
            fit_maps = np.array(fit_maps_ordered)
        self.fit_index = fit_index
        self.fit_maps = fit_maps

        par_names = []
        init_pars = {}
        par_names.append('constant')
        init_pars['constant'] = 1.
        init_pars['error_constant'] = 0.1
        for par in self.fit_maps:
            value = 0
            init_pars[par] = value
            init_pars['error_'+par] = abs(value)/10. if value!=0 else 0.1
            par_names.append(par)

        self.fixes = fixes
        if fixes:
            for key in fixes.keys():
                init_pars[key] = fixes[key]
                init_pars['fix_'+key] = True 
        if limits:
            for key in limits.keys():
                init_pars['limit_'+key] = (limits[key][0], limits[key][1])

        self.priors = priors

        mig = iminuit.Minuit(self.get_chi2, throw_nan=False, \
                             forced_parameters=par_names, \
                             print_level=1, errordef=1, \
                             frontend=iminuit.frontends.ConsoleFrontend(), \
                             **init_pars)

        mig.tol = 1.0 
        imin = mig.migrad()
        self.mig = mig
        self.imin = imin
        self.is_valid = imin[0]['is_valid']
        self.best_pars = mig.values 
        self.errors = mig.errors
        self.chi2min = mig.fval
        self.ndata = self.dens.size
        self.npars = mig.narg
        self.covariance = mig.covariance
        for par in par_names:
            if mig.fitarg['fix_'+par]:
                self.npars -= 1
        self.rchi2min = self.chi2min/(self.ndata-self.npars)
        print('chi2 = %.2f   ndata = %d   npars = %d   rchi2 = %.4f'%\
                (self.chi2min, self.ndata, self.npars, self.rchi2min))

    def export(self, fout):

        fout = open(fout, 'w')
        nbins = self.centers[0].size
       
        dens_before, edens_before = self.get_histograms()
        dens_after,  edens_after  = self.get_histograms(pars=self.pars)
        chi2_before = np.sum( (dens_before-1)**2/edens_before**2)
        chi2_after  = np.sum( (dens_after -1)**2/edens_after**2 )
        rchi2_before = chi2_before/(self.ndata)
        rchi2_after  = chi2_after /(self.ndata-self.npars)
        
        print('#- Photometric systematic fit output by Julian Bautista', file=fout) 
        print('#- Number of systematic maps read: %d'%self.nsyst, file=fout)
        print('#- Number of systematic maps fitted: %d'%len(self.fit_maps), file=fout)
        print('#- Number of bins per systematic: %d'%nbins, file=fout)
        print('#- Maps read:', file=fout)
        print('#- '+' '.join(self.syst_names), file=fout)
        print('#- Maps fitted:', file=fout)
        print('#- '+' '.join(self.fit_maps), file=fout)
        print('#- chi2 ndata npars rchi2 (before fit)', file=fout)
        print('#- %f %d %d %f'%(chi2_before, self.ndata, 0, rchi2_before), file=fout)
        print('#- chi2 ndata npars rchi2 (after fit)', file=fout)
        print('#- %f %d %d %f'%(chi2_after, self.ndata, self.npars, rchi2_after), file=fout)
        print('#-- Parameters (const + slope per fitted systematic)', file=fout)
        print('#--', self.pars, file=fout) 

        
 
        for j in range(self.nsyst):
            sname = self.syst_names[j]
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

