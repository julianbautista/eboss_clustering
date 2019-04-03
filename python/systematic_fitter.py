from __future__ import print_function

import os, sys
import numpy as np
import pylab as plt
import healpy as hp
from astropy.io import fits
from astropy.table import Table
from iminuit import Minuit
import iminuit.frontends
from scipy.optimize import minimize

class Syst:

    def __init__(self, data_we, rand_we):
        self.data_we = data_we 
        self.rand_we = rand_we
        self.data_syst = {}
        self.rand_syst = {}
        self.syst_names = []
        self.nsyst = 0
        self.ndata = data_we.size
        self.nrand = rand_we.size

    def add_syst(self, name, data_syst, rand_syst):
        assert( data_syst.size == self.ndata ) 
        assert( rand_syst.size == self.nrand )
        assert( name not in self.syst_names )
        #-- checking bad values of systematics
        wd = np.isnan(data_syst)|np.isinf(data_syst)
        data_syst[wd] = hp.UNSEEN 
        wr = np.isnan(rand_syst)|np.isinf(rand_syst)
        rand_syst[wr] = hp.UNSEEN 

        self.syst_names.append(name)
        self.data_syst[name] = data_syst
        self.rand_syst[name] = rand_syst
        self.nsyst += 1

    def cut_outliers(self, p=1., verbose=False):
        ''' Cut galaxies and randoms with extreme values of systematics '''

        if p==0:
            return 

        print(f'Cutting {p}% of extreme values in each map:') 
        w_data = np.ones(self.ndata, dtype=bool)
        w_rand = np.ones(self.nrand, dtype=bool) 
        for name in self.syst_names:
            data_syst = self.data_syst[name]
            rand_syst = self.rand_syst[name]
    
            w = data_syst!=hp.UNSEEN
            syst_min = np.percentile(data_syst[w], p/2) #22.4
            syst_max = np.percentile(data_syst[w], 100-p/2) #23.8

            w_data &= (data_syst >= syst_min) & \
                      (data_syst <= syst_max)
            w_rand &= (rand_syst >= syst_min) & \
                      (rand_syst <= syst_max)
            if verbose:
                print(' ', name, 'from', syst_min, 'to', syst_max)
            
        
        #-- Applying cuts and updating arrays
        for name in self.syst_names:
            self.data_syst[name] = self.data_syst[name][w_data] 
            self.rand_syst[name] = self.rand_syst[name][w_rand] 
        self.data_we = self.data_we[w_data]
        self.rand_we = self.rand_we[w_rand]
        self.w_data = w_data
        self.w_rand = w_rand
        self.ndata = self.data_we.size
        self.nrand = self.rand_we.size
        self.factor = np.sum(self.rand_we)/np.sum(self.data_we)
        if verbose:
            print('Number of galaxies before/after cutting outliers: ', 
                  w_data.size, np.sum(w_data))
            print('Number of randoms  before/after cutting outliers: ', 
                  w_rand.size, np.sum(w_rand))

    def prepare(self, nbins=10):

        nsyst = self.nsyst
        data_syst = self.data_syst
        rand_syst = self.rand_syst

        #-- compute histograms        
        edges      = {}
        centers    = {}
        h_rand     = {}
        h_index    = {}
        edelta     = {}
        
        for name in data_syst:
            syst = data_syst[name]
            edg = np.linspace(syst.min()-1e-7, syst.max()+1e-7, nbins+1)
            cen = 0.5*(edg[:-1]+edg[1:]) 
            h_index[name] = np.floor((syst   -edg[0])/\
                                     (edg[-1]-edg[0]) * nbins).astype(int).T
            h_rand[name], _ = np.histogram(rand_syst[name], bins=edg, 
                                        weights=self.rand_we)
            h_dat = np.bincount(h_index[name], weights=self.data_we)
                
            edges[name] = edg
            centers[name] = cen
            edelta[name] =  np.sqrt((h_dat   /h_rand[name]**2 + \
                                     h_dat**2/h_rand[name]**3 )) * self.factor \
                            + 1e10*(h_dat==0)

        self.edges = edges
        self.centers = centers
        self.h_index = h_index
        self.h_rand = h_rand
        self.edelta = edelta
        self.nbins = nbins
        
    def get_subsample(self, wd):
        wd = wd[self.w_data]
        s = Syst(self.data_we[wd], self.rand_we)
        for name in self.data_syst:
            s.add_syst(name, self.data_syst[name][wd], self.rand_syst[name])
        s.nbins = self.nbins
        s.factor = np.sum(s.rand_we)/np.sum(s.data_we)
        s.edges = self.edges
        s.centers = self.centers
        s.h_rand = self.h_rand
        s.h_index = {}
        s.edelta  = {}
        for name in s.data_syst:
            syst = s.data_syst[name]
            edg = s.edges[name]
            s.h_index[name] = np.floor((syst   -edg[0])/\
                                       (edg[-1]-edg[0]) * s.nbins).astype(int).T
            h_dat = np.bincount(s.h_index[name], weights=s.data_we,
                                minlength=s.nbins)
            s.edelta[name] =  np.sqrt((h_dat   /s.h_rand[name]**2 + \
                                       h_dat**2/s.h_rand[name]**3 )) * s.factor \
                            + 1e10*(h_dat==0)

        return s
        
    def get_model(self, pars, syst): 
        ''' Compute model from parameters and systematic values
            Input
            ------
            pars : dictionary containing parameters of fit
            syst : dictionary containing systematic values
        '''

        #-- same but using dictionary
        model = 1.+pars['constant']
        for p in pars:
            if p == 'constant': continue
            edges = self.edges[p]
            edgemin, edgemax = edges[0], edges[-1]
            model += pars[p]* (syst[p]-edgemin)/(edgemax-edgemin)
        return model

    def get_histograms(self, pars=None):
        data_syst = self.data_syst
        data_we = self.data_we
        
        h_rand = self.h_rand
        h_index = self.h_index

        h_data = {} 
        delta = {}

        if pars is None:
            we_model = data_we*0+1
        else:
            we_model = 1/self.get_model(pars, data_syst)

        #-- doing histograms with np.bincount, it's faster
        for name in data_syst:
            h_dat = np.bincount(h_index[name], weights=data_we*we_model, 
                                minlength=self.nbins)
            h_ran = h_rand[name]
            #-- computing overdensity and error assuming poisson
            delt = h_dat/h_ran * self.factor
            #edelt = np.sqrt((h_dat   /h_ran**2 + \
            #                 h_dat**2/h_ran**3 )) * self.factor
            h_data[name] = h_dat
            delta[name] = delt
            #edelta[name] = edelt

        self.h_data = h_data
        self.delta = delta


    def get_chi2(self, *pars):
        ''' Computes chi2 for a set of parameters 
            - for minuit fitter, pars is a tuple
            - but usually it is easy to give a dictionary
            - if no argument is give, compute chi2 for constant=1 and zero slopes
        '''
        if len(pars) == 0: 
            self.get_histograms()
        elif isinstance(pars[0], dict):
            self.get_histograms(pars=pars[0])
        else:
            pars_dict = {}
            for par_name, p in zip(self.par_names, list(pars)):
                pars_dict[par_name] = p
            self.get_histograms(pars=pars_dict)

        chi2 = 0.
        for name in self.syst_names:
            chi2+= np.sum( (self.delta[name]-1)**2/self.edelta[name]**2)
        return chi2

    def fit_minuit(self, fit_maps=None, fixes=None, limits=None, priors=None):

        #-- If fit_maps is None, fit all maps 
        #-- Otherwise, define indices of maps to be fitted
        if fit_maps is None:
            fit_maps = self.syst_names
            #fit_index = np.arange(len(fit_maps), dtype=int)
        else:
            maps = self.syst_names
            for fit_map in fit_maps:
                if fit_map not in maps:
                    print(fit_map, 'not available for fitting')
                    fit_maps.remove(fit_map)
            #fit_index = []
            #fit_maps_ordered = []
            #for i in range(len(maps)):
            #    if maps[i] in fit_maps:
            #        fit_index.append(i)
            #        fit_maps_ordered.append(maps[i])
            #fit_index = np.array(fit_index, dtype=int)
            #fit_maps = np.array(fit_maps_ordered)
        #self.fit_index = fit_index
        self.fit_maps = fit_maps

        par_names = []
        init_pars = {}
        par_names.append('constant')
        init_pars['constant'] = 0.
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
        self.par_names = par_names
        

        print('Maps available for chi2:')
        print(self.syst_names)
        print('Fitting for:')
        print(self.par_names)

        mig = Minuit(self.get_chi2, throw_nan=False, \
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
        self.ndata = self.nbins*self.nsyst
        self.npars = mig.narg
        self.covariance = mig.covariance
        for par in par_names:
            if mig.fitarg['fix_'+par]:
                self.npars -= 1
        self.rchi2min = self.chi2min/(self.ndata-self.npars)
        self.chi2_before = self.get_chi2()
        self.rchi2_before =  self.get_chi2()/self.ndata
        print('chi2 (before fit) = %.2f   ndata = %d                rchi2 = %.4f'%\
                (self.chi2_before, self.ndata, self.rchi2_before))
        print('chi2 (after  fit) = %.2f   ndata = %d   npars = %d   rchi2 = %.4f'%\
                (self.chi2min, self.ndata, self.npars, self.rchi2min))

    def plot_overdensity(self, pars=[None], ylim=[0.75, 1.25], 
        nbinsh=50, title=None):

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
        centers = self.centers

        for par in pars:
            self.get_histograms(pars=par)
            for i in range(nmaps):
                name   = self.syst_names[i]
                delta  = self.delta[name]
                edelta = self.edelta[name]
                chi2   = np.sum( (delta-1.)**2/edelta**2)
                label = r'$\chi^2_{r}  = %.1f/%d = %.2f$'%\
                          (chi2, nbins, chi2/nbins)
                ax[i].errorbar(centers[name], delta, edelta, fmt='.', label=label)
                ax[i].axhline( 1.0, color='k', ls='--')
                ax[i].locator_params(axis='x', nbins=5, tight=True)
                
                #-- add title and legend
                ax[i].legend(loc=0, numpoints=1, fontsize=8)
                ax[i].set_xlabel(name)

        #-- overplot histogram (normalizing to the 1/3 of the y-axis)
        for i in range(nmaps):
            name = self.syst_names[i]
            h_syst, bins = np.histogram(self.data_syst[name], bins=nbinsh)
            x = 0.5*(bins[:-1]+bins[1:])
            y = h_syst/h_syst.max()*0.3*(ylim[1]-ylim[0])+ylim[0]
            ax[i].step(x, y, where='mid', color='g')

        ax[0].set_ylabel('Density fluctuations')
        if title:
            f.subplots_adjust(top=0.9)
            plt.suptitle(title)

    def export(self, fout):

        fout = open(fout, 'w')
        nbins = self.centers[0].size
       
        delta_before, edelta_before = self.get_histograms()
        delta_after,  edelta_after  = self.get_histograms(pars=self.pars)
        chi2_before = np.sum( (delta_before-1)**2/edelta_before**2)
        chi2_after  = np.sum( (delta_after -1)**2/edelta_after**2 )
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
                den0 =   delta_before[j, i]
                eden0 = edelta_before[j, i]
                den1 =   delta_after[j, i]
                eden1 = edelta_after[j, i]
                line = '%f \t %f \t %f \t %f \t %f \t %f \t %f'%\
                       (smin, scen, smax, den0, eden0, den1, eden1)
                print(line, file=fout)

        fout.close()

