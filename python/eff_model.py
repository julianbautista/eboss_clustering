from __future__ import print_function
import os
import sys
import numpy as N
import pylab as P

from pbs import queue
from ebosscat import Catalog, Cosmo, Mask

class efficiency:

    def __init__(self, cat, syst=0):
        
        wd = (cat.PLATE!=0)
        #wr = (ran.Z>zmin)&(ran.Z<zmax)

        self.plates = N.unique(cat.PLATE[wd])
        self.xcat = cat.XFOCAL[wd]
        self.ycat = cat.YFOCAL[wd]
        self.z = cat.Z[wd]
        self.imatch = cat.IMATCH[wd]
        self.wcat = cat.get_weights(cp=1, noz=0, fkp=1, syst=syst)[wd]
        self.cat_plate = cat.PLATE[wd]
        self.cat_spectro = (cat.FIBERID[wd]>500)*1

        #self.xran = ran.XFOCAL[wr]
        #self.yran = ran.YFOCAL[wr]
        #self.wran = ran.get_weights(cp=0, noz=0, fkp=1)[wr]
        #self.ran_plates = ran.PLATE[wr]
        #self.ran_spectro = (ran.YFOCAL[wr]>0.)*1

        nspec = self.imatch.size
        nplates = self.plates.size
        ngals = sum(self.imatch==1)
        nfail = sum((self.imatch==5)|(self.imatch==7))
        print('Nspec   =', nspec )
        print('Nplates =', nplates)
        print('Ngals   =', ngals)
        print('Nfail   =', nfail)
        print('Average efficiency =', ngals*1./(nfail+ngals) )

        self.nspec = nspec
        self.nplates = nplates
        self.read_xy()
        self.read_spectro(cat)
        self.read_spectro_info()
        self.fit_spectro_eff()
    
    def read_spectro_info(self):

        info = os.environ['EBOSS_CLUSTERING_DIR']+'/etc/spPlate_spectroinfo_v5_10_0.txt'
        print('Using S/N estimates from ', info)
        spec = N.loadtxt(info, unpack=0, skiprows=1)
        keys = N.array(open(info).readline().split())
        wk1 = keys == 'SPEC1_I'
        wk2 = keys == 'SPEC2_I'
        platescol = spec[:, 0]

        spectro_sn2 = N.zeros((self.nplates, 2))
        spectro_sn2s = N.zeros(self.nspec)
        wp = N.zeros(self.nplates, dtype=int)
        for i, p in enumerate(self.plates):
            w = N.where(platescol == p)[0]
            if len(w)==0:
                print(p, 'not found in', info)
                continue
            wp[i] = w[0]
            spectro_sn2[i, 0] = spec[w, wk1] 
            spectro_sn2[i, 1] = spec[w, wk2] 

            wg1 = (self.cat_plate == p) & (self.cat_spectro == 0)
            wg2 = (self.cat_plate == p) & (self.cat_spectro == 1)
            spectro_sn2s[wg1] = spectro_sn2[i, 0]
            spectro_sn2s[wg2] = spectro_sn2[i, 1]


        spec = spec[wp] 

        table = dict()
        for i, k in enumerate(keys):
            table[k] = spec[:, i]

        self.spectro_sn2 = spectro_sn2
        self.spectro_sn2s = spectro_sn2s
        self.table = table
        self.keys = keys


    def read_xy_eff(self,  fin=os.environ['EFF_DIR']+'/eff_xyfocal_redmonster_35.txt'):

        i, j, x, y, n, eff = N.loadtxt(fin, unpack=1)
        nx = N.unique(i).size
        ny = N.unique(j).size
        x = N.unique(x)
        y = N.unique(y)
        n = N.reshape(n, (nx, ny))
        eff = N.reshape(eff, (nx, ny))
        self.x = x
        self.y = y
        self.xy_n = n
        self.xy_eff = eff
        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.xmin = x[0]
        self.ymin = y[0]
        
    def read_xy(self, nx=20, ny=20):
       
        self.xmin=-326.
        self.ymin=-326.
        xedges = N.linspace(self.xmin, -self.xmin, nx+1)
        yedges = N.linspace(self.ymin, -self.ymin, ny+1)

        xcat = self.xcat 
        ycat = self.ycat 
        imatch = self.imatch 

        wgals = (imatch==1)*1.
        wfail = ((imatch==5)|(imatch==7))*1.

        ngals, x, y = N.histogram2d(xcat, ycat, bins=[xedges, yedges], weights=wgals)
        nfail, x, y = N.histogram2d(xcat, ycat, bins=[xedges, yedges], weights=wfail)

        eff = ngals*0.
        w = (ngals+nfail)>10
        eff[w] = ngals[w]/(ngals[w]+nfail[w])

        self.nx = nx
        self.ny = ny
        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.x = xedges
        self.y = yedges
        self.xcen = 0.5*(xedges[1:]+xedges[:-1])
        self.ycen = 0.5*(yedges[1:]+yedges[:-1])
        self.xy_eff = eff
        self.xy_n = (ngals+nfail)
        self.xy_mean_eff= N.mean(eff[eff>0])

    def read_xy_delta(self, nx=20, ny=20, zmin=0.6, zmax=1.0):

        self.xmin=-325.
        self.ymin=-325.
        xedges = N.linspace(self.xmin, -self.xmin, nx+1)
        yedges = N.linspace(self.ymin, -self.ymin, ny+1)

        norm = sum(self.wran)/sum(self.wcat)

        ngals, x, y = N.histogram2d(self.xcat, self.ycat,  \
                                    bins=[xedges, yedges], \
                                    weights=self.wcat)
        nrand, x, y = N.histogram2d(self.xran, self.yran,  \
                                    bins=[xedges, yedges], \
                                    weights=self.wran)

        dens = ngals/nrand*norm
        dens[nrand==0] = 0

        self.nx = nx
        self.ny = ny
        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.x = xedges
        self.y = yedges
        self.xcen = 0.5*(xedges[1:]+xedges[:-1])
        self.ycen = 0.5*(yedges[1:]+yedges[:-1])
        self.xy_eff = dens
        self.xy_n = (ngals)


    def get_xy(self, x, y):
        
        ix = (N.floor( (x - self.xmin)/self.dx ) ).astype(int)
        iy = (N.floor( (y - self.ymin)/self.dy ) ).astype(int)
        
        eff = self.xy_eff[ix, iy]
        #-- check for points outside grid
        w = (ix<0)|(ix>=self.x.size)|(iy<0)|(iy>=self.y.size)
        eff[w] = 0.
        return eff

    def plot_xy(self, vmin=0.5, vmax=1.0):
        P.figure()
        cmap = P.get_cmap('jet')
        cmap.set_under('white')
        P.pcolormesh(self.x, self.y, self.xy_eff.T, vmin=vmin, vmax=vmax, cmap=cmap)
        P.xlabel('XFOCAL [mm]')
        P.ylabel('YFOCAL [mm]')
        P.colorbar(extend='min')

    def plot_xy_n(self):
        P.figure()
        cmap = P.get_cmap('jet')
        cmap.set_under('white')
        P.pcolormesh(self.x, self.y, self.xy_n.T, vmin=1, cmap=cmap)
        P.xlabel('XFOCAL [mm]')
        P.ylabel('YFOCAL [mm]')
        P.colorbar(extend='min')

    def read_spectro(self, cat):
    
        plates = self.plates 
        
        ngals = N.zeros((plates.size, 2))
        nfail = N.zeros((plates.size, 2))
        spectro_eff = N.zeros((plates.size, 2))
        spectro_eff_err = N.zeros((plates.size, 2))
                
        for i in range(plates.size):
            wplate = self.cat_plate==plates[i]
            spectro = self.cat_spectro[wplate]
            imatch = self.imatch[wplate]
            
            wgood = (imatch==1)*1.#*weights[wplate]
            wfail = ((imatch==5)|(imatch==7))*1.

            ngals[i, 0] = N.sum(wgood[spectro==0])
            ngals[i, 1] = N.sum(wgood[spectro==1])
            nfail[i, 0] = N.sum(wfail[spectro==0])
            nfail[i, 1] = N.sum(wfail[spectro==1])

        w = (nfail)!=0
        spectro_eff[w] = ngals[w]/nfail[w]
        spectro_eff_err[w] = N.sqrt( ngals[w]/nfail[w]**2 + ngals[w]**2/nfail[w]**3)
        self.spectro_eff = spectro_eff
        self.spectro_eff_err = spectro_eff_err
        self.ngals = ngals
        self.nfail = nfail
    
    def fit_spectro_eff(self, npoly=1, plotit=0, nbins=20, pmin=3, pmax=97, per_plate=0):

        if per_plate:
            w = self.spectro_eff_err.ravel() >0
            x = self.spectro_sn2.ravel()[w]
            ngals = self.ngals.ravel()[w]
            nfail = self.nfail.ravel()[w]
            wg = ngals
            wf = nfail
        else:
            x = self.spectro_sn2s
            wg = (self.imatch==1)
            wf = (self.imatch==5) | (self.imatch==7)
        
        edges = N.array([N.percentile(x, i) \
                         for i in N.linspace(pmin, pmax, nbins+1)])
        centers = 0.5*(edges[1:]+edges[:-1])

        hgals, _ = N.histogram(x, bins=edges, weights=1.*wg)
        hfail, _ = N.histogram(x, bins=edges, weights=1.*wf)

        x = centers
        y = hgals/hfail
        dy = N.sqrt( hgals/hfail**2 + hgals**2/hfail**3)

        coeff = N.polyfit(x, y, npoly, w=1./dy)
        ymodel = N.polyval(coeff, x) 

        chi2 = N.sum( (y-ymodel)**2/dy**2 )
        ndata = y.size
        npars = coeff.size
        rchi2 = chi2/(ndata-npars)

        print('Fit of efficiency vs S/N:')
        print('chi2 =', chi2, 'ndata =', ndata, 'npars =', npars, 'rchi2 =',rchi2)
    
        self.coeff = coeff
        self.pmin = pmin
        self.pmax = pmax
        self.nbins = nbins
        self.per_plate = per_plate

    def plot_spectro_eff(self):

        w = self.spectro_eff_err.ravel() > 0
        sn2 = self.spectro_sn2.ravel()[w]
        ngals = self.ngals.ravel()[w]
        nfail = self.nfail.ravel()[w]
        
        #-- individual plates
        eff = ngals/(ngals+nfail)
        deff = N.sqrt( nfail*1./(ngals+nfail)**2 + nfail**2*1./(ngals+nfail)**3)
        P.errorbar(sn2, eff, deff, fmt='.', color='k', alpha=0.1)
       
        #-- binned values
        if self.per_plate:
            wg = ngals
            wf = nfail 
        else:
            sn2 = self.spectro_sn2s
            wg = self.imatch==1
            wf = (self.imatch==5)|(self.imatch==7)
        
        edges = N.array([N.percentile(sn2, i) \
                     for i in N.linspace(self.pmin, self.pmax, self.nbins+1)])
        centers = 0.5*(edges[1:]+edges[:-1])
        hgals, _ = N.histogram(sn2, bins=edges, weights=1.*wg)
        hfail, _ = N.histogram(sn2, bins=edges, weights=1.*wf)
            

        heff = hgals/(hgals+hfail)
        dheff = N.sqrt( hfail*1./(hgals+hfail)**2 + hfail**2*1./(hgals+hfail)**3)
        P.errorbar(centers, heff, dheff, fmt='o', color='b')

        #-- model
        xmodel = N.linspace(sn2.min(), sn2.max(), 50)
        ymodel = self.get_spectro_eff(xmodel)
        P.plot(xmodel, ymodel, 'r')
        P.xlabel(r'Spectro $(S/N)^2$')
        P.ylabel(r'Efficiency')

        

    def read_spectro_delta(self):
        
        uplates = self.plates 

        norm = sum(self.wran)/sum(self.wcat)
        ngals = N.zeros((uplates.size, 2))
        nrand = N.zeros((uplates.size, 2))
        
        for i in range(uplates.size):
            wd = self.cat_plates==uplates[i]
            wr = self.ran_plates==uplates[i]

            ngals[i, 0] = N.sum(self.wcat[wd]*(self.cat_spectro[wd]==0))
            ngals[i, 1] = N.sum(self.wcat[wd]*(self.cat_spectro[wd]==1))

            nrand[i, 0] = N.sum(self.wran[wr]*(self.ran_spectro[wr]==0))
            nrand[i, 1] = N.sum(self.wran[wr]*(self.ran_spectro[wr]==1))

        eff = (ngals/nrand*norm)
        eff_err = N.sqrt( ngals/nrand**2 + ngals**2/nrand**3)*norm
        w = nrand==0
        eff[w] = 0
        eff_err[w] = 0

        self.spectro_eff = eff
        self.spectro_eff_err = eff_err
        self.ngals = ngals
        self.nrand = nrand
        self.norm = norm

    def stack_delta(self, nbins=10):

        f, ax = P.subplots(3, 4, figsize=(20, 14), sharey=True)

        iplot=0
        for k in self.keys:
            if k in ['PLATE', 'MJD', 'RA', 'DEC', 'CARTID']:
                continue
            elif 'SPEC1' in k:
                wgal = self.ngals[:, 0]
                wran = self.nrand[:, 0]
            elif 'SPEC2' in k:
                wgal = self.ngals[:, 1]
                wran = self.nrand[:, 1]
            else:
                wgal = N.sum(self.ngals, axis=1)
                wran = N.sum(self.nrand, axis=1)

            x = self.table[k]
            edges = N.array([N.percentile(x, i, interpolation='nearest') \
                             for i in N.linspace(0, 100, nbins+1)])
            edges = N.unique(edges)
            edges -= 0.01
            edges[-1] += 0.02
            centers = 0.5*(edges[:-1]+edges[1:])
            ngals, _ = N.histogram(x, bins=edges, weights=wgal)
            nrand, _ = N.histogram(x, bins=edges, weights=wran)
            
            dens = ngals/nrand*self.norm
            dens_err = N.sqrt( ngals/nrand**2 + ngals**2/nrand**3)*self.norm
            chi2 = N.sum( (dens-1.0)**2/dens_err**2)
            rchi2 = chi2/dens.size
            ix = iplot/4
            iy = iplot%4
            ax[ix, iy].errorbar(centers, dens, dens_err, fmt='.', \
                       label=r'$\chi^2 = %.1f/%d = %.1f$'%(chi2, dens.size, rchi2))
            ax[ix, iy].axhline(1.0, color='k', ls='--')
            ax[ix, iy].set_xlabel(k)
            ax[ix, iy].set_ylim(0.85, 1.15)
            ax[ix, iy].legend(loc=0, numpoints=1, fontsize=10)
            iplot+=1
        f.tight_layout()

    def get_spectro_eff(self, specsn2):
        ymodel = N.polyval(self.coeff, specsn2)
        return 1./(1+1/ymodel)

    def read_fiberid(self, cat, nbins=25, syst=1):
        fiberid = cat.FIBERID[cat.PLATE!=0]
        imatch = cat.IMATCH[cat.PLATE!=0]
        weights_cat = cat.get_weights(cp=1, fkp=1, noz=0, syst=syst)[cat.PLATE!=0]

        wgals = (imatch==1)*weights_cat
        wfail = ((imatch==5)|(imatch==7))*weights_cat
        edges = N.linspace(1, 1001, nbins+1)
        fibers = 0.5*(edges[1:]+edges[:-1])
        ngals, _ = N.histogram(fiberid, bins=edges, weights=wgals)
        nfail, _ = N.histogram(fiberid, bins=edges, weights=wfail)
        
        #P.figure()
        #_ = P.hist(fiberid, bins=edges, weights=wgals, histtype='step')
        #_ = P.hist(fiberid, bins=edges, weights=wfail, histtype='step')
        #P.xlim(0, 1001) 
        
        eff = ngals/(ngals+nfail)
        deff = N.sqrt( nfail*1./(ngals+nfail)**2 + nfail**2/(ngals+nfail)**3)
        P.errorbar(fibers, eff, deff, fmt='o')
        P.xlim(0, 1001) 

    def plot_zdist(self, percentile=20, nbins=20, zmin=0.6, zmax=1.0, use_weights=1, spectro=0):
        
        w = (self.imatch==1)
        if spectro:
            w = w & (self.cat_spectro == spectro-1)

        hper = N.percentile(self.spectro_sn2s[w], 100-percentile)
        lper = N.percentile(self.spectro_sn2s[w], percentile)

        wh = w & (self.spectro_sn2s > hper)
        wl = w & (self.spectro_sn2s < lper)

        
        print(sum(wh), sum(wl))
        if use_weights:
            weights = self.wcat
        else:
            weights = N.ones(sum(self.imatch==1))

        hlow, _ = N.histogram(self.z[wl], bins=N.linspace(zmin, zmax, nbins+1),\
                              weights=weights[wl])
        hupp, _ = N.histogram(self.z[wh], bins=N.linspace(zmin, zmax, nbins+1),\
                              weights=weights[wh])
        bins = N.linspace(zmin, zmax, nbins+1)
        centers = 0.5*(bins[:-1]+bins[1:])
        P.figure(figsize=(6, 6))
        P.subplot(211)
        P.errorbar(centers, hlow, N.sqrt(hlow), fmt='.', \
                   label='Lower %d S/N'%percentile)
        P.errorbar(centers, hupp, N.sqrt(hupp), fmt='.', \
                   label='Upper %d S/N'%percentile)
        P.legend(loc=0, numpoints=1)
        P.subplot(212)
        ratio = hlow*1./hupp
        dratio = N.sqrt( hlow*1./hupp**2 + hlow**2*1./hupp**3)
        P.errorbar(centers, ratio, dratio, fmt='.') 

        rchi2 = N.sum( (ratio-1)**2/dratio**2)#/(ratio.size)
        print('rchi2 =', rchi2 )
        P.axhline(1., color='k', ls='--', label=r'$\chi^2_r= %.2f$'%rchi2)

        for order in [0, 1, 2]:
            coeff = N.polyfit(centers, ratio, order, w=1/dratio)
            model = N.polyval(coeff, centers) 
            rchi2 = N.sum( (ratio-model)**2/dratio**2)#/(ratio.size-order-1)
            print('order =', order, 'rchi2 =', rchi2 )
            P.plot(centers, model, label=r'$\chi^2_r= %.2f$'%rchi2)

        P.ylim(0.4, 1.8)
        P.ylabel('Ratio Low/High S/N z-dist')
        P.xlabel('Redshift')
        P.legend(loc=0, numpoints=1, fontsize=10)

    def read_zdist(self, npercentiles=4, nbins=20, zmin=0.6, zmax=1.0):

        w = (self.imatch==1)
        z  = self.z[w]
        sn2 = self.spectro_sn2s[w]

        weights = self.wcat[w]
        #weights /= self.get_spectro_eff(sn2)
        #weights /= self.get_xy(self.xcat[w], self.ycat[w])

        percentiles = N.linspace(0, 100, npercentiles+1)
        bins = N.linspace(zmin, zmax, nbins+1)
        sn2_bins = N.zeros(npercentiles+1)
        hist = N.zeros((npercentiles, nbins))
        ratio = N.zeros((npercentiles-1, nbins))
        ratio_err = N.zeros((npercentiles-1, nbins))
        
        for i in range(npercentiles):
            per_low = percentiles[i]
            per_upp = percentiles[i+1]

            sn2_low = N.percentile(sn2, per_low)
            sn2_upp = N.percentile(sn2, per_upp)
            sn2_bins[i] = sn2_low
            sn2_bins[i+1] = sn2_upp

            ww = (sn2 > sn2_low) & (sn2<= sn2_upp)

            print(sn2_low, sn2_upp, sum(ww))

            hist[i], _ = N.histogram(z[ww], bins=bins, \
                                  weights=weights[ww])
        
        for i in range(npercentiles-1):
            ratio[i] = hist[i]*1./hist[-1]
            ratio_err[i] = N.sqrt( hist[i]*1./hist[-1]**2 + hist[i]**2*1./hist[-1]**3)

        self.zbins = bins
        self.zsn2_bins = sn2_bins
        self.zcenters = 0.5*(bins[:-1]+bins[1:])
        self.zhist = hist
        self.zratio = ratio
        self.zratio_err = ratio_err
        self.znpercentiles = npercentiles
        self.zmin = zmin
        self.zmax = zmax
        self.znbins = nbins

    def plot_zdist2(self):

        

        colors = ['b', 'g', 'r', 'c', 'm']

        P.figure(figsize=(8, 8))
        P.subplot(211)
        for i in range(self.znpercentiles):
            P.errorbar(self.zcenters, self.zhist[i],\
                       N.sqrt(self.zhist[i]), fmt='.', \
                       color=colors[i], \
                       label=r'$%.1f < (S/N)^2 < %.1f$'%\
                             (self.zsn2_bins[i], self.zsn2_bins[i+1]) )
        P.legend(loc=0, numpoints=1, fontsize=10)
       
        P.xlim(self.zmin, self.zmax+0.3*(self.zmax-self.zmin))

        zcenters = self.zcenters
        ratio = self.zratio
        dratio = self.zratio_err

        P.subplot(212)
        linestyles = ['--', '-']
        for i in range(len(ratio)):
            
            P.errorbar(zcenters, ratio[i], dratio[i], fmt='.') 
            
            rchi2 = N.sum( (ratio[i]-1)**2/dratio[i]**2)#/(ratio.size)
            print('rchi2 (null) =', rchi2 )
            P.axhline(1., color=colors[i], ls=':', \
                      label=r'$\chi^2_r= %.2f$'%rchi2)

            for order in [0, 1]:
                coeff = N.polyfit(zcenters, ratio[i], order, w=1/dratio[i])
                model = N.polyval(coeff, zcenters) 
                rchi2 = N.sum( (ratio[i]-model)**2/dratio[i]**2)#/(ratio.size-order-1)
                print('order =', order, 'rchi2 =', rchi2 )
                P.plot(zcenters, model, color=colors[i], \
                        ls=linestyles[order], \
                        label=r'$\chi^2_r= %.2f$'%rchi2)

        P.ylim(0.4, 1.8)
        P.ylabel('Ratio Low/High S/N z-dist')
        P.xlabel('Redshift')
        P.xlim(self.zmin, self.zmax+0.3*(self.zmax-self.zmin))
        P.legend(loc=0, numpoints=1, fontsize=10)
            
    def plot_zdist_rand(self, ran, nsnbins=4, nzbins=15, zmin=0.6, zmax=1.0,\
                        pmin=0, pmax=100):

        #-- read randoms and assign S/N values
        ran_sn2s = N.zeros(ran.size)
        for i, p in enumerate(self.plates):
            wg1 = (ran.PLATE == p) & (ran.YFOCAL < 0)
            wg2 = (ran.PLATE == p)& (ran.YFOCAL > 0)
            ran_sn2s[wg1] = self.spectro_sn2[i, 0]
            ran_sn2s[wg2] = self.spectro_sn2[i, 1] 

        #-- data redshifts and S/N        
        z = self.z[self.imatch==1]
        sn2 = self.spectro_sn2s[self.imatch==1]

        #-- define bins for histograms
        zbins = N.linspace(zmin, zmax, nzbins)
        zcenter = 0.5*(zbins[1:]+zbins[:-1])
        snbins = N.array([N.percentile(ran_sn2s, i) \
                          for i in N.linspace(pmin, pmax, nsnbins+1)])

        norm = sum((ran.Z>zmin) & (ran.Z<zmax))*1./ \
               sum((z>zmin)&(z<zmax))

        colors = ['b', 'g', 'r', 'c', 'm']


        P.figure()
        for i in range(nsnbins):
            #-- randoms
            wsn = (ran_sn2s > snbins[i])&(ran_sn2s<=snbins[i+1])
            hran, zbins = N.histogram(ran.Z[wsn], bins=zbins)
            #-- data
            wsn = (sn2 > snbins[i]) & (sn2<=snbins[i+1])
            hcat, zbins = N.histogram(z[wsn], bins=zbins)

            ratio = hcat*1./hran*norm
            dratio = N.sqrt(hcat*1./hran**2 + hcat**2*1./hran**3)*norm
            #P.plot(zcenter, hcat*1./hran*norm, \
            #       label=r'$%.1f<(S/N)^2<%.1f$'%(snbins[i], snbins[i+1]))
            P.errorbar(zcenter, ratio, dratio, fmt='.', \
                   color=colors[i], \
                   label=r'$%.1f<(S/N)^2<%.1f$'%(snbins[i], snbins[i+1]))
            coeff = N.polyfit(zcenter, ratio, 1, w=1/dratio)
            P.plot(zcenter, N.polyval(coeff, zcenter), color=colors[i])
        P.xlabel('Redshift')
        P.ylabel(r'Data/Random z-distributions')
        P.legend(loc=0, fontsize=12)


    def get_eff(self, plate, x, y, xyfocal=1, spectro=1):


        spec_eff = N.ones(plate.size)

        if spectro:

            #-- assume that positive YFOCAL goes to spectrograph 2
            #-- this is not exactly true though
            spectroid = (y>0.)*1
            for i, p in enumerate(self.plates):
                w = plate == p
                spec_eff[w] *= self.get_spectro_eff(self.spectro_sn2[i, spectroid[w]])

        if xyfocal:
            xy_eff = self.get_xy(x, y)
            spec_eff *= xy_eff/self.xy_mean_eff
        

        return spec_eff


def add_failures_to_mock(mock, eff, seed=0):

    xy_eff = eff.get_eff(mock.PLATE, mock.XFOCAL, mock.YFOCAL, \
                         xyfocal=1, spectro=1)

    if seed:
        N.random.seed(seed)

    r = N.random.rand(mock.size)
    w = (r > xy_eff) 

    mock.SPECTRO_EFF = xy_eff
    mock.IMATCH[w] = 7
    #-- adding a fake fiberid needed on the efficiency constructor
    mock.FIBERID = 501*(mock.YFOCAL>0)

    return mock

def add_and_correct_failures(i=1):

    cosmo = Cosmo(OmegaM=0.29, h=0.7)

    #-- read mock
    indir = os.environ['CLUSTERING_DIR']+'/mocks/lrgs/1.8/eboss-veto-trim2/catalogs/'
    mock = Catalog(indir+'mock-1.8-LRG-North-%04d.dat.fits'%i)

    #-- read data and compute efficiency model 
    cat = Catalog(os.environ['CLUSTERING_DIR']+'/catalogs/bautista/test12'+\
            '/ebosscat-test12-LRG-North-dectrim.dat.fits')
    eff = efficiency(cat)
    
    
    #-- add failures to mock
    add_failures_to_mock(mock, eff, seed=i)
    
    #-- apply nearest neighbor correction
    mock.fiber_collision()

    #-- export
    mock.export(indir+'mock-1.8-LRG-North-%04d-wnoz2.dat.fits'%i, \
            cosmo=cosmo)

def correct_failures_batch():
   
    qe = queue()
    qe.verbose = True
    qe.create(label='correct_failures_wnoz2', alloc='sdss-kp', \
              nodes=8, ppn=16, walltime='100:00:00',umask='0027')
    for i in range(1, 1001):
        script =  'cd %s '%os.environ['EBOSS_CLUSTERING_DIR']
        script = script + '; python python/eff_model.py %d ' %i
        #script = script + '; python bin/downsample_randoms_mocks.py %d ' %i
        print(script)
        qe.append(script)
    qe.commit(hard=True,submit=True)



if __name__ == '__main__':
    if len(sys.argv)>1:
        add_and_correct_failures(i=int(sys.argv[1]))


