import os
import numpy as N
import pylab as P
import healpy as hp
from astropy.io import fits

from scipy.optimize import minimize

class MultiFit:

    def __init__(self, nside=256, index=N.arange(7), \
                 fit_index=N.arange(7) ):
        self.syst_maps = self.read_systematic_maps_fits(nside=nside, index=index)
        self.maps = self.syst_maps['values']
        self.nside=nside
        self.nsyst=index.size
        self.nsyst_fit = fit_index.size
        self.fit_index=fit_index
        
    def read_data(self, ra, dec, weights):
        self.ra_data=ra
        self.dec_data=dec
        self.weights_data = weights
        self.pix_data = self.make_healmap(ra, dec, weights) 
        print 'Number of galaxies', ra.size
        print 'Weight of galaxies', sum(weights)

    def read_randoms(self, ra, dec, weights):
        self.ra_rand=ra
        self.dec_rand=dec
        self.weights_rand=weights
        self.pix_rand = self.make_healmap(ra, dec, weights) 
        print 'Number of randoms', ra.size
        print 'Weight of randoms', sum(weights)

    def read_systematic_maps_fits(self, nside=256, index=N.array([0, 6]), \
            infits=os.environ['EBOSS_CLUSTERING_DIR']+'/etc/systematic_maps_256nest.fits'):
        ''' read FITS file containing all healpix maps
                 
            nside: integer
                 nside of the output (should be less or equal to 256)
            index: integer array
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
            print 'nside should be at most 256'
            return

        a = fits.open(infits)
        allsyst = a[0].data[index]
        nsideh = a[0].header['NSIDE']
        names = a[1].data.NAMES[index]
        suffix = a[1].data.SUFFIX[index]
        nsyst = len(index)

        #-- normalize star density into units per deg2 
        allsyst[0] *= (12*nside**2)/(4*N.pi*(180**2/N.pi**2))

        if nside!=nsideh:
            nallsyst = N.zeros((nsyst, 12*nside**2))
            for i in range(nsyst):
                nallsyst[i] = hp.ud_grade(allsyst[i], nside, order_in='NESTED', order_out='NESTED')
            allsyst = nallsyst
            
        syst_maps = {'values':allsyst, 'names':names, 'suffix':suffix, 'nsyst':nsyst, 'nside':nside}

        return syst_maps

    def read_systematic_maps_ascii(nside=256, index=N.array([0, 6])):
        '''Similar to read_systematic_maps_fits() but reads from ASCII files. It's longer! '''

        print 'Reading healpix maps of systematics:'

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

        nsyst = len(healpixfiles)
        allsyst = N.zeros((nsyst, 12*nside**2))
        for i in range(nsyst):
            print '   %s'%os.path.basename(healpixfiles[i])
            onemap = N.loadtxt(healpixfiles[i])
            onemap = hp.ud_grade(onemap, nside, order_in='NESTED', order_out='NESTED')
            allsyst[i] = onemap

        syst_maps = {'values':allsyst, 'names':names, 'suffix':suffix, 'nsyst':nsyst, 'nside':nside}

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
        collist.append(fits.Column(name='NAMES', format='20A', array=syst['names']))
        collist.append(fits.Column(name='SUFFIX', format='20A', array=syst['suffix']))
        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(collist))

        hdulist.append(prihdu)
        hdulist.append(tbhdu)
        hdulist.writeto(os.environ['MKESAMPLE_DIR']+\
                'inputFiles/systematic_maps_%dnest.fits'%syst['nside'], clobber=True)


    def plot_systematic_map(self, index):

        syst = self.syst_maps
        maps = syst['values']
        names=syst['names']
        nmaps = maps.shape[0]
        i = index
        w = (maps[i] != 0.)
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
        

    def make_healmap(self, ra, dec, weights):
        '''Makes healpix map from RA and DEC and weights'''

        ngal_map = N.zeros( 12*self.nside**2)
        pix = hp.ang2pix(self.nside, (-dec+90.)*N.pi/180, ra*N.pi/180, nest=1)
        for p, w in zip(pix, weights):
            ngal_map[p] += w
        return ngal_map

    def dens_model(self, pars, ra, dec, maps):
        ''' Gets individual systematic weights from a set of parameters
            and a set of systematic maps. 
            
            Parameters:
            pars: array
            ra, dec: arrays containing coordinates in degrees
            maps: 2d array containing systematic maps of the form [n_syst, n_pix]
            nside: passed as well to increase speed
        '''
        
        pix = hp.ang2pix(self.nside, (-dec+90.)*N.pi/180, ra*N.pi/180, nest=1)
        model = pars[0] + N.sum((pars[1:]*maps[:, pix].T).T, axis=0)
        return model

    def dens_model_map(self, pars, maps, edges, norm):
        
        x = (maps.T-edges[:, 0])/norm

        return pars[0] + N.sum((pars[1:]*x).T, axis=0)

    def chi2(self, pars, maps, edges, norm, pix_data, hist_rand):
        
        model_map = self.dens_model_map(pars, maps, edges, norm)

        #-- loop over systematic maps
        chi=0
        for i in range(pars.size-1):
            #-- compute histogram of galaxy count map applying weights_sys = 1./model_map 
            hist_data, bins = N.histogram(maps[i], bins=edges[i], \
                                          weights = pix_data/model_map) 
            #-- density and its error
            dens = hist_data/hist_rand[i]*self.factor
            vdens = (hist_data/hist_rand[i]**2 + \
                     hist_data**2/hist_rand[i]**3) * self.factor**2
            #-- chi2 defined as the distance to unity
            chi+= sum( (dens-1.)**2/vdens )

        return chi

    def prepare(self,  nbins=10):
        
        maps = self.syst_maps['values']
        names = self.syst_maps['names']
        suffix = self.syst_maps['suffix']

        nsyst = self.nsyst 
        nsyst_fit = self.nsyst_fit
        npix = maps.shape[1]
        nside = hp.npix2nside(npix)

        self.nbins=nbins
        self.factor=sum(self.weights_rand)/sum(self.weights_data) 

       
        #-- only pixels with randoms will be used
        wmap = (self.pix_rand>0)
        wfit = self.fit_index
        
        #-- computing edges for histograms
        self.edges      = N.zeros((nsyst, nbins+1))
        self.centers    = N.zeros((nsyst, nbins)) #-- mean value of systematic in each bin
        self.norm       = N.zeros(nsyst) #-- normalization for systematic values 
        self.hist_data0 = N.zeros((nsyst, nbins)) #-- before fitting
        self.hist_data1 = N.zeros((nsyst, nbins)) #-- after fitting
        self.hist_rand  = N.zeros((nsyst, nbins))
        self.dens0  = N.zeros((nsyst, nbins))
        self.edens0 = N.zeros((nsyst, nbins))
        self.dens1  = N.zeros((nsyst, nbins))
        self.edens1 = N.zeros((nsyst, nbins))

        for i in range(nsyst):
            #-- pick one map
            onemap = maps[i]
            #-- pick where the map is valid or has galaxies
            w = (onemap>0) & wmap 
            #-- cut the lower and upper 1% pixels
            sysmin = N.percentile(onemap[w], 1.0)
            sysmax = N.percentile(onemap[w], 99.0)
            #-- do the histograms
            w = (onemap >= sysmin) & (onemap <= sysmax) & wmap 
            self.edges[i] = N.array([N.percentile(onemap[w], j*100./nbins) \
                                     for j in range(nbins+1)])
            self.norm[i] = self.edges[i, -1]-self.edges[i, 0]
            self.hist_rand[i], _= N.histogram(onemap[w], bins=self.edges[i], \
                                                 weights=self.pix_rand[w])
            self.centers[i], _= N.histogram(onemap[w], bins=self.edges[i], \
                                                weights=self.pix_rand[w]*onemap[w])
            self.centers[i] /= self.hist_rand[i]
        

        #-- array with parameters. The first is the constant and the others are the slopes
        pars0 = N.zeros(self.nsyst_fit +1)
        pars0[0] = 1.

        maps_fit = maps[wfit]
        maps_fit = maps_fit[:, wmap]
        edges_fit = self.edges[wfit]
        norm_fit = self.norm[wfit]

        print 'Starting minimizer....'
        self.wfit = wfit
        self.wmap = wmap
        pars_object = minimize(self.chi2, pars0, \
                               args = (maps_fit, edges_fit, norm_fit, \
                                       self.pix_data[wmap], self.hist_rand[wfit]), \
                               method='Nelder-Mead')
        print pars_object['message']
        print pars_object['success']
        print pars_object['x']

        pars1 = pars_object['x']

        pars_all0 = N.zeros(self.nsyst+1)
        pars_all1 = N.zeros(self.nsyst+1) 
        pars_all0[0] = pars0[0]
        pars_all0[self.fit_index+1] = pars0[1:]
        pars_all1[0] = pars1[0]
        pars_all1[self.fit_index+1] = pars1[1:]
        chi2_null= self.chi2(pars_all0, maps[:, wmap], self.edges, self.norm, \
                             self.pix_data[wmap], self.hist_rand)
        chi2_fit = self.chi2(pars_all1, maps[:, wmap], self.edges, self.norm, \
                             self.pix_data[wmap], self.hist_rand)

        print ''
        print '     chi2_null = %.3f/%d = %.3f'%\
                    (chi2_null, (nsyst*nbins), chi2_null/(nsyst*nbins))
        print '     chi2_fit  = %.3f/(%d-%d) = %.3f'%\
                    (chi2_fit, (nsyst*nbins), pars1.size, \
                     chi2_fit/(nsyst*nbins-pars1.size))

        self.pars1 = pars1
        self.ndata = nsyst*nbins
        self.npars = pars1.size
        self.chi2_null=chi2_null
        self.chi2_fit = chi2_fit
        self.rchi2_null=chi2_null/(nsyst*nbins)
        self.rchi2_fit = chi2_fit/(nsyst*nbins-pars1.size)
        self.maps_fit = maps_fit
        self.edges_fit = edges_fit
        self.norm_fit = norm_fit
            
        model = self.dens_model_map(pars1, maps_fit, edges_fit, norm_fit)
        for i in range(nsyst):
            self.hist_data0[i], _ = N.histogram(maps[i, wmap], \
                    bins=self.edges[i], weights=self.pix_data[wmap])
            self.hist_data1[i], _ = N.histogram(maps[i, wmap], \
                    bins=self.edges[i], weights=self.pix_data[wmap]/model)

            self.dens0[i] = self.hist_data0[i]/self.hist_rand[i]*self.factor
            self.dens1[i] = self.hist_data1[i]/self.hist_rand[i]*self.factor
            
            self.edens0[i] = N.sqrt(self.hist_data0[i]   /self.hist_rand[i]**2 + \
                                    self.hist_data0[i]**2/self.hist_rand[i]**3) * \
                                    self.factor
            self.edens1[i] = N.sqrt(self.hist_data1[i]   /self.hist_rand[i]**2 + \
                                    self.hist_data1[i]**2/self.hist_rand[i]**3) * \
                                    self.factor


    def get_weights(self, ra, dec):
        
        model = self.dens_model_map(self.pars1, self.maps[self.wfit], \
                                    self.edges_fit, self.norm_fit)
        pix = hp.ang2pix(self.nside, (-dec+90.)*N.pi/180, ra*N.pi/180, nest=1)
        return 1./model[pix]
        

    def plot_syst_model(self, vmin=0.8, vmax=1.2):

        wmap = self.pix_rand>0
        map_model = self.dens_model_map(self.pars1, self.maps[self.fit_index], \
                                        self.edges_fit, self.norm_fit)
        map_model[~wmap] = 0.
        hp.mollview(map_model, min=vmin, max=vmax, nest=1)
        

    def plot_density_vs_syst(self, plotroot=''):

        f, ax = P.subplots(1, self.nsyst, sharey=True, figsize=(15, 3))
        f.subplots_adjust(wspace=0.05, left=0.05, right=0.98, top=0.98, bottom=0.15)

        maps = self.maps[:, self.wmap]
        names = self.syst_maps['names']
        nbins = self.nbins

        ylim=[0.75, 1.25]
        ax[0].set_ylim(ylim)
        for i in range(self.nsyst):
        
            chi2_0 = N.sum( (self.dens0[i]-1.)**2/self.edens0[i]**2)
            chi2_1 = N.sum( (self.dens1[i]-1.)**2/self.edens1[i]**2)
            label0 = r'$\chi^2_{\rm before} = %.2f/%d = %.2f$'%\
                     (chi2_0, nbins, chi2_0/nbins)
            label1 = r'$\chi^2_{\rm after}  = %.2f/%d = %.2f$'%\
                     (chi2_1, nbins, chi2_1/nbins)

            ax[i].plot(    self.centers[i], self.dens0[i], 'ro', alpha=0.5,\
                            label=label0)
            ax[i].errorbar(self.centers[i], self.dens1[i], self.edens1[i], \
                            fmt='bo', label=label1)
            ax[i].axhline( 1.0, color='k', ls='--')
            ax[i].locator_params(axis='x', nbins=5, tight=True)

            #-- overplot histogram (normalizing to the 1/3 of the y-axis)
            hist_syst, bins = N.histogram(maps[i], \
                                          bins=N.linspace(self.edges[i, 0], self.edges[i, -1], 80), \
                                          weights=self.pix_rand[self.wmap])
            syst_center = 0.5*(bins[:-1]+bins[1:])
            ax[i].step(syst_center, hist_syst/hist_syst.max()*0.3*(ylim[1]-ylim[0])+ylim[0], \
                        where='mid', color='g')

            #-- add title and legend
            ax[i].legend(loc=0, numpoints=1, fontsize=8)
            ax[i].set_xlabel(names[i])

        ax[0].set_ylabel('Density fluctuations')
        if plotroot!='':
                P.savefig(plotroot+'.pdf', bbox_inches='tight')


    @staticmethod
    def LRGs_zbins(cat, ran, zs=[0.6, 0.67, 0.74, 0.81, 1.0], nbins=10, \
                   plotit=0, plotroot='', cp=1, noz=1, fkp=1):
        
        ''' Computes ystematic weights using multi-linear regression a la Abhi 

            Parameters:
            cat:  data Catalog object
            ran: random Catalog object
            zs: array containing the edges of the redshift bins. E.g. zs=[0.6, 1.0]
            nbins: integer with number of bins per systematic map
            cp, noz, fkp: 1 or 0. Defines which weights are used in the fits
            plotit: 1 or 0, if you want to plot the regression result
            plotroot: root (no extension) of filename if you want to save these plots
            
        '''

        cat.WEIGHT_SYSTOT=N.ones(cat.size)
        ran.WEIGHT_SYSTOT=N.ones(ran.size)

        m = MultiFit()
        for i in range(len(zs)-1):
            zmin=zs[i]
            zmax=zs[i+1]
            
            wd = (cat.Z>zmin)&(cat.Z<zmax)&((cat.IMATCH==1)|(cat.IMATCH==2))
            wr = (ran.Z>zmin)&(ran.Z<zmax)

            m.read_data(cat.RA[wd], cat.DEC[wd], \
                        cat.get_weights(cp=cp, fkp=fkp, noz=noz, syst=0)[wd])
            m.read_randoms(ran.RA[wr], ran.DEC[wr], \
                           ran.get_weights(fkp=fkp, cp=0, noz=0, syst=0)[wr])
            
            m.prepare(nbins=nbins)
            wsys_data = m.get_weights(m.ra_data, m.dec_data)
            wsys_rand = m.get_weights(m.ra_rand, m.dec_rand)

            cat.WEIGHT_SYSTOT[wd] = wsys_data
            ran.WEIGHT_SYSTOT[wr] = wsys_rand

            if plotit:
                m.plot_density_vs_syst()
                plotrootzs = plotroot+('-zmin%.2f-zmax%.2f'%(zmin, zmax))*(plotroot!='')
                if plotrootzs!='':
                    P.savefig(plotrootzs+'.pdf', bbox_inches='tight')





