from __future__ import print_function
import numpy as np
import pylab as plt
import os
import copy 
import glob

from scipy.optimize import minimize
#from astropy.io import fits
import iminuit

class Corr:

    def __init__(self, fin, rebin_r=1, shift_r=0, cute=1, ezmock=0):
        if ezmock:
            self.read_ezmock(fin, rebin_r=rebin_r, shift_r=shift_r)
        elif cute:
            self.read_cute(fin, rebin_r=rebin_r, shift_r=shift_r)
        else:
            print('Which format? CUTE or EZMOCK?')

    def read_cute(self, fin, rebin_r=1, shift_r=0):

        ff = open(fin)
        self.wd, self.wd2, self.wr, self.wr2 = [float(s) for s in ff.readline().split()]
        self.header = ff.readline()[:-2]

        self.mu2d, self.r2d, self.cf, self.dcf, self.dd, self.dr, self.rr = \
            np.loadtxt(fin, unpack=1, skiprows=2)
        
        self.mu = np.unique(self.mu2d)
        self.r = np.unique(self.r2d)
        self.nmu = self.mu.size
        self.nr = self.r.size
        self.mu2d = self.shape2d(self.mu2d)
        self.r2d = self.shape2d(self.r2d)
        self.dd = self.shape2d(self.dd)
        self.dr = self.shape2d(self.dr)
        self.rr = self.shape2d(self.rr)
        self.cf = self.shape2d(self.cf)
        self.dcf = self.shape2d(self.dcf)
        dd_norm = 0.5*(self.wd**2-self.wd2)
        rr_norm = 0.5*(self.wr**2-self.wr2)
        dr_norm = self.wd*self.wr
        self.norm_dd = dd_norm
        self.norm_dr = dr_norm
        self.norm_rr = rr_norm

        if rebin_r>1:
            self.rebin(rebin_r=rebin_r, shift_r=shift_r)

    def read_ezmock(self, root, rebin_r=1, shift_r=0):
        mu0, mu1, s0, s1, dd, wdd = np.loadtxt(root+'.dd', unpack=1)
        dr, wdr = np.loadtxt(root+'.dr', unpack=1, usecols=[4, 5])
        rr, wrr = np.loadtxt(root[:-5]+'.rr', unpack=1, usecols=[4, 5])
        self.mu2d = 0.5*(mu0+mu1)
        self.r2d = 0.5*(s0+s1)
        self.mu = np.unique(self.mu2d)
        self.r = np.unique(self.r2d)  
        self.nmu = self.mu.size
        self.nr = self.r.size
        self.cf = (wdd - 2*wdr + wrr)/wrr
        self.dd = dd
        self.dr = dr
        self.rr = rr
        self.mu2d = self.shape2d(self.mu2d)
        self.r2d = self.shape2d(self.r2d)
        self.dd = self.shape2d(self.dd)
        self.dr = self.shape2d(self.dr)
        self.rr = self.shape2d(self.rr)
        self.cf = self.shape2d(self.cf)
        self.norm_dd = np.mean((dd/wdd)[dd>0])
        self.norm_dr = np.mean((dr/wdr)[dr>0])
        self.norm_rr = np.mean((rr/wrr)[rr>0])
       
        if rebin_r>1:
            self.rebin(rebin_r=rebin_r, shift_r=shift_r) 

    def rebin(self, rebin_r=1, shift_r=0):

        nr = (self.nr-rebin_r)//rebin_r
        self.mu2d = np.mean(np.reshape(
                        self.mu2d[shift_r:-rebin_r+shift_r],
                                     (nr, rebin_r, -1)), axis=1)
        self.r2d  = np.mean(np.reshape(
                        self.r2d[shift_r:-rebin_r+shift_r], 
                                     (nr, rebin_r, -1)), axis=1)
        self.dd =   np.sum( np.reshape(
                        self.dd[shift_r:-rebin_r+shift_r], 
                                     (nr, rebin_r, -1)), axis=1) 
        self.dr =   np.sum( np.reshape(
                        self.dr[shift_r:-rebin_r+shift_r], 
                                     (nr, rebin_r, -1)), axis=1) 
        self.rr =   np.sum( np.reshape(
                        self.rr[shift_r:-rebin_r+shift_r],
                                     (nr, rebin_r, -1)), axis=1) 
        self.r = np.unique(self.r2d)
        self.mu = np.unique(self.mu2d)
        self.nr = self.r.size
        self.nmu = self.mu.size
        self.cf = self.dd/self.rr*self.norm_rr/self.norm_dd \
                  - 2.*self.dr/self.rr*self.norm_rr/self.norm_dr + 1.
        self.dcf = (1+self.cf)*(1./np.sqrt(self.dd)+\
                                1./np.sqrt(self.dr)+\
                                1./np.sqrt(self.rr))

    def shape2d(self, x):
        return np.reshape(x, (self.nr, self.nmu))

    def transpose(self, x):
        return self.shape2d(x).T.ravel()

    def compute_cf(self):
       
        #-- sum of pair counts
        dd_norm = 0.5*(self.wd**2-self.wd2)
        rr_norm = 0.5*(self.wr**2-self.wr2)
        dr_norm = self.wd*self.wr
        edd = self.dd*0
        edr = self.dd*0
        err = self.dd*0
        w = self.dd > 0
        edd[w] = 1./np.sqrt(self.dd[w])
        edr[w] = 1./np.sqrt(self.dr[w])
        err[w] = 1./np.sqrt(self.rr[w])
        self.cf = self.dd/self.rr*rr_norm/dd_norm-2.*self.dr/self.rr*rr_norm/dr_norm+1.
        self.dcf = (1+self.cf)*(edd+edr+err)

    def export(self, fout):
        fout = open(fout, 'w')
        print(self.wd, self.wd2, self.wr, self.wr2, file=fout)
        print(self.header, file=fout)
        for i in range(self.nr):
            for j in range(self.nmu):
                print( self.mu[j], self.r[i], self.cf[i, j], self.dcf[i, j], \
                       self.dd[i, j], self.dr[i, j], self.rr[i, j] , file=fout)
        fout.close()


    @staticmethod
    def coadd(c1, c2):

        c = copy.deepcopy(c1)

        #-- ratio of (number of randoms)/(number of galaxies)
        f = (c1.wr/c1.wd)/(c2.wr/c2.wd) 
        print((c1.wr/c1.wd), (c2.wr/c2.wd), f)

        dd = c1.dd + c2.dd
        dr = c1.dr + c2.dr*f
        rr = c1.rr + c2.rr*f**2
        wd = c1.wd + c2.wd
        wd2 = c1.wd2 + c2.wd2
        wr = c1.wr + c2.wr*f
        wr2 = c1.wr2 + c2.wr2*f
        norm_dd = 0.5*(wd**2-wd2)
        norm_rr = 0.5*(wr**2-wr2)
        norm_dr = wr*wd
        
        cf = dd/norm_dd/(rr/norm_rr) - 2*(dr/norm_dr)/(rr/norm_rr) + 1.

        c.dd = dd
        c.dr = dr
        c.rr = rr
        c.wd = wd
        c.wd2 = wd2
        c.wr = wr
        c.wr2 = wr2
        c.norm_dd = norm_dd
        c.norm_rr = norm_rr
        c.norm_dr = norm_dr
        c.cf = cf
        c.dcf = np.ones(cf.shape)

        return c

    @staticmethod
    def combine_mocks(root):
        allm = glob.glob(root)
        nmocks = len(allm)

        cs = [Corr(m) for m in allm]
    
        cfs = np.mean(np.array([c.cf for c in cs]), axis=0)
        dds = np.sum( np.array([c.dd for c in cs]), axis=0)
        drs = np.sum( np.array([c.dr for c in cs]), axis=0)
        rrs = np.sum( np.array([c.rr for c in cs]), axis=0)
        wd = np.sum( np.array([c.wd for c in cs]))
        wd2 = np.sum( np.array([c.wd for c in cs]))
        wr = np.sum( np.array([c.wd for c in cs]))
        wr2 = np.sum( np.array([c.wd for c in cs]))
        norm_dd = np.sum(np.array([0.5*(c.wd**2-c.wd2) for c in cs]))
        norm_rr = np.sum(np.array([0.5*(c.wr**2-c.wr2) for c in cs]))
        norm_dr = np.sum(np.array([c.wd*c.wr for c in cs]))
        cf = dds/rrs*norm_rr/norm_dd-2.*drs/rrs*norm_rr/norm_dr+1.
        dcf = np.std(np.array([c.cf for c in cs]), axis=0)

        c = cs[0]
        c.cf = cf
        c.cfs = cfs
        c.dcf = dcf
        c.wd = wd
        c.wd2 = wd2
        c.wr = wr
        c.wr2 = wr2
        c.dd = dds
        c.dr = drs
        c.rr = rrs
        c.nmocks = nmocks

        return c

    @staticmethod
    def get_full_covariance(root, rebin_r=1, shift_r=0):
        allm = glob.glob(root)
        nmocks = len(allm)
       
        cs = []
        for m in allm:
            c = Corr(m, rebin_r=rebin_r, shift_r=shift_r)
            cs.append(c.cf.ravel())
        cs = np.array(cs)
        coss = np.cov(cs.T)
        del(cs)
        return coss
        
class Wedges:

    def __init__(self, fin=None,
                       cute=0, rebin_r=1, shift_r=0,
                       muranges = [[0., 0.5], [0.5, 0.8], [0.8, 1.]]): 

        if fin is not None:
            if cute:
                c = Corr(fin, rebin_r=rebin_r, shift_r=shift_r)
                weds = self.compute_wedges(c.mu, c.cf, muranges=muranges)
                r = c.r
            else:
                r, muranges, weds = self.read_wedges(fin) 
           
            self.r = r
            self.weds = weds
            self.muranges = muranges
            self.nmocks = 1

    def read_wedges(self, fin):
   
        ff = open(fin)
        #-- Reading header with mu ranges
        mus = np.array([float(s) for s in ff.readline().split()[1:]]) 
        muranges = [ [mus[i], mus[i+1]] for i in range(mus.size-1)] 

        data = np.loadtxt(fin, unpack=1)
        r = data[0]
        weds = data[1:]
        return r, muranges, weds

    def compute_wedges(self, mu, cf2d, muranges=[[0., 0.5], [0.5, 0.8], [0.8, 1.]]):

        weds = list()
        for murange in muranges:
            w = (mu > murange[0])&(mu<=murange[1])
            wed = np.mean(cf2d[:, w], axis=1)
            weds.append(wed)

        return np.array(weds)

    def export(self, fout):
        
        fout = open(fout, 'w')
        r = self.r
        weds = self.weds
        nweds = weds.shape[0]
        muranges = np.unique(self.muranges)
       
        line = '#mus '
        for i in range(nweds+1):
            line+= f'{muranges[i]} '
        print(line, file=fout)
 
        line = '#r(Mpc/h) '
        for i in range(nweds):
            line+= f'w{i} '
        print(line, file=fout)

        for i in range(r.size):
            line = f'{r[i]} '
            for j in range(nweds):
                line+= f'{weds[j, i]} '
            print(line, file=fout)

        fout.close()


    def plot(self, scale_r=2, n=-1, color=None, alpha=None, label=None):

        r = self.r
       
        weds = self.weds*1.
        if n!=-1:
            weds = weds[n, None]
        nweds = weds.shape[0]
        
        for i in range(nweds):
            plt.subplot(nweds, 1, 1+i)
            y = weds[i]*r**scale_r
            plt.plot(r, y, label=label, color=color, alpha=alpha)
            if scale_r == 0:
                plt.ylabel(r'$\xi_{\mu%d}$'%i)
            else:
                plt.ylabel(r'$r^{%d} \xi_{\mu%d}$ [$h^{%d}$ Mpc$^{%d}]$'%\
                         (scale_r, i, -scale_r, scale_r))
            plt.title(r'$%.2f < \mu < %.2f$'%(self.muranges[i][0], self.muranges[i][1]))
        plt.xlabel(r'$r$ [$h^{-1}$ Mpc]')
        plt.tight_layout()

    def plot_many(self, label=None, n=-1, scale_r=2, alpha=1.0, color=None):

        if not hasattr(self, 'nmocks'):
            print('This is a single mock')
            return

        cc = Wedges()
        cc.muranges = self.muranges
        for i in range(self.nmocks):
            cc.r = self.r
            cc.weds = self.weds_many[i]
            cc.plot(n=n, scale_r=scale_r, alpha=alpha, color=color)

    @staticmethod
    def combine_mocks(root,  muranges = [[0., 0.5], [0.5, 0.8], [0.8, 1.]], rebin_r=5):

        allm = glob.glob(root)
        nmocks = len(allm)

        cs = [Wedges(m, muranges=muranges, rebin_r=rebin_r) for m in allm]
    
        weds_many = np.array([c.weds for c in cs])
        weds_flat = np.array([w.ravel() for w in weds_many])
        coss = np.cov(weds_flat.T)
        
        #-- correlation matrix
        corr = coss/np.sqrt(np.outer(np.diag(coss), np.diag(coss)))

        c = cs[0]
        c.weds = np.mean(weds_many, axis=0)
        c.dweds = np.reshape(np.sqrt(np.diag(coss))/np.sqrt(nmocks), c.weds.shape)
        c.coss = coss
        c.corr = corr
        c.nmocks = nmocks
        c.weds_many = weds_many

        return c



class Multipoles:
    
    def __init__(self, fin=None,  cute=0, rebin_r=1, shift_r=0):

        if fin:
            if cute: 
                c = Corr(fin, rebin_r=rebin_r, shift_r=shift_r)
                self.mono, self.quad, self.hexa = self.compute_multipoles(c.mu2d, c.cf)
                self.r = c.r
                self.cute = c
            else:
                self.read_multipoles(fin)
            self.nmocks = 1

    def read_multipoles(self, fin):

        data = np.loadtxt(fin)
        r = data[:, 0]
        mono = data[:, 1]
        try:
            quad = data[:, 2]
        except:
            quad = None
        try:
            hexa = data[:, 3]
        except:
            hexa = None
        self.r = r
        self.mono = mono
        self.quad = quad
        self.hexa = hexa

    def read_cov(self, cov_file):

        i, _, r1, _, coss = np.loadtxt(cov_file, unpack=1)
        r = np.unique(r1)
        nr = r.size
        if (r != self.r).any():
            print( 'Warning: covariance matrix is uncompatible with multipoles')

        ni = np.unique(i).size
        coss = np.reshape(coss, (ni, ni))
        dmono = np.sqrt(np.diag(coss)[:nr])
        if ni>=2*nr:
            dquad = np.sqrt(np.diag(coss)[nr:(2*nr)])
        else:
            dquad = None
        if ni>=3*nr:
            dhexa = np.sqrt(np.diag(coss)[(2*nr):])
        else:
            dhexa = None

        self.dmono = dmono
        self.dquad = dquad
        self.dhexa = dhexa
        self.coss = coss
        self.corr = coss/np.sqrt( np.outer(np.diag(coss), np.diag(coss)) )
        
    def shape2d(self, x):
        return np.reshape(x, (self.nr, self.nmu))

    def compute_multipoles(self, mu2d, cf2d):
       
        mu = np.unique(mu2d) 
        dmu = np.gradient(mu)[0]
        mono = np.sum(cf2d*dmu, axis=1)
        quad = np.sum(cf2d*0.5*(3*mu2d**2-1)*dmu, axis=1)*5.
        hexa = np.sum(cf2d*1/8*(35*mu2d**4-30*mu2d**2+3)*dmu, axis=1)*9.
        return mono, quad, hexa

    def compute_multipoles_trapz(self, mu2d, cf2d):

        mu = np.unique(mu2d)
        dmu = np.gradient(mu)[0]
        xi2 = cf2d * 2.5 * (3 * mu2d**2 - 1)
        xi4 = cf2d * 1.125 * (35 * mu2d**4 - 30 * mu2d**2 + 3)
        mono = np.trapz(cf2d, dx=dmu, axis=1)
        quad = np.trapz(xi2, dx=dmu, axis=1)
        hexa = np.trapz(xi4, dx=dmu, axis=1)
        return mono, quad, hexa
    
    def fit_multipoles(self, cov=None, mu_min=0., mu_max=1.0, verbose=False):
        
        try:
            cf2d = self.cute.cf.ravel()
            r2d = self.cute.r2d.ravel()
            mu = self.cute.mu
        except:
            print('The 2D information is not available')
            return

        self.mono = np.zeros(self.r.size)
        self.quad = np.zeros(self.r.size)
        self.hexa = np.zeros(self.r.size)
        if cov is None:
            cov = np.diag(np.ones(cf.size))


        for i in range(self.r.size):
            r = self.r[i]
            w = r2d == r
            cov_cut = cov[:, w]
            cov_cut = cov_cut[w] 
            inv_cov = np.linalg.inv(cov_cut)
            cf = cf2d[w]
 

            def get_model(mu, p):
                model = p[0] + \
                        p[1]*0.50*(3*mu**2-1) + \
                        p[2]*0.25*(35**mu**4-30*mu**2+3)
                return model

            def chi2(p):
                model = get_model(mu, p)
                residual = cf-model
                chi2 = np.dot(residual, np.dot(inv_cov, residual))
                return chi2

            #par = minimize(chi2, np.array([0., 0., 0.]), 
            #               method='Nelder-Mead')
            #model0 = get_model(mu, par['x'])
        
            par_names = ['mono', 'quad', 'hexa']
            init_pars = {par:0 for par in par_names}
            for par in par_names:
                init_pars['error_'+par] = 10 
            mig = iminuit.Minuit(chi2, throw_nan=False, 
                         forced_parameters=par_names, 
                         print_level=0, errordef=1,
                         use_array_call=True,
                          **init_pars)
            mig.tol = 1.0 
            imin = mig.migrad()
            is_valid = imin[0]['is_valid']
            best_pars = mig.values 
            errors = mig.errors
            chi2min = mig.fval
            ndata = mu.size
            npars = mig.narg
            if verbose:
                print(f'{is_valid} {chi2min} {ndata} {npars}')
                if i%5==0:
                    plt.figure()
                    plt.errorbar(mu, cf, np.sqrt(np.diag(cov_cut)), fmt='.')
                    plt.plot(mu, get_model(mu, best_pars.values()), label='iminuit')
                    plt.plot(mu, model0, '--', label='minimize')
                    plt.title(f'r = {r} chi2 = {chi2min}')
            self.mono[i] = best_pars['mono']
            self.quad[i] = best_pars['quad']
            self.hexa[i] = best_pars['hexa']


    def plot(self, fig=None, figsize=(12, 5), errors=0, scale_r=2, **kwargs):

        has_hexa = True if not self.hexa is None else False

        r = self.r
        y1 = self.mono  * (1 + r**scale_r) 
        y2 = self.quad * (1 + r**scale_r)
        y = [y1, y2]
        if has_hexa:
            y3 = self.hexa * (1 + r**scale_r)
            y = [y1, y2, y3]

        if errors:
            dy1 = self.dmono * (1 + r**scale_r)
            dy2 = self.dquad * (1 + r**scale_r)
            dy = [dy1, dy2]
            if has_hexa:
                dy3 = self.dhexa * (1 + r**scale_r)
                dy = [dy1, dy2, dy3] 

        if fig is None:
            fig, axes = plt.subplots(nrows=1, ncols=len(y), figsize=figsize)
        else: 
            axes = fig.get_axes()

        for i in range(len(y)):
            ax = axes[i]
            if errors:
                ax.errorbar(r, y[i], dy[i], **kwargs)
            else:
                ax.plot(r, y[i], **kwargs)
 
            if scale_r == 0:
                ax.set_ylabel(r'$\xi_%d$'%(i*2))
            else:
                ax.set_ylabel(r'$r^{%d} \xi_{%d} \ [h^{-%d} {\rm Mpc}^{%d}]$'%\
                            (scale_r, 2*i, scale_r, scale_r))
            ax.axhline(0, color='k', ls=':')
            ax.set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
        plt.tight_layout()

        return fig

    def plot_many(self, fig=None, figsize=(12, 5), scale_r=2, **kwargs):

        if not hasattr(self, 'nmocks'):
            print('This is a single mock')
            return

        if fig is None:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        else: 
            axes = fig.get_axes()

        cc = Multipoles()
        for i in range(self.nmocks):
            cc.r = self.r
            cc.mono = self.monos[i]
            cc.quad = self.quads[i]
            cc.hexa = self.hexas[i]
            cc.plot(fig=fig, errors=0, scale_r=scale_r, **kwargs)

    def export(self, fout):
        r = self.r
        fout = open(fout, 'w')
        print( '#r(Mpc/h) mono quad hexa', file=fout)
        for i in range(r.size):
            line = f'{r[i]} {self.mono[i]}'
            if not self.quad is None:
                line+= f' {self.quad[i]}'
            if not self.hexa is None:
                line+= f' {self.hexa[i]}'
            print(line, file=fout)
        fout.close()
    
    def export_cov(self, fout):
        coss = self.coss
        fout = open(fout, 'w')
        r = self.r

        for i in range(coss.shape[0]):
            for j in range(coss.shape[1]):
                print( i, j, r[i%r.size], r[j%r.size], coss[i, j], file=fout)
        fout.close()

    @staticmethod
    def combine_mocks(root, cute=0, rebin_r=1, shift_r=0):

        allm = np.sort(glob.glob(root))
        nmocks = len(allm)
        if nmocks < 1:
            print('No mocks found!')
            return

        cs = [Multipoles(m, cute=cute, \
              rebin_r=rebin_r, shift_r=shift_r) for m in allm]

        has_hexa = True if not cs[0].hexa is None else False
    
        monos = np.array([c.mono for c in cs])
        quads = np.array([c.quad for c in cs])
        hexas = np.array([c.hexa for c in cs])
      
        #-- covariance matrix 
        x = np.append(monos, quads, axis=1)
        if has_hexa:
            x = np.append(x, hexas, axis=1) 
        coss = np.cov(x.T)
        
        #-- correlation matrix
        corr = coss/np.sqrt(np.outer(np.diag(coss), np.diag(coss)))

        c = cs[0]
        c.mono = np.mean(monos, axis=0)
        c.quad = np.mean(quads, axis=0)
        if has_hexa:
            c.hexa = np.mean(hexas, axis=0)
        else:
            c.hexa = None 

        nbins = c.mono.size
        c.dmono = np.sqrt(np.diag(coss)[:nbins])/np.sqrt(nmocks)
        c.dquad = np.sqrt(np.diag(coss)[nbins:(2*nbins)])/np.sqrt(nmocks)
        if has_hexa:
            c.dhexa = np.sqrt(np.diag(coss)[(2*nbins):(3*nbins)])/np.sqrt(nmocks)
        c.coss = coss
        c.corr = corr
        c.nmocks = nmocks
        c.monos = monos
        c.quads = quads
        c.hexas = hexas

        return c



