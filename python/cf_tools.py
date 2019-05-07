from __future__ import print_function
import numpy as np
import pylab as plt
import os
import copy 
import glob

from scipy.optimize import minimize
#from astropy.io import fits

class Corr:

    def __init__(self, fin):

        ff = open(fin)
        self.wd, self.wd2, self.wr, self.wr2 = [float(s) for s in ff.readline().split()]
        self.header = ff.readline()[:-2]

        self.rp, self.rt, self.cf, self.dcf, self.dd, self.dr, self.rr = \
            np.loadtxt(fin, unpack=1, skiprows=2)
        
        rper = np.unique(self.rt)
        rpar = np.unique(self.rp)
        nper = rper.size
        npar = rpar.size
        self.nper = nper
        self.npar = npar
        self.rper = rper
        self.rpar = rpar
        self.filename = fin
        #self.wp = self.get_wp()

    def shape2d(self, x):
        return np.reshape(x, (self.nper, self.npar))

    def transpose(self, x):
        return self.shape2d(x).T.ravel()

    def get_wp(self):
        return 2 * np.sum(self.shape2d(self.cf)*np.gradient(self.rpar), axis=1)

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
        for i in range(self.cf.size):
            print( self.rp[i], self.rt[i], self.cf[i], self.dcf[i], \
                   self.dd[i], self.dr[i], self.rr[i] , file=fout)
        fout.close()


    @staticmethod
    def coadd_old(c1, c2):
        c = copy.deepcopy(c1)                

        c.dd = c1.dd+c2.dd
        c.dr = c1.dr+c2.dr
        c.rr = c1.rr+c2.rr
        c.wd = c1.wd+c2.wd
        c.wd2= c1.wd2+c2.wd2
        c.wr = c1.wr+c2.wr
        c.wr2 = c1.wr2+c2.wr2
        
        #-- from CUTE
        edd = np.zeros(c.dd.size)
        edr = np.zeros(c.dd.size)
        err = np.zeros(c.dd.size)
        w = (c.dd>0)
        edd[w] = 1./np.sqrt(c.dd[w])
        edr[w] = 1./np.sqrt(c.dr[w])
        err[w] = 1./np.sqrt(c.rr[w])
        
        dd_norm = 0.5*(c.wd**2-c.wd2)
        rr_norm = 0.5*(c.wr**2-c.wr2)
        dr_norm = c.wd*c.wr
        ddn = c.dd/dd_norm
        rrn = c.rr/rr_norm
        drn = c.dr/dr_norm
        c.cf = (ddn/rrn-2*drn/rrn+1)
        c.dcf = (1+c.cf)*(edd+edr+err)

        return c 

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
    def fill_rr(root, nmocks=1000):
        
        c0 = Corr(root%1+'.corr2drmu')
       
        for r in range(2, nmocks+1):
            c = Corr(root%r+'.dddr')
            c.rr = c0.rr
            c.compute_cf()
            c.export( (root%r)+'.corr2drmu')

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
        #dcf = (1+cf)*(1./np.sqrt(dds)+1./np.sqrt(drs)+1./np.sqrt(rrs))
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

        

class Wedges:

    def __init__(self, fin=None, muranges = [[0., 0.5], [0.5, 0.8], [0.8, 1.]], rebin_r=8):
        if fin is not None:
            self.wd, self.wd2, self.wr, self.wr2 = [float(s) for s in open(fin).readline().split()]
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
            if rebin_r>1:
                nr = self.nr//rebin_r
                self.mu2d = np.mean(np.reshape(self.mu2d, (nr, rebin_r, -1)), axis=1)
                self.r2d  = np.mean(np.reshape(self.r2d,  (nr, rebin_r, -1)), axis=1)
                self.dd = np.sum(np.reshape(self.dd,  (nr, rebin_r, -1)), axis=1) 
                self.dr = np.sum(np.reshape(self.dr,  (nr, rebin_r, -1)), axis=1) 
                self.rr = np.sum(np.reshape(self.rr,  (nr, rebin_r, -1)), axis=1) 
                self.r = np.unique(self.r2d)
                self.mu = np.unique(self.mu2d)
                self.nr = self.r.size
                self.nmu = self.mu.size
                dd_norm = 0.5*(self.wd**2-self.wd2)
                rr_norm = 0.5*(self.wr**2-self.wr2)
                dr_norm = self.wd*self.wr
                self.cf = self.dd/self.rr*rr_norm/dd_norm \
                          -2.*self.dr/self.rr*rr_norm/dr_norm + 1.
                self.dcf = (1+self.cf)*(1./np.sqrt(self.dd) + \
                                        1./np.sqrt(self.dr) + \
                                        1./np.sqrt(self.rr))

            self.compute_wedges(muranges=muranges)
            self.nmocks = 1
            self.weds_many = None

    def shape2d(self, x):
        return np.reshape(x, (self.nr, self.nmu))

    def compute_wedges(self, muranges=[[0., 0.5], [0.5, 0.8], [0.8, 1.]]):

        weds = list()
        for murange in muranges:
            w = (self.mu > murange[0])&(self.mu<=murange[1])
            wed = np.mean(self.cf[:, w], axis=1)
            weds.append(wed)

        self.muranges=muranges
        self.weds = np.array(weds)

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
    
    def __init__(self, fin=None,  multipoles=0, rebin_r=1, shift_r=0):

        if fin:
            if multipoles:
                self.read_multipoles(fin)
                return
            try:
                self.mu2d, self.r2d, self.cf, self.dcf, self.dd, self.dr, self.rr = \
                     np.loadtxt(fin, unpack=1)
            except:
                self.wd, self.wd2, self.wr, self.wr2 = \
                     [float(s) for s in open(fin).readline().split()]
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
            if rebin_r>1:
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
                self.cf = self.dd/self.rr*rr_norm/dd_norm \
                          - 2.*self.dr/self.rr*rr_norm/dr_norm + 1.
                self.dcf = (1+self.cf)*(1./np.sqrt(self.dd)+\
                                        1./np.sqrt(self.dr)+\
                                        1./np.sqrt(self.rr))
            self.norm_dd = dd_norm
            self.norm_dr = dr_norm
            self.norm_rr = rr_norm
            self.compute_multipoles()
            self.nmocks = 1
            self.monos = None
            self.quads = None
            self.hexas = None

    def read_multipoles(self, fin):
        data = np.loadtxt(fin)
        if data.shape[1] == 2:
            self.r, self.mono = data[:, 0], data[:, 1]
        else:
            self.r, self.mono, self.quad, self.hexa = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    def read_cov(self, cov_file):

        i, _, r1, _, coss = np.loadtxt(cov_file, unpack=1)
        r = np.unique(r1)
        nr = r.size
        if (r != self.r).any():
            print( 'Warning: covariance matrix is uncompatible with multipoles')

        ni = np.unique(i).size
        coss = np.reshape(coss, (ni, ni))
        dmono = np.sqrt(np.diag(coss)[:nr])
        dquad = np.sqrt(np.diag(coss)[nr:(2*nr)])
        dhexa = np.sqrt(np.diag(coss)[(2*nr):])
        self.dmono = dmono
        self.dquad = dquad
        self.dhexa = dhexa
        self.coss = coss
        self.corr = coss/np.sqrt( np.outer(np.diag(coss), np.diag(coss)) )
        
    def shape2d(self, x):
        return np.reshape(x, (self.nr, self.nmu))

    def compute_multipoles(self):
        
        mu2d = self.mu2d
        cf = self.cf
        dcf = self.dcf 
        dmu = np.gradient(self.mu)[0]
        self.mono = np.sum(cf*(dcf>0)*dmu, axis=1)
        self.quad = np.sum(cf*0.5*(3*mu2d**2-1)*(dcf>0)*dmu, axis=1)*5.
        self.hexa = np.sum(cf*1/8*(35*mu2d**4-30*mu2d**2+3)*(dcf>0)*dmu, axis=1)*9.

    def fit_multipoles(self, mu_min=0., mu_max=1.0):

        def chi2(p, mu, cf, mu_max):
            w = (mu<mu_max)
            model = p[0] + p[1]*0.5*(3*mu[w]**2-1) \
                    #+ p[2]*0.25*(35**mu**4-30*mu**2+3)
            return sum((cf[w]-model)**2)

        self.mono = np.zeros(self.r.size)
        self.quad = np.zeros(self.r.size)

        for i in range(self.r.size):
            par = minimize(chi2, np.array([0., 0.]), \
                           args=(self.mu, self.cf[i], mu_max), \
                           method='Nelder-Mead')
            self.mono[i] = par['x'][0]
            self.quad[i] = par['x'][1]

    def plot(self, fig=None, figsize=(12, 5), errors=0, scale_r=2, **kwargs):

        r = self.r
        y1 = self.mono  * (1 + r**scale_r) 
        y2 = self.quad * (1 + r**scale_r)
        y3 = self.hexa * (1 + r**scale_r)
        y = [y1, y2, y3]
        if errors:
            dy1 = self.dmono * (1 + r**scale_r)
            dy2 = self.dquad * (1 + r**scale_r)
            dy3 = self.dhexa * (1 + r**scale_r)
            dy = [dy1, dy2, dy3]

        if fig is None:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
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
            print(r[i], self.mono[i], self.quad[i], self.hexa[i], file=fout)
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
    def combine_mocks(root, multipoles=0, cov_with_quad=0, \
            rebin_r=1, shift_r=0):

        allm = glob.glob(root)
        nmocks = len(allm)
        if nmocks < 1:
            print('No mocks found!')
            return

        cs = [Multipoles(m, multipoles=multipoles, \
              rebin_r=rebin_r, shift_r=shift_r) for m in allm]
    
        monos = np.array([c.mono for c in cs])
        quads = np.array([c.quad for c in cs])
        hexas = np.array([c.hexa for c in cs])
        
        if cov_with_quad:
            coss = np.cov(np.append(monos, quads, hexas, axis=1).T)
        else:
            coss = np.cov(monos.T)
        
        #-- correlation matrix
        corr = coss/np.sqrt(np.outer(np.diag(coss), np.diag(coss)))

        c = cs[0]
        c.mono = np.mean(monos, axis=0)
        c.quad = np.mean(quads, axis=0)
        if cov_with_quad:
            nbins = c.mono.size
            c.dmono = np.sqrt(np.diag(coss)[:nbins])/np.sqrt(nmocks)
            c.dquad = np.sqrt(np.diag(coss)[nbins:])/np.sqrt(nmocks)
        else:
            c.dmono = np.sqrt(np.diag(coss))/np.sqrt(nmocks)
            c.dquad = np.std(quads, axis=0)/np.sqrt(nmocks)
        c.coss = coss
        c.corr = corr
        c.nmocks = nmocks
        c.monos = monos
        c.quads = quads

        return c

    @staticmethod
    def plot_north_south_combined(root, multipoles=0, rebin_r=8):
        mn = Multipoles(root%'North', \
                multipoles=multipoles, rebin_r=rebin_r)
        ms = Multipoles(root%'South', \
                multipoles=multipoles, rebin_r=rebin_r)
        m =  Multipoles(root%'Combined', \
                multipoles=multipoles, rebin_r=rebin_r)
        plt.figure()
        mn.plot(quad=1, label='North')
        ms.plot(quad=1, label='South')
        m.plot(quad=1, label='Combined')
        plt.subplot(211)
        plt.legend(loc=3, numpoints=1, fontsize=12)
        plt.draw()

        
    @staticmethod
    def coadd(m1, m2):
        m = copy.deepcopy(m1)

        #-- ratio of number of galaxy pairs
        f = m2.norm_dd/m1.norm_dd
        
        m.dd = m1.dd/m1.norm_dd + f*m2.dd/m2.norm_dd
        m.dr = m1.dr/m1.norm_dr + f*m2.dr/m2.norm_dr
        m.rr = m1.rr/m1.norm_rr + f*m2.rr/m2.norm_rr
        m.cf = m.dd/m.rr - 2*m.dr/m.rr + 1.
        m.dcf = m.cf*0+1
        m.compute_multipoles()
        return m


