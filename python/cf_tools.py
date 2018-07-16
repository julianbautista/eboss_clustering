from __future__ import print_function
import numpy as N
import pylab as P
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
            N.loadtxt(fin, unpack=1, skiprows=2)
        
        rper = N.unique(self.rt)
        rpar = N.unique(self.rp)
        nper = rper.size
        npar = rpar.size
        self.nper = nper
        self.npar = npar
        self.rper = rper
        self.rpar = rpar
        self.filename = fin
        #self.wp = self.get_wp()

    def shape2d(self, x):
        return N.reshape(x, (self.nper, self.npar))

    def transpose(self, x):
        return self.shape2d(x).T.ravel()

    def get_wp(self):
        return 2 * N.sum(self.shape2d(self.cf)*N.gradient(self.rpar), axis=1)

    def compute_cf(self):
       
        #-- sum of pair counts
        dd_norm = 0.5*(self.wd**2-self.wd2)
        rr_norm = 0.5*(self.wr**2-self.wr2)
        dr_norm = self.wd*self.wr
        edd = self.dd*0
        edr = self.dd*0
        err = self.dd*0
        w = self.dd > 0
        edd[w] = 1./N.sqrt(self.dd[w])
        edr[w] = 1./N.sqrt(self.dr[w])
        err[w] = 1./N.sqrt(self.rr[w])
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

    def export_pylya(self, fout='', zeff=0.74):

        if fout=='':
            fout = self.filename

        #-- need to transpose everything
        #-- convetion CUTE != convention baofit or pylya

        da = self.transpose(self.cf) 
        co = N.diag(self.transpose(self.dcf**2))
        dm = N.diag(N.ones(da.size))
        rp = self.transpose(self.rp)
        rt = self.transpose(self.rt)
        z = N.ones(da.size)*zeff
        col1=fits.Column(name="DA",format='D',array=da)
        col2=fits.Column(name="CO",format=str(len(da))+'D',array=co)
        col3=fits.Column(name="DM",format=str(len(da))+'D',array=dm)
        col4=fits.Column(name="RP",format='D',array=rp)
        col5=fits.Column(name="RT",format='D',array=rt)
        col6=fits.Column(name="Z",format='D',array=z)
        cols=fits.ColDefs([col1,col2,col3,col4,col5,col6])
        tbhdu=fits.BinTableHDU.from_columns(cols)
        tbhdu.writeto(fout+"-exp.fits", clobber=True)

    @staticmethod
    def coadd_old(c1, c2):
        c = copy.deepcopy(c1)                

        f = c2.wd/c1.wd

        c.dd = c1.dd+c2.dd
        c.dr = c1.dr+c2.dr
        c.rr = c1.rr+c2.rr
        c.wd = c1.wd+c2.wd
        c.wd2= c1.wd2+c2.wd2
        c.wr = c1.wr+c2.wr
        c.wr2 = c1.wr2+c2.wr2
        
        #-- from CUTE
        edd = N.zeros(c.dd.size)
        edr = N.zeros(c.dd.size)
        err = N.zeros(c.dd.size)
        w = (c.dd>0)
        edd[w] = 1./N.sqrt(c.dd[w])
        edr[w] = 1./N.sqrt(c.dr[w])
        err[w] = 1./N.sqrt(c.rr[w])
        
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
        c.dcf = N.ones(cf.shape)

        return c

    @staticmethod
    def coadd_north_south(root):
        ''' Coadds North and South
            
            Input
            -----
            root: string
                on the format path/to/mock-1.8-LRG-%s.corr2drmu 
        '''

        cn = Corr(root%'North')
        cs = Corr(root%'South')
        c = Corr.coadd(cn, cs)
        c.export(root%'Combined')

    @staticmethod
    def coadd_north_south_mocks(root, nmocks=1000):
        for r in range(1, nmocks+1):
            cn = Corr(root%('North', r))
            cs = Corr(root%('South', r%nmocks+1))
            c = Corr.coadd(cn, cs)
            cname = root%('Combined', r)
            c.export(cname)

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
    
        cfs = N.mean(N.array([c.cf for c in cs]), axis=0)
        dds = N.sum( N.array([c.dd for c in cs]), axis=0)
        drs = N.sum( N.array([c.dr for c in cs]), axis=0)
        rrs = N.sum( N.array([c.rr for c in cs]), axis=0)
        wd = N.sum( N.array([c.wd for c in cs]))
        wd2 = N.sum( N.array([c.wd for c in cs]))
        wr = N.sum( N.array([c.wd for c in cs]))
        wr2 = N.sum( N.array([c.wd for c in cs]))
        norm_dd = N.sum(N.array([0.5*(c.wd**2-c.wd2) for c in cs]))
        norm_rr = N.sum(N.array([0.5*(c.wr**2-c.wr2) for c in cs]))
        norm_dr = N.sum(N.array([c.wd*c.wr for c in cs]))
        cf = dds/rrs*norm_rr/norm_dd-2.*drs/rrs*norm_rr/norm_dr+1.
        #dcf = (1+cf)*(1./N.sqrt(dds)+1./N.sqrt(drs)+1./N.sqrt(rrs))
        dcf = N.std(N.array([c.cf for c in cs]), axis=0)

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
                 N.loadtxt(fin, unpack=1, skiprows=2)           

            self.mu = N.unique(self.mu2d)
            self.r = N.unique(self.r2d)
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
                nr = self.nr/rebin_r
                self.mu2d = N.mean(N.reshape(self.mu2d, (nr, rebin_r, -1)), axis=1)
                self.r2d  = N.mean(N.reshape(self.r2d,  (nr, rebin_r, -1)), axis=1)
                self.dd = N.sum(N.reshape(self.dd,  (nr, rebin_r, -1)), axis=1) 
                self.dr = N.sum(N.reshape(self.dr,  (nr, rebin_r, -1)), axis=1) 
                self.rr = N.sum(N.reshape(self.rr,  (nr, rebin_r, -1)), axis=1) 
                self.r = N.unique(self.r2d)
                self.mu = N.unique(self.mu2d)
                self.nr = self.r.size
                self.nmu = self.mu.size
                dd_norm = 0.5*(self.wd**2-self.wd2)
                rr_norm = 0.5*(self.wr**2-self.wr2)
                dr_norm = self.wd*self.wr
                self.cf = self.dd/self.rr*rr_norm/dd_norm-2.*self.dr/self.rr*rr_norm/dr_norm+1.
                self.dcf = (1+self.cf)*(1./N.sqrt(self.dd)+1./N.sqrt(self.dr)+1./N.sqrt(self.rr))

            self.compute_wedges(muranges=muranges)

    def shape2d(self, x):
        return N.reshape(x, (self.nr, self.nmu))

    def compute_wedges(self, muranges=[[0., 0.5], [0.5, 0.8], [0.8, 1.]]):

        nwed = len(muranges)
        weds = list()

        for murange in muranges:
            w = (self.mu > murange[0])&(self.mu<=murange[1])
            wed = N.mean(self.cf[:, w], axis=1)
            weds.append(wed)

        self.muranges=muranges
        self.weds = N.array(weds)

    def plot(self, scale_r=2, n=-1, color=None, alpha=None, label=None):

        r = self.r
       
        weds = self.weds*1.
        if n!=-1:
            weds = weds[n, None]
        nweds = weds.shape[0]
        
        for i in range(nweds):
            P.subplot(nweds, 1, 1+i)
            y = weds[i]*r**scale_r
            P.plot(r, y, label=label, color=color, alpha=alpha)
            if scale_r == 0:
                P.ylabel(r'$\xi_{\mu%d}$'%i)
            else:
                P.ylabel(r'$r^{%d} \xi_{\mu%d}$ [$h^{%d}$ Mpc$^{%d}]$'%\
                         (scale_r, i, -scale_r, scale_r))
        P.xlabel(r'$r$ [$h^{-1}$ Mpc]')

    def plot_many(self, label=None, n=-1, scale_r=2, alpha=1.0, color=None):

        if not hasattr(self, 'nmocks'):
            print('This is a single mock')
            return

        cc = Wedges()
        for i in range(self.nmocks):
            cc.r = self.r
            cc.weds = self.weds_many[i]
            cc.plot(n=n, scale_r=scale_r, alpha=alpha, color=color)

    @staticmethod
    def combine_mocks(root, rebin_r=8):

        allm = glob.glob(root)
        nmocks = len(allm)

        cs = [Wedges(m, rebin_r=rebin_r) for m in allm]
    
        weds_many = N.array([c.weds for c in cs])
        weds_flat = N.array([w.ravel() for w in weds_many])
        coss = N.cov(weds_flat.T)
        
        #-- correlation matrix
        corr = coss/N.sqrt(N.outer(N.diag(coss), N.diag(coss)))

        c = cs[0]
        c.weds = N.mean(weds_many, axis=0)
        c.dweds = N.reshape(N.sqrt(N.diag(coss))/N.sqrt(nmocks), c.weds.shape)
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
                     N.loadtxt(fin, unpack=1)
            except:
                self.wd, self.wd2, self.wr, self.wr2 = \
                     [float(s) for s in open(fin).readline().split()]
                self.mu2d, self.r2d, self.cf, self.dcf, self.dd, self.dr, self.rr = \
                     N.loadtxt(fin, unpack=1, skiprows=2)           
 
            self.mu = N.unique(self.mu2d)
            self.r = N.unique(self.r2d)
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
                self.mu2d = N.mean(N.reshape(
                                self.mu2d[shift_r:-rebin_r+shift_r],
                                             (nr, rebin_r, -1)), axis=1)
                self.r2d  = N.mean(N.reshape(
                                self.r2d[shift_r:-rebin_r+shift_r], 
                                             (nr, rebin_r, -1)), axis=1)
                self.dd =   N.sum( N.reshape(
                                self.dd[shift_r:-rebin_r+shift_r], 
                                             (nr, rebin_r, -1)), axis=1) 
                self.dr =   N.sum( N.reshape(
                                self.dr[shift_r:-rebin_r+shift_r], 
                                             (nr, rebin_r, -1)), axis=1) 
                self.rr =   N.sum( N.reshape(
                                self.rr[shift_r:-rebin_r+shift_r],
                                             (nr, rebin_r, -1)), axis=1) 
                self.r = N.unique(self.r2d)
                self.mu = N.unique(self.mu2d)
                self.nr = self.r.size
                self.nmu = self.mu.size
                self.cf = self.dd/self.rr*rr_norm/dd_norm \
                          - 2.*self.dr/self.rr*rr_norm/dr_norm + 1.
                self.dcf = (1+self.cf)*(1./N.sqrt(self.dd)+\
                                        1./N.sqrt(self.dr)+\
                                        1./N.sqrt(self.rr))
            self.norm_dd = dd_norm
            self.norm_dr = dr_norm
            self.norm_rr = rr_norm
            self.compute_multipoles()

    def read_multipoles(self, fin):
        data = N.loadtxt(fin)
        if data.shape[1] == 2:
            self.r, self.mono = data[:, 0], data[:, 1]
        else:
            self.r, self.mono, self.quad = data[:, 0], data[:, 1], data[:, 2]

    def read_cov(self, cov_file):

        i, j, r1, r2, coss = N.loadtxt(cov_file, unpack=1)
        r = N.unique(r1)
        nr = r.size
        if (r != self.r).any():
            print( 'Warning: covariance matrix is uncompatible with multipoles')

        ni = N.unique(i).size
        coss = N.reshape(coss, (ni, ni))
        dmono = N.sqrt(N.diag(coss)[:nr])
        dquad = N.sqrt(N.diag(coss)[nr:])
        self.dmono = dmono
        self.dquad = dquad
        self.coss = coss
        self.corr = coss/N.sqrt( N.outer(N.diag(coss), N.diag(coss)) )
        
    def shape2d(self, x):
        return N.reshape(x, (self.nr, self.nmu))

    def compute_multipoles(self):
        
        mu2d = self.mu2d
        cf = self.cf
        dcf = self.dcf 

        self.mono = N.sum(cf*(dcf>0), axis=1)/N.sum(dcf>0., axis=1)
        self.quad = N.sum(cf*0.5*(3*mu2d**2-1)*(dcf>0), axis=1)/\
                    N.sum(dcf>0., axis=1)*5.

    def fit_multipoles(self, mu_min=0., mu_max=1.0):

        def chi2(p, mu, cf, mu_max):
            w = (mu<mu_max)
            model = p[0] + p[1]*0.5*(3*mu[w]**2-1) \
                    #+ p[2]*0.25*(35**mu**4-30*mu**2+3)
            return sum((cf[w]-model)**2)

        self.mono = N.zeros(self.r.size)
        self.quad = N.zeros(self.r.size)

        for i in range(self.r.size):
            par = minimize(chi2, N.array([0., 0.]), \
                           args=(self.mu, self.cf[i], mu_max), \
                           method='Nelder-Mead')
            self.mono[i] = par['x'][0]
            self.quad[i] = par['x'][1]



    def plot(self, label=None, quad=0, errors=0, scale_r=2, \
             alpha=1.0, color=None):

        r = self.r
        y1 = self.mono  * (1 + r**scale_r) 
        if quad:
            y2 = self.quad  * (1 + r**scale_r)
        if errors:
            dy1 = self.dmono * (1 + r**scale_r)
            dy2 = self.dquad * (1 + r**scale_r)
        
        if quad:
            P.subplot(211)
        if errors:
            P.errorbar(r, y1, dy1, fmt='o', label=label, \
                       color=color, alpha=alpha)
        else:
            P.plot(r, y1, label=label, color=color, alpha=alpha)
        if scale_r == 0:
            P.ylabel(r'$\xi_0$$')
        else:
            P.ylabel(r'$r^{%d} \xi_0$ [$h^{%d}$ Mpc$^{%d}]$'%\
                     (scale_r, -scale_r, scale_r))
        if quad:
            P.subplot(212)
            if errors:
                P.errorbar(r, y2, dy2, fmt='o', color=color, alpha=alpha)
            else:
                P.plot(r, y2, color=color, alpha=alpha)
            if scale_r == 0:
                P.ylabel(r'$\xi_2$$')
            else:
                P.ylabel(r'$r^{%d} \xi_2$ [$h^{%d}$ Mpc$^{%d}]$'%\
                         (scale_r, -scale_r, scale_r))
        P.xlabel(r'$r$ [$h^{-1}$ Mpc]')

    def plot_many(self, label=None, quad=0, errors=1, \
                  scale_r=2, alpha=1.0, color=None):

        if not hasattr(self, 'nmocks'):
            print('This is a single mock')
            return

        cc = Multipoles()
        for i in range(self.nmocks):
            cc.r = self.r
            cc.mono = self.monos[i]
            cc.quad = self.quads[i]
            cc.plot(quad=quad, errors=0, scale_r=scale_r, alpha=alpha, color=color)

    def export(self, fout):
        r = self.r
        fout = open(fout, 'w')
        print( '#r(Mpc/h) mono quad', file=fout)
        for i in range(r.size):
            print(r[i], self.mono[i], self.quad[i], file=fout)
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
    
        monos = N.array([c.mono for c in cs])
        quads = N.array([c.quad for c in cs])
        
        if cov_with_quad:
            coss = N.cov(N.append(monos, quads, axis=1).T)
        else:
            coss = N.cov(monos.T)
        
        #-- correlation matrix
        corr = coss/N.sqrt(N.outer(N.diag(coss), N.diag(coss)))

        c = cs[0]
        c.mono = N.mean(monos, axis=0)
        c.quad = N.mean(quads, axis=0)
        if cov_with_quad:
            nbins = c.mono.size
            c.dmono = N.sqrt(N.diag(coss)[:nbins])/N.sqrt(nmocks)
            c.dquad = N.sqrt(N.diag(coss)[nbins:])/N.sqrt(nmocks)
        else:
            c.dmono = N.sqrt(N.diag(coss))/N.sqrt(nmocks)
            c.dquad = N.std(quads, axis=0)/N.sqrt(nmocks)
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
        P.figure()
        mn.plot(quad=1, label='North')
        ms.plot(quad=1, label='South')
        m.plot(quad=1, label='Combined')
        P.subplot(211)
        P.legend(loc=3, numpoints=1, fontsize=12)
        P.draw()

        
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


