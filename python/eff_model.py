import os
import sys
import numpy as N
import pylab as P

from pbs import queue
from ebosscat import Catalog

class efficiency:

    def __init__(self, cat):
        #self.read_xy_eff2(cat)
        #self.read_spec_eff(cat)
        #self.read_spec_info()
        #self.fit_spec_eff()
        pass

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
        
    def read_xy_eff2(self, cat, nx=25, ny=25):
       
        self.xmin=-325.
        self.ymin=-325.
        xedges = N.linspace(self.xmin, -self.xmin, nx+1)
        yedges = N.linspace(self.ymin, -self.ymin, ny+1)

        xcat = cat.XFOCAL[cat.PLATE!=0]
        ycat = cat.YFOCAL[cat.PLATE!=0]
        imatch = cat.IMATCH[cat.PLATE!=0]

        wgals = (imatch==1)*1.
        wfail = ((imatch==5)|(imatch==7))*1.

        ngals, x, y = N.histogram2d(xcat, ycat, bins=[xedges, yedges], weights=wgals)
        nfail, x, y = N.histogram2d(xcat, ycat, bins=[xedges, yedges], weights=wfail)

        eff = ngals/(ngals+nfail)
        w = (ngals+nfail)<10
        eff[w] = 0

        self.nx = nx
        self.ny = ny
        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.x = xedges
        self.y = yedges
        self.xy_eff = eff
        self.xy_n = (ngals+nfail)

    def get_xy_eff(self, x, y):
        
        ix = (N.floor( (x - self.xmin)/self.dx ) ).astype(int)
        iy = (N.floor( (y - self.ymin)/self.dy ) ).astype(int)
        eff = self.xy_eff[ix, iy]
        #-- check for points outside grid
        w = (ix<0)|(ix>=self.x.size)|(iy<0)|(iy>=self.y.size)
        eff[w] = 0.
        return eff

    def plot_xy_eff(self):
        P.figure()
        cmap = P.get_cmap('jet')
        cmap.set_under('white')
        P.pcolormesh(self.x, self.y, self.xy_eff.T, vmin=0.5, vmax=1.0, cmap=cmap)
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


    def read_spec_eff(self, cat):
    
        wgood = cat.PLATE!=0
        plates = N.unique(cat.PLATE[wgood])
        
        ngals = N.zeros((plates.size, 2))
        nfail = N.zeros((plates.size, 2))
        for i in range(plates.size):
            wplate = cat.PLATE==plates[i]
            fiberid = cat.FIBERID[wplate]
            imatch = cat.IMATCH[wplate]

            w1 = (fiberid<=500)
            w2 = (fiberid> 500)

            ngals[i, 0] = N.sum(imatch[w1]==1)
            ngals[i, 1] = N.sum(imatch[w2]==1)
            nfail[i, 0] = N.sum((imatch[w1]==5)|(imatch[w1]==7))
            nfail[i, 1] = N.sum((imatch[w2]==5)|(imatch[w2]==7))
            #print plates[i], sum(w1), sum(w2), ngals[i], nfail[i]

        self.plates = plates

        #self.spec_eff = ngals*1./(nfail+ngals)
        #self.spec_eff_err = N.sqrt( nfail)*1./(nfail+ngals) 
        self.spec_eff = ngals/nfail
        self.spec_eff_err = N.sqrt( ngals/nfail**2 + ngals**2/nfail**3)
        w = (nfail)==0
        self.spec_eff[w] = 0.
        self.spec_eff_err[w] = 0.
        self.ngals = ngals
        self.nfail = nfail

    def read_spec_info(self):

        spec = N.loadtxt('spPlate_spectroinfo_v5_10_0.txt', unpack=0, skiprows=1)
        keys = N.array(open('spPlate_spectroinfo_v5_10_0.txt').readline().split())
        wk1 = keys == 'SPEC1_I'
        wk2 = keys == 'SPEC2_I'
       
        spec_sn2 = N.zeros((self.plates.size, 2))
        for i, p in enumerate(self.plates):
            w = (spec[:, 0] == p)
            spec_sn2[i, 0] = spec[w, wk1] 
            spec_sn2[i, 1] = spec[w, wk2] 

        self.spec_sn2=spec_sn2

    def fit_spec_eff(self, npoly=1, plotit=0):

        w = self.spec_eff_err.ravel() >0
        x = self.spec_sn2.ravel()[w]
        y = self.spec_eff.ravel()[w]
        dy = self.spec_eff_err.ravel()[w]

        coeff = N.polyfit(x, y, npoly)#, w=1./dy)
        ymodel = N.polyval(coeff, x) 

        chi2 = N.sum( (y-ymodel)**2/dy**2 )
        ndata = y.size
        npars = coeff.size
        rchi2 = chi2/(ndata-npars)

        print chi2, ndata, npars, rchi2
    
        self.coeff = coeff

        if plotit:
            ngals = self.ngals.ravel()
            nfail = self.nfail.ravel()
            eff = ngals/(ngals+nfail)
            #deff = N.sqrt( ngals*1./(ngals+nfail)**2 + ngals**2/(ngals+nfail)**3)
            deff = N.sqrt( nfail*1./(ngals+nfail)**2 + nfail**2/(ngals+nfail)**3)
            P.errorbar(x, eff[w], deff[w], fmt='.', alpha=0.3)
            #P.plot(x, eff[w], '.', alpha=0.5)
            xmodel = N.linspace(x.min(), x.max(), 20)
            ymodel = N.polyval(coeff, xmodel)
            effmodel = self.get_spec_eff(xmodel)
            P.plot(xmodel, effmodel, 'g')
            P.plot(xmodel, 1/(1+1/ymodel), 'r')
            P.xlabel(r'Spectro $(S/N)^2$')
            P.ylabel(r'Efficiency')

    def get_spec_eff(self, specsn2):
        ymodel = N.polyval(self.coeff, specsn2)
        return 1/(1+1/ymodel)

    def read_fiberid(self, cat, nbins=25):
        fiberid = cat.FIBERID[cat.PLATE!=0]
        imatch = cat.IMATCH[cat.PLATE!=0]

        wgals = (imatch==1)*1.
        wfail = ((imatch==5)|(imatch==7))*1.
        edges = N.linspace(1, 1001, nbins+1)
        fibers = 0.5*(edges[1:]+edges[:-1])
        ngals, _ = N.histogram(fiberid, bins=edges, weights=wgals)
        nfail, _ = N.histogram(fiberid, bins=edges, weights=wfail)
        P.figure()
        _ = P.hist(fiberid, bins=edges, weights=wgals, histtype='step')
        _ = P.hist(fiberid, bins=edges, weights=wfail, histtype='step')
        P.xlim(0, 1001) 

    def get_eff(self, plate, x, y):

        xyeff = self.get_xy_eff(x, y)

        spec_eff = N.zeros(plate.size)
        spectroid = (y>0.)*1
        for i, p in enumerate(self.plates):
            w = plate == p
            spec_eff[w] = self.get_spec_eff(self.spec_sn2[i, spectroid[w]])

        return spec_eff*xyeff


def add_failures_to_mock(mock, plates_per_sector, xys, eff, seed=0):

    mock.PLATE, mock.XFOCAL, mock.YFOCAL = \
            get_xy_and_plates_mock(mock.RA, mock.DEC, plates_per_sector, xys, seed=seed)

    xy_eff = eff.get_eff(mock.XFOCAL, mock.YFOCAL)

    r = N.random.rand(mock.size)
    w = (r > xy_eff) 

    mock.IMATCH[w] = 7

    return mock

def add_failures(plates_per_sector, xys, eff):
   
    cosmo = Cosmo(OmegaM=0.29, h=0.7)

    for i in range(1, 501):
        mock = Catalog('mocks/lrgs/1.5/eboss_boss_veto/mock-1.5-LRG-North-%04d.dat.fits'%i)
        mock = add_failures_to_mock(mock, plates_per_sector, xys, eff, seed=i)
        mock.export('mocks/lrgs/1.5/eboss_boss_veto_with_xy_failures_spec1d/mock-1.5-LRG-North-%04d.dat.fits'%i, \
                     cosmo=cosmo)
        print i

def correct_failures(i=1):

    cosmo = Cosmo(OmegaM=0.29, h=0.7)

    geometry = read_mangle_mask(os.environ['MKESAMPLE_DIR']+'/geometry/eboss_geometry_eboss0_eboss5')

    mock = Catalog(os.environ['CLUSTERING_DIR']+\
            '/mocks/lrgs/1.5/eboss_boss_veto_with_xy_failures_spec1d/mock-1.5-LRG-North-%04d.dat.fits'%i)
    mock.SECTOR = get_sectors(mock.RA, mock.DEC, mask=geometry)
    mock.fiber_collision()
    mock.export(os.environ['CLUSTERING_DIR']+\
            '/mocks/lrgs/1.5/eboss_boss_veto_with_xy_failures_spec1d/mock-1.5-LRG-North-%04d-wnoz.dat.fits'%i, \
            cosmo=cosmo)

def correct_failures_batch():
   
    qe = queue()
    qe.verbose = True
    qe.create(label='correct_failures', alloc='sdss-kp', nodes=8, ppn=16, walltime='100:00:00',umask='0027')
    for i in range(1, 501):
        if os.path.exists(os.environ['CLUSTERING_DIR']+\
                '/mocks/lrgs/1.5/eboss_boss_veto_with_xy_failures_spec1d/mock-1.5-LRG-North-%04d-wnoz.dat.fits'%i):
            continue
        script = 'cd %s; python python/xyradec.py %d' %(os.environ['CLUSTERING_DIR'], i)
        qe.append(script)
    qe.commit(hard=True,submit=True)



if __name__ == '__main__':
    if len(sys.argv)>1:
        correct_failures(i=int(sys.argv[1]))


