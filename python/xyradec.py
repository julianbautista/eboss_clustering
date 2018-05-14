from __future__ import print_function
from scipy.optimize import minimize
import numpy as N
import pylab as P
import os
import sys
from ebosscat import Catalog, Mask
#from pbs import queue


class xytrans:

    def __init__(self, *args):
        if len(args)==4:
            self.read(args[0], args[1], args[2], args[3])
        
    def read(self, ra, dec, x, y):
        self.ra = ra - 360*(ra>300.)
        self.dec = dec
        self.med_ra = N.median(self.ra)
        self.med_dec = N.median(dec)
        self.x = x
        self.y = y
        self.fit_xy()

    def radec2pos(self, c, ra, dec):
        mra = self.med_ra
        mdec = self.med_dec
        newpos = c[0]*(ra-mra)**2 + c[1]*(dec-mdec)**2 + c[2]*(ra-mra)*(dec-mdec) \
                + c[3]*(ra-mra) + c[4]*(dec-mdec) + c[5]
        return newpos

    def chi2(self, c, ra, dec, pos):
        newpos = self.radec2pos(c, ra, dec)
        return sum((newpos-pos)**2)/(pos.size-6)

    def fit_xy(self):
        self.res_x = minimize(self.chi2, N.zeros(6), \
                              args=(self.ra, self.dec, self.x), tol=0.1)
        self.res_y = minimize(self.chi2, N.zeros(6), \
                              args=(self.ra, self.dec, self.y), tol=0.1)
        self.cx = self.res_x['x']
        self.cy = self.res_y['x']

    def radec2xy(self, ra, dec):
        ra = ra*1. - 360*(ra>300.)
        x = self.radec2pos(self.cx, ra, dec)
        y = self.radec2pos(self.cy, ra, dec)
        return x, y

    @staticmethod 
    def get_fits(spall, test=0):

        xys = dict()
        for p in N.unique(spall.PLATE):
            w = spall.PLATE == p
            a = spall[w]
            xymap = xytrans(a.RA, a.DEC, a.XFOCAL, a.YFOCAL)
            xymap.plate = p
            xys[p]  = xymap
            rx = xymap.res_x
            ry = xymap.res_y
            print(p, sum(w), rx['fun'], ry['fun'], rx['success'], ry['success'])
        return xys

    @staticmethod
    def export_fits(xys, fout):

        fout = open(fout, 'w')
        plates = N.unique( [xy.plate for xy in xys.values()])
        
        header = '# plate  med_ra  med_dec  coeffs_x  coeffs_y'
        print(header, file=fout)

        for plate in plates:
            xy = xys[plate]
            line = '%d  %f  %f  '%(plate, xy.med_ra, xy.med_dec)
            line+= '  '.join(['%f'%c for c in xy.cx])
            line+='  '+'  '.join(['%f'%c for c in xy.cy])
            print(line)
            print(line, file=fout)

        fout.close()

    @staticmethod
    def read_fits(fin):

        fin = open(fin)
        header = fin.readline()
        xys = dict()
        for line in fin:
            line = line.split()
            xy = xytrans()
            xy.plate = int(line[0])
            xy.med_ra = float(line[1])
            xy.med_dec = float(line[2])
            xy.cx = N.array([float(s) for s in line[3:9]])
            xy.cy = N.array([float(s) for s in line[9:]])
            xys[xy.plate] = xy
        return xys

    @staticmethod
    def get_xy(ra, dec, plates, xys):

        xfocal = N.zeros(ra.size)
        yfocal = N.zeros(ra.size)

        for plate in N.unique(plates):
            #- get xytrans object corresponding to this plate
            if plate not in xys:
                print('plate not found', plate)
                continue
            w = (plates == plate)
            xfocal[w], yfocal[w] = xys[plate].radec2xy(ra[w], dec[w])

        return xfocal, yfocal



def assign_plates_and_xy(cat, ran, mask):
    ''' Adds SECTOR, PLATE, XFOCAL and YFOCAL to random catalog 

    Inputs
    -----
    cat, ran: Catalog objects
    mask: Mangle mask object

    '''

    maskcut = mask[mask.weights>0]

    cat.get_plates_per_sector()
    ran.SECTOR, ran.PLATE = cat.assign_plates_to_ra_dec(ran.RA, ran.DEC, maskcut)
    
    xys = xytrans.read_fits(os.environ['EBOSS_CLUSTERING_DIR']+'/etc/xyradec_v5_10_0.txt')
    ran.XFOCAL, ran.YFOCAL = xytrans.get_xy(ran.RA, ran.DEC, ran.PLATE, xys)


    #-- hack for bad sectors
    dist = N.sqrt(ran.XFOCAL**2+ran.YFOCAL**2)
    w = (dist>326.5)
    if sum(w)>0:
        print(sum(w), 'galaxies outside plates, distance =', dist[w].min())
        bad_sectors = N.unique(ran.SECTOR[w])
        print('Sectors where this happens: ', bad_sectors)

    return

    wr = N.ones(ran.size)==1
    wd = N.ones(cat.size)==1
    wm = N.ones(mask.sector.size)==1
    for sec in bad_sectors:
        wr &= (ran.SECTOR!=sec)
        wd &= (cat.SECTOR!=sec)
        wm &= (mask.sector!=sec)

    print('%.3f of the footprint removed'%(1 - sum(wr)*1./ran.size))
    cat.cut(wd)
    ran.cut(wr)
    mask = Mask.cut(mask, wm)

