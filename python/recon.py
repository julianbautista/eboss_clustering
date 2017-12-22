import numpy as N
import pylab as P
import os
import json
from numba import jit
from scipy.ndimage.filters import gaussian_filter
from scipy.fftpack import fftn, ifftn, fftshift, fftfreq
from ebosscat import Catalog, Cosmo
import pyfftw

class Recon:

    def __init__(self, cat, ran, bias=2.3, f=0.817, smooth=15., nbins=256, \
                 redshift_max=1.0, padding=200., opt_box=1, nthreads=1):

        plotit=0
        beta = f/bias

        #-- parameters of box
        cosmo = Cosmo(OmegaM=0.31)
        print 'Num bins:', nbins
        print 'Smoothing [Mpc/h]:', smooth

        #-- getting weights
        cat.weight = cat.get_weights(fkp=1, noz=1, cp=1, syst=1)
        ran.weight = ran.get_weights(fkp=1, noz=0, cp=0, syst=0)

        #-- computing cartesian positions
        cat.dist = cosmo.get_comoving_distance(cat.Z)
        ran.dist = cosmo.get_comoving_distance(ran.Z)
        cat.x = cat.dist * N.cos(cat.DEC*N.pi/180)*N.sin(cat.RA*N.pi/180)
        cat.y = cat.dist * N.cos(cat.DEC*N.pi/180)*N.cos(cat.RA*N.pi/180)
        cat.z = cat.dist * N.sin(cat.DEC*N.pi/180) 
        cat.newx = cat.x*1.
        cat.newy = cat.y*1.
        cat.newz = cat.z*1.
        ran.x = ran.dist * N.cos(ran.DEC*N.pi/180)*N.sin(ran.RA*N.pi/180)
        ran.y = ran.dist * N.cos(ran.DEC*N.pi/180)*N.cos(ran.RA*N.pi/180)
        ran.z = ran.dist * N.sin(ran.DEC*N.pi/180) 
        ran.newx = ran.x*1.
        ran.newy = ran.y*1.
        ran.newz = ran.z*1.

        print 'Randoms min of x, y, z', min(ran.x), min(ran.y), min(ran.z)
        print 'Randoms max of x, y, z', max(ran.x), max(ran.y), max(ran.z)

        sum_wgal = N.sum(cat.weight)
        sum_wran = N.sum(ran.weight)
        alpha = sum_wgal/sum_wran
        ran_min = 0.01*sum_wran/ran.size

        self.nbins=nbins
        self.bias=bias
        self.f=f
        self.beta=beta
        self.smooth=smooth
        self.cat = cat
        self.ran = ran
        self.ran_min = ran_min
        self.alpha=alpha
        self.cosmo = cosmo
        self.nthreads = nthreads

        self.compute_box(padding=padding, optimize_box=opt_box)

    def compute_box(self, padding=200., optimize_box=1):
    
        if optimize_box:
            dx = max(self.ran.x)-min(self.ran.x)
            dy = max(self.ran.y)-min(self.ran.y)
            dz = max(self.ran.z)-min(self.ran.z)
            x0 = 0.5*(max(self.ran.x)+min(self.ran.x)) 
            y0 = 0.5*(max(self.ran.y)+min(self.ran.y)) 
            z0 = 0.5*(max(self.ran.z)+min(self.ran.z)) 

            box = max([dx, dy, dz])+2*padding
            self.xmin = x0-box/2 
            self.ymin = y0-box/2 
            self.zmin = z0-box/2 
            self.box = box
            self.binsize = box/self.nbins
        else:
            box = self.cosmo.get_comoving_distance(1.05)
            self.xmin=-box
            self.ymin=-box
            self.zmin=-box
            self.box = box*2.
            self.binsize = 2.*box/self.nbins
        
        print 'Box size [Mpc/h]:', self.box
        print 'Bin size [Mpc/h]:', self.binsize

    #@profile
    def iterate(self, iloop, save_wisdom=1):
        cat = self.cat
        ran = self.ran
        smooth = self.smooth
        binsize = self.binsize
        beta = self.beta
        bias = self.bias
        f = self.f
        nbins = self.nbins

        #-- Creating arrays for FFTW
        if iloop==0:
            delta  = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            deltak = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            psi_x  = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            psi_y  = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            psi_z  = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            #delta = N.zeros((nbins, nbins, nbins), dtype='complex128')
            #deltak= N.zeros((nbins, nbins, nbins), dtype='complex128')
            #psi_x = N.zeros((nbins, nbins, nbins), dtype='complex128')
            #psi_y = N.zeros((nbins, nbins, nbins), dtype='complex128')
            #psi_z = N.zeros((nbins, nbins, nbins), dtype='complex128')
            
            print 'Allocating randoms in cells...'
            deltar = self.allocate_gal_cic(ran)
            print 'Smoothing...'
            deltar = gaussian_filter(deltar, smooth/binsize)

            #-- Initialize FFT objects and load wisdom if available
            wisdomFile = "wisdom."+str(nbins)+"."+str(self.nthreads)               
            if os.path.isfile(wisdomFile) :
                print 'Reading wisdom from ',wisdomFile
                g = open(wisdomFile, 'r')
                wisd=json.load(g)
                pyfftw.import_wisdom(wisd)
                g.close()
            print 'Creating FFTW objects...'
            fft_obj = pyfftw.FFTW(delta, delta, axes=[0, 1, 2], threads=self.nthreads)
            ifft_obj = pyfftw.FFTW(deltak, psi_x, axes=[0, 1, 2], \
                                   threads=self.nthreads, \
                                   direction='FFTW_BACKWARD')
        else:
            delta = self.delta
            deltak = self.deltak
            deltar=self.deltar
            psi_x = self.psi_x
            psi_y = self.psi_y
            psi_z = self.psi_z
            fft_obj = self.fft_obj
            ifft_obj = self.ifft_obj

        #fft_obj = pyfftw.FFTW(delta, delta, threads=self.nthreads, axes=[0, 1, 2])
        #-- Allocate galaxies and randoms to grid with CIC method
        #-- using new positions
        print 'Allocating galaxies in cells...'
        deltag = self.allocate_gal_cic(cat)
        print 'Smoothing...'
        deltag = gaussian_filter(deltag, smooth/binsize)

        print 'Computing fluctuations...'
        delta[:]  = deltag - self.alpha*deltar
        w=N.where(deltar>self.ran_min)
        delta[w] = delta[w]/(self.alpha*deltar[w])
        w2=N.where((deltar<=self.ran_min)) 
        delta[w2] = 0.
        w3=N.where(delta>N.percentile(delta[w].ravel(), 99))
        delta[w3] = 0.
        del(w)
        del(w2)
        del(w3)
        del(deltag)

        print 'Fourier transforming delta field...'
        norm_fft = 1.#binsize**3 
        fft_obj(input_array=delta, output_array=delta)
        #delta = pyfftw.builders.fftn(\
        #                delta, axes=[0, 1, 2], \
        #                threads=self.nthreads, overwrite_input=True)()

        #-- delta/k**2 
        k = fftfreq(self.nbins, d=binsize)*2*N.pi
        delta /= k[:, None, None]**2 + k[None, :, None]**2 + k[None, None, :]**2
        delta[0, 0, 0] = 1 

        #-- Estimating the IFFT in Eq. 12 of Burden et al. 2015
        print 'Inverse Fourier transforming to get psi...'
        norm_ifft = 1.#(k[1]-k[0])**3/(2*N.pi)**3*nbins**3

        deltak[:] = delta*-1j*k[:, None, None]/bias
        ifft_obj(input_array=deltak, output_array=psi_x)
        deltak[:] = delta*-1j*k[None, :, None]/bias
        ifft_obj(input_array=deltak, output_array=psi_y)
        deltak[:] = delta*-1j*k[None, None, :]/bias
        ifft_obj(input_array=deltak, output_array=psi_z)

        #psi_x = pyfftw.builders.ifftn(\
        #                delta*-1j*k[:, None, None]/bias, \
        #                axes=[0, 1, 2], \
        #                threads=self.nthreads, overwrite_input=True)().real
        #psi_y = pyfftw.builders.ifftn(\
        #                delta*-1j*k[None, :, None]/bias, \
        #                axes=[0, 1, 2], \
        #                threads=self.nthreads, overwrite_input=True)().real
        #psi_z = pyfftw.builders.ifftn(\
        #                delta*-1j*k[None, None, :]/bias, \
        #                axes=[0, 1, 2], \
        #                threads=self.nthreads, overwrite_input=True)().real
        #psi_x = ifftn(-1j*delta*k[:, None, None]/bias).real*norm_ifft
        #psi_y = ifftn(-1j*delta*k[None, :, None]/bias).real*norm_ifft
        #psi_z = ifftn(-1j*delta*k[None, None, :]/bias).real*norm_ifft

        #-- removing RSD from galaxies
        shift_x, shift_y, shift_z =  \
                self.get_shift(cat, psi_x.real, psi_y.real, psi_z.real, \
                               use_newpos=True)
        for i in range(10):
            print shift_x[i], shift_y[i], shift_z[i], cat.x[i]

        #-- for first loop need to approximately remove RSD component 
        #-- from Psi to speed up calculation
        #-- first loop so want this on original positions (cp), 
        #-- not final ones (np) - doesn't actualy matter
        if iloop==0:
            psi_dot_rhat = (shift_x*cat.x + \
                            shift_y*cat.y + \
                            shift_z*cat.z ) /cat.dist
            shift_x-= beta/(1+beta)*psi_dot_rhat*cat.x/cat.dist
            shift_y-= beta/(1+beta)*psi_dot_rhat*cat.y/cat.dist
            shift_z-= beta/(1+beta)*psi_dot_rhat*cat.z/cat.dist
        #-- remove RSD from original positions (cp) of 
        #-- galaxies to give new positions (np)
        #-- these positions are then used in next determination of Psi, 
        #-- assumed to not have RSD.
        #-- the iterative procued then uses the new positions as 
        #-- if they'd been read in from the start
        psi_dot_rhat = (shift_x*cat.x+shift_y*cat.y+shift_z*cat.z)/cat.dist
        cat.newx = cat.x + f*psi_dot_rhat*cat.x/cat.dist 
        cat.newy = cat.y + f*psi_dot_rhat*cat.y/cat.dist 
        cat.newz = cat.z + f*psi_dot_rhat*cat.z/cat.dist 

        self.deltar = deltar
        self.delta = delta
        self.deltak = deltak
        self.psi_x = psi_x
        self.psi_y = psi_y
        self.psi_z = psi_z
        self.fft_obj = fft_obj
        self.ifft_obj = ifft_obj



        #-- save wisdom
        wisdomFile = "wisdom."+str(nbins)+"."+str(self.nthreads)               
        if iloop==0 and save_wisdom and not os.path.isfile(wisdomFile):
            wisd=pyfftw.export_wisdom()
            f = open(wisdomFile, 'w')
            json.dump(wisd,f)
            f.close()
            print 'Wisdom saved at', wisdomFile

    def apply_shifts(self, verbose=1):
        
        for c in [self.cat, self.ran]:
            shift_x, shift_y, shift_z =  \
                self.get_shift(c, \
                    self.psi_x.real, self.psi_y.real, self.psi_z.real, \
                    use_newpos=False)
            c.newx += shift_x 
            c.newy += shift_y 
            c.newz += shift_z

    def summary(self):

        cat = self.cat
        sx = cat.newx-cat.x
        sy = cat.newy-cat.y
        sz = cat.newz-cat.z
        print 'Shifts stats:'
        for s in [sx, sy, sz]:
            print N.std(s), N.percentile(s, 16), N.percentile(s, 84), \
                    N.min(s), N.max(s)


    def allocate_gal_cic(self, c):
        xmin=self.xmin
        ymin=self.ymin
        zmin=self.zmin
        binsize=self.binsize
        nbins=self.nbins

        xpos = (c.newx-xmin)/binsize
        ypos = (c.newy-ymin)/binsize
        zpos = (c.newz-zmin)/binsize

        i = xpos.astype(int)
        j = ypos.astype(int)
        k = zpos.astype(int)

        ddx = xpos-i
        ddy = ypos-j
        ddz = zpos-k

        delta = N.zeros((nbins, nbins, nbins))
        edges = [N.linspace(0, nbins, nbins+1), \
                 N.linspace(0, nbins, nbins+1), \
                 N.linspace(0, nbins, nbins+1)]

        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    pos = N.array([i+ii, j+jj, k+kk]).transpose()
                    weight = ( ((1-ddx)+ii*(-1+2*ddx))*\
                               ((1-ddy)+jj*(-1+2*ddy))*\
                               ((1-ddz)+kk*(-1+2*ddz)) ) *c.weight
                    delta_t, edges = N.histogramdd(pos, \
                                     bins=edges, weights=weight) 
                    delta+=delta_t

        return delta

    def get_shift(self, c, f_x, f_y, f_z, use_newpos=False):

        xmin = self.xmin
        ymin = self.ymin
        zmin = self.zmin
        binsize = self.binsize

        if use_newpos:
            xpos = (c.newx-xmin)/binsize
            ypos = (c.newy-ymin)/binsize
            zpos = (c.newz-zmin)/binsize
        else: 
            xpos = (c.x-xmin)/binsize
            ypos = (c.y-ymin)/binsize
            zpos = (c.z-zmin)/binsize

        i = xpos.astype(int)
        j = ypos.astype(int)
        k = zpos.astype(int)

        ddx = xpos-i
        ddy = ypos-j
        ddz = zpos-k

        shift_x = N.zeros(c.size)
        shift_y = N.zeros(c.size)
        shift_z = N.zeros(c.size)

        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    weight = ( ((1-ddx)+ii*(-1+2*ddx))*\
                               ((1-ddy)+jj*(-1+2*ddy))*\
                               ((1-ddz)+kk*(-1+2*ddz))  )
                    pos = (i+ii, j+jj, k+kk)
                    shift_x += f_x[pos]*weight
                    shift_y += f_y[pos]*weight
                    shift_z += f_z[pos]*weight

        return shift_x, shift_y, shift_z

    def export_cart(self, root):
        
        lines = ['%f %f %f %f'%(xx, yy, zz, ww) for (xx, yy, zz, ww) \
               in zip(self.cat.x, self.cat.y, self.cat.z, self.cat.weight)]
        fout = open(root+'.dat.txt', 'w') 
        fout.write('\n'.join(lines))
        fout.close()
        lines = ['%f %f %f %f'%(xx, yy, zz, ww) for (xx, yy, zz, ww) \
               in zip(self.ran.x, self.ran.y, self.ran.z, self.ran.weight)]
        fout = open(root+'.ran.txt', 'w') 
        fout.write('\n'.join(lines))
        fout.close()


    def cart_to_radecz(self, x, y, z):

        dist = N.sqrt(x**2+y**2+z**2)
        dec = N.arcsin(z/dist)*180./N.pi
        ra = N.arctan(x/y)*180./N.pi + 180 
        redshift = self.cosmo.get_redshift(dist)
        return ra, dec, redshift

    def get_new_radecz(self, c):

        return self.cart_to_radecz(c.newx, c.newy, c.newz) 




