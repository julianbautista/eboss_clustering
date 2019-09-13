from __future__ import print_function
import numpy as np
import os
from scipy.fftpack import fftfreq
from cosmo import CosmoSimple
import pyfftw
import fastmodules
import sys

class MiniCat:

    def __init__(self, ra, dec, z, we):
        self.ra = ra
        self.dec = dec
        self.Z = z
        self.we = we
        self.size = ra.size

    def get_cart(self, cosmo=None):
        if cosmo is None:
            cosmo = CosmoSimple(omega_m=0.31)
        dist = cosmo.get_comoving_distance(self.Z)
        dec = np.radians(self.dec)
        ra = np.radians(self.ra)
        x = dist * np.cos(dec) * np.cos(ra)
        y = dist * np.cos(dec) * np.sin(ra)
        z = dist * np.sin(dec)
        self.dist = dist
        self.x = x
        self.y = y
        self.z = z
        self.newx = x*1
        self.newy = y*1
        self.newz = z*1 

    def get_zeff(self):
        return np.sum(self.we**2*self.Z)/np.sum(self.we**2)

class Recon:

    def __init__(self, 
                 data_ra, data_dec, data_z, data_we, 
                 rand_ra, rand_dec, rand_z, rand_we, 
                 bias=2.3, f=0.817, smooth=15., nbins=256, 
                 padding=200., opt_box=1, nthreads=1, omega_m=0.31):
        ''' RA, DEC, Z and WE arrays should all be in np.float64 format
            for fastmodules to work 
        '''


        #-- parameters of box
        cosmo = CosmoSimple(omega_m=omega_m)
        

        print('Num bins:', nbins)
        print('Smoothing [Mpc/h]:', smooth)

        dat = MiniCat(data_ra, data_dec, data_z, data_we)
        ran = MiniCat(rand_ra, rand_dec, rand_z, rand_we)
        
        #-- computing effective redshift and growth rate
        z_eff = dat.get_zeff()
        f = cosmo.get_growth_rate(z_eff)
        print('Effective redshift sum(we**2*z)/sum(we**2) = :', z_eff)
        print('Growth rate:', f)
        beta = f/bias

        #-- computing cartesian positions
        dat.get_cart(cosmo=cosmo)
        ran.get_cart(cosmo=cosmo)

        print('Randoms min of x, y, z', np.min(ran.x), np.min(ran.y), np.min(ran.z))
        print('Randoms max of x, y, z', np.max(ran.x), np.max(ran.y), np.max(ran.z))

        sum_wgal = np.sum(dat.we)
        sum_wran = np.sum(ran.we)
        alpha = sum_wgal/sum_wran
        ran_min = 0.01*sum_wran/ran.size

        self.nbins=nbins
        self.bias=bias
        self.f=f
        self.beta=beta
        self.smooth=smooth
        self.dat = dat
        self.ran = ran
        self.ran_min = ran_min
        self.alpha=alpha
        self.cosmo = cosmo
        self.nthreads = nthreads

        self.compute_box(padding=padding, optimize_box=opt_box)

    def compute_box(self, padding=200., optimize_box=1):
    
        if optimize_box:
            dx = np.max(self.ran.x)-np.min(self.ran.x)
            dy = np.max(self.ran.y)-np.min(self.ran.y)
            dz = np.max(self.ran.z)-np.min(self.ran.z)
            x0 = 0.5*(np.max(self.ran.x)+np.min(self.ran.x)) 
            y0 = 0.5*(np.max(self.ran.y)+np.min(self.ran.y)) 
            z0 = 0.5*(np.max(self.ran.z)+np.min(self.ran.z)) 

            box = np.max([dx, dy, dz])+2*padding
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
        
        print('Box size [Mpc/h]:', self.box)
        print('Bin size [Mpc/h]:', self.binsize)

    def iterate(self, iloop, save_wisdom=1, verbose=1):
        dat = self.dat
        ran = self.ran
        smooth = self.smooth
        binsize = self.binsize
        beta = self.beta
        bias = self.bias
        f = self.f
        nbins = self.nbins

        print("Loop %d" % iloop)
        #-- Creating arrays for FFTW
        if iloop==0:
            delta  = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            deltak = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            rho    = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            rhok   = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            psi_x  = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            psi_y  = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')
            psi_z  = pyfftw.empty_aligned((nbins, nbins, nbins), dtype='complex128')

            #-- Initialize FFT objects and load wisdom if available
            wisdom_file = "wisdom."+str(nbins)+"."+str(self.nthreads)+'.npy'
            if os.path.isfile(wisdom_file) :
                print('Reading wisdom from ', wisdom_file)
                wisd = tuple(np.load(wisdom_file))
                print('Status of importing wisdom', pyfftw.import_wisdom(wisd))
                sys.stdout.flush()
            print('Creating FFTW objects...')
            fft_obj = pyfftw.FFTW(delta, delta, axes=[0, 1, 2], threads=self.nthreads)
            ifft_obj = pyfftw.FFTW(deltak, psi_x, axes=[0, 1, 2], \
                                   threads=self.nthreads, \
                                   direction='FFTW_BACKWARD')
            kr = fftfreq(nbins, d=binsize) * 2 * np.pi * self.smooth
            norm = np.exp(-0.5 * (  kr[:, None, None] ** 2 \
                                  + kr[None, :, None] ** 2 \
                                  + kr[None, None, :] ** 2))
            
            if verbose:
                print('Allocating randoms...')
                sys.stdout.flush()
            deltar = np.zeros((nbins, nbins, nbins), dtype='float64')
            fastmodules.allocate_gal_cic(deltar, ran.x, ran.y, ran.z, ran.we, 
                                         ran.size, self.xmin, self.ymin, self.zmin, 
                                         self.box, nbins, 1)
            if verbose:
                print('Smoothing...')
                sys.stdout.flush()
            #  We do the smoothing via FFTs rather than scipy's gaussian_filter 
            #  because if using several threads for pyfftw it is much faster this way 
            #  (if only using 1 thread gains are negligible)
            rho = deltar + 0.0j
            fft_obj(input_array=rho, output_array=rhok)
            fastmodules.mult_norm(rhok, rhok, norm)
            ifft_obj(input_array=rhok, output_array=rho)
            deltar = rho.real


        else:
            delta  = self.delta
            deltak = self.deltak
            deltar = self.deltar
            rho    = self.rho
            rhok   = self.rhok
            psi_x  = self.psi_x
            psi_y  = self.psi_y
            psi_z  = self.psi_z
            fft_obj  = self.fft_obj
            ifft_obj = self.ifft_obj
            norm = self.norm

        #fft_obj = pyfftw.FFTW(delta, delta, threads=self.nthreads, axes=[0, 1, 2])
        #-- Allocate galaxies and randoms to grid with CIC method
        #-- using new positions
        if verbose:
            print('Allocating galaxies in cells...')
            sys.stdout.flush()
        deltag = np.zeros((nbins, nbins, nbins), dtype='float64')
        fastmodules.allocate_gal_cic(deltag, dat.newx, dat.newy, dat.newz, dat.we, 
                                     dat.size, self.xmin, self.ymin, self.zmin, self.box, 
                                     nbins, 1)
        #deltag = self.allocate_gal_cic(dat)
        if verbose:
            print('Smoothing...')
            sys.stdout.flush()
        #deltag = gaussian_filter(deltag, smooth/binsize)
        ##-- Smoothing via FFTs
        rho = deltag + 0.0j
        fft_obj(input_array=rho, output_array=rhok)
        fastmodules.mult_norm(rhok, rhok, norm)
        ifft_obj(input_array=rhok, output_array=rho)
        deltag = rho.real

        if verbose:
            print('Computing density fluctuations, delta...')
            sys.stdout.flush()
        # normalize using the randoms, avoiding possible divide-by-zero errors
        fastmodules.normalize_delta_survey(delta, deltag, deltar, self.alpha, self.ran_min)
        del(deltag)  # deltag no longer required anywhere

        #delta[:]  = deltag - self.alpha*deltar
        #w=np.where(deltar>self.ran_min)
        #delta[w] = delta[w]/(self.alpha*deltar[w])
        #w2=np.where((deltar<=self.ran_min)) 
        #delta[w2] = 0.
        #w3=np.where(delta>np.percentile(delta[w].ravel(), 99))
        #delta[w3] = 0.
        #del(w)
        #del(w2)
        #del(w3)
        #del(deltag)

        if verbose:
            print('Fourier transforming delta field...')
        sys.stdout.flush()
        fft_obj(input_array=delta, output_array=delta)
        ## -- delta/k**2
        k = fftfreq(self.nbins, d=binsize) * 2 * np.pi
        fastmodules.divide_k2(delta, delta, k)
        #delta /= (k[:, None, None]**2 + k[None, :, None]**2 + k[None, None, :]**2 + 1e-100)
        #delta[0, 0, 0] = 0 #-- was 1. (changed just it case) 

        # now solve the basic building block: IFFT[-i k delta(k)/(b k^2)]
        if verbose:
            print('Inverse Fourier transforming to get psi...')
        sys.stdout.flush()
        fastmodules.mult_kx(deltak, delta, k, bias)
        ifft_obj(input_array=deltak, output_array=psi_x)
        fastmodules.mult_ky(deltak, delta, k, bias)
        ifft_obj(input_array=deltak, output_array=psi_y)
        fastmodules.mult_kz(deltak, delta, k, bias)
        ifft_obj(input_array=deltak, output_array=psi_z)
        
        ##-- Estimating the IFFT in Eq. 12 of Burden et al. 2015
        #print('Inverse Fourier transforming to get psi...')
        #norm_ifft = 1.#(k[1]-k[0])**3/(2*np.pi)**3*nbins**3
        #deltak[:] = delta*-1j*k[:, None, None]/bias
        #ifft_obj(input_array=deltak, output_array=psi_x)
        #deltak[:] = delta*-1j*k[None, :, None]/bias
        #ifft_obj(input_array=deltak, output_array=psi_y)
        #deltak[:] = delta*-1j*k[None, None, :]/bias
        #ifft_obj(input_array=deltak, output_array=psi_z)
        #psi_x = ifftn(-1j*delta*k[:, None, None]/bias).real*norm_ifft
        #psi_y = ifftn(-1j*delta*k[None, :, None]/bias).real*norm_ifft
        #psi_z = ifftn(-1j*delta*k[None, None, :]/bias).real*norm_ifft
        
        # from grid values of Psi_est = IFFT[-i k delta(k)/(b k^2)], compute the values at the galaxy positions
        if verbose:
            print('Calculating shifts...')
        sys.stdout.flush()
        shift_x, shift_y, shift_z = self.get_shift(dat.newx, dat.newy, dat.newz, 
                                                   psi_x.real, psi_y.real, psi_z.real)


        #-- for first loop need to approximately remove RSD component 
        #-- from Psi to speed up calculation
        #-- first loop so want this on original positions (cp), 
        #-- not final ones (np) - doesn't actualy matter
        if iloop==0:
            psi_dot_rhat = (shift_x*dat.x + \
                            shift_y*dat.y + \
                            shift_z*dat.z ) /dat.dist
            shift_x-= beta/(1+beta) * psi_dot_rhat * dat.x/dat.dist
            shift_y-= beta/(1+beta) * psi_dot_rhat * dat.y/dat.dist
            shift_z-= beta/(1+beta) * psi_dot_rhat * dat.z/dat.dist

        #-- remove RSD from original positions (cp) of 
        #-- galaxies to give new positions (np)
        #-- these positions are then used in next determination of Psi, 
        #-- assumed to not have RSD.
        #-- the iterative procued then uses the new positions as 
        #-- if they'd been read in from the start
        psi_dot_rhat = (shift_x*dat.x+shift_y*dat.y+shift_z*dat.z)/dat.dist
        dat.newx = dat.x + f * psi_dot_rhat * dat.x/dat.dist 
        dat.newy = dat.y + f * psi_dot_rhat * dat.y/dat.dist 
        dat.newz = dat.z + f * psi_dot_rhat * dat.z/dat.dist 

        if verbose:
            print('Debug: first 10 x,y,z, and total shifts')
            for i in range(10):
                shift = np.sqrt(shift_x[i]**2 + shift_y[i]**2 + shift_z[i]**2)
                print('%.3f %.3f %.3f %.3f ' % (shift_x[i], shift_y[i], shift_z[i], shift))


        self.deltar = deltar
        self.delta = delta
        self.deltak = deltak
        self.rho = rho
        self.rhok = rhok
        self.psi_x = psi_x
        self.psi_y = psi_y
        self.psi_z = psi_z
        self.fft_obj = fft_obj
        self.ifft_obj = ifft_obj
        self.norm = norm


        #-- save wisdom
        wisdom_file = "wisdom."+str(nbins)+"."+str(self.nthreads)+'.npy'
        if iloop==0 and save_wisdom and not os.path.isfile(wisdom_file):
            wisd=pyfftw.export_wisdom()
            np.save(wisdom_file, wisd)
            print('Wisdom saved at', wisdom_file)

    def apply_shifts_rsd(self):
        """ Subtract RSD to get the estimated real-space positions of randoms
        (no need to do this for galaxies, since it already happens during the iteration loop)
        """
        
        shift_x, shift_y, shift_z = \
            self.get_shift(self.ran.x, self.ran.y, self.ran.z, 
                           self.psi_x.real, self.psi_y.real, self.psi_z.real)
        psi_dot_rhat = (shift_x * self.ran.x + shift_y * self.ran.y + shift_z * self.ran.z) / self.ran.dist
        self.ran.newx = self.ran.x + self.f * psi_dot_rhat * self.ran.x / self.ran.dist
        self.ran.newy = self.ran.y + self.f * psi_dot_rhat * self.ran.y / self.ran.dist
        self.ran.newz = self.ran.z + self.f * psi_dot_rhat * self.ran.z / self.ran.dist

    def apply_shifts_full(self):
        """ Use the estimated displacement field to shift the positions of galaxies (and randoms).
        This method subtracts full displacement field as in standard BAO reconstruction"""

        for c in [self.dat, self.ran]:
            shift_x, shift_y, shift_z = \
                self.get_shift(c.newx, c.newy, c.newz, 
                               self.psi_x.real, self.psi_y.real, self.psi_z.real)
            c.newx += shift_x
            c.newy += shift_y
            c.newz += shift_z

    def summary(self):

        dat = self.dat
        sx = dat.newx-dat.x
        sy = dat.newy-dat.y
        sz = dat.newz-dat.z
        axis = ['x', 'y', 'z']
        print('Shifts stats:')
        print('RMS   16th-perc  84th-perc  min  max')
        for ax, s in zip(axis, [sx, sy, sz]):
            print(ax+' %.3f %.3f %.3f %.3f %.3f'%\
                  (np.std(s), np.percentile(s, 16), np.percentile(s, 84), np.min(s), np.max(s)))


    def get_shift(self, x, y, z, f_x, f_y, f_z):

        xmin = self.xmin
        ymin = self.ymin
        zmin = self.zmin
        binsize = self.binsize

        xpos = (x-xmin)/binsize
        ypos = (y-ymin)/binsize
        zpos = (z-zmin)/binsize

        i = xpos.astype(int)
        j = ypos.astype(int)
        k = zpos.astype(int)

        ddx = xpos-i
        ddy = ypos-j
        ddz = zpos-k

        shift_x = np.zeros(x.size)
        shift_y = np.zeros(x.size)
        shift_z = np.zeros(x.size)

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
               in zip(self.dat.x, self.dat.y, self.dat.z, self.dat.we)]
        fout = open(root+'.dat.txt', 'w') 
        fout.write('\n'.join(lines))
        fout.close()
        lines = ['%f %f %f %f'%(xx, yy, zz, ww) for (xx, yy, zz, ww) \
               in zip(self.ran.x, self.ran.y, self.ran.z, self.ran.we)]
        fout = open(root+'.ran.txt', 'w') 
        fout.write('\n'.join(lines))
        fout.close()


    def cart_to_radecz(self, x, y, z):

        dist = np.sqrt(x**2+y**2+z**2)
        dec = 90 - np.degrees(np.arccos(z / dist))
        ra = np.degrees(np.arctan2(y, x))
        ra[ra < 0] += 360
        redshift = self.cosmo.get_redshift(dist)
        return ra, dec, redshift

    def get_new_radecz(self, c):

        return self.cart_to_radecz(c.newx, c.newy, c.newz) 




