from __future__ import print_function
import camb
import numpy as np
import pylab as plt
import fftlog
import iminuit
import sys
import scipy.interpolate 
import scipy.linalg
import copy
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

class Cosmo:

    def __init__(self, z=0.0, name='challenge', pars=None, 
                 norm_pk=False, non_linear=False, 
                 nk=2048, kmax=100., nmu=101):
        self.get_matter_power_spectrum(z=z, name=name, norm_pk=norm_pk, pars=pars,
                                       non_linear=non_linear, kmax=kmax, nk=nk)
        self.r, self.xi = self.get_correlation_function()
        #self.get_sideband()
        self.get_sideband_scipy()
        self.get_sideband_power()
        self.set_2d_arrays(nmu=nmu)

    def get_matter_power_spectrum(self, pars=None, z=0.0, non_linear=0, 
                                        name='challenge', norm_pk=0, 
                                        kmax=100., nk=4098):

        #-- to set sigma8 value, scale As by the square of ratio of sigma8
        if pars is None:
            pars = camb.CAMBparams()
            if name == 'challenge':
                pars.set_cosmology(H0=67.6, ombh2=0.0220,  
                                    omch2=0.11901745, 
                                    YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.06)
                pars.InitPower.set_params(As=2.0406217009089533e-09, ns=0.97)
            elif name == 'cosmo1':
                pars.set_cosmology(H0=67.6, ombh2=0.0220,  
                                    omch2=0.10073838, 
                                    YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.06)
                pars.InitPower.set_params(As=2.0406217009089533e-09, ns=0.97)
            elif name == 'cosmo2':
                pars.set_cosmology(H0=67.6, ombh2=0.0220,  
                                    omch2=0.13729646, 
                                    YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.06)
                pars.InitPower.set_params(As=2.0406217009089533e-09, ns=0.97)
            elif name == 'qpm':
                pars.set_cosmology(H0=70., ombh2=0.022470,  
                                    omch2=0.11963, 
                                    YHe=0.24,nnu=3.04,  mnu=0, 
                                    TCMB=2.7250, num_massive_neutrinos=0)
                pars.InitPower.set_params(As=2.3e-09, ns=0.97)
            elif name == 'planck':
                pars.set_cosmology(H0=67.31, ombh2=0.02222, 
                                   omch2=0.1197, 
                                   YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.06)
                pars.InitPower.set_params(As=2.198e-09, ns=0.9655)
            elif name == 'outerrim':
                pars.set_cosmology(H0=71., ombh2=0.022584, 
                                   omch2=0.10848, 
                                   YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.0,
                                    num_massive_neutrinos=0)
                pars.InitPower.set_params(As=2.224615e-09, ns=0.963)
            elif name == 'ezmock':
                pars.set_cosmology(H0=67.77, ombh2=0.0221399210, 
                                   omch2=0.1189110239, 
                                   YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.0,
                                    num_massive_neutrinos=0)
                pars.InitPower.set_params(As=2.11622e-09, ns=0.9611)
            elif name == 'nseries':
                #-- Om=0.286, h=0.7, ns=0.96, Ob=0.047, s8=0.820
                pars.set_cosmology(H0=70, ombh2=0.02303, 
                                   omch2=0.11711, 
                                   YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.0,
                                    num_massive_neutrinos=0)
                pars.InitPower.set_params(As=2.14681e-09, ns=0.96)
            else: 
                print('Error: name of cosmology should be one of the following')
                print('challenge qpm planck outerrim ezmock')
                sys.exit(0)
 
                

        pars.set_dark_energy()
        
        #-- compute power spectrum
        pars.set_matter_power(redshifts=[z], kmax=2*kmax, k_per_logint=None)

        #-- set non-linear power spectrum
        if non_linear:
            pars.NonLinear = camb.model.NonLinear_both
        else:
            pars.NonLinear = camb.model.NonLinear_none

        results = camb.get_results(pars)
        kh, z, pk = results.get_matter_power_spectrum(\
                        minkh=1.05e-5, maxkh=kmax, npoints = nk)

        
        sigma8 = results.get_sigma8()
        print('sigma_8(z=%.3f) = %.4f'%(z[0],sigma8[0]) )
        print( 'H(z)   = ', results.hubble_parameter(z[0]) )
        print( 'D_A(z) = ', results.angular_diameter_distance(z[0]) )
        print( 'rdrag  = ', results.get_derived_params()['rdrag'] )

        if norm_pk:
            pk /= sigma8[0]**2

        self.z = z[0]
        self.name=name
        self.norm_pk=norm_pk
        self.camb_pars = pars
        self.results = results
        self.k = kh
        self.pk = pk[0]
        self.sigma8 = sigma8[0]
        self.H_z = results.hubble_parameter(z[0])
        self.D_A = results.angular_diameter_distance(z[0])
        self.D_M = self.D_A*(1+z[0]) 
        self.D_H = 299792.458/self.H_z
        self.r_drag = results.get_derived_params()['rdrag']
        
        print(f'D_M(z)/r_d = {self.D_M/self.r_drag:.3f}')
        print(f'D_H(z)/r_d = {self.D_H/self.r_drag:.3f}')

        return kh, pk[0]
        
    def get_correlation_function(self, k=None, pk=None,  
                                 sigma_nl=0., r=None, 
                                 r0=1., inverse=0):
        ''' This computes isotropic xi(r) from isotropic P(k)
            Currently not used by fitter but useful for tests 
        '''

        if k is None or pk is None:
            k = self.k
            pk = self.pk

        #-- transform to log space for Hankel Transform
        klog = 10**np.linspace( np.log10(k.min()), np.log10(k.max()), k.size)
        pklog = np.interp(klog, k, pk)

        #-- apply isotropic damping
        pklog *= np.exp(-0.5*klog**2*sigma_nl**2)

        rout, xiout = fftlog.HankelTransform(klog, pklog, 
                                             q=1.5, mu=0.5, 
                                             output_r_power=-3, 
                                             output_r=r, r0=r0)
        norm = 1/(2*np.pi)**1.5
        if inverse:
            xiout /= norm
        else:
            xiout *= norm

        return rout, xiout

    def get_sideband(self, 
                     fit_range=[[50., 80.], [160., 190.]], 
                     poly_order=4):

        r = self.r
        xi = self.xi

        peak_range = [fit_range[0][1], fit_range[1][0]]

        w = ((r>fit_range[0][0])&(r<fit_range[0][1])) | \
            ((r>fit_range[1][0])&(r<fit_range[1][1]))
        x_fit = r[w]
        y_fit = xi[w]*r[w]**3

        coeff = np.polyfit(x_fit, y_fit, poly_order)
        
        xi_sideband = xi*1.
        w_peak = (r>peak_range[0])&(r<peak_range[1])
        xi_sideband[w_peak] = np.polyval(coeff, r[w_peak])/r[w_peak]**3

        self.xi_model = np.polyval(coeff, r)/r**3
        self.xi_sideband = xi_sideband
        self.peak_range = peak_range
        self.fit_range = fit_range

        return xi_sideband

    def get_sideband_scipy(self, fit_range=[[50., 80.], [160., 190.]], 
                            plotit=False):
        ''' Gets correlation function without BAO peak using 
            scipy.optimize.minimize function 
        '''

        r = self.r*1
        xi = self.xi*1

        peak_range = [fit_range[0][1], fit_range[1][0]]
        w = ((r>fit_range[0][0])&(r<fit_range[0][1])) | \
            ((r>fit_range[1][0])&(r<fit_range[1][1]))
        x_fit = r[w]
        y_fit = xi[w]

        def broadband(x, *pars):
            xx = x/100
            return pars[0]*xx + pars[1] + pars[2]/xx + pars[3]/xx**2 + \
                   pars[4]*xx**2 + pars[5]*xx**3 + pars[6]*xx**4  

        popt, pcov = curve_fit(broadband, x_fit, y_fit,
                                p0=[0, 0, 0, 0, 0, 0, 0])
       
        xi_sideband = xi*1.
        w_peak = (r>peak_range[0])&(r<peak_range[1])
        xi_sideband[w_peak] = broadband(r[w_peak], *popt)
        
        self.xi_model = broadband(r, *popt)
        self.xi_sideband = xi_sideband
        self.peak_range = peak_range
        self.fit_range = fit_range

        return xi_sideband

    def plot_sideband_residuals(self):
        
        r = self.r*1
        xi = self.xi*1
        xis = self.xi_sideband*1
        xim = self.xi_model*1
        fit_range = self.fit_range

        w = ((r>fit_range[0][0])&(r<fit_range[1][1]))
        x = r[w]
        y = xi[w]
        ys = xis[w]
        ym = xim[w] 
        plt.plot(x, y*x**2)
        plt.plot(x, ys*x**2)
        plt.plot(x, ym*x**2)

    def get_sideband_power(self):
        ''' Get power spectrum of sideband '''

        ks, pks = self.get_correlation_function(k=self.r, pk=self.xi_sideband,
                                                inverse=1, r=self.k)
        self.pk_sideband = pks

    def plots_pk_cf_sidebands(self):
        k = self.k
        pk = self.pk
        pks = self.pk_sideband
        r = self.r
        xi = self.xi
        xis = self.xi_sideband

        plt.figure(figsize=(6,4))
        plt.plot(k, pk*k, 'k', lw=2)
        plt.plot(k, pks*k, 'r--', lw=2)
        plt.xscale('log')
        plt.xlabel(r'$k \ [h \ \rm{Mpc}^{-1}]$')
        plt.ylabel(r'$kP_{\rm lin}(k) \ [h^{-2}\mathrm{Mpc}^2]$')
        plt.xlim(1e-3, 10)
        plt.tight_layout()

        plt.figure(figsize=(6,4))
        plt.plot(r, xi*r**2, 'k', lw=2)
        plt.plot(r, xis*r**2, 'r--', lw=2)
        plt.xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')
        plt.ylabel(r'$r^2 \xi_{\rm lin} \ [h^{-2} \mathrm{Mpc}^{2}]$')
        plt.xlim(0, 200)
        plt.tight_layout()

    def smooth(self, xi_peak, dr=1., sigma_nl=4.):
        ''' Gaussian smoothing
            dr: size of bins of xi_peak
            sigma_nl: smoothing length
        '''
        xi_smooth = gaussian_filter1d(xi_peak, sigma=sigma_nl/dr)
        return xi_smooth 

    #def get_multipoles(self, r, xi, f):
    #    ''' Compute multipoles from isotropic correlation function 
    #        with linear redshift-space distortions
    #        following Hamilton 1992
    #
    #        Currently not used but here for reference
    #    '''
    #    xib   = np.array([ np.sum(xi[:i]*r[:i]**2) for i in range(r.size)])\
    #            * 3./r**3 * np.gradient(r)
    #    xibb = np.array([ np.sum(xi[:i]*r[:i]**4) for i in range(r.size)])\
    #            * 5./r**5 * np.gradient(r)
    #    xi0 = (1+2./3*f+1./5*f**2)*xi
    #    xi2 = (4./3*f + 4./7*f**2)*(xi-xib)
    #    xi4 = 8./35*f**2*(xi + 2.5*xib - 3.5*xibb)
    #    return xi0, xi2, xi4

    def set_2d_arrays(self, nmu=201):

        self.mu = np.linspace(0, 1., nmu)
        self.mu2d = np.outer(self.mu, np.ones(self.k.size))
        self.k2d  = np.outer(np.ones(nmu), self.k)

    def get_2d_power_spectrum(self, pars, 
        ell_max=2, no_peak=False, decoupled=False, window=None):

        #-- If all parameters are the same as the previous calculation,
        #-- simply return the same power spectrum (no broadband)
        #if hasattr(self, 'pars') and pars==self.pars:
        #    return self.pk2d_out

        #-- Read alphas and BAO damping terms
        if 'aiso' in pars:
            at = pars['aiso']
            ap = pars['aiso']
            sigma_par = pars['sigma_nl']
            sigma_per = pars['sigma_nl']
        elif 'ap' in pars:
            at = pars['at']
            ap = pars['ap']
            sigma_par = pars['sigma_par']
            sigma_per = pars['sigma_per']
        else:
            at = 1.
            ap = 1.
            sigma_par = 0
            sigma_per = 0
       
        #-- Read bias and growth rate / RSD parameter
        bias = pars['bias']
        if 'beta' in pars:
            beta = pars['beta']
        else:
            f = pars['f']
            beta = f/bias

        #-- Read reconstruction damping parameter
        sigma_rec = pars['sigma_rec']

        #-- Read parameters for cross-correlation
        #-- or making them equal if fitting auto-correlation
        if 'bias2' in pars:
            bias2 = pars['bias2']
        else:
            bias2 = bias*1.
        if 'beta2' in pars:    
            beta2 = pars['beta2']
        else:
            beta2 = beta*bias/bias2
 

        k = self.k
        mu = self.mu
        pk = self.pk
        pk_sideband = self.pk_sideband

        #-- These have shape = (nmu, nk) 
        mu2d = self.mu2d 
        k2d = self.k2d 
   
        #-- Scale k and mu by alphas 
        if 'aiso' in pars:
            ak2d = k2d/at
            amu = mu*1.
        else:
            #-- This is the correct formula (Eq. 58 and 59 from Beutler et al. 2014)
            F = ap/at
            ak2d = k2d/at * np.sqrt( 1 + mu2d**2 * (1/F**2 - 1) )
            amu  = mu/F   / np.sqrt( 1 + mu**2   * (1/F**2 - 1) )


        #-- Sideband model (no BAO peak)
        #-- If decoupled, do not scale sideband by alpha
        if decoupled:
            pk2d_nopeak = np.outer(np.ones_like(mu), pk_sideband)
        else:
            pk2d_nopeak = np.interp(ak2d, k, pk_sideband)

        if no_peak:
            pk2d = pk2d_nopeak 
        else:
            #-- Anisotropic damping applied to BAO peak only
            sigma_nl = (1-mu**2)*sigma_per**2/2+ mu**2*sigma_par**2/2 
            sigma_nl_k2 = np.outer(sigma_nl, k**2)
            #-- Scale BAO peak part by alpha
            pk2d_peak = np.interp(ak2d, k, pk-pk_sideband)
            pk2d  = pk2d_peak * np.exp(-sigma_nl_k2)
            pk2d += pk2d_nopeak

        
        #-- Compute Kaiser redshift space distortions with reconstruction damping
        if sigma_rec == 0:
            recon_damp = np.ones(k.size)
        else:
            recon_damp = 1 - np.exp(-k**2*sigma_rec**2/2) 
        
        recon_damp_mu2 = np.outer(mu**2, recon_damp)
        kaiser = bias * bias2 * (1+beta*recon_damp_mu2) * (1+beta2*recon_damp_mu2)

        #-- Fingers of God
        if pars['sigma_s'] != 0:
            fog = 1./( 1 + np.outer(mu**2, k**2)*pars['sigma_s']**2/2)
        else:
            fog = 1

        #-- This parameters is for intensity mapping only
        if 'beam' in pars:
            pk2d *= np.exp( - pars['beam']**2*np.outer(1-mu**2, k**2)/2) 
        
        #-- This parameters is for intensity mapping only
        if 'THI' in pars:
            pk2d *= pars['t_hi']
        
        pk2d *= kaiser
        pk2d *= fog**2 

        #if not decoupled:
        #    pk2d_out /= (at**2*ap)

        self.ak2d = ak2d
        self.pk2d = pk2d

        return pk2d



 

    def legendre(self, ell, mu):

        if ell == 0:
            return mu*0+1
        elif ell == 2:
            return 0.5 * (3*mu**2-1)
        elif ell == 4:
            return 1/8 * (35*mu**4 - 30*mu**2 +3)
        elif ell == 6:
            return 1/16 * (231*mu**6 - 315*mu**4 + 105*mu**2 - 5)
        elif ell == 8:
            return 1/128* (6435*mu**8 - 12012*mu**6 + 6930*mu**4 - 1260*mu**2 + 35)
        else:
            return -1

    def get_multipoles(self, mu, f2d, ell_max=4):
        ''' Get multipoles of any function of ell 
            Input
            -----
            mu: np.array with shape (nmu) where nmu is the number of mu bins
            f2d: np.array with shape (nmu, nx) from which the multipoles will be computed    
            
            Returns
            ----
            f_mult: np.array with shape (nell, nx) 
        '''
        
        nx = f2d.shape[1]
        nl = ell_max//2+1
        f_mult = np.zeros((nl, nx))
        for ell in range(0, ell_max+2, 2):
            leg = self.legendre(ell, mu)
            f_mult[ell//2] = (2*ell+1)*np.trapz(f2d*leg[:, None], x=mu, axis=0)
        return f_mult

    def get_xi_multipoles_from_pk(self, k, pk_mult, output_r=None, r0=1.):

        nell = len(pk_mult)
        xi_mult = []
        for i in range(nell):
            pk = pk_mult[i]
            ell = i*2
            r, xi = fftlog.HankelTransform(k, pk, mu=0.5+ell, output_r=output_r,
                                           output_r_power=-3, q=1.5, r0=r0)
            norm = 1/(2*np.pi)**1.5 * (-1)**(ell/2)
            xi_mult.append(xi*norm)
        xi_mult = np.array(xi_mult)
        return r, xi_mult 

    def get_pk_multipoles_from_xi(self, r, xi_mult, output_k=None, r0=1.):

        nell = len(xi_mult)
        pk_mult = []
        for i in range(nell):
            xi = xi_mult[i]
            ell = i*2
            k, pk = fftlog.HankelTransform(r, xi, mu=0.5+ell, output_r=output_k,
                                                 output_r_power=-3, q=1.5, r0=r0)
            #norm = 1/(2*np.pi)**1.5 * (-1)**(ell/2)
            norm = 1/(2*np.pi)**1.5 * (-1)**(ell/2) * (2*np.pi)**3
            pk_mult.append(pk*norm)
        pk_mult = np.array(pk_mult)
        return k, pk_mult

    def get_xi_multipoles(self, rout, pars, ell_max=4, decoupled=False, no_peak=False, r0=1.):

        pk2d = self.get_2d_power_spectrum(pars, 
                                          ell_max = ell_max, 
                                          no_peak = no_peak, 
                                          decoupled = decoupled)
        pk_mult = self.get_multipoles(self.mu, pk2d, ell_max=ell_max)
        _, xi_mult = self.get_xi_multipoles_from_pk(self.k, pk_mult, 
                                                    output_r=rout, r0=r0)
        return xi_mult

    def get_pk_multipoles(self, kout, pars, ell_max=4, decoupled=False, no_peak=False, r0=1., apply_window=False):

        pk2d = self.get_2d_power_spectrum(pars, 
                                          ell_max = ell_max, 
                                          no_peak = no_peak, 
                                          decoupled = decoupled)
        pk_mult = self.get_multipoles(self.mu, pk2d, ell_max=ell_max)

        if apply_window:
            _, xi_mult = self.get_xi_multipoles_from_pk(self.k, pk_mult, output_r=self.r) 
            xi_convol = self.get_convoled_xi(xi_mult, self.window_mult)
            _, pk_mult_out = self.get_pk_multipoles_from_xi(self.r, xi_convol, output_k=kout)
        else:
            pk_mult_out = []
            for pk in pk_mult:
                pk_mult_out.append(np.interp(kout, self.k, pk))
        pk_mult_out = np.array(pk_mult_out)

        return pk_mult_out

    def read_window_function(self, window_file):
        data = np.loadtxt(window_file)
        r_window = data[0]
        window = data[1:]

        window_mult = []
        for win in window:
           window_spline = scipy.interpolate.InterpolatedUnivariateSpline(r_window, win)
           window_mult.append(window_spline(self.r))
        window_mult = np.array(window_mult)
        self.window_mult = window_mult

    def get_convolved_xi(self, xi_mult, window_mult):
        ''' Compute convolved multipoles of correlation function 
            given Eq. 19, 20 and 21 of Beutler et al. 2017 
        ''' 
        xi = xi_mult
        win = window_mult

        #-- Mono
        xi_mono = xi[0]*win[0] + xi[1]*(1/5 * win[1]) + xi[2]*(1/9*win[2])
        #-- Quad 
        xi_quad = xi[0]*win[1] + xi[1]*(      win[0] + 2/7    *win[1] + 2/7     *win[2]) \
                               + xi[2]*(2/7 * win[1] + 100/693*win[2] + 25/143  *win[3])
        #-- Hexa
        xi_hexa = xi[0]*win[2] + xi[1]*(18/35*win[1] + 20/77  *win[2] + 45/143  *win[3]) \
                + xi[2]*(win[0] + 20/77 *win[1] + 162/1001*win[2] + 20/143*win[3] + 490/2431*win[4])
    
        xi_conv = np.array([xi_mono, xi_quad, xi_hexa])
        return xi_conv


    def get_dist_rdrag(self):
        
        self.DH_rd = 300000./self.H_z/self.r_drag
        self.DM_rd = self.D_A*(1+self.z)/self.r_drag
        self.DV_rd = (300000.*self.z*(self.D_A*(1+self.z))**2/self.H_z)**(1./3)/self.r_drag
        print('D_H(z=%.2f)/r_d = %.2f'%(self.z, self.DH_rd))
        print('D_M(z=%.2f)/r_d = %.2f'%(self.z, self.DM_rd))
        print('D_V(z=%.2f)/r_d = %.2f'%(self.z, self.DV_rd))

    @staticmethod
    def get_alphas(cosmo, cosmo_fid):

        at = (cosmo.D_M/cosmo.r_drag)/(cosmo_fid.D_M/cosmo_fid.r_drag)
        ap = (cosmo.D_H/cosmo.r_drag)/(cosmo_fid.D_H/cosmo_fig.r_drag)
        #-- Padmanabhan & White 2009
        alpha = at**(2./3.)*ap**(1./3)
        epsilon = (ap/at)**(1./3) - 1
        print('at =', at, ' ap =', ap)
        print('aiso =', alpha, ' epsilon =', epsilon)
        return at, ap, alpha, epsilon   
    
class Data: 

    def __init__(self, r, mono, coss, quad=None, hexa=None, rmin=40., rmax=180., \
                    nmocks=None):

        cf = mono
        rr = r
        if not quad is None:
            rr = np.append(rr, r)
            cf = np.append(cf, quad)
        if not hexa is None:
            rr = np.append(rr, r)
            cf = np.append(cf, hexa)
   
        
        ncf = cf.size
        ncov = coss.shape[0]
        
        print(' Size r:', r.size) 
        print(' Size cf:', ncf)
        print(' Size cov:', ncov)
       
        if ncf > ncov or ncov % mono.size > 0:
            print('Problem: covariance shape is not compatible '+
                  f'with correlation function. CF size: {ncf}  COV shape: {coss.shape}')
            
        if ncf < ncov:
            print('Covariance matrix is larger than correlation function. Trying to cut')
            coss = coss[:, :ncf]
            coss = coss[:ncf, :]

        w = (rr>rmin) & (rr<rmax)
        rr = rr[w]
        cf = cf[w]
        coss = coss[:, w]
        coss = coss[w, :]
        
        self.rr = rr
        self.r = np.unique(rr)
        self.cf = cf
        self.coss = coss
        self.nmul = rr.size//self.r.size
        print('Covariance matrix is positive definite?', np.all(np.linalg.eigvals(coss)>0))
        self.icoss = np.linalg.inv(coss)
        if nmocks:
            correction = (1 - (cf.size + 1.)/(nmocks-1))
            self.icoss *= correction
    

class Model:

    def __init__(self, name='challenge', z = 0, 
                 fit_broadband=True, bb_min=-2, bb_max=0, 
                 norm_pk=False, non_linear=False, no_peak=False, decoupled=False,
                 fit_quad=False, fit_hexa=False,
                 fit_iso=False, 
                 fit_beta=False, fit_cross=False, 
                 fit_amp=False, fit_beam=False):

        cosmo = Cosmo(z=z, name=name, norm_pk=norm_pk, non_linear=non_linear)

        #-- define parameter dictionary
        pars = {}
        pars_names = []
        
        if fit_iso:
            pars_names += ['aiso']
            pars['aiso'] = 1.0
            pars_names += ['sigma_nl']
            pars['sigma_nl'] = 6.
        else:
            pars_names += ['at', 'ap']
            pars['at'] = 1.0
            pars['ap'] = 1.0
            pars_names += ['sigma_per', 'sigma_par']
            pars['sigma_per'] = 6.
            pars['sigma_par'] = 10.
         
        pars_names += ['bias', 'sigma_s', 'sigma_rec']
        pars['bias'] = 3.0
        pars['sigma_s'] = 4.
        pars['sigma_rec'] = 15.
        
        if fit_beta:
            pars['beta'] = 0.3
            pars_names += ['beta']
        else:
            pars['f'] = 0.8
            pars_names += ['f']

        if fit_cross:
            pars_names += ['bias2']        
            pars['bias2'] = 1.
            if fit_beta:
                pars_names += ['beta2']
                pars['beta2'] = 0.5

        if fit_amp:
            pars['t_hi'] = 0.3e-3
            pars_names += ['t_hi']

        if fit_beam:
            pars_names += ['beam']
            pars['beam'] = 4.

        #if fit_broadband:
        #    for i, bb_power in enumerate(np.arange(bb_min, bb_max+1)):
        #        pars_names.append('bb_%d_mono'%i)
        #        pars['bb_%d_mono'%i] = 0.
        #        if fit_quad:
        #            pars_names.append('bb_%d_quad'%i)
        #            pars['bb_%d_quad'%i] = 0.
        #        if fit_hexa:
        #            pars_names.append('bb_%d_hexa'%i)
        #            pars['bb_%d_hexa'%i] = 0.

        self.bb_min = bb_min
        self.bb_max = bb_max
        self.pars = pars
        self.pars_names = pars_names
        self.fit_broadband = fit_broadband
        self.fit_quad = fit_quad
        self.no_peak = no_peak
        self.decoupled = decoupled
        self.fit_cross = fit_cross
        self.fit_beam = fit_beam
        self.fit_beta = fit_beta
        self.fit_amp = fit_amp
        self.fit_hexa = fit_hexa
        self.cosmo = cosmo
        
    def value(self, rout, pars):

        ell_max = 0 + 2*(self.fit_quad) + 2*(self.fit_hexa)
        cf_out = self.cosmo.get_xi_multipoles(rout,  pars, 
                            ell_max=ell_max, no_peak=self.no_peak,
                            decoupled=self.decoupled)
        return cf_out.ravel()

    def get_broadband(self, rout, pars):
  
        monobb = rout*0.
        quadbb = rout*0.
        hexabb = rout*0.
       
        if hasattr(self, 'powers'):
            powers = self.powers
        else: 
            power_min = self.bb_min
            power_max = self.bb_max
            powers = np.arange(power_min, power_max+1)
            self.powers = powers
        for i in range(powers.size):
            power = powers[i]
            #monobb += pars['bb_%d_mono'%i]*((rout)**power)
            coeff = np.prod([pars[f'bb_{j}_mono'] for j in range(i+1)])
            monobb += coeff * rout**power 
            if self.fit_quad:
                coeff = np.prod([pars[f'bb_{j}_quad'] for j in range(i+1)])
                quadbb += coeff * rout**power 
                #quadbb += pars['bb_%d_quad'%i]*((rout)**power)
            if self.fit_hexa: 
                coeff = np.prod([pars[f'bb_{j}_hexa'] for j in range(i+1)])
                hexabb += coeff * rout**power 
                #hexabb += pars['bb_%d_hexa'%i]*((rout)**power)

        bb = monobb
        if self.fit_quad:
            bb = np.append(bb, quadbb)
        if self.fit_hexa:
            bb = np.append(bb, hexabb)
        self.bb = bb

        return bb




class Chi2: 

    def __init__(self, data=None, model=None, fin=None, z=0.72):
        if data:
            self.data = data
        if model:
            self.model = model
        if fin:
            self.read_galaxy_pars(fin, z=z)


    def read_galaxy_pars(self, fin, z=0.71):

        f = open(fin)
        f.readline()
        line = f.readline().split()
        chi2min, ndata, npars, rchi2min = \
                float(line[0]), int(line[1]), int(line[2]), float(line[3])
        f.readline()
    
        best_pars = {}
        errors = {}
        for line in f.readlines():
            line = line.split()
            parname = line[0]
            best_pars[parname] = float(line[1])
            errors[parname] = float(line[2])

        fit_iso = True if 'aiso' in best_pars else False
        no_peak = True if '-nopeak' in fin else False
        fit_quad = True if '-quad' in fin else False
        fit_hexa = True if '-hexa' in fin else False

        if 'bb_0_mono' in best_pars.keys():
            fit_broadband = True
        else:
            fit_broadband = False


        self.model = Model(fit_broadband=fit_broadband, fit_iso=fit_iso,
                           fit_quad=fit_quad, fit_hexa=fit_hexa, 
                           no_peak=no_peak, decoupled=decoupled, norm_pk=0,
                           z=z)
        self.model.cosmo.get_dist_rdrag()
        self.DM_rd = self.model.cosmo.DM_rd
        self.DH_rd = self.model.cosmo.DH_rd
        self.DV_rd = self.model.cosmo.DV_rd
        self.model.pars = best_pars
        self.chi2min = chi2min
        self.ndata = ndata
        self.npars = npars
        self.rchi2min = rchi2min
        self.best_pars=best_pars
        self.errors= errors
    
    def get_model(self, r, pars=None):
        if pars is None:
            pars = self.best_pars

        pars_cosmo = {par: pars[par] for par in pars if not par.startswith('bb')}
        model = self.model.value(r, pars_cosmo)

        #if self.model.fit_broadband:
            
         #   bb = self.fit_broadband(cf-model) 
            #pars_bb = {par: pars[par] for par in pars if par.startswith('bb')}
            #bb = self.model.get_broadband(r, pars_bb)
         #   model += bb

        return model
    
    def setup_broadband_H(self, r=None, bb_min=None, bb_max=None):
        if r is None:
            r = self.data.rr
        if bb_min is None:
            bb_min = self.model.bb_min
        if bb_max is None:
            bb_max = self.model.bb_max

        rr = np.unique(r)
        nmul = r.size//rr.size
        power = np.arange(bb_min, bb_max+1) 
        H = rr[:, None]**power 
        H = np.kron(np.eye(nmul), H)
        self.H = H
        return H
    

    def get_broadband(self, bb_pars, r=None, H=None):

        H = self.setup_broadband_H(r) if H is None else H 
        return H.dot(bb_pars)

    def fit_broadband(self, residual, icoss, H):
       
        if hasattr(self, 'inv_HWH'):
            inv_HWH = self.inv_HWH 
        else:
            inv_HWH = np.linalg.inv(H.T.dot(icoss.dot(H)))
            self.inv_HWH = inv_HWH

        bb_pars = inv_HWH.dot(H.T.dot(icoss.dot(residual)))

        return bb_pars

    def __call__(self, *p):
        pars = {}
        for i, name in enumerate(self.model.pars_names):
            pars[name] = p[i]

        model = self.get_model(self.data.r, pars)
        residual = self.data.cf - model
        inv_cov = self.data.icoss
        if self.model.fit_broadband:
            bb_pars = self.fit_broadband(residual, inv_cov, self.H)
            bb = self.get_broadband(bb_pars, H=self.H)
            #self.bb_pars = bb_pars
            residual -= bb

        chi2 = np.dot(residual, np.dot(inv_cov, residual))

        if self.priors:
            for key in self.priors.keys():
                mean = self.priors[key][0]
                sig = self.priors[key][1]
                chi2 += ((pars[key]-mean)/sig)**2

        return chi2

    def fit(self, limits=None, fixes=None, priors=None):

        init_pars = {}
        for par in self.model.pars:
            value = self.model.pars[par]
            init_pars[par] = value
            init_pars['error_'+par] = abs(value)/10. if value!=0 else 0.1
       
        self.fixes = fixes
        if fixes:
            for key in fixes.keys():
                init_pars[key] = fixes[key]
                init_pars['fix_'+key] = True 
        if limits:
            for key in limits.keys():
                init_pars['limit_'+key] = (limits[key][0], limits[key][1])


        self.priors = priors
        
        self.init_pars = init_pars

        mig = iminuit.Minuit(self, throw_nan=False, 
                             forced_parameters=self.model.pars_names, 
                             print_level=1, errordef=1, 
                             **init_pars)
        #print(mig.get_param_states())
        #mig.tol = 0.01
        imin = mig.migrad()
        print(mig.get_param_states())

        if self.model.fit_broadband:
            print('Broadband terms')    
            model = self.get_model(self.data.r, mig.values)
            residual = self.data.cf - model
            inv_cov = self.data.icoss
            bb_pars = self.fit_broadband(residual, inv_cov, self.H)
            print(bb_pars)

        #mig.hesse()
        print(mig.matrix(correlation=True))
        self.mig = mig
        self.imin = imin
        self.is_valid = imin[0]['is_valid']
        self.best_pars = mig.values 
        self.errors = mig.errors
        self.chi2min = mig.fval
        self.ndata = self.data.cf.size
        self.npars = mig.narg
        if self.model.fit_broadband:
            self.npars += bb_pars.size
            self.bb_pars = bb_pars
        self.covariance = mig.covariance
        for par in self.model.pars_names:
            if mig.fitarg['fix_'+par]:
                self.npars -= 1
        self.rchi2min = self.chi2min/(self.ndata-self.npars)
        print(f'chi2/(ndata-npars) = {self.chi2min:.2f}/({self.ndata}-{self.npars}) = {self.rchi2min:.2f}') 

    def get_correlation_coefficient(self, par_name1, par_name2):

        if not hasattr(self, 'covariance'):
            print('Chi2 was not yet minimized')
            return
        
        cov = self.covariance
        var1 = cov[par_name1, par_name1]
        var2 = cov[par_name2, par_name2]
        cov12 = cov[par_name1, par_name2]
        corr_coeff = cov12/np.sqrt(var1*var2)
        return corr_coeff
        

    def plot_bestfit(self, fig=None, model_only=0, scale_r=2, label=None, figsize=(12, 5)):

        data = self.data
        model = self.model
        nmul = 1+1*model.fit_quad+1*model.fit_hexa
        r = data.r
        cf = data.cf
        dcf = np.sqrt(np.diag(data.coss))
        r_model = np.linspace(r.min(), r.max(), 200)
        cf_model = self.get_model(r_model, self.best_pars)
        if hasattr(self, 'bb_pars'):
            bb_model = self.get_broadband(self.bb_pars, r=np.tile(r_model, nmul))
            cf_model += bb_model
            bb=True
        else:
            bb=False

        if fig is None:
            fig, axes = plt.subplots(nrows=1, ncols=nmul, figsize=figsize)
        else:
            axes = fig.get_axes()

        for i in range(nmul):
            try:
                ax = axes[i]
            except:
                ax = axes
            y_data  =  cf[i*r.size:(i+1)*r.size]
            dy_data = dcf[i*r.size:(i+1)*r.size]
            y_model = cf_model[i*r_model.size:(i+1)*r_model.size]
            y_data *= r**scale_r
            dy_data *= r**scale_r
            y_model *= r_model**scale_r 
            if bb:
                b_model = bb_model[i*r_model.size:(i+1)*r_model.size]
                b_model *= r_model**scale_r

            if not model_only:
                ax.errorbar(r, y_data, dy_data, fmt='o', ms=4)
            color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(r_model, y_model, color=color, label=label)
            if bb:
                ax.plot(r_model, b_model, '--', color=color)

            if scale_r!=0:
                ax.set_ylabel(r'$r^{%d} \xi_{%d}$ [$h^{%d}$ Mpc$^{%d}]$'%\
                              (scale_r, i*2, -scale_r, scale_r))
            else:
                ax.set_ylabel(r'$\xi_{%d}$'%(i*2), fontsize=16)
            ax.set_xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$', fontsize=16)

        return fig

    def scan(self, par_name='alpha', par_min=0.8, par_max=1.2, par_nsteps=400):

        init_pars = {}
        for par in self.model.pars.items():
            name = par[0]
            value = par[1]
            init_pars[name] = value
            init_pars['error_'+name] = abs(value)/10. if value!=0 else 0.1

        init_pars['fix_'+par_name] = True
        par_grid = np.linspace(par_min, par_max, par_nsteps)
        chi2_grid = np.zeros(par_nsteps)
       
        if self.fixes:
            for key in self.fixes:
                init_pars[key] = self.fixes[key]
                init_pars['fix_'+key] = True 

        for i in range(par_nsteps):
            value = par_grid[i]
            init_pars[par_name] = value

            mig = iminuit.Minuit(self, forced_parameters=self.model.pars_names, \
                                 print_level=1, errordef=1, \
                                 **init_pars)
            mig.migrad()
            print( 'scanning: %s = %.5f  chi2 = %.4f'%(par_name, value, mig.fval))
            chi2_grid[i] = mig.fval

        return par_grid, chi2_grid

    def scan_2d(self, par_names=['at','ap'], \
                par_min=[0.8, 0.8], \
                par_max=[1.2, 1.2], \
                par_nsteps=[40, 40] ):

        init_pars = {}
        for par in self.model.pars.items():
            name = par[0]
            value = par[1]
            init_pars[name] = value
            init_pars['error_'+name] = abs(value)/10. if value!=0 else 0.1
    

        init_pars['fix_'+par_names[0]] = True
        par_grid0 = np.linspace(par_min[0], par_max[0], par_nsteps[0])
        init_pars['fix_'+par_names[1]] = True
        par_grid1 = np.linspace(par_min[1], par_max[1], par_nsteps[1])

        chi2_grid = np.zeros(par_nsteps)
       
        if self.fixes:
            for key in self.fixes:
                init_pars[key] = self.fixes[key]
                init_pars['fix_'+key] = True 

        for i in range(par_nsteps[0]):
            value0 = par_grid0[i]
            init_pars[par_names[0]] = value0
            for j in range(par_nsteps[1]):
                value1 = par_grid1[j]
                init_pars[par_names[1]] = value1
                mig = iminuit.Minuit(self,
                         forced_parameters=self.model.pars_names, 
                         print_level=0, errordef=1, 
                         **init_pars)
                mig.migrad()
                print( 'scanning: %s = %.5f   %s = %.5f    chi2 = %.4f'%\
                        (par_names[0], value0, par_names[1], value1, mig.fval))
                chi2_grid[i, j] = mig.fval

        return par_grid0, par_grid1, chi2_grid

    def read_scan1d(self, fin):

        sfin = fin.split('.')
        par_name = sfin[-2]
        x, chi2 = np.loadtxt(fin, unpack=1)
        bestx = x[0]
        chi2min = chi2[0]
        x = np.unique(x[1:])
        chi2scan = chi2[1:]
        self.chi2scan = chi2scan
        self.x=x
        self.par_name=par_name
        self.bestx=bestx
        self.chi2min=chi2min

    def plot_scan1d(self, ls=None, 
                    color=None,  alpha=None, label=None):

        plt.plot(self.x, self.chi2scan-self.chi2min, ls=ls, 
                color=color, alpha=alpha, label=label)

    def read_scan2d(self, fin):

        sfin = fin.split('.')
        
        x, y, chi2 = np.loadtxt(fin, unpack=1)
        bestx = x[0]
        besty = y[0]
        chi2min = chi2[0]
        if chi2min>chi2.min():
            chi2min = chi2.min()
            i=0
        else:
            i=1

        x = np.unique(x[i:])
        y = np.unique(y[i:])
        chi2scan2d = np.reshape(chi2[i:], (x.size, y.size)).transpose()
        
        self.chi2scan2d = chi2scan2d
        self.x=x
        self.y=y
        self.bestx=bestx
        self.besty=besty
        self.chi2min=chi2min

    def plot_scan2d(self, levels=[2.3, 6.18, 11.83], ls=['-', '--', ':'], 
                    color='b',  alpha=1.0, label=None, scale_dist= False):

        for i in range(len(levels)):
            if i!=0:
                label=None
            if scale_dist:
                DM_rd = self.DM_rd
                DH_rd = self.DH_rd
                x = self.x*DM_rd
                y = self.y*DH_rd
            else:
                x = self.x*1.
                y = self.y*1.
            plt.contour(x, y, self.chi2scan2d-self.chi2min, 
                        levels=[levels[i]], 
                        linestyles=[ls[i]], colors=color, alpha=alpha,
                        label=label)

    def export(self, fout):

        fout = open(fout, 'w')
        fout.write('chi2    ndata    npars   rchi2\n')
        fout.write('%f \t %d \t %d \t %f \n'%\
                (self.chi2min, self.ndata, self.npars, self.rchi2min))
        fout.write('param \t value \t error\n')
       
        pars_names = np.sort([p for p in self.best_pars])
        for p in pars_names:
            print(p, self.best_pars[p], self.errors[p], file=fout)
        fout.close()

    def export_covariance(self, fout):

        fout = open(fout, 'w')
        print('# par_name1 par_name2 covariance corr_coeff', file=fout)
        cov = self.covariance
        for k in cov:
            corr = cov[k]/np.sqrt(cov[(k[0], k[0])]*cov[(k[1], k[1])])
            print(f'{k[0]} {k[1]} {cov[k]} {corr}', file=fout)
        fout.close()  


