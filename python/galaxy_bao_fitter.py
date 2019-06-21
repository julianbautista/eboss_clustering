from __future__ import print_function
import camb
import numpy as np
import pylab as plt
import fftlog
import iminuit
import scipy.interpolate 
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

class Cosmo:

    def __init__(self, z=0.0, name='challenge', norm_pk=False, non_linear=False, 
                 nk=2048, kmax=100., nmu=101):
        self.get_matter_power_spectrum(z=z, name=name, norm_pk=norm_pk, 
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
                                    omch2=0.11902256, 
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
            elif name == 'outerim':
                pars.set_cosmology(H0=71., ombh2=0.022584, 
                                   omch2=0.10848, 
                                   YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.0,
                                    num_massive_neutrinos=0)
                pars.InitPower.set_params(As=2.224615e-09, ns=0.963)
            elif name == 'EZmock':
                pars.set_cosmology(H0=67.77, ombh2=0.0221399210, 
                                   omch2=0.1189110239, 
                                   YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.0,
                                    num_massive_neutrinos=0)
                pars.InitPower.set_params(As=2.11622e-09, ns=0.9611)
            
                

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
        self.r_drag = results.get_derived_params()['rdrag']

        return kh, pk[0]
        
    def get_correlation_function(self, k=None, pk=None,  
                                 Sigma_nl=0., r=None, 
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
        pklog *= np.exp(-0.5*klog**2*Sigma_nl**2)

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

        ks, pks = self.get_correlation_function(k=self.r, pk=self.xi_sideband,\
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

    def smooth(self, xi_peak, dr=1., Sigma_nl=4.):
        ''' Gaussian smoothing
            dr: size of bins of xi_peak
            Sigma_nl: smoothing length
        '''
        xi_smooth = gaussian_filter1d(xi_peak, sigma=Sigma_nl/dr)
        return xi_smooth 

    def get_multipoles(self, r, xi, f):
        ''' Compute multipoles from isotropic correlation function 
            with linear redshift-space distortions
            following Hamilton 1992

            Currently not used but here for reference
        '''
        xib   = np.array([ np.sum(xi[:i]*r[:i]**2) for i in range(r.size)])\
                * 3./r**3 * np.gradient(r)
        xibb = np.array([ np.sum(xi[:i]*r[:i]**4) for i in range(r.size)])\
                * 5./r**5 * np.gradient(r)
        xi0 = (1+2./3*f+1./5*f**2)*xi
        xi2 = (4./3*f + 4./7*f**2)*(xi-xib)
        xi4 = 8./35*f**2*(xi + 2.5*xib - 3.5*xibb)
        return xi0, xi2, xi4

    def set_2d_arrays(self, nmu=201):

        self.mu = np.linspace(0, 1., nmu)
        self.mu2d = np.tile(self.mu[:, None], (1, self.k.size))
        self.k2d = np.tile(self.k, (self.mu.size, 1))

    def get_2d_power_spectrum(self, pars, ell_max=2, no_peak=0):

        if hasattr(self, 'pars') and pars==self.pars:
            return self.pk2d_out

        if 'aiso' in pars:
            at = pars['aiso']
            ap = pars['aiso']
            Sigma_par = pars['Sigma_NL']
            Sigma_per = pars['Sigma_NL']
        else:
            at = pars['at']
            ap = pars['ap']
            Sigma_par = pars['Sigma_par']
            Sigma_per = pars['Sigma_per']
        

        k = self.k
        mu = self.mu
        
        #-- These have shape = (nmu, nk) 
        mu2d = self.mu2d 
        k2d = self.k2d 
   
        #-- scale k and mu by alphas 
        if 'aiso' in pars:
            ak2d = k2d/at
            amu = mu*1.
        else:
            #-- this is the correct formula (Beutler et al. 2013)
            F = ap/at
            ak2d = k2d/at * np.sqrt( 1 + mu2d**2 * (1/F**2 - 1) )
            amu   = mu/F  * np.sqrt( 1 + mu**2   * (1/F**2 - 1) )**(-1) 

        #-- This has shape = (nmu, nk) 
        amu2d = np.broadcast_to( amu, (k.size, amu.size)).T 


        bias = pars['bias']
        if 'beta' in pars:
            beta = pars['beta']
            fit_beta = True
        else:
            f = pars['f']
            fit_beta = False

        Sigma_rec = pars['Sigma_rec']

        #-- dealing with cross-correlation
        if 'bias2' in pars:
            bias2 = pars['bias2']
            if fit_beta:
                beta2 = pars['beta2']
        else:
            bias2 = bias*1.
            #f2 = f*1.
            beta2 = beta*1.

        #-- linear Kaiser redshift space distortions with reconstruction damping
        if Sigma_rec == 0:
            if fit_beta:
                Kaiser = bias*bias2*(1+beta*amu2d**2)*(1+beta2*amu2d**2)
            else:
                Kaiser = (bias+f*amu2d**2)*(bias2+f*amu2d**2)
        else:
            if fit_beta:
                Kaiser = bias*bias2*\
                         (1+beta *(1.-np.exp(-ak2d**2*Sigma_rec**2/2))*amu2d**2) * \
                         (1+beta2*(1.-np.exp(-ak2d**2*Sigma_rec**2/2))*amu2d**2)
            else:
                Kaiser = (bias +f *(1.-np.exp(-ak2d**2*Sigma_rec**2/2))*amu2d**2) * \
                         (bias2+f *(1.-np.exp(-ak2d**2*Sigma_rec**2/2))*amu2d**2)

        #-- Fingers of God
        if pars['Sigma_s'] != 0:
            Dnl = 1./( 1 + ak2d**2*amu2d**2*pars['Sigma_s']**2/2)
        else:
            Dnl = 1

        #-- Sideband model (no BAO peak)
        apk2d_s = np.interp(ak2d, self.k, self.pk_sideband)

        if no_peak:
            pk2d_out = apk2d_s
        else:
            #-- Anisotropic damping applied to BAO peak only
            sigma_v2 = (1-amu**2)*Sigma_per**2/2+ amu**2*Sigma_par**2/2 
            apk2d = np.interp(ak2d, self.k, self.pk)
            pk2d_out = ( (apk2d - apk2d_s)*np.exp(-ak2d**2*sigma_v2[:, None]) + apk2d_s)

        #-- this parameters is for intensity mapping only
        if 'beam' in pars:
            pk2d_out *= np.exp( -ak2d**2*pars['beam']**2*(1-amu2d**2)/2) 
        
        pk2d_out *= Kaiser
        pk2d_out *= Dnl**2 
        pk2d_out /= (at**2*ap)

        #-- this parameters is for intensity mapping only
        if 'THI' in pars:
            pk2d_out *= pars['THI']

        self.amu2d = amu2d
        self.ak2d = ak2d
        self.pk2d_out = pk2d_out
        self.pars = pars

        return pk2d_out

    def get_pk_multipoles(self, mu2d, pk2d, ell_max=4):
        
        nk = pk2d.shape[1]
        pk_mult = np.zeros((ell_max//2+1, nk))
        for ell in range(0, ell_max+2, 2):
            Leg = self.Legendre(ell, mu2d)
            pk_mult[ell//2] = (2*ell+1)*np.trapz(pk2d*Leg, x=mu2d, axis=0)
        self.pk_mult = pk_mult

        return pk_mult

    def Legendre(self, ell, mu):

        if ell == 0:
            return 1
        elif ell == 2:
            return 0.5*(3*mu**2-1)
        elif ell == 4:
            return 0.125*(35*mu**4 - 30*mu**2 +3)
        else:
            return -1

    def get_xi_multipoles_from_pk(self, k, pk_mult, output_r=None, r0=1.):

        xi_mult = []
        ell = 0 
        for pk in pk_mult:
            rout, xiout = fftlog.HankelTransform(k, pk, mu=0.5+ell, output_r=output_r,
                                                 output_r_power=-3, q=1.5, r0=r0)
            norm = 1/(2*np.pi)**1.5 * (-1)**(ell/2)
            xi_mult.append(xiout*norm)
            ell+=2 

        return rout, np.array(xi_mult)

    def get_xi_multipoles(self, rout, pars, ell_max=4, no_peak=False, r0=1.):

        pk2d = self.get_2d_power_spectrum(pars, 
                            ell_max=ell_max, no_peak=no_peak)
        pk_mult = self.get_pk_multipoles(self.mu2d, pk2d, ell_max=ell_max)
        r, xi_out = self.get_xi_multipoles_from_pk(self.k, pk_mult, 
                        output_r=rout, r0=r0)
        return xi_out

    @staticmethod
    def test(z=0, 
             pars_to_test= {'aiso': [0.95, 1.0, 1.05], 
                            'epsilon': [1.0, 1.0201, 0.9799], 
                            'beta': [0.35, 0.45, 0.25], 
                            'Sigma_rec': [0, 5, 10],
                            'Sigma_s': [0, 2, 4]},
             pars_center = {'ap': 1.0, 'at': 1.0, 
                            'bias': 1.0, 'beta': 0.6, 
                            'Sigma_par': 10., 'Sigma_per': 6., 
                            'Sigma_s': 4., 'Sigma_rec': 0.},
             rmin=1., rmax=200., scale_r=2, ell_max=4, figsize=(6, 8)):

        r = np.linspace(rmin, rmax, 2000)
        cosmo = Cosmo(z=z, name='planck')
        nell = ell_max//2+1
        lss = ['-', '--', ':', '-.']

        if 'aiso' in pars_to_test:
            plt.figure(figsize=figsize)
            pars = pars_center.copy()
            for i, ap in enumerate(pars_to_test['aiso']):
                pars['at'] = ap
                pars['ap'] = ap
                aiso = ap
                xi_mult = cosmo.get_xi_multipoles(r, pars, ell_max=ell_max)
                for j in range(nell):
                    plt.subplot(nell, 1, j+1)
                    plt.plot(r, xi_mult[j]*r**scale_r, 
                             ls=lss[i], color='k', lw=2, \
                             label=r'$\alpha_{\rm iso} = %.2f$'%aiso)
                    ylabel = r'$\xi_{%d}$'%(j*2)
                    if scale_r:
                        ylabel+= r'$r^%d [h^{-%d} \mathrm{Mpc}^{%d}]$'%(scale_r, scale_r, scale_r)
                    plt.ylabel(ylabel)
            plt.xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')
            plt.legend(loc=0, fontsize=10)
            plt.tight_layout()

        if 'epsilon' in pars_to_test: 
            plt.figure(figsize=figsize)
            pars = pars_center.copy()
            for i, ap in enumerate(pars_to_test['epsilon']):
                pars['at'] = 1./np.sqrt(ap)
                pars['ap'] = ap
                epsilon = (ap*np.sqrt(ap))**(1./3)-1
                xi_mult = cosmo.get_xi_multipoles(r, pars)
                for j in range(nell):
                    plt.subplot(nell, 1, j+1)
                    plt.plot(r, xi_mult[j]*r**scale_r, ls=lss[i], color='k', lw=2, \
                           label=r'$\epsilon = %.3f$'%epsilon)
                    ylabel = r'$\xi_{%d}$'%(j*2)
                    if scale_r:
                        ylabel+= r'$r^%d [h^{-%d} \mathrm{Mpc}^{%d}]$'%(scale_r, scale_r, scale_r)
                    plt.ylabel(ylabel)
            plt.xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')
            plt.legend(loc=0, fontsize=10)
            plt.tight_layout()

        for par in pars_to_test:
            if par == 'aiso' or par == 'epsilon': continue
            plt.figure(figsize=figsize)
            pars = pars_center.copy()
            values = pars_to_test[par]
            for i, val in enumerate(values):
                pars[par] = val
                xi_mult = cosmo.get_xi_multipoles(r, pars)
                for j in range(nell):
                    plt.subplot(nell, 1, j+1)
                    plt.plot(r, xi_mult[j]*r**scale_r, ls=lss[i], color='k', lw=2, \
                           label=f'{par} = {val:.3f}')
                    ylabel = r'$\xi_{%d}$'%(j*2)
                    if scale_r:
                        ylabel+= r'$r^%d [h^{-%d} \mathrm{Mpc}^{%d}]$'%(scale_r, scale_r, scale_r)
                    plt.ylabel(ylabel)
            plt.xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')
            plt.legend(loc=0, fontsize=10)
            plt.tight_layout() 

    @staticmethod 
    def test_xu2012():

        Cosmo.test(z=0.35, 
                   pars_to_test={'epsilon': [1.0, 1.0201, 0.9799]}, 
                   pars_center={'ap': 1.0, 'at': 1.0, 'bias': 2.2, 'beta': 0.35, 
                                'Sigma_par': 0.0, 'Sigma_per': .0, 'Sigma_s': 0, 
                                'Sigma_rec': 0.0})
         

    def get_dist_rdrag(self):
        
        self.DH_rd = 300000./self.H_z/self.r_drag
        self.DM_rd = self.D_A*(1+self.z)/self.r_drag
        self.DV_rd = (300000.*self.z*(self.D_A*(1+self.z))**2/self.H_z)**(1./3)/self.r_drag
        print('D_H(z=%.2f)/r_d = %.2f'%(self.z, self.DH_rd))
        print('D_M(z=%.2f)/r_d = %.2f'%(self.z, self.DM_rd))
        print('D_V(z=%.2f)/r_d = %.2f'%(self.z, self.DV_rd))

    @staticmethod
    def get_alphas(cosmo, cosmo_fid):

        at = (cosmo.D_A/cosmo.r_drag)/(cosmo_fid.D_A/cosmo_fid.r_drag)
        ap = (cosmo_fid.r_drag*cosmo_fid.H_z)/(cosmo.r_drag*cosmo.H_z)
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
        print(r.size, ncf, ncov)
       
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
        
        self.r = np.unique(rr)
        self.cf = cf
        self.coss = coss
        print('Covariance matrix is positive definite?', np.all(np.linalg.eigvals(coss)>0))
        self.icoss = np.linalg.inv(coss)
        if nmocks:
            correction = (1 - (cf.size + 1.)/(nmocks-1))
            self.icoss *= correction

class Model:

    def __init__(self, name='challenge', z = 0, 
                 fit_broadband=True, bb_min=-2, bb_max=0, 
                 norm_pk=False, non_linear=False, no_peak=False,
                 fit_quad=False, fit_hexa=False,
                 fit_iso=False, 
                 fit_beta=False, fit_cross=False, 
                 fit_amp=False, fit_beam=False):

        cosmo = Cosmo(z=z, name=name, norm_pk=norm_pk, non_linear=non_linear)

        #-- define parameter dictionary
        pars = {}
        
        if fit_iso:
            pars_names = ['aiso']
            pars['aiso'] = 1.0
            pars_names+= ['Sigma_NL']
            pars['Sigma_NL'] = 6.
        else:
            pars_names = ['at', 'ap']
            pars['at'] = 1.0
            pars['ap'] = 1.0
            pars_names += ['Sigma_per', 'Sigma_par']
            pars['Sigma_per'] = 6.
            pars['Sigma_par'] = 10.
       
         
        pars_names += ['bias', 'Sigma_s', 'Sigma_rec']
        pars['bias'] = 3.0
        pars['Sigma_s'] = 4.
        pars['Sigma_rec'] = 15.
        
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
            pars['THI'] = 0.3e-3
            pars_names += ['THI']

        if fit_beam:
            pars_names += ['beam']
            pars['beam'] = 4.

        if fit_broadband:
            for i, bb_power in enumerate(np.arange(bb_min, bb_max+1)):
                pars_names.append('bb_%d_mono'%i)
                pars['bb_%d_mono'%i] = 0.
                if fit_quad:
                    pars_names.append('bb_%d_quad'%i)
                    pars['bb_%d_quad'%i] = 0.
                if fit_hexa:
                    pars_names.append('bb_%d_hexa'%i)
                    pars['bb_%d_hexa'%i] = 0.

        self.bb_min = bb_min
        self.bb_max = bb_max
        self.pars = pars
        self.pars_names = pars_names
        self.fit_broadband = fit_broadband
        self.fit_quad = fit_quad
        self.no_peak = no_peak
        self.fit_cross = fit_cross
        self.fit_beam = fit_beam
        self.fit_beta = fit_beta
        self.fit_amp = fit_amp
        self.fit_hexa = fit_hexa
        self.cosmo = cosmo
        
    def value(self, rout, pars):

        ell_max = 0 + 2*(self.fit_quad) + 2*(self.fit_hexa)
        cf_out = self.cosmo.get_xi_multipoles(rout,  pars, 
                            ell_max=ell_max, no_peak=self.no_peak)
        return cf_out.ravel()

    def get_broadband(self, rout, pars):
        
        if hasattr(self, 'pars_bb') and self.pars_bb==pars:
            return self.bb
  
        monobb = rout*0.
        quadbb = rout*0.
        hexabb = rout*0.
        for i in range(self.bb_max-self.bb_min+1):
            power = self.bb_min + i
            monobb += pars['bb_%d_mono'%i]*(rout**power)
            if self.fit_quad:
                quadbb += pars['bb_%d_quad'%i]*(rout**power)
            if self.fit_hexa: 
                hexabb += pars['bb_%d_hexa'%i]*(rout**power)

        bb = monobb
        if self.fit_quad:
            bb = np.append(bb, quadbb)
        if self.fit_hexa:
            bb = np.append(bb, hexabb)
        self.pars_bb = pars
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
                           no_peak=no_peak, norm_pk=0,
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

        if self.model.fit_broadband:
            pars_bb = {par: pars[par] for par in pars if par.startswith('bb')}
            bb = self.model.get_broadband(r, pars_bb)
            model += bb

        return model

    def __call__(self, *p):
        pars = {}
        for i, name in enumerate(self.model.pars_names):
            pars[name] = p[i]

        model = self.get_model(self.data.r, pars)
        residual = self.data.cf - model
        inv_cov = self.data.icoss

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
        #mig.tol = 10.0 
        imin = mig.migrad()
        #mig.hesse()
        self.mig = mig
        self.imin = imin
        self.is_valid = imin[0]['is_valid']
        self.best_pars = mig.values 
        self.errors = mig.errors
        self.chi2min = mig.fval
        self.ndata = self.data.cf.size
        self.npars = mig.narg
        self.covariance = mig.covariance
        for par in self.model.pars_names:
            if mig.fitarg['fix_'+par]:
                self.npars -= 1
        self.rchi2min = self.chi2min/(self.ndata-self.npars)
        print(f'chi2/(ndata-npars) = {self.chi2min:.2f}/({self.ndata}-{self.npars}) = {self.rchi2min:.2f}') 


    def plot_bestfit(self, fig=None, model_only=0, scale_r=2, label=None, figsize=(12, 5)):

        data = self.data
        model = self.model
        r = data.r
        cf = data.cf
        dcf = np.sqrt(np.diag(data.coss))
        r_model = np.linspace(r.min(), r.max(), 200)
        cf_model = self.get_model(r_model, self.best_pars)

        nmul = 1+1*model.fit_quad+1*model.fit_hexa

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

            if not model_only:
                ax.errorbar(r, y_data, dy_data, fmt='o', ms=4)
            ax.plot(r_model, y_model, label=label)

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
                mig = iminuit.Minuit(self, \
                         forced_parameters=self.model.pars_names, \
                         print_level=1, errordef=1, \
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

    def plot_scan1d(self, ls=None, \
                    color=None,  alpha=None, label=None):

        plt.plot(self.x, self.chi2scan-self.chi2min, ls=ls, \
                color=color, alpha=alpha, label=label)

    def read_scan2d(self, fin):

        sfin = fin.split('.')
        #par_name0 = sfin[-3]
        #par_name1 = sfin[-2]

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
        #self.par_name0=par_name0
        #self.par_name1=par_name1
        self.bestx=bestx
        self.besty=besty
        self.chi2min=chi2min

    def plot_scan2d(self, levels=[2.3, 6.18, 11.83, 19.33], ls=['-', '--', '-.', ':'], \
                    color='b',  alpha=1.0, label=None, scale_dist=0,
                    transverse=False, DM_rd=None, DH_rd=None):

        if transverse:
            x = self.y
            y = self.x
            chi2 = self.chi2scan2d.T
        else:
            x = self.x
            y = self.y
            chi2 = self.chi2scan2d

        for i in range(len(levels)):
            if i!=0:
                label=None
            if scale_dist:
                x *= DM_rd
                y *= DH_rd
            plt.contour(x, y, chi2-self.chi2min,
                        levels=[levels[i]], linestyles=[ls[i]], colors=color, alpha=alpha)

    def get_parameter(self, par_name='alpha'):

        if par_name in self.minos:
            par_dict = self.minos[par_name]
            return par_dict['min'], par_dict['lower'], par_dict['upper']
        else:
            mig = self.mig
            return mig.values[par_name], mig.errors[par_name]

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




