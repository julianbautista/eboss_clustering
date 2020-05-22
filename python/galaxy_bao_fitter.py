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

class PowerSpectrum:

    def __init__(self, pk_file=None, 
                 z=0.0, par='challenge', camb_pars=None, 
                 non_linear=False, 
                 nk=2048, kmax=100., nmu=101):

        if pk_file:
            self.k, self.pk = np.loadtxt(pk_file, unpack=True)
        else:
            import cosmo
            c = cosmo.Cosmo(z=z, par=par, camb_pars=camb_pars, 
                    non_linear=non_linear, kmax=kmax, nk=nk)
            self.k = c.k*1.
            self.pk = c.pk*1.

        self.r, self.xi = self.get_correlation_function()
        self.get_sideband_xi()
        self.get_sideband_pk()
        self.set_2d_arrays(nmu=nmu)

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

    def get_sideband_xi(self, fit_range=[[50., 80.], [160., 190.]], 
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

    def get_sideband_pk(self):
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

    def get_2d_power_spectrum(self, pars, no_peak=False, decoupled=False):
        ''' Compute P(k, mu) for a set of parameters and
            pk, pk_sideband
        Input
        -----
        pars (dict): available parameters are:
                     at - alpha transverse
                     ap - alpha radial
                     aiso - alpha isotropic
                     epsilon - anisotropic parameter
                     bias - linear large scale bias parameter
                     beta - redshift space distortion parameter
                     f - growth rate of structures
                     sigma_nl - isotropic damping
                     sigma_per - transverse damping of BAO
                     sigma_par - radial damping of BAO
                     sigma_s - Finger's of God damping
                     sigma_rec - Reconstruction damping
                     bias2 - linear bias for second tracer
                     beta2 - RSD parameter for second tracer
                     beam - transverse damping (for 21cm data)
        '''

        if 'aiso' in pars:
            at = pars['aiso']/(1+pars['epsilon'])
            ap = pars['aiso']*(1+pars['epsilon'])**2
        elif 'ap' in pars:
            at = pars['at']
            ap = pars['ap']
        else:
            at = 1.
            ap = 1.
       
        if 'sigma_nl' in pars:
            sigma_par = pars['sigma_nl']
            sigma_per = pars['sigma_nl']
        elif 'sigma_par' in pars:
            sigma_par = pars['sigma_par']
            sigma_per = pars['sigma_per']
        else:
            sigma_par = 0
            sigma_per = 0

        #-- Read bias and growth rate / RSD parameter
        bias = pars['bias']
        beta = pars['beta'] if 'beta' in pars else pars['f']/bias

        #-- Read parameters for cross-correlation
        #-- or making them equal if fitting auto-correlation
        bias2 = pars['bias2'] if 'bias2' in pars else bias*1
        beta2 = pars['beta2'] if 'beta2' in pars else beta*bias/bias2
 
        k = self.k
        mu = self.mu
        pk = self.pk
        pk_sideband = self.pk_sideband

        #-- These have shape = (nmu, nk) 
        mu2d = self.mu2d 
        k2d = self.k2d 
   
        #-- Scale k and mu by alphas 
        #-- This is the correct formula (Eq. 58 and 59 from Beutler et al. 2014)
        F = ap/at
        ak2d = k2d/at * np.sqrt( 1 + mu2d**2 * (1/F**2 - 1) )
        #amu  = mu/F   / np.sqrt( 1 + mu**2   * (1/F**2 - 1) )


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
        sigma_rec = pars['sigma_rec'] if 'sigma_rec' in pars else 0.
        if sigma_rec == 0:
            recon_damp = np.ones(k.size)
        else:
            recon_damp = 1 - np.exp(-k**2*sigma_rec**2/2) 
        
        recon_damp_mu2 = np.outer(mu**2, recon_damp)
        kaiser = bias * bias2 * (1+beta*recon_damp_mu2) * (1+beta2*recon_damp_mu2)

        #-- Fingers of God
        if pars['sigma_s'] != 0:
            fog = 1./( 1 + 0.5*np.outer(mu**2, k**2)*pars['sigma_s']**2)
        else:
            fog = 1

        #-- This parameters is for intensity mapping only
        if 'beam' in pars:
            pk2d *= np.exp( - 0.5*pars['beam']**2*np.outer(1-mu**2, k**2)) 
        
        pk2d *= kaiser
        pk2d *= fog**2 

        #if not decoupled:
        #    pk2d_out /= (at**2*ap)

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
        """ Perform Hankel Transform to compute \\xi_\\ell(r) from P_\\ell(k) 
        """
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
        """ Perform Hankel Transform to compute P_\\ell(k) from \\xi_\\ell(r) 
        """
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

    def get_xi_multipoles(self, rout, pars, 
        ell_max=4, decoupled=False, no_peak=False, r0=1.):
        """ Compute \\xi_\\ell(r) from a set of parameters
        Input
        -----
        rout (np.array): contains the separation values in Mpc/h
        pars (dict): contains the parameters required for P(k, \\mu) 

        Output
        -----
        xi_mult (np.array): array with shape (n_ell, rout.size) with the \\xi_\\ell(r)
        """
        pk2d = self.get_2d_power_spectrum(pars, 
                                          no_peak = no_peak, 
                                          decoupled = decoupled)
        pk_mult = self.get_multipoles(self.mu, pk2d, ell_max=ell_max)
        _, xi_mult = self.get_xi_multipoles_from_pk(self.k, pk_mult, 
                                                    output_r=rout, r0=r0)
        return xi_mult

    def get_pk_multipoles(self, kout, pars, 
        ell_max=4, decoupled=False, no_peak=False, r0=1., apply_window=False):
        ''' Compute P_\ell(k) from a set of parameters
        Input
        -----
        kout (np.array): contains the wavevector values in h/Mpc
        pars (dict): contains the parameters required for P(k, \mu) 

        Output
        -----
        pk_mult_out (np.array): array with shape (n_ell, kout.size) with the P_\ell(k)
        '''
        pk2d = self.get_2d_power_spectrum(pars, 
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
    
class Chi2: 

    def __init__(self, data=None, model=None, parameters=None, options=None):
        self.data = data
        self.model = model
        self.parameters = parameters
        self.options = options
        self.setup_broadband_H()
        #print(parameters)

    def get_model(self, r, pars=None):
        if pars is None:
            pars = self.best_pars
        model = self.model.get_xi_multipoles(r, pars, 
            ell_max  =self.options['ell_max'], 
            decoupled=self.options['decouple_peak'], 
            no_peak  =self.options['fit_nopeak'])
        return model.ravel()
    
    def setup_broadband_H(self, r=None, bb_min=None, bb_max=None):
        if r is None:
            r = self.data.rr
        if bb_min is None:
            bb_min = self.options['bb_min']
        if bb_max is None:
            bb_max = self.options['bb_max']

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

    def __call__(self, p):
        ''' Compute chi2 for a set of free parameters (and only the free parameters!)
        '''
        pars = {}
        i = 0
        for par in self.parameters:
            if self.parameters[par]['fixed']:
                pars[par] = self.parameters[par]['value']
            else:
                pars[par] = p[i]
                limit_low = self.parameters[par]['limit_low']
                if not limit_low is None and p[i]<limit_low:
                    return np.inf
                limit_upp = self.parameters[par]['limit_upp']
                if not limit_upp is None and p[i]>limit_upp:
                    return np.inf
                i+=1

        model = self.get_model(self.data.r, pars)
        residual = self.data.cf - model
        inv_cov = self.data.icoss

        if self.options['fit_broadband']:
            bb_pars = self.fit_broadband(residual, inv_cov, self.H)
            bb = self.get_broadband(bb_pars, H=self.H)
            residual -= bb

        chi2 = np.dot(residual, np.dot(inv_cov, residual))

        for par in pars:
            if par in self.parameters and 'prior_mean' in self.parameters[par]:
                mean = self.parameters[par]['prior_mean']
                sigma = self.parameters[par]['prior_sigma']
                chi2 += ((pars[par]-mean)/sigma)**2

        return chi2

    def log_prob(self, p):
        #print(p)
        return -0.5*self.__call__(p)    

    def fit(self):

        #-- Initialise iMinuit dictionaty for initial guess of parameters
        #-- to be fitted, excluding those fixed.
        minuit_options = {}
        pars_to_fit_values = []
        pars_to_fit_name = []
        for par in self.parameters:
            if self.parameters[par]['fixed'] == True: 
                continue
            pars_to_fit_name.append(par)
            pars_to_fit_values.append(self.parameters[par]['value'])
            minuit_options['error_'+par] = self.parameters[par]['error']
            minuit_options['limit_'+par] = (self.parameters[par]['limit_low'], 
                                            self.parameters[par]['limit_upp'])

        mig = iminuit.Minuit.from_array_func(self, tuple(pars_to_fit_values),
                                            name = tuple(pars_to_fit_name),
                             print_level=1, errordef=1, throw_nan=False,
                             **minuit_options)
        #print(mig.get_param_states())
        #mig.tol = 0.01
        imin = mig.migrad()
        print(mig.get_param_states())

        best_pars = {}
        for par in self.parameters:
            best_pars[par] = {}
            if self.parameters[par]['fixed']:
                best_pars[par]['value'] = self.parameters[par]['value']
                best_pars[par]['error'] = 0
            else:
                best_pars[par]['value'] = mig.values[par]
                best_pars[par]['error'] = mig.errors[par]

        if self.options['fit_broadband']==True:
            print('\nBroadband terms')
            pars = {par: best_pars[par]['value'] for par in best_pars}    
            model = self.get_model(self.data.r, pars)
            residual = self.data.cf - model
            inv_cov = self.data.icoss
            bb_pars = self.fit_broadband(residual, inv_cov, self.H)
            self.bb_pars = bb_pars
            ibb = np.arange(self.options['bb_max']-self.options['bb_min']+1)
            bb_name = []
            bb_name+= [f'bb_{i}_mono' for i in ibb]
            if self.options['ell_max']>=2:
                bb_name+= [f'bb_{i}_quad' for i in ibb]
            if self.options['ell_max']>=4:
                bb_name+= [f'bb_{i}_hexa' for i in ibb]
            for bb, bbn in zip(bb_pars, bb_name):
                best_pars[bbn] = {'value': bb, 'error': 0}
                print(bbn, bb)

        #mig.hesse()
        print('\nApproximate correlation coefficients:')
        print(mig.matrix(correlation=True))
        #self.mig = mig
        #self.imin = imin
        self.is_valid = imin[0]['is_valid']
        self.best_pars = best_pars
        self.chi2min = mig.fval
        self.ndata = self.data.cf.size
        self.npars = len(pars_to_fit_name)
        if self.options['fit_broadband']:
            self.npars += bb_pars.size
        #self.covariance = mig.covariance
        self.rchi2min = self.chi2min/(self.ndata-self.npars)
        print(f'\n chi2/(ndata-npars) = {self.chi2min:.2f}/({self.ndata}-{self.npars}) = {self.rchi2min:.2f}') 

    def get_correlation_coefficient(self, par_par1, par_par2):

        if not hasattr(self, 'covariance'):
            print('Chi2 was not yet minimized')
            return
        
        cov = self.covariance
        var1 = cov[par_par1, par_par1]
        var2 = cov[par_par2, par_par2]
        cov12 = cov[par_par1, par_par2]
        corr_coeff = cov12/np.sqrt(var1*var2)
        return corr_coeff
        

    def plot_bestfit(self, fig=None, model_only=0, scale_r=2, label=None, figsize=(10, 4)):

        data = self.data
        model = self.model
        nmul = self.options['ell_max']//2+1
        r = data.r
        cf = data.cf
        dcf = np.sqrt(np.diag(data.coss))
        r_model = np.linspace(r.min(), r.max(), 200)
        pars = {par: self.best_pars[par]['value'] for par in self.best_pars}
        cf_model = self.get_model(r_model, pars)
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
            #if bb:
            #    ax.plot(r_model, b_model, '--', color=color)

            if scale_r!=0:
                ax.set_ylabel(r'$r^{%d} \xi_{%d}$ [$h^{%d}$ Mpc$^{%d}]$'%\
                              (scale_r, i*2, -scale_r, scale_r))
            else:
                ax.set_ylabel(r'$\xi_{%d}$'%(i*2), fontsize=16)
            ax.set_xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$', fontsize=12)

        return fig

    def scan1d(self, par_name='alpha', par_min=0.8, par_max=1.2, par_nsteps=400):

        #-- Initialise chi2 grid
        par_grid = np.linspace(par_min, par_max, par_nsteps)
        chi2_grid = np.zeros(par_nsteps)

        for i in range(par_nsteps):
            self.parameters[par_name]['value'] = par_grid[i]
            self.parameters[par_name]['fixed'] = True

            minuit_options = {}
            pars_to_fit_values = []
            pars_to_fit_name = []
            for par in self.parameters:
                if self.parameters[par]['fixed'] == True: 
                    continue
                pars_to_fit_name.append(par)
                pars_to_fit_values.append(self.best_pars[par]['value'])
                minuit_options['error_'+par] = self.best_pars[par]['error']
                minuit_options['limit_'+par] = (self.parameters[par]['limit_low'], 
                                                self.parameters[par]['limit_upp'])

            mig = iminuit.Minuit.from_array_func(self, tuple(pars_to_fit_values),
                                            name = tuple(pars_to_fit_name),
                             print_level=0, errordef=1, throw_nan=False,
                             **minuit_options)
            mig.migrad()
            print( 'scanning: %s = %.5f  chi2 = %.4f'%(par_name, par_grid[i], mig.fval))
            chi2_grid[i] = mig.fval

        return par_grid, chi2_grid

    def scan_2d(self, par_names=['at','ap'], \
                par_min=[0.8, 0.8], \
                par_max=[1.2, 1.2], \
                par_nsteps=[40, 40] ):

        #-- Initialise chi2 grid
        par0 = par_names[0]
        par1 = par_names[1]
        par_grid0 = np.linspace(par_min[0], par_max[0], par_nsteps[0])
        par_grid1 = np.linspace(par_min[1], par_max[1], par_nsteps[1])
        chi2_grid = np.zeros(par_nsteps)

        for i in range(par_nsteps[0]):
            self.parameters[par0]['value'] = par_grid0[i]
            self.parameters[par0]['fixed'] = True
            for j in range(par_nsteps[1]):
                self.parameters[par1]['value'] = par_grid1[j]
                self.parameters[par1]['fixed'] = True

                minuit_options = {}
                pars_to_fit_values = []
                pars_to_fit_name = []
                for par in self.parameters:
                    if self.parameters[par]['fixed'] == True: 
                        continue
                    pars_to_fit_name.append(par)
                    pars_to_fit_values.append(self.best_pars[par]['value'])
                    minuit_options['error_'+par] = self.best_pars[par]['error']
                    minuit_options['limit_'+par] = (self.parameters[par]['limit_low'], 
                                                    self.parameters[par]['limit_upp'])

                mig = iminuit.Minuit.from_array_func(self, tuple(pars_to_fit_values),
                                                name = tuple(pars_to_fit_name),
                                print_level=0, errordef=1, throw_nan=False,
                                **minuit_options)
                mig.migrad()
                print( 'scanning: %s = %.5f   %s = %.5f    chi2 = %.4f'%\
                        (par0, par_grid0[i], par0, par_grid1[j], mig.fval))
                chi2_grid[i, j] = mig.fval

        return par_grid0, par_grid1, chi2_grid

    def export_bestfit_parameters(self, fout):

        fout = open(fout, 'w')
        print(f'chi2  {self.chi2min}', file=fout)
        print(f'ndata {self.ndata}', file=fout)
        print(f'npars {self.npars}', file=fout)
        print(f'rchi2 {self.rchi2min}', file=fout)
       
        for p in self.best_pars:
            print(p,          self.best_pars[p]['value'], file=fout)
            print(p+'_error', self.best_pars[p]['error'], file=fout)

        fout.close()

    def export_covariance(self, fout):

        fout = open(fout, 'w')
        print('# par_par1 par_par2 covariance corr_coeff', file=fout)
        cov = self.covariance
        for k in cov:
            corr = cov[k]/np.sqrt(cov[(k[0], k[0])]*cov[(k[1], k[1])])
            print(f'{k[0]} {k[1]} {cov[k]} {corr}', file=fout)
        fout.close()  

    def export_model(self, fout):

        nmul = self.options['ell_max']//2+1
        nr = 200
        r_model = np.linspace(self.data.r.min(), self.data.r.max(), nr)
        pars = {par: self.best_pars[par]['value'] for par in self.best_pars}  
        cf_model = self.get_model(r_model, pars)

        if hasattr(self, 'bb_pars'):
            bb_model = self.get_broadband(self.bb_pars, r=np.tile(r_model, nmul))
            cf_model += bb_model
            bb=True
        else:
            bb=False

        cf_model = cf_model.reshape((nmul, nr)) 
        if bb:
            bb_model = bb_model.reshape((nmul, nr)) 
        
        fout = open(fout, 'w')
        line = '#r mono '
        line += 'quad '*(self.options['ell_max']>=2)
        line += 'hexa '*(self.options['ell_max']>=4)
        if bb:
            line += 'bb_mono '
            line += 'bb_quad '*(self.options['ell_max']>=2)
            line += 'bb_hexa '*(self.options['ell_max']>=4)
        print(line, file=fout)

        for i in range(nr):
            line = f'{r_model[i]}  '
            for l in range(nmul):
                line += f'{cf_model[l, i]}  ' 
            if bb:
                for l in range(nmul):
                    line += f'{bb_model[l, i]}  ' 
            print(line, file=fout)
        fout.close()


