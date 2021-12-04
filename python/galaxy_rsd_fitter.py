import numpy as np 
import pylab as plt
from scipy.stats import norm
from scipy import integrate
from scipy import interpolate


def legendre(ell, mu):

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


def multipoles(x, ell_max=8):
    ''' Get multipoles of any function of ell
        It assumes symmetry around mu=0 
        Input
        -----
        x: np.array with shape (nmu, nx) from which the multipoles will be computed    
        mu: np.array with shape (nmu) where nmu is the number of mu bins
        
        Returns
        ----
        f_mult: np.array with shape (nell, nx) 
    '''
    
    n_mu = x.shape[0]
    n_x = x.shape[1]
    n_ell = ell_max//2+1
    x_mult = np.zeros((n_ell, n_x))
    ell = np.arange(0, ell_max+2, 2)
    mu = np.linspace(0, 1, n_mu)
    for i in range(n_ell):
        leg = legendre(ell[i], mu)
        x_mult[i] = np.trapz(x*leg[:, None], x=mu, axis=0)
        x_mult[i] *= (2*ell[i]+1)
    return x_mult

def mass_function(self):        
    ''' Sheth-Tormen mass function
        NOT USED 
    '''
    a = 0.707
    p = 0.3
    delta_c = 1.686
    nu = np.linspace(0, 5., 1000)
    anu2 = a*nu**2
    f1 = 1/delta_c*( anu2 - 1 + 2*p/(1+anu2**p)  )
    f2 = 1/delta_c**2*( anu2**2 - 3*anu2 + 2*p*(2*anu2+2*p-1)/(1+anu2**p)  )

class CLPTGS:

    def __init__(self, xi_file, v12_file, s12_file):

        #-- Reading output of CLPT code by Wang et al. 2013
        #-- https://github.com/wll745881210/CLPT_GSRSD
        #r = np.loadtxt(xi_file, unpack=True)[0]
        #xi = np.loadtxt(xi_file, unpack=True)[2:]
        #v12 = np.loadtxt(v12_file, unpack=True)[2:]
        #s12 = np.loadtxt(s12_file, unpack=True)[1:]

        #-- Reading output of CLPT code by Breton et al. 
        #-- https://github.com/mianbreton/CLPT_GS 
        xi = np.loadtxt(xi_file, unpack=True)
        v12 = np.loadtxt(v12_file, unpack=True)[1:]
        s12 = np.loadtxt(s12_file, unpack=True)[1:]

        #-- Integration parameters
        lo_integ = -100 
        hi_integ = 300
        nbins_integ = 500
        nbins_mu = 101
        rpar_integ = np.linspace(lo_integ, hi_integ, nbins_integ)
        mu_mod = np.linspace(0, 1, nbins_mu)

        self.r = xi[0]
        self.xi = xi[1:]
        self.v12 = v12
        self.s12 = s12
        self.mu_mod = mu_mod
        self.rpar_integ = rpar_integ

    def get_xi_2d(self, r_output, parameters):

        r = self.r
        xi = self.xi
        v12 = self.v12
        s12 = self.s12
        mu_mod = self.mu_mod
        rpar_integ = self.rpar_integ

        for par in ['f', 'f1', 'f2']:
            if par not in parameters:
                print(f'ERROR: {par} is a required parameter. Only found:')
                print(parameters)
                return

        xi_s = (xi[0] 
                + parameters['f1']*xi[1] 
                + parameters['f2']*xi[2] 
                + parameters['f1']**2*xi[3] 
                + parameters['f1']*parameters['f2']*xi[4]
                + parameters['f2']**2*xi[5])
    
        v12_s = parameters['f']/(1+xi_s) * (v12[0] 
                + parameters['f1']*v12[1] 
                + parameters['f2']*v12[2] 
                + parameters['f1']**2*v12[3] 
                + parameters['f1']*parameters['f2']*v12[4])

        s12_para = parameters['f']**2/(1+xi_s) * (s12[0] 
                + parameters['f1']*s12[1] 
                + parameters['f2']*s12[2] 
                + parameters['f1']**2*s12[3])

        s12_perp = parameters['f']**2/(1+xi_s) * (s12[4] 
                + parameters['f1']*s12[5] 
                + parameters['f2']*s12[6] 
                + parameters['f1']**2*s12[7])
  
        if 'alpha_iso' in parameters:
            at = parameters['alpha_iso']/(1+parameters['epsilon'])
            ap = parameters['alpha_iso']*(1+parameters['epsilon'])**2
        elif 'alpha_perp' in parameters:
            at = parameters['alpha_perp']
            ap = parameters['alpha_para']
        else:
            at = 1
            ap = 1
        
        if 'sigma_fog' in parameters:
            sigma_fog = parameters['sigma_fog']
        else:
            sigma_fog = 0.

        #-- Gaussian Streaming Model

        #-- Scale separations by alphas 
        s_perp = at*np.sqrt(1-mu_mod**2)[:, None]*r_output
        s_para = ap*mu_mod[:, None]*r_output
        #-- Make a 3D matrix for integral using np.trapz
        #-- This is a hack and can be improved
        r_integ = np.sqrt(s_perp[:, :, None]**2 + rpar_integ**2)
        mu_integ = rpar_integ/r_integ
        #-- Interpolate CLPT functions for all separations in 3D matrix
        #-- This can be improved given that many are repeated
        xi_integ = np.interp(r_integ, r, xi_s)
        v12_integ = np.interp(r_integ, r, v12_s)
        s12_para_integ = np.interp(r_integ, r, s12_para)
        s12_perp_integ = np.interp(r_integ, r, s12_perp)
        v12_para_integ = v12_integ*mu_integ
        sigma_2 = mu_integ**2 * s12_para_integ + (1-mu_integ**2)* s12_perp_integ
        sigma2_tot = sigma_2 + sigma_fog**2
        #-- Define Gaussian for integration
        func = np.exp( -0.5*(s_para[:, :, None]-rpar_integ-v12_para_integ)**2/sigma2_tot)
        func *= 1/np.sqrt(2*np.pi*sigma2_tot)
        func *= (1+xi_integ)
        #-- Integrate to obtain xi(s_perp, s_para)
        xi_2d = -1 + np.trapz(func, x=rpar_integ, axis=2)
        #-- Jacobian 
        #xi_2d/=at**2*ap
        
        #self.v12_para_integ = v12_para_integ
        #self.sigma2_tot = sigma2_tot
        #self.func = func
        
        self.xi_s = xi_s
        self.v12_s = v12_s
        self.s12_para = s12_para
        self.s12_perp = s12_perp

        return xi_2d

    def get_xi_multipoles(self, r_output, parameters, ell_max=4):
        """ Compute \\xi_\\ell(r) from a set of parameters
            This is the function used in the class Chi2

        Input
        -----
        rout (np.array): contains the separation values in Mpc/h
        parameters (dict): contains the parameters required for xi_s(r, \\mu) 
        options (dict): contains options for the calculation including
                        ell_max
        Output
        -----
        xi_mult (np.array): array with shape (n_ell, rout.size) with the \\xi_\\ell(r)
        """
        xi_2d = self.get_xi_2d(r_output, parameters)
        xi_mult = multipoles(xi_2d, ell_max=ell_max)
        return xi_mult
