import numpy as np 
import pylab as plt
from scipy.stats import norm
from scipy import integrate
from scipy import interpolate

class CLPTGS:

    def __init__(self, xi_file, v12_file, s12_file):

        self.r = np.loadtxt(xi_file, unpack=True)[0]
        self.xi = np.loadtxt(xi_file, unpack=True)[2:]
        self.v12 = np.loadtxt(v12_file, unpack=True)[2:]
        self.s12 = np.loadtxt(s12_file, unpack=True)[1:]

        #- Sheth-Tormen mass function
        a = 0.707
        p = 0.3
        delta_c = 1.686
        nu = np.linspace(0, 5., 1000)
        anu2 = a*nu**2
        f1 = 1/delta_c*( anu2 - 1 + 2*p/(1+anu2**p)  )
        f2 = 1/delta_c**2*( anu2**2 - 3*anu2 + 2*p*(2*anu2+2*p-1)/(1+anu2**p)  )

        ####################################################
        # Integration parameters
        ####################################################
        lo_integ = -100 
        hi_integ = 300
        nbins_integ = 500
        nbins_mu = 101

        rpar_integ = np.linspace(lo_integ, hi_integ, nbins_integ)
        mu_mod = np.linspace(0, 1, nbins_mu)

        self.mu_mod = mu_mod
        self.rpar_integ = rpar_integ

    def get_xi(self, rout, pars):

        xi = self.xi
        v12 = self.v12
        s12 = self.s12

        xi_s = xi[0] + pars['f1']*xi[1] + pars['f2']*xi[2] \
               + pars['f1']**2*xi[3] + pars['f1']*pars['f2']*xi[4] \
               + pars['f2']**2*xi[5]
    
        v12_s = pars['f']*( v12[0] + pars['f1']*v12[1] + pars['f2']*v12[2] \
                    + (pars['f1']**2)*v12[3] + pars['f1']*pars['f2']*v12[4] ) / (1+xi_s) # + pars['f2']**2*v12[5] ?
        s12_para  = pars['f']**2*( s12[0] + pars['f1']*s12[1] + pars['f2']*s12[2] + (pars['f1']**2)*s12[3] ) / (1+xi_s)
        s12_perp = pars['f']**2*0.5*( s12[4] + pars['f1']*s12[5] + pars['f2']*s12[6] + (pars['f1']**2)*s12[7] ) / (1+xi_s)
  
        self.xi_s = xi_s
        self.v12_s = v12_s
        self.s12_para = s12_para
        self.s12_perp = s12_perp

        ####################################################
        # Gaussian Streaming Model
        ####################################################
        r_mod = rout
        nbins_r = len(r_mod)
        mu_mod = self.mu_mod
        rpar_integ = self.rpar_integ

        if 'aiso' in pars:
            at = pars['aiso']/(1+pars['epsilon'])
            ap = pars['aiso']*(1+pars['epsilon'])**2
        else:
            at = pars['at']
            ap = pars['ap']

        s_perp = at*np.sqrt(1-mu_mod**2)[:, None]*r_mod
        s_para = ap*mu_mod[:, None]*r_mod
        r_integ = np.sqrt(s_perp[:, :, None]**2+rpar_integ**2)
        mu_integ = rpar_integ/r_integ
        xi_integ = np.interp(r_integ, self.r, xi_s)
        v12_integ = np.interp(r_integ, self.r, v12_s)
        s12_para_integ = np.interp(r_integ, self.r, s12_para)
        s12_perp_integ = np.interp(r_integ, self.r, s12_perp)
        v12_para_integ = v12_integ*mu_integ
        sigma2_clpt =    mu_integ**2 * s12_para_integ + \
                        (1-mu_integ**2)* s12_perp_integ
        sigma2_tot = sigma2_clpt + pars['sigma_fog']**2
        func = np.exp( -0.5*(s_para[:, :, None]-rpar_integ-v12_para_integ)**2/sigma2_tot)
        func *= 1/np.sqrt(2*np.pi*sigma2_tot)
        func *= (1+xi_integ)
        xi2d = -1 + np.trapz(func, x=rpar_integ, axis=2)
        
        #xi2d/=at**2*ap
        #self.v12_para_integ = v12_para_integ
        #self.sigma2_tot = sigma2_tot
        #self.func = func
        return xi2d

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

    def get_xi_multipoles(self, rout, pars, options):
        """ Compute \\xi_\\ell(r) from a set of parameters
            This is the function used in the class Chi2

        Input
        -----
        rout (np.array): contains the separation values in Mpc/h
        pars (dict): contains the parameters required for xi_s(r, \\mu) 
        options (dict): contains options for the calculation including
                        ell_max
        Output
        -----
        xi_mult (np.array): array with shape (n_ell, rout.size) with the \\xi_\\ell(r)
        """
        xi2d = self.get_xi(rout, pars)
        xi_mult = self.get_multipoles(self.mu_mod, xi2d, ell_max=options['ell_max'])
        return xi_mult
