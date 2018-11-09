from __future__ import print_function
import camb
import numpy as N
import pylab as P
import fftlog
import iminuit
import iminuit.frontends
import  scipy.interpolate 
from scipy.ndimage import gaussian_filter1d

class Cosmo:

    def __init__(self, z=0.0, name='challenge', norm_pk=0, non_linear=0):
        self.get_matter_power_spectrum(z=z, name=name, norm_pk=norm_pk, 
                                       non_linear=non_linear)
        self.get_correlation_function(update=1)
        self.get_sideband()
        self.get_sideband_power()
        self.set_2d_arrays()

    def get_matter_power_spectrum(self, pars=None, z=0.0, non_linear=0, \
                                        name='challenge', norm_pk=0):

        if pars is None:
            pars = camb.CAMBparams()
            if name == 'challenge':
                pars.set_cosmology(H0=67.6, ombh2=0.0220,  \
                                    omch2=0.11902256, \
                                    YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.06)
                #pars.InitPower.set_params(As=2.039946656721871e-09, ns=0.97)
                pars.InitPower.set_params(As=2.0406217009089533e-09, ns=0.97)
            elif name == 'qpm':
                pars.set_cosmology(H0=70., ombh2=0.022470,  \
                                    omch2=0.11963, \
                                    YHe=0.24,nnu=3.04,  mnu=0, \
                                    TCMB=2.7250, num_massive_neutrinos=0)
                pars.InitPower.set_params(As=2.3e-09, ns=0.97)
            elif name == 'planck':
                pars.set_cosmology(H0=67.31, ombh2=0.02222, \
                                   omch2=0.1197, \
                                   YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.06)
                pars.InitPower.set_params(As=2.198e-09, ns=0.9655)
            elif name == 'outerim':
                pars.set_cosmology(H0=71., ombh2=0.022584, \
                                   omch2=0.10848, \
                                   YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.0,
                                    num_massive_neutrinos=0)
                pars.InitPower.set_params(As=2.224615e-09, ns=0.963)
                

        pars.set_dark_energy()
        
        #-- compute power spectrum
        pars.set_matter_power(redshifts=[z], kmax=100.0, k_per_logint=None)

        #-- set non-linear power spectrum
        if non_linear:
            pars.NonLinear = camb.model.NonLinear_both
        else:
            pars.NonLinear = camb.model.NonLinear_none

        results = camb.get_results(pars)
        kh, z, pk = results.get_matter_power_spectrum(\
                        minkh=1.05e-5, maxkh=100., npoints = 2048)

        
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
        self.k = kh
        self.pk = pk[0]
        self.sigma8 = sigma8[0]
        self.H_z = results.hubble_parameter(z[0])
        self.D_A = results.angular_diameter_distance(z[0])
        self.r_drag = results.get_derived_params()['rdrag']

        return kh, pk[0]
        
    def get_correlation_function(self, k=None, pk=None,  \
                                Sigma_nl=0., r=None, r0=1., inverse=0, update=0):

        if k is None or pk is None:
            k = self.k
            pk = self.pk

        #-- transform to log space for Hankel Transform
        klog = 10**N.linspace( N.log10(k.min()), N.log10(k.max()), k.size)
        pklog = N.interp(klog, k, pk)

        #-- apply isotropic damping
        pklog *= N.exp(-0.5*klog**2*Sigma_nl**2)

        rout, xiout = fftlog.HankelTransform(klog, pklog, q=1.5, mu=0.5, \
                                             output_r_power=-3, output_r=r, r0=r0)
        norm = 1/(2*N.pi)**1.5
        if inverse:
            xiout /= norm
        else:
            xiout *= norm

        if update:
            self.r = rout
            self.xi = xiout

        return rout, xiout

    def get_sideband(self, r=None, xi=None, \
                     fit_range=[[50., 80.], [150., 190.]], poly_order=4):

        if r is None or xi is None:
            r = self.r
            xi = self.xi

        peak_range = [fit_range[0][1], fit_range[1][0]]

        w = ((r>fit_range[0][0])&(r<fit_range[0][1])) | \
            ((r>fit_range[1][0])&(r<fit_range[1][1]))
        x_fit = r[w]
        y_fit = xi[w]*r[w]**3

        coeff = N.polyfit(x_fit, y_fit, poly_order)
        
        xi_sideband = xi*1.
        w_peak = (r>peak_range[0])&(r<peak_range[1])
        xi_sideband[w_peak] = N.polyval(coeff, r[w_peak])/r[w_peak]**3

        self.xi_sideband = xi_sideband
        self.peak_range = peak_range

        return xi_sideband

    def get_sideband_power(self):

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

        P.figure(figsize=(6,4))
        P.plot(k, pk*k, 'k', lw=2)
        P.plot(k, pks*k, 'r--', lw=2)
        P.xscale('log')
        P.xlabel(r'$k \ [h \ \rm{Mpc}^{-1}]$')
        P.ylabel(r'$kP_{\rm lin}(k) \ [h^{-2}\mathrm{Mpc}^2]$')
        P.xlim(1e-3, 10)
        P.tight_layout()

        P.figure(figsize=(6,4))
        P.plot(r, xi*r**2, 'k', lw=2)
        P.plot(r, xis*r**2, 'r--', lw=2)
        P.xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')
        P.ylabel(r'$r^2 \xi_{\rm lin} \ [h^{-2} \mathrm{Mpc}^{2}]$')
        P.xlim(0, 200)
        P.tight_layout()

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
        '''
        xibar   = N.array([ N.sum(xi[:i]*r[:i]**2) for i in range(r.size)])\
                * 3./r**3 * N.gradient(r)
        #xibbar = N.array([ N.sum(xi[:i]*r[:i]**4) for i in range(r.size)])\
                #* 5./r**5 * N.gradient(r)
        xi0 = (1+2./3*f+1./5*f**2)*xi
        xi2 = (4./3*f + 4./7*f**2)*(xi-xibar)
        #xi4 = 8./35*f**2*(xi + 2.5*xibar - 3.5*xibarbar)
        return xi0, xi2#, xi4

    def set_2d_arrays(self):

        self.mu = N.linspace(0, 1., 201)
        self.mu2d = N.tile(self.mu[:, None], (1, self.k.size))
        self.k2d = N.tile(self.k, (self.mu.size, 1))

    def get_2d_power_spectrum(self, pars, ell_max=2, no_peak=0):

        if hasattr(self, 'pars') and pars==self.pars:
            return self.pk_mult

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
        bias = pars['bias']
        beta = pars['f']/pars['bias']
        
        Sigma_stream = pars['Sigma_s']
        Sigma_rec = pars['Sigma_rec']

        k = self.k
        mu = self.mu
        mu2d = self.mu2d 
        k2d = self.k2d 
    
        if 'aiso' in pars:
            ak2d = k2d/at
            amu = mu*1.
        else:
            #-- this is the correct formula (Beutler et al. 2013)
            F = ap/at
            ak2d = k2d/at * N.sqrt( 1 + mu2d**2 * (1/F**2 - 1) )
            amu   = mu/F  * N.sqrt( 1 + mu**2   * (1/F**2 - 1) )**(-1) 
        amu2d = N.tile( amu[:, None], (1, k.size)) 

        #-- anisotropic damping
        sigma_v2 = (1-amu**2)*Sigma_per**2/2+ amu**2*Sigma_par**2/2 

        #-- linear Kaiser redshift space distortions with reconstruction damping
        if Sigma_rec == 0:
            Kaiser = (1+beta*amu2d**2 )**2
        else:
            Kaiser = (1+beta*(1.-N.exp(-ak2d**2*Sigma_rec**2/2))*amu2d**2 )**2 

        #-- Fingers of God
        Dnl = 1./( 1 + ak2d**2*amu2d**2*Sigma_stream**2)

        apk2d_s = N.interp(ak2d, self.k, self.pk_sideband)*bias**2

        if no_peak:
            pk2d_out = apk2d_s
        else:
            apk2d = N.interp(ak2d, self.k, self.pk)*bias**2 
            pk2d_out = ( (apk2d - apk2d_s)*N.exp(-ak2d**2*sigma_v2[:, None]) + apk2d_s)

        pk2d_out *= Dnl**2 * Kaiser #[:, None]
        pk2d_out /= (at**2*ap)

        pk_mult = N.zeros((ell_max//2+1, k.size))
        dmu = N.gradient(mu)
        for ell in range(0, ell_max+2, 2):
            pk_mult[ell//2] = (2.*ell+1.) * N.sum( pk2d_out * \
                             self.Legendre(ell, mu)[:, None] * dmu[:, None], axis=0)  

        self.pars = pars.copy()
        self.pk_mult = pk_mult

        return pk_mult

    def Legendre(self, ell, mu):

        if ell == 0:
            return N.ones(mu.shape)
        elif ell == 2:
            return 0.5*(3*mu**2-1)
        elif ell == 4:
            return 0.125*(35*mu**4 - 30*mu**2 +3)
        else:
            return -1

    def get_multipoles_from_pk(self, k, pk_mult, r0=1., r=None):

        xi_mult = pk_mult*0
        ell_max = xi_mult.shape[0]*2

        for ell in range(0, ell_max, 2):
            rout, xiout = fftlog.HankelTransform(k, pk_mult[ell//2], q=1.5, mu=0.5+ell, \
                                                 output_r_power=-3, output_r=r, r0=r0)
            norm = 1/(2*N.pi)**1.5 * (-1)**(ell/2)
            xi_mult[ell//2] = xiout*norm 

        return rout, xi_mult

    def get_multipoles_2d(self, rout, pars, ell_max=2, no_peak=0):

        pk_multipoles = self.get_2d_power_spectrum(pars, ell_max=ell_max,\
                                                    no_peak=no_peak)
        r, cf_multipoles = self.get_multipoles_from_pk(self.k, pk_multipoles)
       
        nmult = cf_multipoles.shape[0]
        cf_out = N.zeros((nmult, rout.size))
        for i in range(nmult):
            cf_out[i] = N.interp(rout, r, cf_multipoles[i])
        return cf_out

    @staticmethod
    def test():

        cosmo = Cosmo()
        r = N.linspace(40, 180, 100)
        pars0 = {'ap':1.0, 'at': 1.0, 'bias':1.0, 'f':0.6, \
                'Sigma_par':10., 'Sigma_per':6., 'Sigma_s':4., 'Sigma_rec':0.}
        lss = ['-', '--', ':']

        P.figure(figsize=(6, 5))
        pars = pars0.copy()
        for i, ap in enumerate([0.95, 1.0, 1.05]):
            pars['at'] = ap
            pars['ap'] = ap
            aiso = ap
            xi_mult = cosmo.get_multipoles_2d(r, pars)
            for j in range(2):
                P.subplot(2, 1, j+1)
                P.plot(r, xi_mult[j]*r**2, ls=lss[i], color='k', lw=2, \
                       label=r'$\alpha_{\rm iso} = %.2f$'%aiso)
                if i==0:
                    P.ylabel(r'$r^2 \xi_{%d} \ [h^{-2} \mathrm{Mpc}^{2}]$'%(j*2))
        P.xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')
        P.legend(loc=0, fontsize=10)
        
        P.figure(figsize=(6, 5))
        pars = pars0.copy()
        for i, ap in enumerate([0.98, 1.0, 1.02]):
            pars['at'] = 1./N.sqrt(ap)
            pars['ap'] = ap
            epsilon = (ap*N.sqrt(ap))**(1./3)-1
            xi_mult = cosmo.get_multipoles_2d(r, pars)
            for j in range(2):
                P.subplot(2, 1, j+1)
                P.plot(r, xi_mult[j]*r**2, ls=lss[i], color='k', lw=2, \
                       label=r'$\epsilon = %.2f$'%epsilon)
                if i==0:
                    P.ylabel(r'$r^2 \xi_{%d} \ [h^{-2} \mathrm{Mpc}^{2}]$'%(j*2))
        P.xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')
        P.legend(loc=0, fontsize=10)


        P.figure(figsize=(6, 5))
        pars = pars0.copy()
        for i, sigma_rec in enumerate([0., 5.0, 10.]):
            pars['Sigma_rec'] = sigma_rec
            xi_mult = cosmo.get_multipoles_2d(r, pars)
            for j in range(2):
                P.subplot(2, 1, j+1)
                P.plot(r, xi_mult[j]*r**2, ls=lss[i], color='k', lw=2, \
                       label=r'$\Sigma_r = %.1f$'%sigma_rec)
                if i==0:
                    P.ylabel(r'$r^2 \xi_{%d} \ [h^{-2} \mathrm{Mpc}^{2}]$'%(j*2))
        P.xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')
        P.legend(loc=0, fontsize=10)
        

            
        P.figure(figsize=(6, 5))
        pars = pars0.copy()
        for i, s in enumerate([0., 4., 8.]):
            pars['Sigma_s'] = s
            xi_mult = cosmo.get_multipoles_2d(r, pars)
            for j in range(2):
                P.subplot(2, 1, j+1)
                P.plot(r, xi_mult[j]*r**2, ls=lss[i], color='k', lw=2, \
                        label=r'$\Sigma_s = %.1f$'%s)
                if i==0:
                    P.ylabel(r'$r^2 \xi_{%d} \ [h^{-2} \mathrm{Mpc}^{2}]$'%(j*2))
        P.xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')
        P.legend(loc=0, fontsize=10)
            
        P.figure(figsize=(6, 5))
        pars = pars0.copy()
        for i, sig in enumerate([[6, 10], [8, 8], [0., 0]]):
            pars['Sigma_per'] = sig[0]
            pars['Sigma_par'] = sig[1]
            xi_mult = cosmo.get_multipoles_2d(r, pars)
            for j in range(2):
                P.subplot(2, 1, j+1)
                P.plot(r, xi_mult[j]*r**2, ls=lss[i], color='k', lw=2, \
                        label=r'$\Sigma_\perp = %.0f,\Sigma_\parallel = %.0f$'%\
                        (sig[0], sig[1]))
                if i==0:
                    P.ylabel(r'$r^2 \xi_{%d} \ [h^{-2} \mathrm{Mpc}^{2}]$'%(j*2))
        P.xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')
        P.legend(loc=0, fontsize=10)

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

    def __init__(self, r, mono, coss, quad=None, rmin=40., rmax=180., \
                    nmocks=None):

        if quad is not None:
            r = N.append(r, r)
            cf = N.append(mono, quad)
            if cf.size != coss.shape[0]:
                print('Problem: covariance shape is not compatible '+
                      'with mono-quad', \
                      cf.size, coss.shape[0])
        else:
            cf = mono
            if coss.shape[0] == 2*r.size:
                print('covariance matrix contains quad, removing it')
                coss = coss[:r.size, :r.size]

        w = (r>rmin) & (r<rmax)
        r = r[w]
        if quad is not None:
            r = r[:r.size//2]

        cf = cf[w]
        coss = coss[:, w]
        coss = coss[w, :]
        
        self.r = r
        self.cf = cf
        self.coss = coss
        self.icoss = N.linalg.inv(coss)
        if nmocks:
            correction = (1 - (cf.size + 1.)/(nmocks-1))
            self.icoss *= correction

class Model:

    def __init__(self, fit_broadband=1, bb_min=-2, bb_max=0, name='challenge',
                 norm_pk=0, non_linear=0, z=0,
                 fit_iso=0, fit_multipoles=0, no_peak=0):

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
        #pars_names += ['bias', 'beta', 'Sigma_per', 'Sigma_par', 'Sigma_s', 'Sigma_rec']
        pars_names += ['bias', 'f', 'Sigma_s', 'Sigma_rec']
        pars['bias'] = 1.0
        #pars['beta'] = 0.4
        pars['f'] = 0.82
        pars['Sigma_s'] = 4.
        pars['Sigma_rec'] = 1000.
            
        if fit_broadband:
            for i, bb_power in enumerate(N.arange(bb_min, bb_max+1)):
                if fit_multipoles:
                    pars_names.append('bb_%d_mono'%i)
                    pars['bb_%d_mono'%i] = 0.
                    pars_names.append('bb_%d_quad'%i)
                    pars['bb_%d_quad'%i] = 0.
                else:
                    pars_names.append('bb_%d'%i)
                    pars['bb_%d'%i] = 0.

        self.bb_min = bb_min
        self.bb_max = bb_max
        self.pars = pars
        self.pars_names = pars_names
        self.fit_broadband = fit_broadband
        self.fit_multipoles = fit_multipoles
        self.no_peak = no_peak
        self.cosmo = cosmo
        
    def value(self, rout, pars):

        ell_max = 2*self.fit_multipoles
        cf_out = self.cosmo.get_multipoles_2d(rout,  pars, \
                            ell_max=ell_max, no_peak=self.no_peak)
        return cf_out.ravel()

    def get_broadband(self, rout, pars):
        
        if self.fit_multipoles:
            monobb = rout*0.
            quadbb = rout*0.
            for i in range(self.bb_max-self.bb_min+1):
                power = self.bb_min + i
                monobb += pars['bb_%d_mono'%i]*(rout**power)
                quadbb += pars['bb_%d_quad'%i]*(rout**power)
            return N.append(monobb, quadbb)
        else:
            bb = rout*0.
            for i in range(self.bb_max-self.bb_min+1):
                power = self.bb_min + i
                bb += pars['bb_%d'%i]*(rout**power)
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

        if '-nopeak' in fin:
            no_peak = 1
        else:
            no_peak=0

        if 'aiso' in best_pars.keys():
            fit_multipoles=0
            fit_iso=1
        else:
            fit_multipoles=1
            fit_iso=0

        if 'bb_0' in best_pars.keys() or 'bb_0_mono' in best_pars.keys():
            fit_broadband=1
        else:
            fit_broadband=0


        self.model=Model(fit_broadband=fit_broadband, fit_iso=fit_iso,\
                    fit_multipoles=fit_multipoles, no_peak=no_peak, norm_pk=1,
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

        model = self.model.value(r, pars)
        if self.model.fit_broadband:
            model += self.model.get_broadband(r, pars)
        return model

    def __call__(self, *p):
        pars = {}
        for i, name in enumerate(self.model.pars_names):
            pars[name] = p[i]
        #    else:
        #  if name.startswith('bb'): 
        #       parsbb[name] = p[i]

        model = self.get_model(self.data.r, pars)
        residual = self.data.cf - model
        inv_cov = self.data.icoss

        chi2 = N.dot(residual, N.dot(inv_cov, residual))

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

        mig = iminuit.Minuit(self, throw_nan=False, \
                             forced_parameters=self.model.pars_names, \
                             print_level=1, errordef=1, \
                             frontend=iminuit.frontends.ConsoleFrontend(), \
                             **init_pars)
        mig.tol = 10.0 
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
        print('chi2 = %.2f   ndata = %d   npars = %d   rchi2 = %.4f'%\
                (self.chi2min, self.ndata, self.npars, self.rchi2min))

    def plot_bestfit(self, model_only=0, scale_r=2, label=None):

        data = self.data
        model = self.model
        r = data.r
        cf = data.cf
        dcf = N.sqrt(N.diag(data.coss))
        cf_model = self.get_model(r, self.best_pars)

        if model.fit_multipoles:
            mono = cf[:r.size]
            dmono = dcf[:r.size]
            mono_model = cf_model[:r.size]
            quad = cf[r.size:]
            dquad = dcf[r.size:]
            quad_model = cf_model[r.size:]
            P.subplot(211)
            if not model_only:
                P.errorbar(r, mono*r**scale_r, dmono*r**scale_r, fmt='o')
            P.plot(r, mono_model*r**scale_r, label=label)
            P.ylabel(r'$r^{%d} \xi_0$ [$h^{%d}$ Mpc$^{%d}]$'%\
                     (scale_r, -scale_r, scale_r))
            P.subplot(212)
            if not model_only:
                P.errorbar(r, quad*r**scale_r, dquad*r**scale_r, fmt='o')
            P.plot(r, quad_model*r**scale_r)
            P.ylabel(r'$r^{%d} \xi_2$ [$h^{%d}$ Mpc$^{%d}]$'%\
                    (scale_r, -scale_r, scale_r))
        else: 
            if not model_only:
                P.errorbar(r, cf*r**scale_r, dcf*r**scale_r, fmt='o')
            P.plot(r, cf_model*r**scale_r, label=label)
            P.ylabel(r'$r^{%d} \xi_0$ [$h^{%d}$ Mpc$^{%d}]$'%\
                     (scale_r, -scale_r, scale_r))
        P.xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')

    def scan(self, par_name='alpha', par_min=0.8, par_max=1.2, par_nsteps=400):

        init_pars = {}
        for par in self.model.pars.items():
            name = par[0]
            value = par[1]
            init_pars[name] = value
            init_pars['error_'+name] = abs(value)/10. if value!=0 else 0.1

        init_pars['fix_'+par_name] = True
        par_grid = N.linspace(par_min, par_max, par_nsteps)
        chi2_grid = N.zeros(par_nsteps)
       
        if self.fixes:
            for key in self.fixes:
                init_pars[key] = self.fixes[key]
                init_pars['fix_'+key] = True 

        for i in range(par_nsteps):
            value = par_grid[i]
            init_pars[par_name] = value

            mig = iminuit.Minuit(self, forced_parameters=self.model.pars_names, \
                                 print_level=1, errordef=1, \
                                 frontend=iminuit.frontends.ConsoleFrontend(), \
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
        par_grid0 = N.linspace(par_min[0], par_max[0], par_nsteps[0])
        init_pars['fix_'+par_names[1]] = True
        par_grid1 = N.linspace(par_min[1], par_max[1], par_nsteps[1])

        chi2_grid = N.zeros(par_nsteps)
       
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
                         frontend=iminuit.frontends.ConsoleFrontend(), \
                         **init_pars)
                mig.migrad()
                print( 'scanning: %s = %.5f   %s = %.5f    chi2 = %.4f'%\
                        (par_names[0], value0, par_names[1], value1, mig.fval))
                chi2_grid[i, j] = mig.fval

        return par_grid0, par_grid1, chi2_grid

    def read_scan1d(self, fin):

        sfin = fin.split('.')
        par_name = sfin[-2]
        x, chi2 = N.loadtxt(fin, unpack=1)
        bestx = x[0]
        chi2min = chi2[0]
        x = N.unique(x[1:])
        chi2scan = chi2[1:]
        self.chi2scan = chi2scan
        self.x=x
        self.par_name=par_name
        self.bestx=bestx
        self.chi2min=chi2min

    def plot_scan1d(self, ls=None, \
                    color=None,  alpha=None, label=None):

        P.plot(self.x, self.chi2scan-self.chi2min, ls=ls, \
                color=color, alpha=alpha, label=label)

    def read_scan2d(self, fin):

        sfin = fin.split('.')
        par_name0 = sfin[-3]
        par_name1 = sfin[-2]

        x, y, chi2 = N.loadtxt(fin, unpack=1)
        bestx = x[0]
        besty = y[0]
        chi2min = chi2[0]

        x = N.unique(x[1:])
        y = N.unique(y[1:])
        chi2scan2d = N.reshape(chi2[1:], (x.size, y.size)).transpose()
        
        self.chi2scan2d = chi2scan2d
        self.x=x
        self.y=y
        self.par_name0=par_name0
        self.par_name1=par_name1
        self.bestx=bestx
        self.besty=besty
        self.chi2min=chi2min

    def plot_scan2d(self, levels=[2.3, 6.18, 11.83], ls=['-', '--', ':'], \
                    color='b',  alpha=1.0, label=None, scale_dist=0):


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
            P.contour(x, y, self.chi2scan2d-self.chi2min, \
                      levels=[levels[i]], \
                      linestyles=[ls[i]], colors=color, alpha=alpha,\
                      label=label)

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
       
        pars_names = N.sort([p for p in self.best_pars])
        for p in pars_names:
            print(p, self.best_pars[p], self.errors[p], file=fout)
        fout.close()




