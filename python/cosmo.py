import numpy as np
import camb
import sys

#-- cosmology
class CosmoSimple:

    def __init__(self, omega_m=0.31, h=0.676):
        print(f'Initializing cosmology with omega_m = {omega_m:.2f}')
        c = 299792.458 #km/s
        omega_lambda = 1 - omega_m
        ztab = np.linspace(0., 5., 10000)
        E_z = np.sqrt(omega_lambda + omega_m*(1+ztab)**3)
        rtab = np.zeros(ztab.size)
        dz = ztab[1]-ztab[0]
        for i in range(1, ztab.size):
            rtab[i] = rtab[i-1] + c * (1/E_z[i-1]+1/E_z[i])/2. * dz / 100.

        self.h = h
        self.c = c
        self.omega_m = omega_m
        self.omega_lambda = omega_lambda
        self.ztab = ztab
        self.rtab = rtab 
        self.E_z = E_z

    def get_hubble(self, z):
        '''Hubble rate in km/s/Mpc'''
        return np.interp(z, self.ztab, self.E_z*self.h*100.)

    def get_comoving_distance(self, z):
        '''Comoving distance in Mpc/h '''
        return np.interp(z, self.ztab, self.rtab)

    def get_DV(self, z):
        ''' Get spherically averaged distance D_V(z) = (c*z*D_M**2/H)**(1/3) in Mpc'''
        #-- This equation below is only valid in flat space
        #-- Dividing by h to get DV in Mpc to match H which is in km/s/Mpc
        D_M = self.get_comoving_distance(z)/self.h
        H = self.get_hubble(z)
        D_V = (self.c*z*D_M**2/H)**(1/3)
        return D_V

    def get_redshift(self, r):
        '''Get redshift from comoving distance in Mpc/h units'''
        return np.interp(r, self.rtab, self.ztab)

    def shell_vol(self, zmin, zmax):
        '''Comoving spherical volume between zmin and zman in (Mpc/h)**3'''
        rmin = self.get_comoving_distance(zmin)
        rmax = self.get_comoving_distance(zmax)
        return 4*np.pi/3.*(rmax**3-rmin**3)

    def get_box_size(self, ra, dec, zmin=0.5, zmax=1.0):

        dmin = self.get_comoving_distance(zmin)
        dmax = self.get_comoving_distance(zmax)

        theta = np.radians(-dec+90)
        phi = np.radians(ra)

        xmin = dmin * np.sin(theta)*np.cos(phi)
        ymin = dmin * np.sin(theta)*np.sin(phi)
        zmin = dmin * np.cos(theta)
        xmax = dmax * np.sin(theta)*np.cos(phi)
        ymax = dmax * np.sin(theta)*np.sin(phi)
        zmax = dmax * np.cos(theta)

        for pair in [[xmin, xmax], [ymin, ymax], [zmin, zmax]]:
            print( np.abs(np.array(pair).min() - np.array(pair).max()))

    def get_growth_rate(self, z):

        e_z = np.interp(z, self.ztab, self.E_z)
        return (self.omega_m*(1+z)**3/e_z**2)**0.55
        
class Cosmo:

    def __init__(self, camb_pars=None,
                 z=0.0, name='challenge',  
                 non_linear=False, 
                 nk=2048, kmax=100., nmu=101):

        self.get_matter_power_spectrum(camb_pars=camb_pars, 
                                       z=z, name=name, 
                                       non_linear=non_linear, kmax=kmax, nk=nk)

        #self.r, self.xi = self.get_correlation_function()
        #self.get_sideband_scipy()
        #self.get_sideband_power()
        #self.set_2d_arrays(nmu=nmu)

    def get_matter_power_spectrum(self, camb_pars=None, z=0.0, non_linear=0, 
                                        name='challenge', norm_pk=0, 
                                        kmax=100., nk=4098):

        #-- to set sigma8 value, scale As by the square of ratio of sigma8
        if camb_pars is None:
            camb_pars = camb.CAMBparams()
            if name == 'challenge':
                camb_pars.set_cosmology(H0=67.6, ombh2=0.0220,  
                                    omch2=0.11901745, 
                                    YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.06)
                camb_pars.InitPower.set_params(As=2.0406217009089533e-09, ns=0.97)
            elif name == 'cosmo1':
                camb_pars.set_cosmology(H0=67.6, ombh2=0.0220,  
                                    omch2=0.10073838, 
                                    YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.06)
                camb_pars.InitPower.set_params(As=2.0406217009089533e-09, ns=0.97)
            elif name == 'cosmo2':
                camb_pars.set_cosmology(H0=67.6, ombh2=0.0220,  
                                    omch2=0.13729646, 
                                    YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.06)
                camb_pars.InitPower.set_params(As=2.0406217009089533e-09, ns=0.97)
            elif name == 'challenge_omegab1':
                camb_pars.set_cosmology(H0=67.6, ombh2=0.0240,
                                   omch2=0.11701745,
                                   YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.06)
            elif name == 'challenge_omegab2':
                camb_pars.set_cosmology(H0=67.6, ombh2=0.0200,
                                   omch2=0.12101645,
                                   YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.06)
                camb_pars.InitPower.set_params(As=2.0406217009089533e-09, ns=0.97)
            elif name == 'qpm':
                camb_pars.set_cosmology(H0=70., ombh2=0.022470,  
                                    omch2=0.11963, 
                                    YHe=0.24,nnu=3.04,  mnu=0, 
                                    TCMB=2.7250, num_massive_neutrinos=0)
                camb_pars.InitPower.set_params(As=2.3e-09, ns=0.97)
            elif name == 'planck':
                camb_pars.set_cosmology(H0=67.31, ombh2=0.02222, 
                                   omch2=0.1197, 
                                   YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.06)
                camb_pars.InitPower.set_params(As=2.198e-09, ns=0.9655)
            elif name == 'planck_open':
                camb_pars.set_cosmology(H0=63.6, ombh2=0.02249, omk=-0.044, 
                                   omch2=0.1185, 
                                   YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.06)
                camb_pars.InitPower.set_params(As=2.0697e-09, ns=0.9688)
            elif name == 'planck2018':
                camb_pars.set_cosmology(H0=67.66, ombh2=0.02242,
                                   omch2=0.119352571517,
                                   YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.06)
                camb_pars.InitPower.set_params(As=2.105214963579407e-09, ns=0.9665)
            elif name == 'outerrim':
                camb_pars.set_cosmology(H0=71., ombh2=0.02258, 
                                   #omch2=0.10848, 
                                   omch2=0.1109, 
                                   YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.0,
                                    num_massive_neutrinos=0)
                camb_pars.InitPower.set_params(As=2.1604128e-09, ns=0.963)
            elif name == 'ezmock':
                camb_pars.set_cosmology(H0=67.77, ombh2=0.0221399210, 
                                   omch2=0.1189110239, 
                                   YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.0,
                                    num_massive_neutrinos=0)
                camb_pars.InitPower.set_params(As=2.11622e-09, ns=0.9611)
            elif name == 'nseries':
                #-- Om=0.286, h=0.7, ns=0.96, Ob=0.047, s8=0.820
                camb_pars.set_cosmology(H0=70, ombh2=0.02303, 
                                   omch2=0.11711, 
                                   YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.0,
                                    num_massive_neutrinos=0)
                camb_pars.InitPower.set_params(As=2.14681e-09, ns=0.96)
            else: 
                print('Error: name of cosmology should be one of the following')
                print('challenge qpm planck outerrim ezmock')
                sys.exit(0)

        camb_pars.set_dark_energy()
        
        #-- compute power spectrum
        camb_pars.set_matter_power(redshifts=[z], kmax=2*kmax, k_per_logint=None)

        #-- set non-linear power spectrum
        if non_linear:
            camb_pars.NonLinear = camb.model.NonLinear_both
        else:
            camb_pars.NonLinear = camb.model.NonLinear_none

        results = camb.get_results(camb_pars)
        kh, z, pk = results.get_matter_power_spectrum(\
                        minkh=1.05e-5, maxkh=kmax, npoints = nk)

        pars = {}

        pars['z'] = z[0]
        pars['name'] = name
        pars['H0'] = camb_pars.H0
        pars['Omega_k'] = camb_pars.omk
        pars['Omega_b'] = camb_pars.omegab
        pars['Omega_c'] = camb_pars.omegac
        pars['Omega_m'] = camb_pars.omegam
        pars['Omega_de'] = results.omega_de
        pars['Omega_nu'] = camb_pars.omeganu  
        pars['N_eff'] = camb_pars.N_eff
        pars['mass_neutrinos'] = 0.06*(pars['Omega_nu'] > 0)
        pars['n_s'] = camb_pars.InitPower.ns
        pars['A_s'] = camb_pars.InitPower.As
        pars['H_z'] = results.hubble_parameter(z[0])
        pars['D_A'] = results.angular_diameter_distance(z[0])
        pars['D_M'] = pars['D_A']*(1+pars['z'])
        pars['D_H'] = 299792.458/pars['H_z']
        pars['D_V'] = (pars['z']*(pars['D_M'])**2*pars['D_H'])**(1/3) 
        pars['r_drag'] = results.get_derived_params()['rdrag']
        pars['DM_rd'] = pars['D_M']/pars['r_drag']
        pars['DH_rd'] = pars['D_H']/pars['r_drag']
        pars['DV_rd'] = pars['D_V']/pars['r_drag']
        pars['sigma8'] = results.get_sigma8()[0]
        pars['fsigma_8'] = results.get_fsigma8()[0]
        pars['f'] = pars['fsigma_8']/pars['sigma8']

        self.camb_pars = camb_pars
        self.camb_results = results
        self.name=name
        self.k = kh
        self.pk = pk[0]
        self.pars = pars
        
        #return kh, pk[0]

    def print_distances(self, fout=sys.stdout):

        print('z =', self.z)
        print('r_drag = ', self.r_drag) 
        print('H(z)   = ', self.H_z )
        print('D_H(z)/r_d = ', self.DH_rd)
        print('D_M(z)/r_d = ', self.DM_rd)
        print('D_V(z)/r_d = ', self.DV_rd)
        print(f'sigma_8(z) = ', self.sigma8)
        print(f'f(z) = ', self.f)
        print(f'fsig8(z) = ', self.fsigma8)

    
    def get_dist_rdrag(self):
        
        c = 299792.458
        self.DH_rd = c/self.H_z/self.r_drag
        self.DM_rd = self.D_M/self.r_drag
        self.DV_rd = (c*self.z*(self.D_A*(1+self.z))**2/self.H_z)**(1./3)/self.r_drag

    @staticmethod
    def get_alphas(cosmo, cosmo_fid):

        alphas = {}
        alphas['at'] = (cosmo.D_M/cosmo.r_drag)/(cosmo_fid.D_M/cosmo_fid.r_drag)
        alphas['ap'] = (cosmo.D_H/cosmo.r_drag)/(cosmo_fid.D_H/cosmo_fid.r_drag)
        #-- Padmanabhan & White 2009
        alphas['aiso'] = alphas['at']**(2./3.)*alphas['ap']**(1./3)
        alphas['epsilon'] = (alphas['ap']/alphas['at'])**(1./3) - 1
        return alphas 

    def export_linear_power_spectrum(self, fout):

        fout = open(fout, 'w')
        print('# Cosmological parameters', file=fout)
        for p in self.pars:
            print(f'# {p} {self.pars[p]}', file=fout)
        print('# ', file=fout)
        print('# k[h/Mpc]  P_lin(k)', file=fout)
        for i in range(self.k.size):
            print(f'{self.k[i]}    {self.pk[i]}', file=fout)
        fout.close()