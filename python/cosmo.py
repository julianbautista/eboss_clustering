import numpy as np
import camb
import sys

#-- cosmology
class CosmoSimple:

    def __init__(self, omega_m=0.31, flat=True, omega_de=0, h=0.676, 
                mass_neutrinos=0.06, n_eff=3.046, zmax=5., nz=10000):

        c = 299792.458 #km/s
        #-- For T_CMB = 2.7255 Kelvin
        omega_photon = 2.4728018939788232e-05 / h**2
        omega_nu = mass_neutrinos/93.0033/h**2
        n_massless_neutrinos = n_eff - n_eff/3*(mass_neutrinos>0)
        omega_r = omega_photon*(1 + 7/8*n_massless_neutrinos*(4/11)**(4/3))
        if flat:
            omega_de = 1 - omega_m - omega_r
            omega_k = 0
        else:
            omega_k = 1 - omega_de - omega_m - omega_r

        z_tab = np.linspace(0., zmax, nz)
        #-- Dimensionless Hubble expansion rate
        E_z = np.sqrt(omega_de + omega_m*(1+z_tab)**3 + omega_r*(1+z_tab)**4 + omega_k*(1+z_tab)**2)
        
        #-- Comoving distance table
        r_tab = np.zeros(z_tab.size) 
        for i in range(1, z_tab.size):
            r_tab[i] = np.trapz(c/h/100/E_z[:i+1], x=z_tab[:i+1])

        pars = {}
        pars['h'] = h
        pars['c'] = c
        pars['omega_k'] = omega_k
        pars['omega_nu'] = omega_nu
        pars['omega_ph'] = omega_photon
        pars['omega_m'] = omega_m
        pars['omega_de'] = omega_de
        pars['mass_neutrinos'] = mass_neutrinos
        pars['N_eff'] = n_eff
        self.flat = flat
        self.pars = pars
        self.z_tab = z_tab
        self.r_tab = r_tab 
        self.E_z = E_z

    def get_hubble(self, z):
        '''Hubble rate in km/s/Mpc'''
        H0 = self.pars['h']*100
        return np.interp(z, self.z_tab, self.E_z)*H0

    def get_comoving_distance(self, z):
        '''Comoving distance in Mpc '''

        return np.interp(z, self.z_tab, self.r_tab)

    def get_transverse_comoving_distance(self, z):
        '''Transverse comoving distance, including non-flat cosmologies''' 

        r = self.get_comoving_distance(z)
        if self.flat:
            return r
        else:
            dh = self.pars['c']/self.pars['h']/100
            omega_k = self.pars['omega_k']
            return (dh/np.sqrt(omega_k+0j)*np.sinh( np.sqrt(omega_k+0j)*r/dh)).real

    def get_DM(self, z):
        '''Comoving angular diameter distance in Mpc '''

        D_M = self.get_transverse_comoving_distance(z)
        return D_M

    def get_DA(self, z):
        '''Angular diameter distance in Mpc'''

        D_A = self.get_transverse_comoving_distance(z)/(1+z)
        return D_A

    def get_DH(self, z):
        '''Hubble distance c/H(z) in Mpc '''

        D_H = self.pars['c']/self.get_hubble(z)
        return D_H

    def get_DV(self, z):
        ''' Get spherically averaged distance D_V(z) = (c*z*D_M**2/H)**(1/3) in Mpc'''

        D_M = self.get_DM(z)
        D_H = self.get_DH(z)
        D_V = (z*D_M**2*D_H)**(1/3)
        return D_V

    def get_redshift(self, r):
        '''Get redshift from comoving distance in Mpc/h units'''

        return np.interp(r, self.r_tab, self.z_tab)

    def shell_vol(self, zmin, zmax):
        '''Comoving spherical volume between zmin and zman in (Mpc)**3'''

        rmin = self.get_comoving_distance(zmin)
        rmax = self.get_comoving_distance(zmax)
        return 4*np.pi/3.*(rmax**3-rmin**3)

    def get_box_size(self, ra, dec, zmin=0.5, zmax=1.0):
        ''' Provide minimum box dimensions in comoving coordinates 
            from a list of RA and DEC 
        '''

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
        ''' Computes approximated growth-rate of structures
            using f(z) = Omega_m(z)**0.55
        '''

        e_z = np.interp(z, self.z_tab, self.E_z)
        return (self.omega_m*(1+z)**3/e_z**2)**0.55
        
class Cosmo:

    def __init__(self, camb_pars=None,
                 z=0.0, name='challenge',  
                 non_linear=False, 
                 nk=2048, kmax=100., nmu=101):

        self.get_matter_power_spectrum(camb_pars=camb_pars, 
                                       z=z, name=name, 
                                       non_linear=non_linear, kmax=kmax, nk=nk)

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
            elif name == 'om0.22':
                camb_pars.set_cosmology(H0=67.6, ombh2=0.0220,  
                                    omch2=0.07788958160106182, 
                                    YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.06)
                camb_pars.InitPower.set_params(As=2.0406217009089533e-09, ns=0.97)
            elif name == 'om0.26':
                camb_pars.set_cosmology(H0=67.6, ombh2=0.0220,  
                                    omch2=0.09616862160106183, 
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
                camb_pars.InitPower.set_params(As=2.1601810717683e-09, ns=0.963)
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
            elif name == 'xu2013':
                #-- Om=0.25, h=0.7, ns=1.0, Ob=0.04, s8=0.8
                camb_pars.set_cosmology(H0=70, ombh2=0.0196, 
                                   omch2=0.1029, 
                                   YHe=0.24, TCMB=2.7255, nnu=3.046, mnu=0.0,
                                    num_massive_neutrinos=0)
                camb_pars.InitPower.set_params(As=2.2135045586846496e-09, ns=1.)
            else: 
                print('Error: name of cosmology should be one of the following')
                print('challenge qpm planck outerrim ezmock')
                sys.exit(0)

        camb_pars.set_dark_energy()
        #camb_pars.set_accuracy(AccuracyBoost=3, lAccuracyBoost=3)

        #-- compute power spectrum
        camb_pars.set_matter_power(redshifts=[z], kmax=2*kmax, k_per_logint=None,
                                    accurate_massive_neutrino_transfers=True)

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
        pars['mass_neutrinos'] = camb_pars.omeganu*93.0033*(camb_pars.H0/100)**2 
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
