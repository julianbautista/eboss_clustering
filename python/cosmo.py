import numpy as np

#-- cosmology
class CosmoSimple:

    def __init__(self, omega_m=0.31, h=0.676):
        print(f'Initializing cosmology with omega_m = {omega_m:.2f}')
        c = 299792.458 #m/s
        omega_lambda = 1 - omega_m
        ztab = np.linspace(0., 5., 10000)
        E_z = np.sqrt(omega_lambda + omega_m*(1+ztab)**3)
        rtab = np.zeros(ztab.size)
        for i in range(1, ztab.size):
            rtab[i] = rtab[i-1] + c * (1/E_z[i-1]+1/E_z[i])/2. * (ztab[i]-ztab[i-1]) / 100.

        self.h = h
        self.c = c
        self.omega_m = omega_m
        self.omega_lambda = omega_lambda
        self.ztab = ztab
        self.rtab = rtab 
        self.E_z = E_z

    def get_comoving_distance(self, z):
        '''Comoving distance in Mpc/h '''
        return np.interp(z, self.ztab, self.rtab)

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
        


