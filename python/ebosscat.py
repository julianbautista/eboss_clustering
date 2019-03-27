from __future__ import print_function
import os
import sys
import numpy as N
import pylab as P
#import mangle
#import graphmask
import copy
import healpy as hp
from subprocess import call
#from systematics import MultiFit

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u

#-- cosmology
class Cosmo:

    def __init__(self, OmegaM=0.31, h=0.676):
        print('Initializing cosmology with OmegaM = %.2f'%OmegaM)
        c = 299792458. #m/s
        OmegaL = 1.-OmegaM
        ztab = N.linspace(0., 4., 10000)
        E_z = N.sqrt(OmegaL + OmegaM*(1+ztab)**3)
        rtab = N.zeros(ztab.size)
        for i in range(1, ztab.size):
            rtab[i] = rtab[i-1] + c*1e-3 * (1/E_z[i-1]+1/E_z[i])/2. * (ztab[i]-ztab[i-1]) / 100.

        self.h = h
        self.c = c
        self.OmegaM = OmegaM
        self.OmegaL = OmegaL
        self.ztab = ztab
        self.rtab = rtab 

    #-- comoving distance in Mpc/h
    def get_comoving_distance(self, z):
        return N.interp(z, self.ztab, self.rtab)

    def get_redshift(self, r):
        return N.interp(r, self.rtab, self.ztab)


    #-- comoving spherical volume between zmin and zman in (Mpc/h)**3
    def shell_vol(self, zmin, zmax):
        rmin = self.get_comoving_distance(zmin)
        rmax = self.get_comoving_distance(zmax)
        return 4*N.pi/3.*(rmax**3-rmin**3)

    def get_box_size(self, ra, dec, zmin=0.5, zmax=1.0):

        dmin = get_comoving_distance(zmin)
        dmax = get_comoving_distance(zmax)

        theta = (-dec+90)*N.pi/180.
        phi = (ra)*N.pi/180

        xmin = dmin * N.sin(theta)*N.cos(phi)
        ymin = dmin * N.sin(theta)*N.sin(phi)
        zmin = dmin * N.cos(theta)
        xmax = dmax * N.sin(theta)*N.cos(phi)
        ymax = dmax * N.sin(theta)*N.sin(phi)
        zmax = dmax * N.cos(theta)

        for pair in [[xmin, xmax], [ymin, ymax], [zmin, zmax]]:
            print( abs(N.array(pair).min() - N.array(pair).max()))

class Nbar:
    
    def __init__(self, *args):
        if len(args)>0:
            fin = args[0]
            f = open(fin)
            f.readline()
            mask_area_eff, veff_tot = N.array(f.readline().split()).astype(float)

            zcen, zlow, zhigh, nbar, wfkp, shell_vol, wgals = \
                N.loadtxt(fin, skiprows=3, unpack=1)

            self.mask_area_eff = mask_area_eff
            self.veff_tot = veff_tot
            self.zcen = zcen
            self.zlow = zlow
            self.zhigh= zhigh
            self.nbar = nbar
            self.wfkp = wfkp
            self.shell_vol = shell_vol
            self.wgals = wgals
            self.nbins = zcen.size
    
    def plot(self, label=None, color=None, alpha=1.0):
        P.plot(self.zcen, self.nbar/1e-4, label=label, color=color, alpha=alpha)
        P.ylabel(r'$\bar{n}(z)$  $[10^{-4} h^3 \mathrm{Mpc}^{-3}]$', fontsize=16)
        P.xlabel(r'$z$', fontsize=16)
        P.tight_layout()

    def compute_effective_volume(self, P0=10000., cosmo=None, \
                                 zmin=0.6, zmax=1.0):
        if cosmo is None:
            cosmo = Cosmo()

        mask_vol = self.mask_area_eff * cosmo.shell_vol(self.zlow, self.zhigh)\
                   / (4*N.pi*(180./N.pi)**2)
        veff = ((self.nbar*P0)/(1+self.nbar*P0))**2*mask_vol
        w = (self.zcen>=zmin)&(self.zcen<=zmax)
        veff_tot = sum(veff[w])
        return veff_tot 

    def export(self, fout):
        fout = open(fout, 'w')
        print('# effective area (deg^2), effective volume (Mpc/h)^3:', \
              file=fout)
        print(fout, self.mask_area_eff, self.veff_tot, \
              file=fout)
        print('# zcen, zlow, zhigh, nbar, wfkp, shell_vol,'+\
              ' total weighted gals', \
              file=fout)
        for i in range(self.nbins):
            print(self.zcen[i], self.zlow[i], self.zhigh[i], \
                  self.nbar[i], self.wfkp[i], self.shell_vol[i], self.wgals[i],\
                  file=fout)
        fout.close()

class Mask:

    mask_dir = ''
    geometry_file = mask_dir + '' 
    vetos= {'collision':'collision_priority_mask_lrg_eboss_trimmed_v3.ply', \
            'bad_field':'badfield_mask_unphot_seeing_extinction_pixs8_dr12.ply',\
            'bright_star':'allsky_bright_star_mask_pix.ply',\
            'bright_object':'bright_object_mask_rykoff_pix.ply',\
            'centerpost':'eboss_centerpost_mask.ply',\
            'IR_bright':'brightstarmask_tiling_final.ply'}

    @staticmethod
    def read_mangle_mask(root):

        mfits = mangle.Mangle(root+'.fits', keep_ids=1)
        m = mangle.Mangle(root+'.ply', keep_ids=1)
        m.chunk = mfits.chunk
        m.decmid = mfits.decmid
        m.ramid = mfits.ramid
        m.ntiles = mfits.ntiles
        m.sector = mfits.sector
        m.sectorarea = mfits.sectorarea
        m.tiles = mfits.tiles
        m.use_caps = mfits.use_caps
        m.areaname = mfits.areaname
        m.iboss= mfits.iboss
        m.maskedarea = mfits.maskedarea 
        m.names = mfits.names
        m.metadata = mfits.metadata
        return m

    @staticmethod
    def change_mangle_mask_weights(root):
        masktype = os.path.basename(root).split('-')[0]
        outroot = os.path.dirname(root)+'/'+\
                  os.path.basename(root).replace(masktype, masktype+'0')
        mask = read_mangle_mask(root)

        w = mask.weights > 0.
        mask.weights[w] = 1.

        mask.write_fits_file(outroot+'.fits', keep_ids=True)
        mask.writeply(outroot+'.ply', keep_ids=True)
      
    @staticmethod
    def cut(mask, w):
        #newmask = copy.deepcopy(self)
        size = mask.weights.size
        for f in mask.__dict__.items():
            if hasattr(f[1], 'size') and f[1].size % size == 0:
                mask.__dict__[f[0]] = f[1][w]
        newsize = mask.weights.size
        mask.npoly = newsize
        mask.npixels = newsize
        mask.pixel_dict = mask._create_pixel_dict()


    @staticmethod
    def get_sectors(ra, dec, mask=None):
        if mask is None:
            mask = Mask.read_mangle_mask(Mask.geometry_file)
        w_in, sectors = Mask.veto(ra, dec, mask, get_sectors=1)
        return sectors

    @staticmethod
    def get_chunks(ra, dec, mask=None):
        if mask is None:
            mask = Mask.read_mangle_mask(Mask.geometry_file)
        pix = mask.get_polyids(ra, dec)
        if sum(pix<0) > 0:
            print( 'There are points outside mask!')
        return mask.chunk[pix]

    @staticmethod
    def veto(ra, dec, mask, out=0, weight_min=0., get_sectors=0):
        pix = mask.get_polyids(ra, dec)
        w_in = (pix>-1)
        w_weight= (mask.weights[pix]>weight_min)
        if out:
            w = (~w_in)&(w_weight) 
        else:
            w = ( w_in)&(w_weight)

        if get_sectors:
            return w, mask.sector[pix]
        else:
            return w

    @staticmethod
    def run_vetos(ra, dec, masknames=['collision', 'bad_field', 'IR_bright',\
            'bright_star', 'bright_object', 'centerpost']):

        
        w = (N.ones(ra.size)==1)
        bits = N.zeros(ra.size, dtype=int)

        #-- getting all targets inside geometry first
        print(' Applying geometry')
        mask = Mask.read_mangle_mask(Mask.geometry_file)
        w_in = Mask.veto(ra, dec, mask, out=0)
        print('   Inside geometry', sum(w_in), 'of', ra.size)
        w &= w_in

        for i, name in enumerate(masknames):
            print(' Applying veto with:', Mask.vetos[name])
            print('   reading mask')
            mask = mangle.Mangle(Mask.mask_dir+'/'+Mask.vetos[name], \
                                 keep_ids=True)
            print('   getting pixels')
            w_out = Mask.veto(ra, dec, mask, out=1) 
            bits[~w_out] += 2**i
            w &= w_out
            print('   Outside this mask:', sum(w_out))
            print('   Objects left: ', sum(w))
            print('   Objects vetoed: ', sum(~w))
     
        return w, bits

    @staticmethod
    def get_sector_completeness(mask, w_num, w_den, sectors, mincomp=0.):
        ''' Compute completeness over mask 

        Parameters
        ----------
        mask: mangle object
            The original weights of this mask are ignored
        w_num: bool array
            Selection of numerator among galaxies of catalog
        w_den: bool array
            Selection of denominator among galaxies of catalog
        sectors: int array
            Sectors of each galaxy in the catalog
        mincomp: float
            Any sector with completeness strictly less than this will be zeroed

        Returns
        -------
        new_mask: mangle object
            The mask containing the computed completeness
        comps: float array
            Completeness values for all galaxies

        '''


        unique_mask_sectors = N.unique(mask.sector)
        bins = N.append(unique_mask_sectors, unique_mask_sectors.max()+1)
        
        for sec in sectors:
            if sec not in unique_mask_sectors:
                print(sec, 'not in mask')

        hist_den, bins = N.histogram(sectors[w_den], bins=bins)
        hist_num, bins = N.histogram(sectors[w_num], bins=bins)
        w = hist_den > 0
        comp = N.zeros(unique_mask_sectors.size) 
        comp[w] = hist_num[w]*1./hist_den[w]

        print('  Completeness values (min, median, max):', \
                min(comp[w]), N.median(comp[w]), max(comp[w]))

        #-- setting low completeness sectors to zero
        w = (comp < mincomp)
        comp[w] = 0.

        #-- update mask
        new_mask = copy.deepcopy(mask)
        print('  Filling sector completeness in mask')
        for i in range(mask.sector.size):
            w = (unique_mask_sectors == mask.sector[i])
            new_mask.weights[i] = comp[w]

        print('  Getting completeness for each galaxy')
        comps = N.zeros(sectors.size)

        for i, sec in enumerate(sectors):
            w = (unique_mask_sectors == sec)
            comps[i] = comp[w]
            
        return new_mask, comps

    @staticmethod
    def plot_completeness(\
           mask='/uufs/chpc.utah.edu/common/home/sdss00/'+\
                'ebosswork/eboss/lss/catalogs/'+\
                '1.5/mask-lrg-N-eboss_v1.5_IRt.ply', \
           cap='North', save=0, vmin=0.5, vmax=1.0, ptitle=''):

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        if 'North' in os.path.basename(mask):
            cap = 'North'
        elif 'South' in os.path.basename(mask):
            cap = 'South'
        

        my_cmap = P.get_cmap('jet')
       
        #polys= read_mangle_mask(mask)
        polys=mangle.Mangle(mask, keep_ids=True)

        fig = P.figure(figsize=(12,7))

        fig.subplots_adjust(hspace=0)
        if cap=='North':
            p, m = graphmask.plot_mangle_map(\
                     polys, bgcolor=None, facecolor=[.7,.7,.7], \
                     drawgrid=False, linewidth=.2, pointsper2pi=100, \
                     cenaz=191, cenel=55, width=106, \
                     height=46,projection='laea')
        elif cap=='South':
            p, m = graphmask.plot_mangle_map(\
                     polys, bgcolor=None, facecolor=[.7,.7,.7], \
                     drawgrid=False, linewidth=.2, pointsper2pi=100, \
                     cenaz=5, cenel=15, width=80, \
                     height=48,projection='laea')
        else:
            print('cap = North or South. Input:', cap)

        m.drawmeridians(N.arange(0,360,5),linewidth=.1)
        m.drawparallels(N.arange(-10,80,10),linewidth=.1)

        w = polys.weights > 0
        azel, weight, midpoints = m.get_graphics_polygons(polys[w])
        p = graphmask.draw_weighted_polygons(\
                m.world2pix(azel), weight=weight, draw_colorbar=False, \
                cmap=my_cmap, linewidth=0, vmin=vmin, vmax=vmax)

        P.gca().xaxis.set_label_coords(.5,-.06)
        P.gca().yaxis.set_label_coords(-.09,.5)
        P.xlabel(r'$\alpha$ ($^\circ$)')
        P.ylabel(r'$\delta$ ($^\circ$)')
        P.title(ptitle)

        axins=inset_axes(P.gca(), width="40%", height="3%", loc=1)
        P.colorbar(p, cax=axins, orientation='horizontal', \
                   ticks=N.linspace(vmin, vmax, 6))
        if save:
            P.savefig(save, bbox_inches='tight')
        P.draw()

    @staticmethod
    def create_randoms_ransack(nran, mask_file, seed_ransack=323466458):
        ''' Create random distribution of points inside a mask
        Inputs
        ------
        nran: integer
            Number of random galaxies
        mask_file: string
            Name of the mask file (.ply) 
        seed_ransack: integer
            Seed for ransack code

        Output
        ------
        create_randoms_ransack: Catalog
            A Catalog object containing RA, Dec and Z
        '''

        print('''
        =====================================
        ==== Creating randoms over mask  ====
        =====================================
 
        nran: %d
        mask_file: %s
        seed_ransack: %d 
         '''%(nran, mask_file, seed_ransack)) 

        #-- run ransack to generate random points in sectors 
        #-- based on fiber completeness
        ransack_output = 'ransack.tmp'
        cmd = 'ransack -c %d -r %d %s %s' % \
              (seed_ransack, nran, mask_file, ransack_output) 

        os.system(cmd)
        ra, dec, ipoly = N.loadtxt(ransack_output, unpack=1, skiprows=1)
        os.system('rm ransack.tmp')

        rancat = Catalog()
        rancat.RA = ra
        rancat.DEC = dec
        rancat.size = ra.size

        return rancat

class Utils:

    @staticmethod
    def spherematch(ra1, dec1, ra2, dec2, angle=1./3600):
        ''' Implementation of spherematch using SkyCoord from astropy
        Inputs
        ------
        ra1, dec1, ra2, dec2: arrays 
            Coordinates to be matched
        angle: float
            Angle in degrees for defining maximum separation

        Returns
        -------
        idx1, idx2: arrays
            Index of the matching 
        distance: array
            Distance between matches
        '''

        c1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)     
        c2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)
        idx, d2d, d3d = c1.match_to_catalog_sky(c2)  
        w = d2d.value <= angle
        idx[~w] = -1
        
        idx1 = N.where(w)[0]
        idx2 = idx[idx>-1] 
        distance = d2d.value[w]

        return idx1, idx2, distance

    @staticmethod
    def spheregroup(ra, dec, angle=62./3600):
        ''' Gets list of index for (ra, dec) pairs at distances smaller than angle'''

        c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
        idc1, idc2, d2d, d3d = c.search_around_sky(c, angle*u.degree)

        #-- exclude pairs of same object twice
        #ww = (d2d>0.)
        ww = (idc1!=idc2) & (d2d>0.)
        
        #-- if no collisions return empty list
        if sum(ww) == 0: 
            return []

        i1 = idc1[ww]
        i2 = idc2[ww] 
        distance = d2d[ww]

        #-- removing duplicate pairs
        pairs = [ [ii1, ii2] for ii1, ii2 in zip(i1, i2) if ii1<ii2]
        return pairs

    @staticmethod
    def maskbits(value):
        ''' Get mask bit values from integer '''

        if type(value)==list or type(value)==N.ndarray:
            return [Utils.maskbits(v) for v in value]
        else:
            return [pos for pos, char in enumerate(bin(value)[::-1]) if char == '1']

    @staticmethod
    #-- healpix map rotation 
    def rotate_map(m, coord=['C', 'G'], rot=None):
        """
        STOLEN FROM THE GREAT SPIDER COLLABORATION, and simplified to non-polarized data.
        Rotate an input map from one coordinate system to another or to place a
        particular point at centre in rotated map. 

        e.g. m = rotate_map(m, rot=[phi,90.-theta,0.])

        takes point at original theta, phi to new coord ra=dec=0

        Arguments
        ---------
        m : array_like
            A single map
        coord : list of two coordinates, optional.
            Coordinates to rotate between.  Default: ['C', 'G']
        rot : scalar or sequence, optional
            Describe the rotation to apply.
            In the form (lon, lat, psi) (unit: degrees) : the point at
            longitude lon and latitude lat will be at the center of the rotated
            map. An additional rotation of angle psi around this direction is applied
        """

        res = hp.get_nside(m)
        mi = m
        if rot is None:
            R = hp.Rotator(coord=coord, inv=False)
        else:
            R = hp.Rotator(rot=rot, inv=False)

        # rotate new coordinate system to original coordinates
        theta, phi = hp.pix2ang(res, N.arange(len(mi)))
        mtheta, mphi = R(theta, phi)
        mr = hp.get_interp_val(mi, mtheta, mphi)

        return mr



class Catalog(object):
 
    collate = ''

    def __init__(self, cat=None, collate=0, unique=0):
        if collate:
            self.read_collate(unique=unique)

        if cat is not None:
            a = fits.open(cat)[1].data
            header = fits.open(cat)[1].header

            for f in a.names:
                self.__dict__[f] = a.field(f)
            self.size = self.RA.size
            self.names = a.names

            try:
                self.cap = header['CAP']
                self.target = header['TARGET']
                self.version = header['VERSION'] 
                self.P0 = header['P0']

                self.mask_area = header['AREA'] 
                self.mask_area_eff = header['AREA_EFF'] 
                self.vetofraction = header['VETOFRAC']
                self.veff_tot = header['VEFF_TOT']
            except:
                print('Reading a different catalog')
                self.vetofraction = 1 


    def read_collate(self, unique=0):

        print('Reading', Catalog.collate )
        a = fits.open(Catalog.collate)[1].data
    
        fields = [  'RA', 'DEC', 'EBOSS_TARGET0', 'EBOSS_TARGET1', \
                    'EBOSS_TARGET2', 'EBOSS_TARGET_ID', \
                    'CHUNK', 'SECTOR', 'THING_ID_TARGETING', 'BOSSTILE_STATUS' , \
                    'EXTINCTION', 'PSF_FWHM', 'FIBER2FLUX', 'FIBER2FLUX_IVAR', \
                    'MODELMAG', 'MODELMAG_IVAR', 'W1_MAG', 'W1_MAG_ERR' ]
                    
        for field in fields:
            self.__dict__[field] = a.field(field)

        self.size = self.RA.size
       
        print(self.size, 'entries')
 
        #-- entries with THING_ID_TARGETING==0
        w = self.THING_ID_TARGETING > 0
        if sum(~w)>0:
            print('Removing', sum(~w), 'entries with THING_ID_TARGETING==0')
            self.cut(w)

        #-- fixing duplicates
        if unique:
            self.remove_duplicates()

        self.WEIGHT_FKP = N.ones(self.size)
        self.WEIGHT_SYSTOT = N.ones(self.size) 
        self.FIBER_COMP = N.zeros(self.size)
        self.Z_COMP = N.zeros(self.size)
 
    def remove_duplicates(self):
        
        uthid = N.unique(self.THING_ID_TARGETING)
        if self.size == uthid.size:
            print('There are no duplicates')
            return

        print('There are duplicates in the collate file:', self.size-uthid.size)
        print('Removing duplicates...'    )
        w = N.argsort(self.THING_ID_TARGETING)
        index = list()
        for i in range(self.size-1):
            if self.THING_ID_TARGETING[w[i]] != self.THING_ID_TARGETING[w[i+1]]:
                index.append(i)
            elif self.BOSSTILE_STATUS[w[i]] & 2**0 == 0:
                continue
        if self.size-2 in index:
            index.append(self.size-1)

        index = N.array(index)
        print(self.size, index.size)
        self.cut(w[index])

    def read_mock(self, mockfile, target='LRG'):
        
        if target=='LRG':
            self.target = target
            self.P0 = 10000.0


        self.RA, self.DEC, self.Z, self.COMP, nz, self.WEIGHT_FKP = \
                N.loadtxt(mockfile, unpack=True)
        self.size = self.RA.size
        self.WEIGHT_NOZ = N.ones(self.size)
        self.WEIGHT_CP = N.ones(self.size)
        self.WEIGHT_SYSTOT = N.ones(self.size)
        self.IMATCH = N.ones(self.size)
        self.COMP = N.ones(self.size)
        self.FIBER_COMP = N.ones(self.size)
        self.Z_COMP = N.ones(self.size)

    def cut(self, w):
        ''' Trim catalog columns using a boolean array'''

        size = self.size
        for f in self.__dict__.items():
            if hasattr(f[1], 'size') and f[1].size % size == 0:
                self.__dict__[f[0]] = f[1][w]
        self.size = self.RA.size

    def select_targets(self, target):

        if target=='LRG':
            w = self.EBOSS_TARGET1 & 2 > 0
            self.cut(w)
            self.target = 'LRG'
            self.P0 = 10000.
        elif target=='QSO':
            w = self.EBOSS_TARGET1 & 2**10 > 0
            self.cut(w)
            self.target = 'QSO'
            self.P0 = 6000.
        else:
            print('Need to choose between LRG or QSO' )

    def select_galactic_cap(self, cap):

        w = (self.RA>90.) & (self.RA<300.)
        if cap=='North':
            self.cut(w)
        elif cap=='South':
            self.cut(~w)
        else:
            print('Need to choose between North or South')
        self.cap = cap

    def veto(self):
        ''' Run set of veto masks defined in Mask.run_vetos '''

        print(' ')
        print('=========================================')
        print('==== Running veto masks over catalog ====')
        print('=========================================')
        print(' ')
        
        #--- for LRGs
        if self.target=='LRG':
            masknames = ['collision', 'bad_field', 'IR_bright',\
                         'bright_star', 'bright_object', 'centerpost']
        #-- for QSOs
        elif self.target=='QSO':
            masknames = ['bad_field', 'bright_star', 'bright_object', 'centerpost']
        else:
            print('Are these LRG or QSO?')
            return 0

        w, bits = Mask.run_vetos(self.RA, self.DEC, masknames=masknames)
        #self.vetofraction = N.sum(w)*1./w.size
        self.vetobits = bits
        
        if self.target=='LRG' and self.cap=='North':
            #-- Cut problematic region for NGC
            wo = abs(self.DEC-40.55)>0.05
            print('Inside overlap region', sum(~wo))
            self.vetobits[~wo] = 2**10
            #self.cut(wo)

        #self.cut(w)
 

    def match_with_redshifts(self, zcatalog=None, zwar_cut=1):
        print(' ')
        print('=======================================================')
        print('==== Matching catalog with spectroscopic redshifts ====')
        print('=======================================================')
        print(' ')

        self.PLATE = N.zeros(self.size, dtype=int)
        self.FIBERID = N.zeros(self.size, dtype=int)
        self.MJD = N.zeros(self.size, dtype=int)
        self.THING_ID = N.zeros(self.size, dtype=int)
        self.EBOSS_TARGET_ID_SPEC = N.zeros(self.size, dtype=int)
        self.XFOCAL = N.zeros(self.size)
        self.YFOCAL = N.zeros(self.size)

        self.IMATCH = N.zeros(self.size, dtype=int)
        self.Z = N.zeros(self.size)
        self.ZWARNING = N.zeros(self.size, dtype=int)
        self.RCHI2DIFF = N.zeros(self.size)
        self.CLASS = N.zeros(self.size, dtype='15S')
        self.CHUNK_SPEC = N.zeros(self.size, dtype='15S')

        
        spall = fits.open(zcatalog)[1].data


        #w = (spall.SPECPRIMARY == 1)
        #if self.target == 'LRG':
        #    w = w & ((spall.EBOSS_TARGET1 & 2**1 > 0) | \
        #            (spall.EBOSS_TARGET0 & 2**2 > 0))
        #elif self.target == 'QSO':
        #    w = w & (spall.EBOSS_TARGET1 & 2**10 > 0) 
        #spall = spall[w]


        id2, id1, dist = Utils.spherematch(spall.RA, spall.DEC, \
                                           self.RA, self.DEC)

        print('Entries in catalog: ', self.size)
        print('Entries in catalog with spectro info:', id1.size)

        self.PLATE[id1] = spall.PLATE[id2]
        self.FIBERID[id1] = spall.FIBERID[id2]
        self.MJD[id1] = spall.MJD[id2] 
        self.IMATCH[id1] = 1 
        self.XFOCAL[id1] = spall.XFOCAL[id2]
        self.YFOCAL[id1] = spall.YFOCAL[id2]

        print('Entries with different thing_id', \
                sum(self.THING_ID_TARGETING[id1] != spall.THING_ID[id2]))
        self.THING_ID[id1] = spall.THING_ID[id2]
        self.CHUNK_SPEC[id1] = spall.CHUNK[id2]
        self.EBOSS_TARGET_ID_SPEC[id1] = spall.EBOSS_TARGET_ID[id2].astype(int)

        if self.target=='LRG': 
            self.Z[id1] = spall.Z_RM[id2] 
            self.ZWARNING[id1] = spall.ZWARNING_RM[id2] 
            self.CLASS[id1] = spall.CLASS_RM[id2]
            self.RCHI2DIFF[id1] = spall.RCHI2DIFF_1[id2] 
            #-- if redmonster and spec1d disagree for  z<0.4 and 
            #-- z>1.0, pick spec1d classification
            wz = N.where((abs(spall.Z_RM[id2] - spall.Z_NOQSO[id2]) > 0.01)&\
                         (spall.ZWARNING_RM[id2]    == 0) & \
                         (spall.ZWARNING_NOQSO[id2] == 0) & \
                         ( (spall.Z_RM[id2] < 0.4) | \
                           (spall.Z_RM[id2] > 1.0)      ))[0]
            print('   Using spec1d instead of redmonster redshifts for ', \
                   wz.size, 'galaxies')
            if wz.size>0:
                self.Z[id1[wz]] = spall.Z_NOQSO[id2[wz]]
                self.ZWARNING[id1[wz]] = spall.ZWARNING_NOQSO[id2[wz]]
                self.CLASS[id1[wz]] = spall.CLASS_NOQSO[id2[wz]]
                self.RCHI2DIFF[id1[wz]] = spall.RCHI2DIFF_NOQSO[id2[wz]]

        if self.target=='QSO':
            self.Z[id1] = spall.Z[id2] 
            self.ZWARNING[id1] = spall.ZWARNING[id2] 
            self.CLASS[id1] = spall.CLASS[id2]
            self.RCHI2DIFF[id1] = spall.RCHI2DIFF[id2] 

        #-- Mark redshift failures    
        w =  (self.ZWARNING != 0) 
        self.IMATCH[w] = 7


        #-- Mark stars
        w = ((self.CLASS == 'STAR') | (self.CLASS == 'CAP')) 
        if zwar_cut:
            w &= (self.ZWARNING == 0)
        self.IMATCH[w] = 4 

        #-- Mark wrong class
        if self.target=='LRG':
            w = (self.CLASS == 'QSO') 
            if zwar_cut:
                w &= (self.ZWARNING == 0)
            self.IMATCH[w] = 9
        if self.target=='QSO':
            w = ((self.CLASS == 'GALAXY')|(self.CLASS == 'ssp_galaxy_glob')) 
            if zwar_cut:
                w &= (self.ZWARNING == 0)
            self.IMATCH[w] = 9

    def get_plates_per_sector(self, sect_plate=None):

        w = (self.IMATCH != 0) & (self.PLATE > 0)
        sectors = self.SECTOR[w]
        plates = self.PLATE[w]
        unique_sectors = N.unique(sectors)
        
        if sect_plate:
            fsec = open(sect_plate, 'w')
       
        plates_per_sect = dict()

        print('Getting plate numbers in', unique_sectors.size, 'sectors' )
        for sect in unique_sectors:
            w = (sectors == sect) 
            ps = N.unique(plates[w])
            plates_per_sect[sect] = ps

            if sect_plate:
                line = '%d '%(sect)
                for p in ps:
                    line+='%d '%p
                print(line, file=fsec)
                   
                   
        self.plates_per_sect = plates_per_sect 
        if sect_plate:
            fsec.close()

    def get_plates_per_pixel(self, mask):

        w = (self.PLATE>0)
        pix = mask.get_polyids(self.RA[w], self.DEC[w])
        plates = self.PLATE[w]
        plates_per_pixel = dict()
        for px in N.unique(pix):
            w = pix==px
            ps = N.unique(plates[w])
            plates_per_pixel[px] = ps
        self.plates_per_pixel = plates_per_pixel


    def assign_plates_to_ra_dec(self, ra, dec, mask, seed=0, use_pixels=0):
        ''' Assigns a plate number for an array of RA and DEC

            Inputs
            ------
            ra, dec: arrays
            mask: Mangle object with mask containing sectors
            seed: int (optional)
        
            Returns
            ------
            sectors, plates: int arrays 
        
        '''
        if seed!=0:
            N.random.seed(seed)

        if sum(mask.weights == 0)>0:
            print('Warning: mask containing zero weight regions')

        if use_pixels:
            sectors = mask.get_polyids(ra, dec)
            plates_dict = self.plates_per_pixel
        else:
            sectors = Mask.get_sectors(ra, dec, mask)
            plates_dict = self.plates_per_sect
        plates = sectors*0



        for i in range(sectors.size): 

            if sectors[i] in plates_dict:
                splates = plates_dict[sectors[i]]
                plates[i] = splates[N.random.randint(splates.size)] 
            else:
                plates[i] = 0

        return sectors, plates


    def fiber_collision(self, apply_noz=1, dist_root=''):

        print(' ')
        print('========================================')
        print('==== Finding fiber-collision pairs  ====')
        print('========================================')
        print(' ')

        sectors = self.SECTOR
        unique_sectors = N.unique(sectors)
      
        cnt_legacy_removed = 0
     
        #-- making copies for testing purposes
        all_z = N.copy(self.Z)
        all_imatch = N.copy(self.IMATCH)
        all_weight_cp = N.ones(self.size)
        all_weight_noz = N.ones(self.size)

        all_pair_in_sector_good = N.zeros(self.size, dtype=int)
        all_pair_in_sector_tot = N.zeros(self.size, dtype=int)
        all_gal_in_sector_good = N.zeros(self.size, dtype=int)
        all_gal_in_sector_bad = N.zeros(self.size, dtype=int)
        all_cp_pair_over_poss = N.zeros(self.size)
        all_cp_gal_over_poss = N.zeros(self.size)


        if dist_root != '':
            fout = open(dist_root, 'w')

        print('Solving fiber collisions in ', unique_sectors.size, 'sectors' )
        for i, sect in enumerate(unique_sectors):
            w = (sectors == sect) & (all_imatch != 2) & (self.vetobits == 0)

            plates = N.unique(self.PLATE[w])

            z = all_z[w]
            imatch = all_imatch[w]
            weight_cp = N.zeros(z.size)
           
            if (imatch==0).all():
                continue

            print('  %d of %d'%(i, unique_sectors.size), \
                  '\tSector#:', sect,\
                  '\tTargets:', N.sum(w), \
                  '\tnplates:', len(plates),\
                  '\tnspec:', sum(imatch>0))

            #-- call to spheregroup for galaxies (with new redshifts) in this sector
            #-- with linking length of 62''.
            pairs = Utils.spheregroup(self.RA[w], self.DEC[w], angle=62./3600)
           
       

            
            pair_in_sector_good = 0
            pair_in_sector_tot = 0
            gal_in_sector_good = 0
            gal_in_sector_bad = 0
            
            for pair in pairs:
                imatch1 = imatch[pair[0]]
                imatch2 = imatch[pair[1]]
                #print(imatch1, imatch2)

                if imatch1 != 0 or imatch2 != 0:
                    pair_in_sector_tot += 1
                    
                #-- if collision is resolved
                if imatch1 != 0 and imatch1 != 3 and imatch2 != 0 and imatch2 != 3:
                    gal_in_sector_good += 1
                    pair_in_sector_good += 1
                    #print('This collision is solved through many plates')

                #-- solving collision
                elif imatch1 == 0 and imatch2 !=0 and imatch2 != 3:
                    z[pair[0]] = z[pair[1]]
                    imatch[pair[0]] = 3
                    imatch1 = 3
                    weight_cp[pair[1]] += 1
                    gal_in_sector_bad += 1
                    #print('Solving collision: putting right redshift on the left')

                elif imatch2 == 0 and imatch1 !=0 and imatch1 != 3:
                    z[pair[1]] = z[pair[0]]
                    imatch[pair[1]] = 3
                    imatch2 = 3
                    weight_cp[pair[0]] += 1
                    gal_in_sector_bad += 1
                    #print('Solving collision: putting left redshift on the right')
        

            if pair_in_sector_tot > 0: 
                cp_pair_over_poss = pair_in_sector_good*1./pair_in_sector_tot
            else: 
                cp_pair_over_poss = 0
            if gal_in_sector_good+gal_in_sector_bad > 0: 
                cp_gal_over_poss = gal_in_sector_good*1. / \
                                  (gal_in_sector_good + gal_in_sector_bad)
            else:
                cp_gal_over_poss = 0
            if plates.size == 1:
                cp_pair_over_poss = 0.
                cp_gal_over_poss = 0.

            print('  Close-pairs:', len(pairs))
            print('     pairs numbers :', pair_in_sector_good, \
                        pair_in_sector_tot, cp_pair_over_poss )
            print('     galaxy numbers:', gal_in_sector_good,  \
                        gal_in_sector_bad, cp_gal_over_poss)

            all_z[w] = z
            all_imatch[w] = imatch
            all_weight_cp[w] += weight_cp
            all_pair_in_sector_good[w] = pair_in_sector_good
            all_pair_in_sector_tot[w] = pair_in_sector_tot
            all_gal_in_sector_good[w] = gal_in_sector_good
            all_gal_in_sector_bad[w] = gal_in_sector_bad
            all_cp_pair_over_poss[w] = cp_pair_over_poss
            all_cp_gal_over_poss[w] = cp_gal_over_poss

        
            #-- now look for LEGACY close pairs in this sector.
            #-- remove some according to probability cp_gal_over_poss

            ww = (sectors == sect) & (self.vetobits==0)
            imatch = all_imatch[ww]
            z = all_z[ww]
            weight_cp = all_weight_cp[ww]
            weight_noz = all_weight_noz[ww]
            ra = self.RA[ww]
            dec = self.DEC[ww]

            #-- if there is any legacy redshifts do something
            if (all_imatch[ww] == 2).any():
                pairs = Utils.spheregroup(ra, dec, angle=62./3600)
                
                for pair in pairs:
                    imatch1 = imatch[pair[0]]
                    imatch2 = imatch[pair[1]]

                    if  (imatch1 == 2 or imatch2 == 2) and \
                         imatch1 != 3 and imatch2 != 3 and \
                         imatch1 != 0 and imatch2 != 0 and \
                         imatch1 != 8 and imatch2 != 8:
                        r1 = N.random.rand()
                        if r1 < cp_gal_over_poss: continue
                        r2 = N.random.rand() 
                        if r2 > 0.5: 
                            imatch[pair[0]] = 8
                            # this needs to update too so the same galaxy 
                            # doesn't get updated twice.
                            imatch1 = 8 
                            z[pair[0]] = z[pair[1]]
                            weight_cp[pair[1]] += weight_cp[pair[0]]
                        else:
                            imatch[pair[1]] = 8
                            # this needs to update too so the same galaxy 
                            # doesn't get updated twice.
                            imatch2 = 8 
                            z[pair[1]] = z[pair[0]]
                            weight_cp[pair[0]] += weight_cp[pair[1]]
                        cnt_legacy_removed += 1
                #-- end of pair loop            
            #-- if any imatch==2 in this sector
            all_z[ww] = z
            all_imatch[ww] = imatch
            all_weight_cp[ww] = weight_cp


            ##-- Here goes the redshift failure correction ### To be changed hopefully
            if apply_noz == 0:
                continue

            #-- find failed targets in this sector
            wz = (imatch == 7)     
            if sum(wz) == 0:
                continue
          
            #-- find closest neighbor with imatch = 1, 4 (star) or 9 (wrong class)
            wgood = N.where( (imatch == 1) | (imatch == 4) | (imatch == 9))[0]
            if wgood.size == 0:
                print('  No good redshifts found to do \
                         redshift failure correction')
                continue
     
            print('     Failures:', sum(wz), \
                  '\t Number of targets available to apply correction', \
                  wgood.size)
            id1, id2, dist = Utils.spherematch(\
                                ra[wz], dec[wz], ra[wgood], dec[wgood], \
                                angle=2.)
           
            if dist_root!='': 
                for j in range(dist.size): 
                    print(sect, dist[j], file=fout)

            #-- attribute same redshift and transfer weight_cp 
            imatch[wz] = 5
            z[wz] = z[wgood[id2]]
            for i in range(id2.size):
                weight_noz[wgood[id2[i]]] += weight_cp[wz][id1[i]] 
     
            all_z[ww] = z
            all_imatch[ww] = imatch
            all_weight_noz[ww] = weight_noz
       
        ##-- end loop over regions

        self.Z = all_z
        self.IMATCH = all_imatch
        self.WEIGHT_CP = all_weight_cp
        self.WEIGHT_NOZ = all_weight_noz       
        self.PAIRS_GOOD =     all_pair_in_sector_good
        self.PAIRS_TOT =  all_pair_in_sector_tot
        self.GALS_GOOD =     all_gal_in_sector_good
        self.GALS_BAD =     all_gal_in_sector_bad
        self.COMP_CP =    all_cp_pair_over_poss
        self.COMP_GAL=    all_cp_gal_over_poss


        if dist_root != '':
            fout.close()    

    def get_fiber_completeness(self, mask,  mincomp=0.):

        sectors = self.SECTOR
        imatch = self.IMATCH
        
        #-- numerator: objects with fibers plugged or fiber-collision corrected 
        w_num = (imatch != 0) & (imatch != 2)
        #-- denominator: all targets except legacy
        w_den = (imatch != 2) 
    
        new_mask, fiber_comp = \
              Mask.get_sector_completeness(mask, w_num, w_den, \
                                           sectors, mincomp=mincomp)
        return new_mask, fiber_comp

    def get_spectro_completeness(self, mask, mincomp=0.):

        sectors = self.SECTOR
        imatch = self.IMATCH
        #-- numerator: spectroscopic confirmed good targets 
        w_num = (imatch == 1) 
        #-- denominator: confirmed, corrected failures and failures
        w_den = (imatch == 1) | (imatch == 5) | (imatch == 7) 
    
        new_mask, z_comp = Mask.get_sector_completeness(\
                            mask, w_num, w_den, sectors, mincomp=mincomp)
        return new_mask, z_comp

    def get_tinker_completeness(self, mask, mincomp=0.):

        sectors = self.SECTOR
        imatch = self.IMATCH
        #-- numerator: spectroscopic confirmed good targets and close-pairs  
        w_num = (imatch==1) | (imatch==3) 
        #-- denominator: confirmed, corrected failures and failures
        w_den = (imatch!=2) & ( imatch != 4) & (imatch!= 9) 
    
        new_mask, tinker_comp = Mask.get_sector_completeness(\
                                mask, w_num, w_den, sectors, mincomp=mincomp)
        return new_mask, tinker_comp

    def make_mask(self, mask=None, mincomp=0., export_dir=''):    
        ''' Fill the mask weights with the different completeness definitions
            
            Parameters
            ----------
            mask: the mangle object containing sector numbers
            mincomp: zero fiber completeness below this threshold 

        ''' 

        print(' ')
        print('======================================================')
        print('==== Computing fiber and spectro completeness     ====')
        print('======================================================')
        print(' ')

        if mask is None:
            print('   Reading geometry file')
            mask = Mask.read_mangle_mask(Mask.geometry_file)
                    

        fibercomp_mask, fiber_comp = self.get_fiber_completeness(mask, mincomp=mincomp)
        zcomp_mask, z_comp = self.get_spectro_completeness(mask, mincomp=mincomp)
        tink_mask, tink_comp = self.get_tinker_completeness(mask, mincomp=mincomp)

        self.FIBER_COMP = fiber_comp
        self.Z_COMP = z_comp
        self.TINK_COMP = tink_comp

        #-- export masks
        if export_dir!='':

            for m, name in zip([fibercomp_mask, zcomp_mask, tink_mask], \
                               ['fibercomp', 'zcomp', 'tink']):

                mask_root = '/%s-%s-%s-%s'%(name, self.version, self.target, self.cap)

                fout = export_dir+mask_root
                m.write_fits_file(fout+'.fits', keep_ids=True)
                m.writeply(fout+'.ply', keep_ids=True)
                print('  Exported %s completeness to:\n  '%name+fout)
            
    def trim_catalog(self, cp=1, noz=1, comp='fibercomp'):

        print(' ')
        print('=================================')
        print('==== Trimming data catalog  =====')
        print('=================================')
        print(' ')
    
        if comp=='fibercomp':
            self.COMP = self.FIBER_COMP
        elif comp=='zcomp':
            self.COMP = self.Z_COMP
        elif comp=='combined':
            self.COMP = self.FIBER_COMP*self.Z_COMP
        elif comp=='tink':
            self.COMP = self.TINK_COMP
        else:
            print('Need to choose completeness: fibercomp, zcomp or combined or tink')
            

        #-- randomly sub-sample known galaxies to match completeness
        rr = N.random.rand(self.size)
        w = (self.COMP > 0) & ((self.IMATCH != 2) | (rr < self.COMP))
        self.cut(w)

        #-- selecting only confirmed galaxies and sub-sampled known galaxies
        ww = (self.COMP > 0) & ( (self.IMATCH == 1) | (self.IMATCH == 2) )
        ngal = sum(ww)

        weights = self.get_weights(cp=cp, noz=noz, fkp=0, syst=0)[ww] 
       
        #-- weighted number of galaxies 
        wgal = N.sum(weights)
    
        self.ngalaxies = ngal
        self.wgalaxies = wgal

        print(' Total number of galaxies with redshifts:', self.ngalaxies )
        print(' Total weight of galaxies with redshifts:', self.wgalaxies )

    def create_randoms(self, nran, mask_root, do_veto=0, seed_ransack=323466458):
       
        rancat = Mask.create_randoms_ransack(\
                    nran*self.wgalaxies, mask_root+'.ply', \
                    seed_ransack=seed_ransack )

        rancat.target = self.target
        rancat.P0 = self.P0
        rancat.cap = self.cap
        rancat.version = self.version
        if do_veto:
            ransize_before = rancat.size*1.
            rancat.veto()
            rancat.cut(rancat.vetobits==0)
            ransize_after = rancat.size*1.
            rancat.vetofraction = ransize_after/ransize_before
        else:
            rancat.vetofraction = 1.0

        return rancat

    def compute_area(self, rancat, mask):
        '''
        Computes survey area and effective survey area in square degrees 
        
        Parameters
        ---------
        rancat : random Catalog
        mask : mangle object with weighted mask

        '''

        self.vetofraction = rancat.vetofraction

        self.mask_area = sum(mask.areas * (mask.weights > 0.001)) * (180./N.pi)**2 * self.vetofraction 
        self.mask_area_eff = sum(mask.areas * mask.weights) * (180./N.pi)**2 * self.vetofraction 

        rancat.mask_area = self.mask_area
        rancat.mask_area_eff = self.mask_area_eff

        print('Survey area:          ', self.mask_area, ' (sq. deg.)  '  )
        print('Survey area effective:', self.mask_area_eff, ' (sq. deg.)  '  )

    def compute_nbar(self, cosmo=None, zmin=0.0, zmax=3.5, dz=0.005, \
                        export='', cp=1, noz=1, syst=1):
        ''' Computes density as a function of redshift 
            and assigns redshifts for the random catalog objects 
            

            Returns
            -------
            Nbar object

        '''

        print(' ')
        print('=====================================================')
        print('=== Computing nbar and assigning random redshifts ===')
        print('=====================================================')
        print(' ')

        if cosmo is None:
            cosmo = Cosmo()

        nbins = int(N.floor((zmax-zmin)/dz))
        zedges = N.linspace(zmin, zmax, nbins+1)
        zcen = 0.5*(zedges[:-1]+zedges[1:])
        zlow = zedges[:-1]
        zhigh = zedges[1:]

        #-- info from catalog
        mask_area_eff = self.mask_area_eff 
        weights = self.get_weights(cp=cp, noz=noz, fkp=0, syst=syst)
        P0 = self.P0
        
        ww = (self.COMP > 0) & ( (self.IMATCH == 1) | \
             (self.IMATCH == 2) | (self.IMATCH==101) | (self.IMATCH == 102) )
        z = self.Z[ww]
        weights = weights[ww]
        
        total_weight = N.sum(weights)
        cumul_weight = N.cumsum(weights)

        cnt_tot, zedges = N.histogram(z, bins=zedges, weights=weights)
        mask_vol = mask_area_eff * cosmo.shell_vol(zlow, zhigh) / \
                   (4*N.pi * (180./N.pi)**2)
        nbar = cnt_tot / mask_vol
        wfkp = 1./(1+nbar*P0)
        veff = ((nbar*P0)/(1+nbar*P0))**2 * mask_vol
        veff_tot = sum(veff)

        self.veff_tot = veff_tot   
    
        nb = Nbar()
        nb.mask_area_eff = mask_area_eff
        nb.veff_tot = veff_tot
        nb.zcen = zcen
        nb.zlow = zlow
        nb.zhigh=zhigh
        nb.nbar = nbar
        nb.wfkp = wfkp
        nb.shell_vol = mask_vol
        nb.wgals = cnt_tot
        nb.nbins = nbins

        return nb

    def assign_random_redshifts(self, rancat, nbar, cp=1, noz=1, syst=1, seed=0):

        if seed:
            N.random.seed(seed)

        weights = self.get_weights(cp=cp, noz=noz, fkp=0, syst=syst)

        w = (self.COMP > 0) & ( (self.IMATCH == 1) | (self.IMATCH == 2) )
        z = self.Z[w] 
        weights = weights[w]
        total_weight = N.sum(weights)
        cumul_weight = N.cumsum(weights)

        #-- assing redshifts to randoms
        weight_random = N.random.rand(rancat.size)*total_weight
        index0 = N.arange(cumul_weight.size)
        index1 = N.floor(N.interp(weight_random, cumul_weight, index0)).astype(int)
        rancat.Z = z[index1]
        rancat.veff_tot = self.veff_tot
       
    def assign_fkp_weights(self, nbar):

        wfkp = N.ones(self.size)
        for i in range(nbar.zcen.size):
            w = (self.Z >= nbar.zlow[i]) & (self.Z < nbar.zhigh[i])
            wfkp[w] = nbar.wfkp[i]

        self.WEIGHT_FKP = wfkp

    def export(self, fout, cosmo=None):

        collist = []

        for f in self.__dict__.items():
            if hasattr(f[1], 'size') and f[1].size % self.size == 0:
                arraytype = f[1].dtype.name
                if 'int' in arraytype:
                    col_format='K'
                elif 'float' in arraytype:
                    col_format='D'
                elif 'string' in arraytype:
                    col_format='15A'
                else:
                    continue

                if len(f[1].shape)>1:
                    ndim = f[1].size / self.size
                    col_format = '%d'%ndim+col_format

                col = fits.Column(name=f[0], format=col_format, array=self.__dict__[f[0]])
                collist.append(col)

        hdulist = fits.BinTableHDU.from_columns(collist)
        header = hdulist.header
        
        header['VERSION'] = self.version
        header['CAP'] = self.cap
        header['TARGET'] = self.target
        header['P0'] = self.P0
          
        if cosmo is None:
            cosmo = Cosmo() 
        header['OmegaM'] = cosmo.OmegaM
        header['OmegaL'] = cosmo.OmegaL
        header['h'] = cosmo.h

        try:
            header['AREA'] = self.mask_area
            header['AREA_EFF'] = self.mask_area_eff
            header['VETOFRAC'] = self.vetofraction
            header['VEFF_TOT'] = self.veff_tot
        except:
            pass

        hdulist.writeto(fout, clobber=True)
        print('Catalog exported to: ', fout)

    def get_weights(self, cp=1, noz=1, fkp=0, syst=0):
        
        if cp==1 and noz==1:
            weights = (self.WEIGHT_CP + self.WEIGHT_NOZ - 1)
        elif cp==1:
            weights = 1.*self.WEIGHT_CP 
        elif noz==1:
            weights = 1.*self.WEIGHT_NOZ 
        else:
            weights = N.ones(self.size)

        if fkp:
            weights *= self.WEIGHT_FKP
    
        if syst:
            weights *= self.WEIGHT_SYSTOT
    
        return weights

    def plot(self, w=None, fmt='.', alpha=1, color=None, label=None):

        if w is None:
            w = N.ones(self.size)==1

        ra = self.RA[w] -360*(self.RA[w]>300.)
        dec = self.DEC[w]
        P.plot(ra, dec, \
                fmt, alpha=alpha, color=color,  label=label)

    

    @staticmethod
    def merge_catalogs(c1, c2):
        ''' Merges two catalog objects into one, keeping only columns in common
            between the two. This is mostly to merge eBOSS and CMASS catalogs.

        Input
        -----
        c1, c2: Catalog object
            Input catalogs to merge
        Output
        -----
        c: Catalog object
        ''' 

        names1=list()
        names2=list()
        for f in c1.__dict__.items():
            if hasattr(f[1], 'size') and f[1].size > 1:
                names1.append(f[0])

        for f in c2.__dict__.items():
            if hasattr(f[1], 'size') and f[1].size > 1:
                names2.append(f[0])

        fields = set(names1) & set(names2)

        c = Catalog()
        for f in fields:
            print(f)
            x = N.append(c1.__dict__[f], c2.__dict__[f], axis=0)
            c.__dict__[f] = x

        c.size = c.RA.size
        c.target = c1.target
        c.cap = c1.cap
        c.version = c1.version
        c.mask_area = c1.mask_area
        c.mask_area_eff = c1.mask_area_eff
        c.vetofraction = c1.vetofraction
        c.P0 = c1.P0

        return c



def main(outdir=None, \
         collate=None, \
         geometry=None, \
         vetos_dir=None, \
         zcatalog=None, \
         version='test', target='LRG', cap='Both', comp='fibercomp',\
         mincomp=0.5, do_veto=1, zmin=0.6, zmax=1.0,\
         noz=0, unique=0, zwar_cut=1, nran=50, fc=1, OmegaM=0.31):
    ''' Run catalog creation 

        Examples:
        - Produce catalogs similar to official ones
        main(version='my1.5', comp='fibercomp', unique=0, zwar_cut=0)

        - Produce catalogs a la CMASS, using fiber completeness, 
        and fixing redshift
        failures by increasing weight of closest neighbor
        main(version='test7fcomp', comp='fibercomp')

        - Produce catalogs a la Zhai, using a different completeness based on 
        redshift failures
        main(version='test7tink', comp='tink', noz=0)
        


    '''
    

    if cap=='Both':
        for cap in ['North', 'South']:
            main(outdir=outdir, collate=collate, geometry=geometry, \
                 vetos_dir=vetos_dir, \
                 zcatalog=zcatalog, version=version, target=target, cap=cap, \
                 comp=comp,\
                 do_veto=do_veto, zmin=zmin, zmax=zmax,\
                 noz=noz, unique=unique, zwar_cut=zwar_cut, nran=nran, fc=fc,\
                 OmegaM=OmegaM)
        return 

    Mask.mask_dir = vetos_dir
    Mask.geometry_file = geometry
    Catalog.collate = collate

    outdir = os.path.join(outdir, version)
    try: 
        os.makedirs(outdir)
    except:
        pass

    #log_file = '%s/log-%s-%s-%s.txt'%(outdir, version, target, cap)
    info = (outdir, version, target, cap)
    dist_root = '%s/dist-%s-%s-%s.txt'%info
    sect_plate = '%s/sectplate-%s-%s-%s.txt'%(outdir, version, target, cap)
    mask_root = '%s/mask-%s-%s-%s'%(outdir, version, target, cap)
    nbar_file = '%s/nbar-%s-%s-%s.dat'%(outdir, version, target, cap)
    syst_root = '%s/syst-%s-%s-%s'%(outdir, version, target, cap)
    cat_dat_file = '%s/ebosscat-%s-%s-%s.dat.fits'%\
                    (outdir, version, target, cap)
    cat_ran_file = '%s/ebosscat-%s-%s-%s.ran.fits'%\
                  (outdir, version, target, cap)


    if fc:
        #-- read info from collate file
        cat = Catalog(collate=1, unique=0)
        cat.version = version
        cat.outdir = outdir

        #-- cutting chunk 20 by hand now
        w = (cat.CHUNK!=20)
        cat.cut(w)

        #-- select north or south
        cat.select_galactic_cap(cap)
     
        #-- select type of target : lrgs or qsos
        cat.select_targets(target)

        cat.remove_duplicates()

        #-- apply veto masks
        if do_veto:
            cat.veto()
            #cat.cut(cat.vetobits==0)

        cat.export(cat_dat_file+'.temp_veto')

        #-- match to spectro redshifts
        cat.match_with_redshifts(zcatalog=zcatalog, zwar_cut=zwar_cut)

        #-- find plates in sectors
        cat.get_plates_per_sector(sect_plate=sect_plate)

        #-- find fiber collisions
        cat.fiber_collision(apply_noz=noz, dist_root=dist_root)
        cat.export(cat_dat_file+'.temp_fc')
    else: 
        cat = Catalog(cat_dat_file+'.temp_fc')

    #-- fill mask weights with many completeness definitions
    cat.make_mask(mincomp=mincomp, export_dir=outdir) 

    #-- select final targets with good redshifts 
    cat.trim_catalog(cp=1, noz=noz, comp=comp)
   
    #-- build angular random catalog  
    mask_root = mask_root.replace('mask', comp) 
    rancat = cat.create_randoms(nran, mask_root, do_veto=do_veto, \
                                seed_ransack=323466458 )

    #-- compute total effective area in sq degrees (input for compute_nbar)
    mask = Mask.read_mangle_mask(mask_root)
    cat.compute_area(rancat, mask)

    #-- compute nbar, fkp weights and attribute redshifts to randoms
    cosmo = Cosmo(OmegaM=OmegaM)
    nbar = cat.compute_nbar(noz=noz, cosmo=cosmo) 
    nbar.export(nbar_file)

    cat.assign_random_redshifts(rancat, nbar, noz=noz, seed=323466458) 

    cat.assign_fkp_weights(nbar)
    rancat.assign_fkp_weights(nbar)

    #-- compute systematic weights
    if target=='LRG':
        MultiFit.LRGs_zbins(cat, rancat, zs=[0.6, 1.0], \
             nbins=10, plotit=0, plotroot=syst_root, cp=1, noz=noz, fkp=1)
        #MultiFit.LRGs_zbins(cat, rancat, zs=[0.6, 0.67, 0.74, 0.81, 1.0], \
        #     nbins=10, plotit=0, plotroot=syst_root, cp=1, noz=noz, fkp=1)


    #-- export catalogs to fits files
    cat.export(cat_dat_file, cosmo=cosmo)
    rancat.export(cat_ran_file, cosmo=cosmo)





