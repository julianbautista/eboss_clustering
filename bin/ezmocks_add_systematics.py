import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pylab as plt
import healpy as hp
import pymangle
import sys

from nbodykit.algorithms.fibercollisions import FiberCollisions
from astropy.table import Table, vstack, unique
from astropy.coordinates import SkyCoord
from astropy import units

#-- This is from github.com/julianbautista/eboss_clustering.git
#-- The equivalent in mkAllsamples is zfail_JB.py
from cosmo import CosmoSimple
import redshift_failures
import systematic_fitter 

def ra(x):
    return x-360*(x>300)

def plot_survey(surv, *args, **kwargs):
    plt.plot(ra(surv['RA']), surv['DEC'], *args, **kwargs)

def countdup(x, weights=None):
    if weights is None:
        ux, inverse, counts = np.unique(x, return_inverse=True,return_counts=True)
    else:
        ux, inverse = np.unique(x, return_inverse=True)
        counts = np.bincount(inverse, weights=weights)
    return counts[inverse]

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

    c1 = SkyCoord(ra=ra1*units.degree, dec=dec1*units.degree)
    c2 = SkyCoord(ra=ra2*units.degree, dec=dec2*units.degree)
    idx, d2d, d3d = c1.match_to_catalog_sky(c2)
    w = d2d.value <= angle
    idx[~w] = -1

    idx1 = np.where(w)[0]
    idx2 = idx[idx>-1]
    distance = d2d.value[w]

    return idx1, idx2, distance

def add_fiber_collisions(mock_full):

    nmock = len(mock_full)
    mock_full['INGROUP'] = np.zeros(nmock, dtype=int)

    has_fiber = (mock_full['IMATCH'] == 2)

    #-- This information was obtained from the mask file
    ntiles = mock_full['NTILES']

    #-- From Arnaud's code, iterate over ntiles
    for ipass in range(ntiles.max()):
        mask_decollide = (ntiles > ipass) & (has_fiber==False) 
        ndecollide = mask_decollide.sum()
        if ndecollide == 0: break
        if ndecollide == 1:
            has_fiber[mask_decollide] = True
            break

        #-- Run nbodykit algorithm for fiber collisions        
        labels = FiberCollisions(mock_full['RA'][mask_decollide], 
                                 mock_full['DEC'][mask_decollide],
                                 collision_radius=62/3600).labels
        collided = labels['Collided'].compute()
        has_fiber[mask_decollide] = (collided == 0)

        #-- Assign groupid only in the first iteration, needed for corrections later
        if ipass==0:
            groupid = labels['Label'].compute()
            mock_full['INGROUP'][mask_decollide] = groupid
        print(f'  TSR at pass {ipass+1}: {np.sum(has_fiber)/nmock:.4f}')

    #-- At this point, imatch = 0 or -1 for all galaxies
    #-- Assign imatch = 1 for all galaxies with imatch = 0
    w = (mock_full['IMATCH'] == 0)&(has_fiber)
    mock_full['IMATCH'][w] = 1
    #-- Assign imatch = 0 for targets including bad_targets that don't have a fiber 
    w = (mock_full['IMATCH'] != 1)&(~has_fiber)
    mock_full['IMATCH'][w] = 0

def correct_fiber_collisions(mock_full):
   
    #-- Select only targets in collision groups
    w_group = mock_full['INGROUP'] > 0
    groups = mock_full['INGROUP'][w_group]
    imatch = mock_full['IMATCH'][w_group]

    #-- All targers with fiber
    w_spec = (imatch==1) | \
             (imatch==4) | \
             (imatch==7) | \
             (imatch==9)

    #-- All targets 
    w_all =  (imatch!=2) & \
             (imatch!=8) & \
             (imatch!=13)

    count_spec = countdup(groups, weights=w_spec)
    count_all  = countdup(groups, weights=w_all)

    #-- weight_cp = ntargets/nfibers (sets to 1.0 if nfibers == 0)
    weight_cp = count_all/(count_spec  + count_all*(count_spec==0))   
    weight_cp[~w_spec] = 1.

    #-- Fill imatch for targets without fiber but in group with at least one fiber
    imatch[ (imatch== 0) & (count_spec>0) ] = 3 
    imatch[ (imatch==-1) & (count_spec>0) ] = 3 
   
    mock_full['WEIGHT_CP'] = np.ones(len(mock_full))
    mock_full['WEIGHT_CP'][w_group] = weight_cp
    mock_full['IMATCH'][w_group] = imatch

    w = (mock_full['IMATCH'] == 2)
    mock_full['WEIGHT_CP'][w] = 1 


def add_redshift_failures(mock_full, data, fields_redshift_failures, seed=None):

    #-- All targers with fiber
    imatch = data['IMATCH']
    w_spec = (imatch==1) | \
             (imatch==4) | \
             (imatch==7) | \
             (imatch==9) | \
             (imatch==14) | \
             (imatch==15) 
    data_spec = data[w_spec] 

    print(f'  Number of data spectra to match mock: {len(data_spec)}')
    #-- Remove failures already in mock, which come from real data (contaminants)
    w = mock_full['IMATCH'] == 7 
    mock_full['IMATCH'][w] = 6 

    #-- All targets to be matched
    w_nonlegacy = (mock_full['IMATCH'] !=  2) & (mock_full['IMATCH'] != 6) & \
                  (mock_full['IMATCH'] != -1) & (mock_full['IMATCH'] != 0)
    mock = mock_full[w_nonlegacy]

    #-- Match mock and data 
    id1, id2, dist = spherematch(mock['RA'], mock['DEC'], 
                                 data_spec['RA'], data_spec['DEC'], angle=1.)
    nmatch = id1.size
    nmock = len(mock)
    print(f'  Number of mock galaxies matched to data: {nmatch}/{nmock} = {nmatch/nmock:.3f}')
    print(f'  Max distance for matching: {dist.max():.3f} deg')

    #-- Assign to mocks the data quantities 
    for field in fields_redshift_failures:
        mock[field][id1] = data_spec[field][id2]
    mock.rename_column('WEIGHT_NOZ', 'WEIGHT_NOZ_DATA')

    #-- Define ssr: spectroscopic success rate using data weights
    ssr = 1/(mock['WEIGHT_NOZ_DATA'])

    #-- Setting randomly failures to imatch = 7
    if not seed is None:
        np.random.seed(seed)

    w_mock_fail = (np.random.rand(nmock) > ssr)
    mock['IMATCH'][w_mock_fail] = 7

    #-- Compare redshift failure rates
    w_data_fail = (imatch==7)
    w_data = (imatch==1) | \
             (imatch==4) | \
             (imatch==7) | \
             (imatch==9)

    mock_full[w_nonlegacy] = mock

    w_mock = (mock_full['IMATCH'] == 1)|\
             (mock_full['IMATCH'] == 4)|\
             (mock_full['IMATCH'] == 7)|\
             (mock_full['IMATCH'] == 9)
    print(f'  Data redshift failure rate: {np.sum(w_data_fail)/np.sum(w_data):.3f}')
    print(f'  Mock redshift failure rate: {np.sum(w_mock_fail)/np.sum(w_mock):.3f}')

    #-- Setting targets without plate info to imatch = 0 
    w = (mock_full['PLATE'] == 0) & (w_nonlegacy)
    mock_full['IMATCH'][w] = -1


def get_systematic_weights(mock, rand, 
    target='LRG', zmin=0.6, zmax=1.0, 
    random_fraction=1, seed=None, nbins=20, plotit=False):

    wd = (mock['Z'] >= zmin)&(mock['Z'] <= zmax)
    if 'IMATCH' in mock.colnames:
        wd &= ((mock['IMATCH']==1)|(mock['IMATCH']==2))
    if 'COMP_BOSS' in mock.colnames:
        wd &= (mock['COMP_BOSS'] > 0.5)
    if 'sector_SSR' in mock.colnames:
        wd &= (mock['sector_SSR'] > 0.5) 

    wr = (rand['Z'] >= zmin) & (rand['Z'] <= zmax)
    if 'COMP_BOSS' in rand.colnames:
        wr &= (rand['COMP_BOSS'] > 0.5)
    if 'sector_SSR' in rand.colnames:
        wr &= (rand['sector_SSR'] > 0.5)
    if random_fraction != 1:
        wr &= (np.random.rand(len(rand))<random_fraction)

    #-- Defining RA, DEC and weights
    data_ra, data_dec = mock['RA'].data[wd], mock['DEC'].data[wd]
    rand_ra, rand_dec = rand['RA'].data[wr], rand['DEC'].data[wr]
    data_we = mock['WEIGHT_FKP'][wd]
    if 'WEIGHT_CP' in mock.colnames:
        data_we *= mock['WEIGHT_CP'][wd]
    if 'sector_SSR' in mock.colnames:
        data_we /= mock['sector_SSR'][wd]
    rand_we = rand['WEIGHT_FKP'][wr]
    if 'COMP_BOSS' in rand.colnames:
        rand_we *= rand['COMP_BOSS'][wr]

    #-- Read systematic values for data and randoms
    data_syst = systematic_fitter.get_systematic_maps(data_ra, data_dec)
    rand_syst = systematic_fitter.get_systematic_maps(rand_ra, rand_dec)
    map_syst = systematic_fitter.get_systematic_maps(nside=256)

    #-- Create fitter object
    s = systematic_fitter.Syst(data_we, rand_we)

    if target == 'LRG':
        use_maps = ['STAR_DENSITY', 'EBV', 'PSF_I', 'DEPTH_I_MINUS_EBV', 'AIRMASS']
        fit_maps = ['STAR_DENSITY', 'EBV']
    if target == 'QSO':
        use_maps = ['STAR_DENSITY', 'EBV', 'PSF_G', 'SKY_G', 'DEPTH_G_MINUS_EBV', 'AIRMASS']
        fit_maps = ['STAR_DENSITY', 'DEPTH_G_MINUS_EBV']

    #-- Add the systematic maps we want 
    for syst_name in use_maps:
        s.add_syst(syst_name, data_syst[syst_name], rand_syst[syst_name])
    #-- Cut galaxies and randoms with some extreme values
    s.cut_outliers(p=0.5, verbose=1)
    #-- Perform global fit
    s.prepare(nbins=nbins)
    s.fit_minuit(fit_maps=fit_maps)
    if plotit:
        s.plot_overdensity(pars=[None, s.best_pars], ylim=[0.5, 1.5],
                           title='global fit')

    #-- Get weights for global fit
    weight_systot = 1/s.get_model(s.best_pars, data_syst)

    mock_weight_systot     = np.zeros(len(mock))
    rand_weight_systot     = np.zeros(len(rand))
    mock_weight_systot[wd] = weight_systot
    rand_weight_systot[wr] = np.random.choice(weight_systot, 
                                              size=np.sum(wr), replace=True)

    mock['WEIGHT_SYSTOT'] = mock_weight_systot
    rand['WEIGHT_SYSTOT'] = rand_weight_systot
    
    #-- Compute a map with the best-fit density model that will be used to sub-sample mocks
    dens_model = s.get_model(s.best_pars, map_syst)
    w = np.isnan(dens_model) | (dens_model < 0.01) | (dens_model > 2)
    dens_model[w] = hp.UNSEEN 
    norm = np.percentile(dens_model[~w], 99.9)
    dens_model[~w] /= norm
    return s, dens_model 
    
def count_per_sector(mock, zmin, zmax):

    im_values = [-1, 0, 1, 2, 3, 4, 7, 9, 13, 14, 15]
    
    counts = Table()
    counts['SECTOR'] = mock['SECTOR']
    for im in im_values:
        w = mock['IMATCH'] == im
        n = countdup(mock['SECTOR'], weights=w)
        counts[f'N_IMATCH{im}'] = n
        if im==1 or im==2:
            w = w & (mock['Z']>=zmin) & (mock['Z'] <= zmax)
            n = countdup(mock['SECTOR'], weights=w)
            counts[f'N_IMATCH{im}_Z'] = n
    try:
        counts['COMP_BOSS_IN'] = mock['COMP_BOSS_IN']
    except:
        counts['COMP_BOSS_IN'] = mock['COMP_BOSS']

    counts = unique(counts)
    #counts['FRAC_COMP'] = ((counts['N1415']+counts['N479']+counts['NGAL'])*counts['COMP_BOSS_IN']\
    #                        - counts['N479'])/counts['NGAL']

    return counts
 
def downsample_completeness(mock, seed=None):

    if not seed is None:
        np.random.seed(seed)
    w_sub = (np.random.rand(len(mock)) < mock['COMP_BOSS_IN'])
    return w_sub 
    
         
def fiber_completeness(mock):

    sectors = mock['SECTOR']

    im_good = [1, 3, 4, 7, 9]
    im_all = im_good + [0, -1, 14, 15]
    
    nmock = len(mock)
    w_good = np.zeros(nmock, dtype=bool )
    w_all  = np.zeros(nmock, dtype=bool)
    
    for im in im_good:
        w_good |= (mock['IMATCH']==im)
    for im in im_all:
        w_all |= (mock['IMATCH']==im)
 
    count_good = countdup(sectors, weights=w_good)
    count_all  = countdup(sectors, weights=w_all)

    comp_boss = count_good/(count_all+1*(count_all==0))

    return comp_boss

def sector_ssr(mock):

    sectors = mock['SECTOR']

    im_good = [1, 4, 9]
    im_all = im_good + [7]

    nmock = len(mock)
    w_good = np.zeros(nmock, dtype=bool)
    w_all  = np.zeros(nmock, dtype=bool)

    for im in im_good:
        w_good |= (mock['IMATCH']==im)
    for im in im_all:
        w_all  |= (mock['IMATCH']==im)

    count_good = countdup(sectors, weights=w_good)
    count_all  = countdup(sectors, weights=w_all)

    sect_ssr = (count_all>0)*count_good/(count_all+1*(count_all==0))

    return sect_ssr 

def compute_area(rand, mask_dict):
   
    usec = np.unique(rand['SECTOR'])
    sectors = mask_dict['sector']
    
    w = np.in1d(sectors, usec)
    area = np.sum(mask_dict['sector_area'][w])
 
    return area
    
def read_nbar(fin):

    #-- Read first line
    print('Reading density vs z from', fin)
    f = open(fin)
    area_eff, vol_eff = np.array(f.readline().split()[-2:]).astype(float)

    #-- Read the rest of the lines
    zcen, zlow, zhigh, nbar, wfkp, shell_vol, wgals = np.loadtxt(fin, unpack=1) 
    z_edges = np.append(zlow, zhigh[-1])
    
    nbar_dict = {'z_cen': zcen, 'z_edges': z_edges, 'nbar': nbar, 'shell_vol': shell_vol, 
                 'weighted_gals': wgals, 'area_eff': area_eff, 'vol_eff': vol_eff} 
    return nbar_dict

def export_nbar(nbar, fout):
    fout = open(fout, 'w')
    print(f"# effective area (deg^2), effective volume (Mpc/h)^3: {nbar['area_eff']} {np.sum(nbar['vol_eff'])}", file=fout)
    print('# zcen, zlow, zhigh, nbar, wfkp, shell_vol, total weighted gals', file=fout)
    z_cen = nbar['z_cen']    
    z_edges = nbar['z_edges']
 
    for i in range(z_cen.size):
        print(z_cen[i], z_edges[i], z_edges[i+1], 
              nbar['nbar'][i], nbar['w_fkp'][i], 
              nbar['shell_vol'][i], nbar['weighted_gals'][i], 
              file=fout)
    fout.close()


def compute_nbar(mock, z_edges, area_eff, P0=10000. ):
    
    c = CosmoSimple(omega_m=0.31)

    w = np.ones(len(mock), dtype=bool)
    if 'IMATCH' in mock.colnames:
        w &= ((mock['IMATCH'] == 1) | (mock['IMATCH'] == 2)) 
    if 'sector_SSR' in mock.colnames:
        w &= (mock['sector_SSR'] > 0.5) 
    if 'COMP_BOSS' in mock.colnames:
        w &= (mock['COMP_BOSS'] > 0.5)
    
    z = mock['Z'][w]
    weight = np.ones_like(z)
    if 'WEIGHT_CP' in mock.colnames: 
        weight *= mock['WEIGHT_CP'][w]
    if 'WEIGHT_NOZ' in mock.colnames:
        weight *= mock['WEIGHT_NOZ'][w]
    #if 'sector_SSR' in mock.colnames: 
    #    weight /= mock['sector_SSR'][w]
    
    counts, _ = np.histogram(z, bins=z_edges, weights=weight)
    shell_vol = area_eff * c.shell_vol(z_edges[:-1], z_edges[1:]) / (4*np.pi * (180./np.pi)**2)
    nbar = counts/shell_vol
    vol_eff = shell_vol * (nbar * P0/(1+nbar * P0))**2
    w_fkp = 1/(nbar*P0+1)

    z_cen = 0.5*(z_edges[:-1]+z_edges[1:])
   
    nbar_dict = {'z_cen': z_cen, 'z_edges': z_edges, 'nbar': nbar, 'w_fkp': w_fkp, 'shell_vol': shell_vol,
                 'weighted_gals': counts, 'vol_eff': vol_eff, 'area_eff': area_eff}

    return nbar_dict

def compute_weight_fkp(mock, nbar, P0=10000.):

    nz = np.zeros(len(mock))
    z_edges = nbar['z_edges']
    index = np.floor( (mock['Z'] - z_edges[0])/(z_edges[1]-z_edges[0])).astype(int)
    w = (index>=0)&(index<nbar['nbar'].size)
    nz[w] = nbar['nbar'][index[w]]
    mock['NZ'] = nz
    mock['WEIGHT_FKP'] = 1/(1+nz*P0) 
    

    
def downsample_nbar(nbar, nbar_mock, mock, seed=None):

    if not seed is None:
        np.random.seed(seed)

    z_edges = nbar['z_edges']
    prob = nbar['nbar']/nbar_mock
    index = np.floor( (mock['Z'] - z_edges[0])/(z_edges[1]-z_edges[0])).astype(int)
    
    w = np.random.rand(len(mock)) < prob[index] 

    return w

def downsample_photo(dens_model, mock, seed=None):

    if not seed is None:
        np.random.seed(seed)

    nside = hp.get_nside(dens_model)
    pix = systematic_fitter.get_pix(nside, mock['RA'].data, mock['DEC'].data)
    prob = dens_model[pix] 
    w = np.random.rand(len(mock)) < prob
    return w

 
def read_data(target=None, cap=None, data_name=None, rand_name=None, 
    survey_comp=None, survey_geometry=None, nbar_name=None):

    #-- Read real data 
    if data_name is None:
        data_name =   f'/mnt/lustre/eboss/DR16_{target}_data/v7/eBOSS_{target}_full_{cap}_v7.dat.fits'
        rand_name =   f'/mnt/lustre/eboss/DR16_{target}_data/v7/eBOSS_{target}_full_{cap}_v7.ran.fits'
        nbar_name =   f'/mnt/lustre/eboss/DR16_{target}_data/v7/nbar_eBOSS_{target}_{cap}_v7.dat'
        survey_comp = f'/mnt/lustre/eboss/DR16_{target}_data/v7/eBOSS_{target}geometry_v7.fits'
        survey_geometry = '/mnt/lustre/eboss/DR16_geometry/eboss_geometry_eboss0_eboss27'
    
        
    print('Reading data from ', data_name)
    data = Table.read(data_name)
    print('Reading randoms from ', rand_name)
    rand = Table.read(rand_name)

    #-- Cut on COMP_BOSS > 0.5 (there's no mocks outside)
    wd = (data['COMP_BOSS']>0.5) & (data['sector_SSR']>0.5)
    data = data[wd]
    wr = (rand['COMP_BOSS']>0.5) & (rand['sector_SSR']>0.5)
    rand = rand[wr]

    #-- Removing NaN (just to avoid annoying warning messages) 
    w = np.isnan(data['Z'])
    data['Z'][w] = -1

    #-- Read mask files
    print('Reading completeness info from ', survey_comp)
    survey_comp = Table.read(survey_comp) 
    sec_comp = unique(survey_comp['SECTOR', 'COMP_BOSS', 'SECTORAREA'])

    print('Reading survey geometry from ', survey_geometry)
    mask_fits = Table.read(survey_geometry+'.fits')
    mask_ply = pymangle.Mangle((survey_geometry+'.ply'))

    mask_fits['AREA'] = mask_ply.areas*1.
    usect = np.unique(mask_fits['SECTOR'])
    for sec in usect:
        w = mask_fits['SECTOR'] == sec
        area = np.sum(mask_fits['AREA'][w])
        mask_fits['SECTORAREA'][w] = area*1.

    for i in range(len(sec_comp)):
        sec = sec_comp['SECTOR'][i]
        w = mask_fits['SECTOR'] == sec
        sec_comp['SECTORAREA'][i] = mask_fits['SECTORAREA'][w][0]


    mask_dict = {'sector': sec_comp['SECTOR'], 
                 'comp_boss': sec_comp['COMP_BOSS'],
                 'sector_area': sec_comp['SECTORAREA'],
                 'mask_ply': mask_ply,
                 'mask_fits': mask_fits}

    nbar = read_nbar(nbar_name)
        
    return data, rand, mask_dict, nbar

def assign_imatch2(data, mock, zmin=0.75, zmax=2.25, seed=None):

    sectors = data['SECTOR']

    im_good = [2]
    im_all = [1, 2]

    nd = len(data)
    w_good = np.zeros(nd, dtype=bool)
    w_all  = np.zeros(nd, dtype=bool)


    for im in im_good:
        w_good |= (data['IMATCH']==im)
    for im in im_all:
        w_all  |= (data['IMATCH']==im)
    
    w_good &= (data['Z']>=zmin) & (data['Z']<=zmax)
    w_all  &= (data['Z']>=zmin) & (data['Z']<=zmax)

    count_good = countdup(sectors, weights=w_good)
    count_all  = countdup(sectors, weights=w_all)

    imatch2_fraction = (count_all>0)*count_good/(count_all+1*(count_all==0))
    data['IM2_FRAC'] = imatch2_fraction

    im2_frac = Table()
    im2_frac['SECTOR'] = sectors
    im2_frac['IM2_FRAC'] = imatch2_fraction
    im2_frac = unique(im2_frac)

    win = np.in1d(mock['SECTOR'], im2_frac['SECTOR'])
    print(f' Number of mocks with im2_frac information: {np.sum(win)}/{len(mock)}')

    mock['IM2_FRAC'] = np.interp(mock['SECTOR'], im2_frac['SECTOR'], im2_frac['IM2_FRAC'])
    data['IM2_FRAC'] = np.interp(data['SECTOR'], im2_frac['SECTOR'], im2_frac['IM2_FRAC'])
     
    if not seed is None:
        np.random.seed(seed)
    imatch = mock['IMATCH'] 
    w = (imatch != -1)
    w_sub = mock['IM2_FRAC'][w] > np.random.rand(np.sum(w))
    imatch[w] = 2*(w_sub) + imatch[w]*(~w_sub)
    mock['IMATCH'] = imatch
 


    

def get_contaminants(data, fields, zmin, zmax):

    #-- Getting contaminants from data
    imatch = data['IMATCH']
    wbad  = (imatch == 4) #-- stars
    wbad |= (imatch == 7) #-- z-failures
    wbad |= (imatch == 9) #-- wrong class
    wbad |= (imatch == 14) #-- little coverage, unplugged, bad_target, no data
    wbad |= (imatch == 15) #-- not tiled
    
    wbad |= (imatch == 1) & ((data['Z'] < zmin) | (data['Z'] > zmax)) 
    #wbad |= (imatch == 2) & ((data['Z'] < zmin) | (data['Z'] > zmax)) 
 
    bad_targets = data[wbad]
    bad_targets.keep_columns(fields)
    return bad_targets

def assign_sectors(mock, mask_dict):

    #-- Get information from mask 
    mock_polyid = mask_dict['mask_ply'].polyid(mock['RA'], mock['DEC'])
    sector = mask_dict['mask_fits']['SECTOR'][mock_polyid]
    ntiles = mask_dict['mask_fits']['NTILES'][mock_polyid]
    comp_boss = np.interp(sector, mask_dict['sector'], mask_dict['comp_boss'])
    mock['SECTOR'] = sector*(mock_polyid>=0) 
    mock['NTILES'] = ntiles*(mock_polyid>=0)
    mock['COMP_BOSS_IN'] = comp_boss*(mock_polyid>=0) 
 
def make_randoms(mock, rand, seed=None, add_radial_integral_constraint=False):

    w = (mock['IMATCH'] == 1) | (mock['IMATCH'] == 2)
    z = mock['Z'][w]
   
    mock_sector, index = np.unique(mock['SECTOR'], return_index=True)
    mock_comp = mock['COMP_BOSS'][index]
    mock_ssr = mock['sector_SSR'][index]

    rand_sector, inverse = np.unique(rand['SECTOR'], return_inverse=True)
    w = np.in1d(rand_sector, mock_sector)
    print(' number of sectors in random but not in mock:', sum(~w))
    rand_comp = np.zeros(w.size)
    rand_ssr  = np.zeros(w.size)
    rand_comp[w] = np.interp(rand_sector[w], mock_sector, mock_comp)
    rand_ssr[w]  = np.interp(rand_sector[w], mock_sector, mock_ssr)
   
    #-- Creating a new random 
    rand_new = Table()
    for field in ['RA', 'DEC', 'SECTOR']:
        rand_new[field] = rand[field]*1
    rand_new['COMP_BOSS'] = rand_comp[inverse]
    rand_new['sector_SSR'] = rand_ssr[inverse]

    #-- Picking redshifts from data
    if add_radial_integral_constraint:
        if not seed is None:
            np.random.seed(seed)
        rand_new['Z'] = np.random.choice(z, size=len(rand), replace=True)
    else:
        rand_new['Z'] = rand['Z']

    return rand_new

def make_clustering_catalog(mock, zmin, zmax):

    w = ((mock['IMATCH']==1)|(mock['IMATCH']==2))
    w &= (mock['Z'] >= zmin)
    w &= (mock['Z'] <= zmax)
    w &= (mock['COMP_BOSS'] > 0.5)
    w &= (mock['sector_SSR'] > 0.5)
    mock_clust = mock[w]

    mock_clust.keep_columns(['RA', 'DEC', 'Z', 
        'WEIGHT_FKP', 'WEIGHT_NOZ', 'WEIGHT_CP', 'WEIGHT_SYSTOT', 
        'NZ'])

    return mock_clust

def make_clustering_catalog_random(rand, mock, seed=None):
    
    rand_clust = Table()
    rand_clust['RA'] = rand['RA']*1
    rand_clust['DEC'] = rand['DEC']*1
    rand_clust['Z'] = rand['Z']*1
    rand_clust['NZ'] = rand['NZ']*1
    rand_clust['WEIGHT_FKP'] = rand['WEIGHT_FKP']*1
    rand_clust['COMP_BOSS'] = rand['COMP_BOSS']*1
    rand_clust['sector_SSR'] = rand['sector_SSR']*1

    if not seed is None:
        np.random.seed(seed)
    
    index = np.arange(len(mock))
    ind = np.random.choice(index, size=len(rand), replace=True)
    
    fields = ['WEIGHT_NOZ', 'WEIGHT_CP', 'WEIGHT_SYSTOT'] 
    for f in fields:
        rand_clust[f] = mock[f][ind]

    #-- As in real data:
    rand_clust['WEIGHT_SYSTOT'] *= rand_clust['COMP_BOSS']

    w = (rand_clust['COMP_BOSS'] > 0.5) & (rand_clust['sector_SSR'] > 0.5) 

    return rand_clust[w]


def get_imatch_stats(mock, zmin, zmax):

    imatch = np.unique(mock['IMATCH'])
    for im in imatch:
        w = mock['IMATCH'] == im
        print(im, np.sum(w))
        if im == 1 or im == 2:
            ww = (mock['Z'][w] >= zmin)&(mock['Z'][w] <= zmax)
            ngals = np.sum(ww)
            print(im, f'{ngals} at {zmin} <= z <= {zmax}')
            if 'WEIGHT_CP' in mock.colnames:
                wgals = np.sum(mock['WEIGHT_CP'][w][ww])
                print(im, f'{wgals:.1f} sum of (wcp)      at {zmin} <= z <= {zmax}')
                if 'WEIGHT_NOZ' in mock.colnames:
                    wgals = np.sum(mock['WEIGHT_CP'][w][ww]*mock['WEIGHT_NOZ'][w][ww])
                    print(im, f'{wgals:.1f} sum of (wcp*wnoz) at {zmin} <= z <= {zmax}')
                    if 'WEIGHT_SYSTOT' in mock.colnames: 
                        wgals = np.sum(mock['WEIGHT_CP'][w][ww]*mock['WEIGHT_NOZ'][w][ww]*mock['WEIGHT_SYSTOT'][w][ww])
                        print(im, f'{wgals:.1f} sum of (wcp*wnoz*wsystot) at {zmin} <= z <= {zmax}')

def plot_completeness(mock, data, field='COMP_BOSS', vmin=0.5, vmax=1, sub=10):

    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    pcm = ax[0].scatter(ra(data['RA'])[::sub], data['DEC'][::sub], c=data[field][::sub], 
                  s=1, vmin=vmin, vmax=vmax, alpha=0.5)  
    f.colorbar(pcm, ax=ax[0])
    pcm = ax[1].scatter(ra(mock['RA'])[::sub], mock['DEC'][::sub], c=mock[field][::sub], 
                  s=1, vmin=vmin, vmax=vmax, alpha=0.5)  
    f.colorbar(pcm, ax=ax[1])
    ax[0].set_ylabel('DEC [deg]')
    ax[0].set_title('Data')
    ax[1].set_title('Mock')
    ax[0].set_xlabel('RA [deg]')
    ax[1].set_xlabel('RA [deg]')
    plt.tight_layout()

def main(target, cap, ifirst, ilast, zmin, zmax, P0):

    print(f'Target: {target}')
    print(f'cap:    {cap}')
    print(f'zmin:   {zmin}')
    print(f'zmax:   {zmax}')
    print(f'P0:     {P0}')

    #-- Optional systematics
    add_contaminants = False
    add_photosyst = False
    add_fibercompleteness = False
    add_fibercollisions = False
    add_zfailures = False
    add_radial_integral_constraint = False
    
    #-- Read data
    data, rand_data, mask_dict, nbar_data = read_data(target=target, cap=cap)
    
    #-- Get stars, bad targets, unpluggged fibers and the following columns
    fields_redshift_failures = ['XFOCAL', 'YFOCAL', 'PLATE', 'MJD', 'FIBERID', 
                                'SPEC1_G', 'SPEC1_R', 'SPEC1_I', 
                                'SPEC2_G', 'SPEC2_R', 'SPEC2_I', 
                                'WEIGHT_NOZ']
    fields_bad_targets = ['RA', 'DEC', 'Z', 'IMATCH', 'NTILES', 'SECTOR']  \
                         + fields_redshift_failures
    bad_targets = get_contaminants(data, fields_bad_targets, zmin, zmax)
    
    #-- Fit for photo-systematics on data and obtain a density model for sub-sampling mocks
    if add_photosyst:
        syst_obj, density_model_data = get_systematic_weights(data, rand_data, plotit=False,
                                                              target=target, zmin=zmin, zmax=zmax)
    

    #-- Setup mock directories
    input_dir  = f'/mnt/lustre/eboss/EZ_MOCKs/EZmock_{target}_v7.0'
    output_dir = f'/mnt/lustre/eboss/EZ_MOCKs/EZmock_{target}_v7.1_nosyst'
    rand_name  = input_dir + f'/random/random_20x_eBOSS_{target}_{cap}_v7.fits'

    #-- Read randoms just once
    print('')
    print('Reading random from ', rand_name)
    rand0 = Table.read(rand_name)
    if 'SECTOR' not in rand0.colnames:
        assign_sectors(rand0, mask_dict)
        rand0.write(rand_name, overwrite=True)

    area_eff_data = nbar_data['area_eff']
    area_data = area_eff_data * len(rand0) / np.sum(rand0['COMP_BOSS_IN']) 

    for imock in range(ifirst, ilast):
        mock_name_out = output_dir+ f'/eBOSS_{target}/EZmock_eBOSS_{target}_{cap}_v7_{imock:04d}.dat.fits'
        mask_name_out = output_dir+ f'/eBOSS_{target}/EZmock_eBOSS_{target}_{cap}_v7_{imock:04d}.mask.fits'
        nbar_name_out = output_dir+ f'/eBOSS_{target}/EZmock_eBOSS_{target}_{cap}_v7_{imock:04d}.nbar.txt'
        rand_name_out = output_dir+ f'/eBOSS_{target}/EZmock_eBOSS_{target}_{cap}_v7_{imock:04d}.ran.fits'

        #-- Read mock
        mock_name_in  = input_dir + f'/eBOSS_{target}/EZmock_eBOSS_{target}_{cap}_v7_{imock:04d}.dat'
        print('')
        print('Reading mock from', mock_name_in)
        try:
            mock0 = Table.read(mock_name_in+'.fits')
        except:
            mock0 = Table.read(mock_name_in, format='ascii', 
                               names=('RA', 'DEC', 'Z', 'WEIGHT_FKP'))
            w = np.isnan(mock0['RA']) | np.isnan(mock0['DEC']) | np.isnan(mock0['Z'])
            mock0 = mock0[~w]
            mock0.write(mock_name_in+'.fits', overwrite=True)
        
        assign_sectors(mock0, mask_dict)

        #-- Getting an estimate of how many galaxies are lost through photo systematics
        #-- but not yet perform sub-sampling
        if add_photosyst:
            w_photo = downsample_photo(density_model_data, mock0, seed=imock)
            photo_ratio = np.sum(w_photo)/len(mock0)
        else:
            photo_ratio = 1
        #print(f' Fraction kept after photo syst downsampling: {photo_ratio:.4f}')

        #-- Removing this since now we correctly re-compute the effective area 
        #if add_fibercompleteness:
        #    w_comp = downsample_completeness(mock0, seed=imock)
        #    comp_ratio = np.sum(w_comp)/len(mock0)
        #else:
        comp_ratio = 1.
        #print(f' Fraction kept after fiber comp downsampling: {comp_ratio:.4f}')

        #-- Compute original density
        #nbar_mock0 = compute_nbar(mock0, nbar_data['z_edges'], nbar_data['shell_vol'], P0=P0)
        nbar_mock0 = compute_nbar(mock0, nbar_data['z_edges'], area_data, P0=P0)
        w = nbar_mock0['nbar']>0
        nbar_mock0_const = np.median(nbar_mock0['nbar'][w])
        print(f' Original number density (assumed constant): {nbar_mock0_const*1e4:.3f} x 1e-4')

        #-- Divide the original density of the mock by photo_ratio
        nbar_mock0_const *= photo_ratio * comp_ratio
        nbar_rand0_const = nbar_mock0_const 

        #-- Sub-sample to match n(z) of the data
        w_sub = downsample_nbar(nbar_data, nbar_mock0_const, mock0, seed=imock)
        mock = mock0[w_sub]
        w_sub = downsample_nbar(nbar_data, nbar_rand0_const, rand0, seed=imock)
        rand = rand0[w_sub]

        #-- Sub-sample of mock with photometric systematic map
        if add_photosyst:
            w_photo = downsample_photo(density_model_data, mock, seed=imock)
            photo_ratio = np.sum(w_photo)/len(mock)
            mock = mock[w_photo]
            print(f' Fraction kept after photo syst downsampling: {photo_ratio:.4f}')
        
        #-- Add missing fields to mock with zeros 
        for field in fields_bad_targets:
            if field not in mock.colnames:
                mock[field] = np.zeros(len(mock), dtype=data[field].dtype) 
        
        #-- Assign legacy targets for QSO only
        if target=='QSO':
            assign_imatch2(data, mock, zmin=zmin, zmax=zmax, seed=imock)

        #-- Add data contaminants to mock
        if add_contaminants:
            mock_full = vstack([mock, bad_targets])
        else:
            mock_full = mock

        assign_sectors(mock_full, mask_dict)

        #-- Sub-sample following completeness
        if add_fibercompleteness:
            #counts = count_per_sector(mock_full, zmin, zmax)
            #frac_comp = np.interp(mock_full['SECTOR'], counts['SECTOR'], counts['FRAC_COMP']) 
            #np.random.seed(imock)
            #w_comp = (np.random.rand(len(mock_full)) > frac_comp) & (mock_full['IMATCH'] == 0) 
            #comp_ratio = np.sum(~w_comp)/len(mock_full)
            w_comp = downsample_completeness(mock_full, seed=imock)
            comp_ratio = np.sum(w_comp)/len(mock_full)
            mock_full['IMATCH'][~w_comp] = -1
            print(f' Fraction kept after fiber comp downsampling: {comp_ratio:.4f}')

        #-- Add fiber collisions
        if add_fibercollisions:
            print('')
            print('Adding fiber collisions...')
            add_fiber_collisions(mock_full)

            #-- Correcting for fiber collisions
            print('')
            print('Correcting fiber collisions...')
            correct_fiber_collisions(mock_full)
        else:
            w = (mock_full['IMATCH'] == 0)
            mock_full['IMATCH'][w] = 1
            mock_full['WEIGHT_CP'] = 1
        
        #-- Add redshift failures
        if add_zfailures:
            print('')
            print('Adding redshift failures...')
            add_redshift_failures(mock_full, data, fields_redshift_failures, seed=imock)
     
            #-- Correct redshift failures
            print('')
            print('Correcting for redshift failures...')
            redshift_failures.get_weights_noz(mock_full)
            mock_full['WEIGHT_NOZ'][mock_full['IMATCH']==2] = 1  
            #redshift_failures.plot_failures(mock_full)
        else:
            mock_full['WEIGHT_NOZ'] = 1

        mock_full['COMP_BOSS'] = fiber_completeness(mock_full) 
        mock_full['sector_SSR'] = sector_ssr(mock_full) 
        
        w_comp = mock_full['COMP_BOSS']>0.5
        w_ssr  = mock_full['sector_SSR']>0.5
        w_both_mock = w_comp & w_ssr
        print('Galaxies information:')
        print(f' - COMP_BOSS  > 0.5: {np.sum(w_comp)}/{len(mock_full)} = {np.sum(w_comp)/len(mock_full):.3f}') 
        print(f' - sector_SSR > 0.5: {np.sum(w_ssr )}/{len(mock_full)} = {np.sum(w_ssr )/len(mock_full):.3f}')
        print(f' - both :            {np.sum(w_both_mock)}/{len(mock_full)} = {np.sum(w_both_mock)/len(mock_full):.3f}')
        
        mask = unique(mock_full['SECTOR', 'COMP_BOSS', 'sector_SSR'])     
        w_comp = mask['COMP_BOSS']>0.5
        w_ssr  = mask['sector_SSR']>0.5
        w_both = w_comp & w_ssr

        print('Sectors information:')
        print(f' - COMP_BOSS  > 0.5: {np.sum(w_comp)}/{len(mask)}')
        print(f' - sector_SSR > 0.5: {np.sum(w_ssr )}/{len(mask)}')
        print(f' - both :            {np.sum(w_both)}/{len(mask)}')

        #-- Make random catalog for this mock 
        rand_full = make_randoms(mock_full, rand, seed=imock, 
                                 add_radial_integral_constraint=add_radial_integral_constraint)
        w_comp = rand_full['COMP_BOSS']>0.5
        w_ssr  = rand_full['sector_SSR']>0.5
        w_both_rand = w_comp & w_ssr
        print('Randoms information:')
        print(f' - COMP_BOSS  > 0.5: {np.sum(w_comp)}/{len(rand_full)} = {np.sum(w_comp)/len(rand_full):.3f} ')
        print(f' - sector_SSR > 0.5: {np.sum(w_ssr )}/{len(rand_full)} = {np.sum(w_ssr )/len(rand_full):.3f}')
        print(f' - both :            {np.sum(w_both_rand)}/{len(rand_full)} = {np.sum(w_both_rand)/len(rand_full):.3f}')
        frac_area = np.sum(w_both_rand)/len(rand_full)
    
        mock_full = mock_full[w_both_mock]
        rand_full = rand_full[w_both_rand]

        #-- Compute effective area
        area_eff_mock = area_data * frac_area * np.sum(rand_full['COMP_BOSS'])/ len(rand_full)
        print(f'Area effective mock: {area_eff_mock:.2f}')
        print(f'Area effective data: {area_eff_data:.2f}')

        #-- Compute nbar

        #nbar_mock = compute_nbar(mock_full, nbar_data['z_edges'], nbar_data['shell_vol'], P0=P0)
        nbar_mock = compute_nbar(mock_full, nbar_data['z_edges'], area_eff_mock, P0=P0)
        


 
        #-- Compute FKP weights
        compute_weight_fkp(mock_full, nbar_mock, P0=P0)
        compute_weight_fkp(rand_full, nbar_mock, P0=P0)
     
        #-- Correct photometric systematics
        if add_photosyst:
            print('')
            print('Getting photo-systematic weights...')
            s, density_model_mock = get_systematic_weights(mock_full, rand_full,
                                                           target=target, zmin=zmin, zmax=zmax,
                                                           random_fraction=1, seed=imock, nbins=20, 
                                                           plotit=False)
        else: 
            mock_full['WEIGHT_SYSTOT'] = 1.0
            rand_full['WEIGHT_SYSTOT'] = 1.0

        print('')
        print('Mock IMATCH statistics')
        get_imatch_stats(mock_full, zmin, zmax)
        print('')
        print('Data IMATCH statistics')
        get_imatch_stats(data, zmin, zmax)


        #-- Make clustering catalog
        mock_clust = make_clustering_catalog(mock_full, zmin, zmax)
        rand_clust = make_clustering_catalog_random(rand_full, mock_clust, seed=imock)

        #-- Export  
        print('')
        print('Exporting mock to', mock_name_out)
        mock_clust.write(mock_name_out, overwrite=True) 
        print('Exporting random to', rand_name_out)
        rand_clust.write(rand_name_out, overwrite=True) 
        print('Exporting nbar to', nbar_name_out)
        export_nbar(nbar_mock, nbar_name_out)

        #print('Exporting mask to ', mask_name_out)
        #mask.write(mask_name_out, overwrite=True)
       
        if imock < 11:
            plot_completeness(mock_full, data, field='COMP_BOSS', sub=1)
            plt.suptitle(f'COMP_BOSS {target} {cap} mock #{imock:04d}')
            plt.savefig(mock_name_out.replace('dat.fits', 'comp_boss.png'))
            plot_completeness(mock_full, data, field='sector_SSR', sub=1)
            plt.suptitle(f'sector_SSR {target} {cap} mock #{imock:04d}')
            plt.savefig(mock_name_out.replace('dat.fits', 'sector_ssr.png'))
            mock_full.write(mock_name_out.replace('.dat.fits', '_full.dat.fits'), overwrite=True)
            rand_full.write(rand_name_out.replace('.ran.fits', '_full.ran.fits'), overwrite=True)

#-- This should be done with .ini file ... 
if len(sys.argv) == 8:
    matplotlib.use('Agg')
    plt.ioff()

    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), 
         float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]))
else:
    print('python ezmocks_add_systematics.py target cap ifirst ilast zmin zmax P0')




 
