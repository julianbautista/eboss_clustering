import numpy as np
import pylab as plt
import healpy as hp
import pymangle
import sys

from nbodykit.algorithms.fibercollisions import FiberCollisions
from astropy.table import Table, vstack, unique
from astropy.coordinates import SkyCoord
from astropy import units

#-- This is from github.com/julianbautista/eboss_clustering.git
#-- The equivalent in mkAllsamples is zfail_JB.py
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
    has_fiber = np.zeros(nmock, dtype=bool)

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
        print(f'  TSR at pass {ipass+1}: {np.sum(has_fiber)/has_fiber.size:.4f}')

    w = (mock_full['IMATCH'] == 0)&(has_fiber)
    mock_full['IMATCH'][w] = 1

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


def add_redshift_failures(mock_full, data, seed=None):


    #-- All targers with fiber
    imatch = data['IMATCH']
    w_spec = (imatch==1) | \
             (imatch==4) | \
             (imatch==7) | \
             (imatch==9) | \
             (imatch==14) | \
             (imatch==15) 
    data_spec = data[w_spec] 

    #-- Match mock and data 
    id1, id2, dist = spherematch(mock_full['RA'], mock_full['DEC'], 
                                 data_spec['RA'], data_spec['DEC'], angle=0.2)
    nmatch = id1.size
    nmock = len(mock_full)
    print(f'  Number of mock galaxies matched to data: {nmatch}/{nmock} = {nmatch/nmock:.3f}')

    #-- Assign to mocks the following data quantities 
    fields_redshift_failures = ['XFOCAL', 'YFOCAL', 'PLATE', 'MJD', 'FIBERID', 
                                'SPEC1_G', 'SPEC1_R', 'SPEC1_I', 
                                'SPEC2_G', 'SPEC2_R', 'SPEC2_I', 
                                'WEIGHT_NOZ']
    for field in fields_redshift_failures:
        mock_full[field][id1] = data_spec[field][id2]
    mock_full.rename_column('WEIGHT_NOZ', 'WEIGHT_NOZ_DATA')

    #-- Define ssr: spectroscopic success rate using data weights
    ssr = 1/(mock_full['WEIGHT_NOZ_DATA'] + 1*(mock_full['WEIGHT_NOZ_DATA']==0))

    #-- Setting randomly failures to imatch = 7
    if not seed is None:
        np.random.seed(seed)
    w_failure = (np.random.rand(nmock) > ssr) & (mock_full['IMATCH']==1)
    mock_full['IMATCH'][w_failure] = 7

    #-- Setting targets without plate info to imatch = 0 
    w = mock_full['PLATE'] == 0 
    mock_full['IMATCH'][w] = 0


def get_systematic_weights(mock, rand, 
    target='LRG', zmin=0.6, zmax=1.0, 
    random_fraction=1, seed=None, nbins=20):
    

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
    data_ra, data_dec = mock['RA'].data.data[wd], mock['DEC'].data.data[wd]
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
        use_maps = ['STAR_DENTISY', 'EBV', 'PSF_G', 'SKY_G', 'DEPTH_G_MINUS_EBV', 'AIRMASS']
        fit_maps = ['STAR_DENSITY', 'DEPTH_G_MINUS_EBV']

    #-- Add the systematic maps we want 
    for syst_name in use_maps:
        s.add_syst(syst_name, data_syst[syst_name], rand_syst[syst_name])
    #-- Cut galaxies and randoms with some extreme values
    s.cut_outliers(p=0.5, verbose=1)
    #-- Perform global fit
    s.prepare(nbins=nbins)
    s.fit_minuit(fit_maps=fit_maps)
    #s.plot_overdensity(pars=[None, s.best_pars], ylim=[0.5, 1.5],
    #    title='global fit')

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
     
         
def fiber_completeness(mock):

    sectors = mock['SECTOR']

    im_good = [1, 3, 4, 7, 9]
    im_all = im_good + [0, -1, 14, 15]
    
    nmock = len(mock)
    w_good = np.zeros(nmock, dtype=bool)
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
        w_all |= (mock['IMATCH']==im)

    count_good = countdup(sectors, weights=w_good)
    count_all  = countdup(sectors, weights=w_all)

    sect_ssr = count_good/(count_all+1*(count_all==0))

    return sect_ssr 

    
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

def compute_nbar(mock, z_edges, shell_vol, P0=10000.):

    w = np.ones(len(mock), dtype=bool)
    if 'IMATCH' in mock.colnames:
        w = ((mock['IMATCH'] == 1) | (mock['IMATCH'] == 2)) 
        w &= (mock['sector_SSR'] > 0.5) 
        w &= (mock['COMP_BOSS'] > 0.5)
    
    z = mock['Z'][w]
    if 'WEIGHT_CP' in mock.colnames and 'sector_SSR' in mock.colnames:
        weight = mock['WEIGHT_CP'][w]/mock['sector_SSR'][w]
    else:
        weight = None
    
    counts, _ = np.histogram(z, bins=z_edges, weights=weight)
    nbar = counts/shell_vol
    vol_eff = shell_vol * (nbar * P0/(1+nbar * P0))**2

    z_cen = 0.5*(z_edges[:-1]+z_edges[1:])
   
    nbar_dict = {'z_cen': z_cen, 'z_edges': z_edges, 'nbar': nbar, 'shell_vol': shell_vol,
                 'weighted_gals': counts, 'vol_eff': vol_eff}

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

 
def read_data(cap):

    #-- Read real data 
    data_name = f'/mnt/lustre/eboss/DR16_LRG_data/v7/eBOSS_LRG_full_{cap}_v7.dat.fits'
    rand_name = f'/mnt/lustre/eboss/DR16_LRG_data/v7/eBOSS_LRG_full_{cap}_v7.ran.fits'

    print('Reading data from ', data_name)
    data = Table.read(data_name)
    print('Reading randoms from ', rand_name)
    rand = Table.read(rand_name)

    #-- Cut on COMP_BOSS > 0.5 (there's no mocks outside)
    wd = (data['COMP_BOSS']>0.5) & (data['sector_SSR']>0.5)
    data = data[wd]
    wr = (rand['COMP_BOSS']>0.5) & (rand['sector_SSR']>0.5)
    rand = rand[wr]

    #-- Removing NaN 
    w = np.isnan(data['Z'])
    data['Z'][w] = -1

    #-- Read mask files
    geo = '/mnt/lustre/eboss/DR16_geometry/eBOSS_LRGgeometry_v7.fits'
    print('Reading completeness info from ', geo)
    geometry = Table.read(geo) 
    sec_comp = unique(geometry['SECTOR', 'COMP_BOSS'])
    geo_sector = sec_comp['SECTOR']
    geo_comp   = sec_comp['COMP_BOSS']

    survey_geometry = '/mnt/lustre/eboss/DR16_geometry/eboss_geometry_eboss0_eboss27'
    mask_ply = pymangle.Mangle(survey_geometry+'.ply')
    mask_fits = Table.read(survey_geometry+'.fits')

    mask_dict = {'geo_sector': geo_sector, 
                 'geo_comp': geo_comp,
                 'mask_ply': mask_ply,
                 'mask_fits': mask_fits}

    nbar = read_nbar(f'/mnt/lustre/eboss/DR16_LRG_data/v7/nbar_eBOSS_LRG_{cap}_v7.dat')
        
    return data, rand, mask_dict, nbar
   
def get_contaminants(data, fields):

    #-- Getting contaminants from data
    imatch = data['IMATCH']
    wbad  = (imatch == 4) #-- stars
    wbad |= (imatch == 9) #-- qsos
    wbad |= (imatch == 14) #-- little coverage, unplugged, bad_target, no data
    wbad |= (imatch == 15) 
    bad_targets = data[wbad]
    bad_targets.keep_columns(fields)
    return bad_targets

def assign_sectors(mock, mask_dict):

    #-- Get information from mask 
    mock_polyid = mask_dict['mask_ply'].polyid(mock['RA'], mock['DEC'])
    mock['SECTOR'] = mask_dict['mask_fits']['SECTOR'][mock_polyid]
    mock['NTILES'] = mask_dict['mask_fits']['NTILES'][mock_polyid]
    mock['COMP_BOSS_IN'] = np.interp(mock['SECTOR'], 
                                     mask_dict['geo_sector'], 
                                     mask_dict['geo_comp']) 
    
 
def make_randoms(mock, rand, seed=None):

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
    if not seed is None:
        np.random.seed(seed)
    rand_new['Z'] = np.random.choice(z, size=len(rand), replace=True)

    return rand_new

def make_clustering_catalog(mock):

    w = ((mock['IMATCH']==1)|(mock['IMATCH']==2))
    w &= (mock['COMP_BOSS'] > 0.5)
    w &= (mock['sector_SSR'] > 0.5)
    mock_clust = mock[w]

    mock_clust.keep_columns(['RA', 'DEC', 'Z', 
        'WEIGHT_FKP', 'WEIGHT_NOZ', 'WEIGHT_CP', 'WEIGHT_SYSTOT', 
        'NZ'])

    return mock_clust

def make_clustering_catalog_random(rand, mock, seed=None):

    if not seed is None:
        np.random.seed(seed)
    
    index = np.arange(len(mock))
    ind = np.random.choice(index, size=len(rand), replace=True)

    rand_clust = Table()
    rand_clust['RA'] = rand['RA']*1
    rand_clust['DEC'] = rand['DEC']*1
    rand_clust['COMP_BOSS'] = rand['COMP_BOSS']*1
    
    fields = ['WEIGHT_FKP', 'WEIGHT_NOZ', 'WEIGHT_CP', 'WEIGHT_SYSTOT', 'NZ', 'Z']
    for f in fields:
        rand_clust[f] = mock[f][ind]

    w = (rand_clust['COMP_BOSS'] > 0.5)
    return rand_clust[w]


def get_imatch_stats(mock, zmin=0.55, zmax=1.05):

    imatch = np.unique(mock['IMATCH'])
    for im in imatch:
        w = mock['IMATCH'] == im
        print(im, np.sum(w))
        if im == 1:
            ww = (mock['Z'][w] >= zmin)&(mock['Z'][w] <= zmax)
            print(im, np.sum(ww), f'at {zmin} < z < {zmax}')

def main(cap, ifirst, ilast):
    data, rand_data, mask_dict, nbar_data = read_data(cap)

    #-- Get stars, bad targets, unpluggged fibers and the following columns
    fields_redshift_failures = ['XFOCAL', 'YFOCAL', 'PLATE', 'MJD', 'FIBERID', 
                                'SPEC1_G', 'SPEC1_R', 'SPEC1_I', 
                                'SPEC2_G', 'SPEC2_R', 'SPEC2_I', 
                                'WEIGHT_NOZ']
    fields_bad_targets = ['RA', 'DEC', 'Z', 'IMATCH', 'NTILES', 'SECTOR']  \
                         + fields_redshift_failures
    bad_targets = get_contaminants(data, fields_bad_targets)

    #-- Fit for photo-systematics on data and obtain a density model for sub-sampling mocks
    syst_obj, density_model_data = get_systematic_weights(data, rand_data)

    #-- Setup directories
    input_dir  = f'/mnt/lustre/eboss/EZ_MOCKs/EZmock_LRG_v7.0'
    output_dir = f'/mnt/lustre/eboss/EZ_MOCKs/EZmock_LRG_v7.0_syst'
    rand_name  = input_dir + f'/random/random_20x_eBOSS_LRG_{cap}_v7.fits'

    #-- Read randoms just once
    print('')
    print('Reading random from ', rand_name)
    rand0 = Table.read(rand_name)

    for imock in range(ifirst, ilast):
        mock_name_in  = input_dir + f'/eBOSS_LRG/EZmock_eBOSS_LRG_{cap}_v7_{imock:04d}.dat'
        mock_name_out = output_dir+ f'/eBOSS_LRG/EZmock_eBOSS_LRG_{cap}_v7_{imock:04d}.dat.fits'
        rand_name_out = output_dir+ f'/eBOSS_LRG/EZmock_eBOSS_LRG_{cap}_v7_{imock:04d}.ran.fits'

        #-- Read mock
        print('')
        print('Reading mock from', mock_name_in)
        mock0 = Table.read(mock_name_in, format='ascii', names=('RA', 'DEC', 'Z', 'WEIGHT_FKP'))
        w = np.isnan(mock0['RA']) | np.isnan(mock0['DEC']) | np.isnan(mock0['Z'])
        mock0 = mock0[~w]

        #-- Getting an estimate of how many galaxies are lost through photo systematics
        #-- but not yet perform sub-sampling
        w_photo = downsample_photo(density_model_data, mock0, seed=imock)
        photo_ratio = np.sum(w_photo)/len(mock0)
        print(f' Fraction of galaxies after photo syst downsampling: {photo_ratio:.4f}')

        #-- Divide the original density of the mock by photo_ratio
        nbar_mock0 = 3.2e-4 * photo_ratio
        nbar_rand0 = nbar_mock0 

        #-- Sub-sample to match n(z) of the data
        w_sub = downsample_nbar(nbar_data, nbar_mock0, mock0, seed=imock)
        mock = mock0[w_sub]
        w_sub = downsample_nbar(nbar_data, nbar_rand0, rand0, seed=imock)
        rand = rand0[w_sub]

        #-- Now actually perform sub-sampling of mock with photometric systematic map
        w_photo = downsample_photo(density_model_data, mock, seed=imock)
        mock = mock[w_photo]
        
        assign_sectors(mock, mask_dict)
        assign_sectors(rand, mask_dict)

        #-- Add missing fields to mock with zeros 
        for field in fields_bad_targets:
            if field not in mock.colnames:
                mock[field] = np.zeros(len(mock), dtype=data[field].dtype) 
    
       #-- Sub-sample following completeness
        print('')
        print('Sub-sampling by COMP_BOSS of data...')
        np.random.seed(imock)
        w_sub = np.random.rand(len(mock)) > mock['COMP_BOSS_IN']
        mock['IMATCH'][w_sub] = -1
        nlost = np.sum(w_sub)
        nmock = len(mock)
        print(f' Fraction lost by sub-sampling: {nlost}/{nmock} = {nlost/nmock:.3f}')

        #-- Add data contaminants to mock
        mock_full = vstack([mock, bad_targets])

        #-- Add fiber collisions
        print('')
        print('Adding fiber collisions...')
        add_fiber_collisions(mock_full)

        #-- Correcting for fiber collisions
        print('')
        print('Correcting fiber collisions...')
        correct_fiber_collisions(mock_full)

        #-- Add redshift failures
        print('')
        print('Adding redshift failures...')
        add_redshift_failures(mock_full, data, seed=imock)
     
        #-- Correct redshift failures
        print('')
        print('Correcting for redshift failures...')
        redshift_failures.get_weights_noz(mock_full)
        #redshift_failures.plot_failures(mock_full)

        mock_full['COMP_BOSS'] = fiber_completeness(mock_full) 
        mock_full['sector_SSR'] = sector_ssr(mock_full) 
      
        #-- Make random catalog for this mock 
        rand_new = make_randoms(mock_full, rand, seed=imock)

        #-- Compute nbar
        nbar_mock = compute_nbar(mock_full, nbar_data['z_edges'], nbar_data['shell_vol'])
        compute_weight_fkp(mock_full, nbar_mock)
        compute_weight_fkp(rand_new, nbar_mock)
     
        #-- Correct photometric systematics
        print('')
        print('Getting photo-systematic weights...')
        s, density_model_mock = get_systematic_weights(mock_full, rand_new,
                            target='LRG', zmin=0.6, zmax=1.0,
                            random_fraction=1, seed=imock, nbins=20)

        #-- Make clustering catalog
        mock_clust = make_clustering_catalog(mock_full)
        rand_clust = make_clustering_catalog_random(rand_new, mock_clust, seed=imock)

        #-- Export  
        print('')
        print('Exporting mock to', mock_name_out)
        mock_clust.write(mock_name_out, overwrite=True) 
        print('Exporting random to', mock_name_out)
        rand_clust.write(rand_name_out, overwrite=True) 
        

main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))



 
