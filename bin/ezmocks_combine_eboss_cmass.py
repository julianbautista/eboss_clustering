#!/usr/bin/env python3
import sys
import os
import numpy as np
import pymangle
from astropy.table import Table, unique, vstack

#if len(sys.argv) != 8:
#  print('Usage: {} CMASS eBOSS ZMIN ZMAX eBOSS_POLY_BASE COMPLETENESS OUTPUT'.format(sys.argv[0]), file=sys.stderr)
#  sys.exit(1)

cap = sys.argv[1]
ifirst = int(sys.argv[2])
ilast = int(sys.argv[3])
zmin = float(sys.argv[4])
zmax = float(sys.argv[5])
options = sys.argv[6]
P0 = float(10000.)

for imock in range(ifirst, ilast):
    cmass_data = f'/mnt/lustre/eboss/EZ_MOCKs/EZmock_LRG_v5.0/CMASS_{cap}/EZmock_CMASS_LRG_{cap}_DR12v5_{imock:04d}.fits'
    cmass_rand = f'/mnt/lustre/eboss/EZ_MOCKs/EZmock_LRG_v5.0/RANDOM/random_20x_CMASS_LRG_{cap}_DR12v5.fits' 
    eboss_data = f'/mnt/lustre/eboss/EZ_MOCKs/EZmock_v7.1_{options}/eBOSS_LRG/EZmock_eBOSS_LRG_{cap}_v7_{imock:04d}.dat.fits'
    eboss_rand = f'/mnt/lustre/eboss/EZ_MOCKs/EZmock_v7.1_{options}/eBOSS_LRG/EZmock_eBOSS_LRG_{cap}_v7_{imock:04d}.ran.fits'
    eboss_comp = f'/mnt/lustre/eboss/EZ_MOCKs/EZmock_v7.1_{options}/eBOSS_LRG/EZmock_eBOSS_LRG_{cap}_v7_{imock:04d}.mask.fits'
    os.makedirs(f'/mnt/lustre/eboss/EZ_MOCKs/EZmock_v7.1_{options}/eBOSS_CMASS', exist_ok=True)
    out_root =   f'/mnt/lustre/eboss/EZ_MOCKs/EZmock_v7.1_{options}/eBOSS_CMASS/EZmock_eBOSS_LRGpCMASS_{cap}_v7_{imock:04d}'
    #eboss_data = f'/mnt/lustre/eboss/EZ_MOCKs/EZmock_LRG_v5.0/eBOSS_{cap}/EZmock_eBOSS_LRG_{cap}_v5_{imock:04d}.fits'
    #eboss_rand = f'/mnt/lustre/eboss/EZ_MOCKs/EZmock_LRG_v5.0/RANDOM/random_20x_eBOSS_LRG_{cap}_v5.fits'
    #eboss_comp = f'/mnt/lustre/eboss/DR16_LRG_data/v7/eBOSS_LRGgeometry_v7.fits'
    #out_root =   f'/mnt/lustre/eboss/EZ_MOCKs/EZmock_LRG_v7.0_syst/COMB_NOSYST/EZmock_eBOSS_LRGpCMASS_{cap}_v7_{imock:04d}'

    mply  = '/mnt/lustre/eboss/DR16_geometry/eboss_geometry_eboss0_eboss27.ply'
    mfits = '/mnt/lustre/eboss/DR16_geometry/eboss_geometry_eboss0_eboss27.fits'

    zbin = 0.01
    compmin = 0.5
    ssrmin = 0.5


    print('> Reading input files')
    try:
        cm_data = Table.read(cmass_data, format='ascii.no_header', 
            names=['RA','DEC','Z','WEIGHT_FKP','NZ'])
        cm_rand = Table.read(cmass_rand, format='ascii.no_header', 
            names=['RA','DEC','Z','WEIGHT_FKP','NZ'])
    except:
        cm_data = Table.read(cmass_data)
        cm_rand = Table.read(cmass_rand)

    eb_data = Table.read(eboss_data)
    eb_rand = Table.read(eboss_rand)

    print('> Cut in redshift')
    w = (eb_data['Z'] >= zmin) & (eb_data['Z'] <= zmax)
    print(f'  eboss data: {np.sum(w)} {w.size}')
    eb_data = eb_data[w]
    w = (eb_rand['Z'] >= zmin) & (eb_rand['Z'] <= zmax)
    print(f'  eboss rand: {np.sum(w)} {w.size}')
    eb_rand = eb_rand[w]
    w = (cm_data['Z'] >= zmin) & (cm_data['Z'] <= zmax)
    print(f'  cmass data: {np.sum(w)} {w.size}')
    cm_data = cm_data[w]
    w = (cm_rand['Z'] >= zmin) & (cm_rand['Z'] <= zmax)
    print(f'  cmass rand: {np.sum(w)} {w.size}')
    cm_rand = cm_rand[w]
    

    print('> Correcting ndata/nrand ratios')
    def get_ratios(eb_data, eb_rand, cm_data, cm_rand):
        if 'WEIGHT_CP' in eb_data.colnames:
            eb_ndata = np.sum(eb_data['WEIGHT_SYSTOT']*eb_data['WEIGHT_CP']*eb_data['WEIGHT_NOZ'])
            eb_nrand = np.sum(eb_rand['WEIGHT_SYSTOT']*eb_rand['WEIGHT_CP']*eb_rand['WEIGHT_NOZ'])
        else:
            eb_ndata = len(eb_data)
            eb_nrand = len(eb_rand)
        cm_ndata = len(cm_data)
        cm_nrand = len(cm_rand)
        eb_ratio = (eb_nrand/eb_ndata)
        cm_ratio = (cm_nrand/cm_ndata)
        return eb_ratio, cm_ratio

    eb_ratio, cm_ratio = get_ratios(eb_data, eb_rand, cm_data, cm_rand)
    print(f'  ratios before: {eb_ratio:.2f} {cm_ratio:.2f}')
    np.random.seed(imock)
    if eb_ratio > cm_ratio:
        w = np.random.rand(len(eb_rand)) < cm_ratio/eb_ratio
        eb_rand = eb_rand[w]
    else:
        w = np.random.rand(len(cm_rand)) < eb_ratio/cm_ratio
        cm_rand = cm_rand[w]
    eb_ratio, cm_ratio = get_ratios(eb_data, eb_rand, cm_data, cm_rand)
    print(f'  ratios after: {eb_ratio:.2f} {cm_ratio:.2f}')
        

    m = pymangle.Mangle(mply)
    geo_sector = Table.read(mfits, format='fits', hdu=1)['SECTOR']
    try:
        mask = Table.read(eboss_comp, format='fits', hdu=1)['SECTOR','COMP_BOSS','sector_SSR']
    except:
        mask = Table.read(eboss_comp, format='fits', hdu=1)['SECTOR','COMP_BOSS', 'SPEC_SUCCESS_RATE']
        mask.rename_column('SPEC_SUCCESS_RATE', 'sector_SSR')
    mask = unique(mask)

    for cm, eb in zip([cm_data, cm_rand], [eb_data, eb_rand]):

        print('> Getting eBOSS sectors for CMASS data')
        ids = m.polyid(cm['RA'], cm['DEC'])
        sectors = geo_sector[ids]*(ids!=-1) -1*(ids==-1)
        infoot = (ids!=-1)

        print('> Finding completeness of sectors')
        comp_boss = np.interp(sectors, mask['SECTOR'], mask['COMP_BOSS'])*(sectors>0)
        ssr = np.interp(sectors, mask['SECTOR'], mask['sector_SSR'])*(sectors>0)

        complow = ((comp_boss <= compmin) | (ssr <= ssrmin))
        infoot[complow] = False
        nwt = np.sum(infoot) 
        ncm = len(cm)
         
        print('--> {0:d} out of {1:d} CMASS galaxies in eBOSS footprint'.format(nwt, ncm))

        cm['SAMPLE'] = ['CMASS'] * ncm
        cm['INFOOT'] = infoot
        cm['SECTOR'] = sectors
        cm['WEIGHT_CP'] = np.ones(ncm) 
        cm['WEIGHT_NOZ'] = np.ones(ncm)
        cm['WEIGHT_SYSTOT'] = np.ones(ncm)

        neb = len(eb)
        ids = m.polyid(eb['RA'], eb['DEC'])
        eb_sectors = geo_sector[ids]
        eb['SAMPLE'] = ['eBOSS'] * neb
        eb['INFOOT'] = (ids!=-1)
        eb['SECTOR'] = eb_sectors
        if not 'WEIGHT_CP' in eb.colnames:
            eb['WEIGHT_CP'] = np.ones(neb) 
            eb['WEIGHT_NOZ'] = np.ones(neb)
            eb['WEIGHT_SYSTOT'] = np.ones(neb)

    print('> Computing n(z)')
    nzbins = np.int((zmax-zmin)/zbin)
    zbcmeb = np.floor((cm_data['Z']-zmin)/zbin).astype(int)
    nzsum = np.bincount(zbcmeb, minlength=nzbins, weights=cm_data['NZ'])
    nzn = np.bincount(zbcmeb, minlength=nzbins)
    nz_cmass = nzsum/(nzn+1*(nzn==0))*(nzn>0)

    zbeb = np.floor((eb_data['Z']-zmin)/zbin).astype(int)
    nzsum = np.bincount(zbeb, minlength=nzbins, weights=eb_data['NZ'])
    nzn = np.bincount(zbeb, minlength=nzbins)
    nz_eboss = nzsum/(nzn+1*(nzn==0))*(nzn>0)

    nz_comb = nz_eboss + nz_cmass

    for cm, eb, suffix in zip([cm_data, cm_rand], [eb_data, eb_rand], ['.dat.fits', '.ran.fits']):
        combgal = vstack([eb, cm])
        web = (combgal['INFOOT'] == True)

        index = np.floor((combgal['Z'][web]-zmin) / zbin).astype(int)
        combgal['NZ'][web] = nz_comb[index]

        combgal['WEIGHT_FKP'] = 1.0 / (1.0 + combgal['NZ'] * P0)
        print('Exporting to', out_root+suffix)
        combgal['RA','DEC','Z','WEIGHT_FKP','WEIGHT_CP','WEIGHT_NOZ','NZ', \
            'WEIGHT_SYSTOT','SAMPLE'].write(out_root+suffix, format='fits', overwrite=True)


