from __future__ import print_function

import os, sys
import numpy as np
import pylab as plt
import healpy as hp
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import minimize
import argparse

import imaging_systematics


#-- Main starts here
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data',    
    help='Input data catalog')
parser.add_argument('-r', '--randoms', 
    help='Input random catalog', default=None)
parser.add_argument('-o', '--output', default=None,
    help='Output catalogs name without extension (will create .dat.fits and .ran.fits')
parser.add_argument('--read_maps', nargs='+', 
    help='List of maps to be read.', 
    default=['STAR_DENSITY',
             'AIRMASS',
             'EBV',
             'DEPTH_Z',
             'PSF_Z',
             'W1_MED',
             'W1_COVMED'])
parser.add_argument('--fit_maps', nargs='+', 
    help='List of maps to be fitted. Default fits for all maps.', default=None)
parser.add_argument('--input_fits', default=None,
    help='Input fits file with systematics maps')
parser.add_argument('--nbins_per_syst', type=int, default=20, 
    help='Number of bins per systematic quantity')
parser.add_argument('--zmin', type=float, default=0.6,
    help='Minimum redshift')
parser.add_argument('--zmax', type=float, default=1.0,
    help='Maximum redshift')
parser.add_argument('--plot_deltas', action='store_true', default=False,
    help='If set, plots the delta vs systematic')
parser.add_argument('--save_plot_deltas', default=None,
    help='Filename for saving plot of deltas vs systematic (e.g. delta-vs-syst.pdf)')
parser.add_argument('--random_fraction', type=float, default=1., 
    help='Fraction of randoms to be used. Default = 1 (use all of them)')
parser.add_argument('--nest', action='store_true', default=False, 
    help='Include this option if systematic maps in input_fits are in NESTED scheme')
parser.add_argument('--export', type=str, 
    help='Export density vs systematic values into text file')
args = parser.parse_args()


#-- Read data and randoms
print('Reading galaxies from ',args.data)
dat = Table.read(args.data)
if args.randoms is not None:
    ran_file = args.randoms 
else:
    ran_file = args.data.replace('.dat.fits', '.ran.fits')
print('Reading randoms  from ', ran_file)
ran = Table.read(ran_file)


#-- Cut the sample 
print('Cutting galaxies and randoms between zmin=%.3f and zmax=%.3f'%\
      (args.zmin, args.zmax))
wd = ((dat['IMATCH']==1)|(dat['IMATCH']==2))&\
     (dat['Z']>=args.zmin)&\
     (dat['Z']<=args.zmax)&\
     (dat['COMP_BOSS']>0.5)&\
     (dat['sector_SSR']>0.5) 
wr = (ran['Z']>=args.zmin)&\
     (ran['Z']<=args.zmax)&\
     (ran['COMP_BOSS']>0.5)&\
     (ran['sector_SSR']>0.5)&\
     (np.random.rand(len(ran))<args.random_fraction)

#-- Defining RA, DEC and weights
data_ra, data_dec = dat['RA'][wd], dat['DEC'][wd]
rand_ra, rand_dec = ran['RA'][wr], ran['DEC'][wr]
data_we = (dat['WEIGHT_CP']*dat['WEIGHT_FKP']/dat['sector_SSR'])[wd]
rand_we = (ran['COMP_BOSS'])[wr]#*ran['WEIGHT_FKP'])[wr]

m = imaging_systematics.MultiLinearFit(
        data_ra=data_ra, data_dec=data_dec, data_we=data_we,
        rand_ra=rand_ra, rand_dec=rand_dec, rand_we=rand_we,
        maps = args.read_maps, 
        nbins_per_syst = args.nbins_per_syst,
        infits = args.input_fits,
        nest=args.nest)

#-- Perform the fit
m.fit_pars(fit_maps=args.fit_maps)


print('Assigning weights to galaxies' )
dat['WEIGHT_SYSTOT'] = m.get_weights(dat['RA'], dat['DEC'])
#ran['WEIGHT_SYSTOT'] = m.get_weights(ran['RA'], ran['DEC'])

#-- Make plots
if args.plot_deltas:
    print('Plotting deltas versus systematics')
    m.plot_overdensity(ylim=[0.5, 1.5])
    plt.tight_layout()
    if args.save_plot_deltas:
        plt.savefig(args.save_plot_deltas)
    plt.show()

#-- Export to table
if args.export:
    print('Exporting to', args.export)
    m.export(args.export)
    
#-- Export catalogs
if args.output:
    print('Exporting catalogs to ', args.output)
    dat.write(args.output+'.dat.fits', overwrite=True)
    ran.write(args.output+'.ran.fits', overwrite=True)

 
