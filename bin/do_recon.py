import argparse
import numpy as np
from astropy.table import Table
from astropy.io import fits
from recon import Recon

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', help='Input data catalog')
parser.add_argument('-r', '--randoms', help='Input random catalog')
parser.add_argument('-o', '--output', help='Output catalogs root name')
parser.add_argument('--nthreads', \
    help='Number of threads', type=int, default=1)
parser.add_argument('--niter', \
    help='Number of iterations', type=int, default=3)
parser.add_argument('--nbins', \
    help='Number of bins for FFTs', type=int, default=512)
parser.add_argument('--padding', default=200., \
    help='Size in Mpc/h of the zero padding region', type=float)
parser.add_argument('--zmin', help='Redshift lower bound', type=float)
parser.add_argument('--zmax', help='Redshift upper bound', type=float)
parser.add_argument('--smooth', help='Smoothing scale in Mpc/h', \
    type=float, default=15.)
parser.add_argument('--bias', \
    help='Estimate of the bias of the sample', type=float, required=True)
parser.add_argument('--f', \
    help='Estimate of the growth rate', type=float, required=True, default=0.817)
parser.add_argument('--cmass', \
    help='Add this option to work with eboss+cmass catalogs', type=bool, default=False)
args = parser.parse_args()

argnames = np.sort(np.array([arg for arg in args.__dict__]))
for arg in argnames:
    print(arg, ':', args.__dict__[arg])





nbins=args.nbins
nthreads=args.nthreads
padding =args.padding 
zmin=args.zmin
zmax=args.zmax
smooth=args.smooth
bias = args.bias
f = args.f
opt_box = 1 #optimize box dimensions


#-- Reading data and randoms
data = Table.read(args.data)
rand = Table.read(args.randoms)

#-- Defining weights
if args.cmass:
    data_we = data['WEIGHT_FKP']*data['WEIGHT_ALL_NOFKP']
    rand_we = rand['WEIGHT_FKP']*rand['WEIGHT_ALL_NOFKP']
else:
    data_we = data['WEIGHT_FKP']*data['WEIGHT_SYSTOT']*\
              data['WEIGHT_NOZ']*data['WEIGHT_CP']
    rand_we = rand['WEIGHT_FKP']*rand['WEIGHT_SYSTOT']*\
              rand['WEIGHT_NOZ']*rand['WEIGHT_CP']


#-- Performing reconstruction
rec = Recon(data['RA'], data['DEC'], data['Z'], data_we, \
            rand['RA'], rand['DEC'], rand['Z'], rand_we, \
            nbins=nbins, smooth=smooth, f=f, bias=bias, \
            padding=padding, opt_box=opt_box, nthreads=nthreads)
for i in range(args.niter):
    rec.iterate(i)
rec.apply_shifts()
rec.summary()

data['RA'], data['DEC'], data['Z'] = rec.get_new_radecz(rec.dat)
rand['RA'], rand['DEC'], rand['Z'] = rec.get_new_radecz(rec.ran) 

data.write(args.output+'.dat.fits', format='fits', overwrite=True)
rand.write(args.output+'.ran.fits', format='fits', overwrite=True)


