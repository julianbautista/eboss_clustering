from ebosscat import Catalog
from recon import Recon
import argparse

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
args = parser.parse_args()
print args




cat = Catalog(args.data) 
ran = Catalog(args.randoms) 

nbins=args.nbins
nthreads=args.nthreads
padding =args.padding 
zmin=args.zmin
zmax=args.zmax
smooth=args.smooth
bias = args.bias
f = args.f
opt_box = 1 #optimize box dimensions

#-- selecting galaxies
w = (cat.IMATCH==1)|(cat.IMATCH==2)|(cat.IMATCH==101)|(cat.IMATCH==102)
w = w & ((cat.Z>=zmin)&(cat.Z<=zmax))
cat.cut(w)
wr = ((ran.Z>=zmin)&(ran.Z<=zmax))
ran.cut(wr)

rec = Recon(cat, ran, nbins=nbins, smooth=smooth, f=f, bias=bias, \
            padding=padding, opt_box=opt_box, nthreads=nthreads)
for i in range(args.niter):
    rec.iterate(i)
rec.apply_shifts()
rec.summary()

cat.RA, cat.DEC, cat.Z = rec.get_new_radecz(rec.cat) 
ran.RA, ran.DEC, ran.Z = rec.get_new_radecz(rec.ran) 

cat.export(args.output+'.dat.fits')
ran.export(args.output+'.ran.fits')


