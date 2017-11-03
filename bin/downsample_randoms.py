import numpy as N
import pylab as P

from ebosscat import Catalog, Mask

import xyradec
from eff_model import efficiency

cat = Catalog('catalogs/bautista/test12/ebosscat-test12-LRG-South-systz.dat.fits')
ran = Catalog('catalogs/bautista/test12/ebosscat-test12-LRG-South-systz.ran.fits')
mask = Mask.read_mangle_mask('catalogs/bautista/test12/fibercomp-test12-LRG-South')

xyradec.assign_plates_and_xy(cat, ran, mask)

photo=1 
spectro=1


if spectro:
    eff = efficiency(cat, ran, syst=1, zmin=0.6, zmax=1.0)
    eff.read_xy(cat)
    eff.plot_xy()
    eff.read_spectro(cat)
    eff.fit_spectro_eff(npoly=1, plotit=1)

w = (ran.Z>0.6)&(ran.Z<1.0)

photoeff = N.ones(ran.size)
speceff =  N.ones(ran.size)
if photo:
    photoeff = 1/ran.WEIGHT_SYSTOT
if spectro:
    speceff = eff.get_eff(ran.PLATE, ran.XFOCAL, ran.YFOCAL)
speceffnorm = speceff/N.median(speceff[w])

fulleff = speceffnorm*photoeff

P.figure()
if photo:
    P.hist(photoeff[w],    bins=N.linspace(0.5, 1.5, 1000), histtype='step', label='Photometric', normed=1)
if spectro:
    P.hist(speceffnorm[w], bins=N.linspace(0.5, 1.5, 1000), histtype='step', label='Spectroscopic', normed=1)
P.hist(fulleff[w],     bins=N.linspace(0.5, 1.5, 1000), histtype='step', label='Combined', normed=1)
P.legend(loc=0)

r = N.random.rand(ran.size)
wr = (r < fulleff/N.percentile(fulleff[w], 99) )
fulleff = fulleff[wr]
print 'Fraction of randoms after cut:', sum(w&wr)*1./sum(w)
ran.cut(wr)

#-- setting systematic weights to ones
cat.WEIGHT_SYSTOT = N.ones(cat.size)

suffix='s'*(spectro==1)+'p'*(photo==1)+'sub'
cat.export('catalogs/bautista/test12/ebosscat-test12-LRG-South-systz-%s.dat.fits'%suffix)
ran.export('catalogs/bautista/test12/ebosscat-test12-LRG-South-systz-%s.ran.fits'%suffix)

zs=[0.6, 0.67, 0.74, 0.81, 1.0] 
for i in range(len(zs)-1):
    wz = (ran.Z>zs[i])&(ran.Z<zs[i+1])
    figure()
    scatter(ra(ran.RA[wz]), ran.DEC[wz], c=fulleff[wz], lw=0, s=2, vmin=0.7, vmax=1.3)
    colorbar()
    title('%.2f < z < %.2f'%(zs[i], zs[i+1]))



