import numpy as N
import pylab as P
import os
import sys

from ebosscat import Catalog, Mask, Cosmo
from eff_model import efficiency

imock = int(sys.argv[1])

indir = os.environ['CLUSTERING_DIR']+'/mocks/lrgs/1.8/eboss-veto-trim2/catalogs'
mock = Catalog(indir+'/mock-1.8-LRG-North-%04d-wnoz.dat.fits'%imock)
ran = Catalog(indir+'/mock-1.8-LRG-North-50x.ran.fits')

photo=0 
spectro=1

correct_xyfocal=0
correct_spectro=1

plotit=0

if spectro:
    eff = efficiency(mock)
    if plotit:
        eff.plot_xy()
        P.figure()
        eff.plot_spectro_eff()

w = (ran.Z>0.6)&(ran.Z<1.0)

photoeff = N.ones(ran.size)
speceff =  N.ones(ran.size)
if photo:
    photoeff = 1/ran.WEIGHT_SYSTOT
if spectro:
    speceff = eff.get_eff(ran.PLATE, ran.XFOCAL, ran.YFOCAL, \
                          xyfocal=correct_xyfocal, \
                          spectro=correct_spectro)
speceffnorm = speceff/N.median(speceff[w])

fulleff = speceffnorm*photoeff

if plotit:
    P.figure()
    if photo:
        P.hist(photoeff[w],    bins=N.linspace(0.5, 1.5, 1000), \
                histtype='step', label='Photometric', normed=1)
    if spectro:
        P.hist(speceffnorm[w], bins=N.linspace(0.5, 1.5, 1000), \
                histtype='step', label='Spectroscopic', normed=1)
    P.hist(fulleff[w],     bins=N.linspace(0.5, 1.5, 1000), \
                histtype='step', label='Combined', normed=1)
    P.legend(loc=0)


N.random.seed(imock)
r = N.random.rand(ran.size)
wr = (r < fulleff/N.percentile(fulleff[w], 99) )
print 'Fraction of randoms after cut:', sum(w&wr)*1./sum(w)
ran.cut(wr)

if plotit:
    P.show()

suffix='s'*(spectro==1)+'p'*(photo==1)+'sub'
if spectro and ((correct_xyfocal==0) or (correct_spectro==0)):
    if correct_xyfocal:
        suffix=suffix+'-xyonly'
    if correct_spectro:
        suffix=suffix+'-speconly'

ran.export(indir+'/mock-1.8-LRG-North-%04d-%s.ran.fits'%(imock, suffix) )

