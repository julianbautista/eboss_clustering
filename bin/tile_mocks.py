from ebosscat import Catalog, Mask
from xyradec import *
import os


#-- the only part needed from the data to get plate numbers
cat = Catalog(os.environ['CLUSTERING_DIR']+\
    '/catalogs/bautista/test12/ebosscat-test12-LRG-South.dat.fits')


mockdir = os.environ['CLUSTERING_DIR']+'/mocks/lrgs/1.8'

#-- the mask is used only to get sector numbers
mask = Mask.read_mangle_mask(mockdir+'/eboss-veto/mask-1.8-LRG-South')


#-- trim randoms
if 1==0:
    mock = Catalog(mockdir+'/eboss-veto/catalogs/mock-1.8-LRG-South-0001.ran.fits')
    
    #-- Cut randoms outside mask (not sure why there are some...)
    wm = Mask.veto(mock.RA, mock.DEC, mask)
    print 'Outside mask', sum(~wm)
    mock.cut(wm)

    #-- Cut problematic region for NGC
    #wo = abs(mock.DEC-40.55)>0.05
    #print 'Inside overlap region', sum(wo)
    #mock.cut(wo)

    assign_plates_and_xy(cat, mock, mask)
    
    #-- Check for mock galaxies in sectors where there are no spectra on data
    wp = mock.PLATE>0
    print 'Without plate number', sum(~wp)
    mock.cut(wp)

    mock.export(mockdir+'/eboss-veto-trim2/catalogs/mock-1.8-LRG-South-50x.ran.fits')

#-- trim mock realizations
for i in range(2, 1001):
    mock = Catalog(mockdir+'/eboss-veto/catalogs/mock-1.8-LRG-South-%04d.dat.fits'%i)

    wm = Mask.veto(mock.RA, mock.DEC, mask)
    print 'Outside mask', sum(~wm)
    mock.cut(wm)
    
    #wo = abs(mock.DEC-40.55)>0.05
    #print 'Inside overlap region', sum(wo)
    #mock.cut(wo)
    
    assign_plates_and_xy(cat, mock, mask)
    
    wp = mock.PLATE>0
    print 'Without plate number', sum(~wp)
    mock.cut(wp)
    
    mock.export(mockdir+'/eboss-veto-trim2/catalogs/mock-1.8-LRG-South-%04d.dat.fits'%i)
    print i

