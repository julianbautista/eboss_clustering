
#-- removes SEQUELS spectra (boss214 and boss217) from the region outside eboss0
from mangle import Mangle
from astropy.io import fits
import os

ebosstiledir = '/uufs/chpc.utah.edu/common/home/sdss05/software/svn.sdss.org/repo/eboss/ebosstilelist/trunk'
eb0 = Mangle(ebosstiledir+'/outputs/eboss0/geometry-eboss0.ply')


rmall = fits.open(os.environ['EBOSS_CLUSTERING_DIR']+\
        '/redmonster_spAll_v5_10_0_v1_2_0.fits')[1].data


pix = eb0.get_polyids(rmall.RA, rmall.DEC)

w = ((rmall.CHUNK == 'boss214')|(rmall.CHUNK=='boss217'))
wout = (pix<0)&w

print rmall.size, 'spectra',  sum(~wout), 'being cut'
rmall = rmall[~wout]

fits.writeto(os.environ['EBOSS_CLUSTERING_DIR']+\
        '/redmonster_spAll_v5_10_0_v1_2_0_eboss0corrected.fits', data=rmall)

