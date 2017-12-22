
import os

from astropy.io import fits
from ebosscat import *


def trim_cmass():
    ''' Cuts CMASS sample into this mask
    Input
    -----
    mask: string
        Mask filename with format path/masktype-version-target-cap (no extension) 

    '''
  
    outdir = os.environ['MKESAMPLE_DIR']+'/inputFiles/DR12'
    mask = Mask.read_mangle_mask(Mask.geometry_file)

    for cap in ['North', 'South']: 

        cmass_data_in  = os.environ['MKESAMPLE_DIR']+\
                        '/inputFiles/DR12/galaxy_DR12v5_CMASSLOWZTOT_%s.fits'%cap
        cmass_data_out = os.environ['MKESAMPLE_DIR']+\
                        '/inputFiles/DR12/galaxy_DR12v5_CMASSLOWZTOT_eboss0-22_%s.fits'%cap

        data = fits.open(cmass_data_in)[1].data

        print 'CMASS data sample:', data.size

        w = (data.Z > 0.6)
        data = data[w]
        print '  at z>0.6: ', data.size

        w = Mask.veto(data.RA, data.DEC, mask)
        data = data[w]
        print '  inside eboss geometry:', data.size
       
        print '  adding 100 for imatch values'
        data.IMATCH += 100

        tbhdu = fits.BinTableHDU(data=data)
        tbhdu.writeto(cmass_data_out, overwrite=True)

        cmass_rand_in  = os.environ['MKESAMPLE_DIR']+\
                        '/inputFiles/DR12/random1_DR12v5_CMASSLOWZTOT_%s.fits.gz'%cap
        cmass_rand_out = os.environ['MKESAMPLE_DIR']+\
                        '/inputFiles/DR12/random1_DR12v5_CMASSLOWZTOT_eboss0-22_%s.fits.gz'%cap

        print 'Reading random catalog...'
        data = fits.open(cmass_rand_in)[1].data

        print 'CMASS random sample:', data.size

        w = data.Z>0.6 
        data = data[w]
        print '  at z>0.6: ', data.size

        w = Mask.veto(data.RA, data.DEC, mask)
        data = data[w] 
        print '  inside eboss geometry:', data.size

        tbhdu = fits.BinTableHDU(data=data)
        tbhdu.writeto(cmass_rand_out, overwrite=True)
        print 'Done!'

def merge_eboss_cmass(seed=1):

    print 'Merging CMASS with eBOSS catalogs'

    indir = os.environ['CLUSTERING_DIR']+'/catalogs/bautista/v1.0'

    cat_root = '/ebosscat-v1.0-LRG-%s-spsub'
    ran_root = '/ebosscat-v1.0-LRG-%s-spsub'
    mask_root = '/fibercomp-v1.0-LRG-%s'
    nbar_root = '/nbar-v1.0-LRG-%s.dat'

    for cap in ['North', 'South']:
    
        print 'eboss catalog: %s'%(indir+cat_root%cap+'.dat.fits')
        #-- read eboss catalog
        cat0 = Catalog(indir+cat_root%cap+'.dat.fits')
        ran0 = Catalog(indir+ran_root%cap+'.ran.fits')
        mask = Mask.read_mangle_mask(indir+mask_root%cap)

        w0 = (cat0.IMATCH==1)|(cat0.IMATCH==2)
        cat0.cut(w0)
        print 'eBOSS objects in mask:', cat0.size

        #-- read cmass catalog
        cmass_data = os.environ['MKESAMPLE_DIR']+\
                     '/inputFiles/DR12/'+\
                     'galaxy_DR12v5_CMASSLOWZTOT_eboss0-22_%s.fits'%cap
        cmass_randoms = os.environ['MKESAMPLE_DIR']+\
                     '/inputFiles/DR12/'+\
                     'random1_DR12v5_CMASSLOWZTOT_eboss0-22_%s.fits.gz'%cap
        print 'cmass catalog: %s'%cmass_data
        cat1 = Catalog(cmass_data)
        ran1 = Catalog(cmass_randoms)

        #-- selecting CMASS objects in eBOSS mask only
        win, sectors = Mask.veto(cat1.RA, cat1.DEC, mask, get_sectors=1)
        cat1.SECTOR = sectors
        cat1.cut(win)
        cat1.version = cat0.version
        cat1.cap = cap
        cat1.target = 'LRG'
        cat1.P0 = cat0.P0
        print 'CMASS objects in mask:', cat1.size
        
        win, sectors = Mask.veto(ran1.RA, ran1.DEC, mask, get_sectors=1)
        ran1.SECTOR = sectors
        ran1.cut(win)
        ran1.version = cat0.version
        ran1.cap = cap
        ran1.target = 'LRG'
        ran1.P0 = cat0.P0


        weights0 = cat0.get_weights()
        weights1 = cat1.get_weights()
        frac0 = ran0.size*1./cat0.size
        frac1 = ran1.size*1./cat1.size
        wfrac0 = ran0.size/sum(weights0)
        wfrac1 = ran1.size/sum(weights1)
        print 'eboss data/random:', frac0, wfrac0 
        print 'cmass data/random:', frac1, wfrac1

        #-- sum nbar and compute new FKP weights
        n_boss = Nbar(os.environ['MKESAMPLE_DIR']+\
                    '/inputFiles/DR12/nbar_DR12v5_CMASSLOWZ_%s_om0p31_Pfkp10000.dat'%cap )
        n_eboss = Nbar(indir+nbar_root%cap)

        n_tot = copy.deepcopy(n_eboss)
        n_tot.nbar += N.interp(n_eboss.zcen, n_boss.zcen, n_boss.nbar) 
        n_tot.wfkp = 1./(1+n_tot.nbar*cat0.P0) 
        
        n_tot.export(indir+nbar_root.replace('nbar-', 'nbar-ebosscmass-')%cap) 

        #-- trim duplicates (neglect for now!)
        
        #-- sub-sample randoms s
        N.random.seed(seed)
        if wfrac1 > wfrac0:
            rr = N.random.rand(ran1.size)
            wsub = (rr < wfrac0/wfrac1)
            print 'Downsampling cmass randoms by', sum(wsub)*1./ran1.size
            ran1.cut(wsub)
        else:
            rr = N.random.rand(ran0.size)
            wsub = (rr < wfrac1/wfrac0)
            print 'Downsampling eboss randoms by', sum(wsub)*1./ran0.size
            ran0.cut(wsub)

        frac0 = ran0.size*1./cat0.size
        frac1 = ran1.size*1./cat1.size
        wfrac0 = ran0.size/sum(weights0)
        wfrac1 = ran1.size/sum(weights1)
        print 'eboss data/random:', frac0, wfrac0 
        print 'cmass data/random:', frac1, wfrac1

        #-- merge catalogs
        cat = Catalog.merge_catalogs(cat0, cat1)
        cat.assign_fkp_weights(n_tot)
        
        ran = Catalog.merge_catalogs(ran0, ran1)
        ran.assign_fkp_weights(n_tot)
       
        #-- exporting merged catalog
        outroot = cat_root.replace('ebosscat', 'cmassebosscat')
        cat.export(indir+outroot%cap+'.dat.fits')
        ran.export(indir+outroot%cap+'.ran.fits')

        #-- exporting CMASS only 
        cmassroot= cat_root.replace('ebosscat', 'cmass')
        cat1.export(indir+cmassroot%cap+'.dat.fits')
        ran1.export(indir+cmassroot%cap+'.ran.fits')

if __name__=='__main__':
    merge_eboss_cmass()

