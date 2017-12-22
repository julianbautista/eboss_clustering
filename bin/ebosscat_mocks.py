from ebosscat import *
from pbs import queue
import sys

def make_mock_survey(realization, maskroot, outdir, do_randoms=True, rancat_in='', \
                        do_veto=0, OmegaM=0.29, nran=50., cmass=0):
    ''' Run make_survey and create mock catalogs from HOD populated QPM halos
        
        Parameters:
        ----------
        realization: integer from 1 to 1000
        maskroot : string
            Mask name containing completeness used to subsample the mocks and create randoms
        outdir : string
            Folder where catalogs will be written as
            outdir/mock-VERSION-TARGET-CAP-NNNN.dat.fits
            outdir/mock-VERSION-TARGET-CAP-NNNN.ran.fits
            outdir/nbar-VERSION-TARGET-CAP-NNNN.txt
        do_randoms : bool 
            If True, will create random catalog from mask provided in maskroot
            If False, needs an input random catalog passed through `rancat_in`
        rancat_in : string
            If do_randoms=False, a symlink will be created pointing to this random catalog
        do_veto : bool
            If True, run vetos
        OmegaM : value for the cosmology used in the nbar calculation
        nran: float
            Factor for random catalog generation
        cmass: bool
            If True, make combined CMASS+eBOSS mocks

        Examples:
        ---------
        1)
        clustdir = '/uufs/astro.utah.edu/common/uuastro/astro_data/kdawson/bautista/clustering'
        maskroot = clustdir+'/catalogs/bautista/test5/mask-test5-LRG-North'
        outdir = clustdir+'/mocks/lrgs/test5'
        rancat = outdir+'/mock-test5-LRG-North-0001.ran.fits'
        make_mock_survey(2, maskroot, outdir, do_veto=0, do_randoms=0, rancat_in=rancat)
        
        2)
        clustdir = '/uufs/astro.utah.edu/common/uuastro/astro_data/kdawson/bautista/clustering'
        maskroot = clustdir+'/catalogs/bautista/test5/mask-test5-LRG-South'
        outdir = clustdir+'/mocks/lrgs/test5'
        make_mock_survey(1, maskroot, outdir)

    '''

    #-- reading info from maskroot name
    masktype, version, target, cap = os.path.basename(maskroot).split('-')
  
    #-- output files
    nbar_file = outdir+'/nbar-%s-%s-%s-%04d.txt' % \
                   (version, target, cap, realization) 
    cat_dat_file = outdir+'/mock-%s-%s-%s-%04d.dat.fits' % \
                   (version, target, cap, realization)
    cat_ran_file = outdir+'/mock-%s-%s-%s-%04d.ran.fits' % \
                   (version, target, cap, realization)
    
    #-- make surveyfile from mockfile
    surveyfile = 'survey-%s-%s-%s-%04d.dat'%(version, target, cap, realization)  

    paramfile = 'eboss'
    if cmass:
        paramfile += '_boss'
    if cap=='North':
        paramfile += '_ngc_%s.param'%version
    elif cap=='South':
        paramfile += '_sgc_%s.param'%version
    else: 
        print 'What cap is this?', cap
        return     

    #-- run make_survey
    mockfolder = os.environ['ASTRODATA']+'/software/mockFactory/QPM_mocks/'
    if cmass:
        mockfolder += 'cmass-eboss-mocks'
    else:
        mockfolder += 'eboss-mocks'

    mockfile= mockfolder+'/a0.5882_%04d.mock'%realization
    script = 'run_make_survey %s %s %s %s' % \
             (paramfile, maskroot+'.ply', mockfile, surveyfile)
    print script
    os.system(script)

    #-- read make_survey output
    cat = Catalog()
    cat.read_mock(surveyfile) 
    cat.version = version
    cat.target = target
    cat.cap = cap
    if do_veto:
        cat.veto()
        w = cat.vetobits==0
        cat.cut(w)
    cat.trim_catalog()

    #-- cleaning raw output from make_survey
    os.system('rm %s'%surveyfile)
   
    if do_randoms: 
        rancat = cat.create_randoms(nran, maskroot, do_veto=do_veto)
    else:
        rancat = Catalog(rancat_in)

    #-- compute effective area (needed for nbar to assign random redshifts)
    mask = Mask.read_mangle_mask(maskroot) 
    cat.compute_area(rancat, mask) 
    

    #-- compute nbar, fkp weights and attribute redshifts to randoms
    cosmo = Cosmo(OmegaM=OmegaM)
    nbar = cat.compute_nbar(cosmo=cosmo, noz=0)
    nbar.export(nbar_file)
    cat.assign_fkp_weights(nbar)
    if do_randoms:
        cat.assign_random_redshifts(rancat, nbar, noz=0, seed=323466458)
        rancat.assign_fkp_weights(nbar)

    #-- writting fits files
    cat.export(cat_dat_file, cosmo=cosmo)
    if do_randoms:
        rancat.export(cat_ran_file, cosmo=cosmo)


def make_many_mocks(outdir, mask_root=os.environ['EBOSS_LSS_CATALOGS']+\
                    '/1.5/mask-lrg-N-eboss_v1.5_IRt', \
                    version='1.5', target='LRG', cap='North',  \
                    nmocks=1000, cmass=0, do_veto=0):

    #-- making symlink of mask with name matching convention of make_mock_survey()
    mask_out = outdir+'/mask-%s-%s-%s'%(version, target, cap)

    for suffix in ['.fits', '.ply']:
        print mask_root+suffix
        os.system('ln -sf %s %s'%(mask_root+suffix, mask_out+suffix))

    #-- make first mock and random catalog
    make_mock_survey(1,  mask_out, outdir, do_randoms=True, \
                     do_veto=do_veto, OmegaM=0.29, nran=50., cmass=cmass)


    #-- make other mocks and symlink the random catalog
    rancat_in = outdir+'/mock-%s-%s-%s-0001.ran.fits'%(version, target, cap)
    for r in range(2, nmocks+1):
        make_mock_survey(r,  mask_out, outdir, do_randoms=False, \
                         rancat_in=rancat_in, do_veto=do_veto,   \
                         OmegaM=0.29, cmass=cmass)


def make_many_mocks_batch(outdir, version='1.8', target='LRG', \
                            nmocks=1000, cmass=0, do_veto=0):

    qe = queue()
    qe.verbose = True
    qe.create(label='galaxy_mocks_1.8', alloc='sdss-kp', nodes=10, ppn=16, \
              walltime='200:00:00',umask='0027')

    for cap in ['North', 'South']:
        mask = outdir+'/mask-%s-%s-%s'%(version, target, cap)

        for realiz in range(2, nmocks+1):
            rancat_in = outdir+'/mock-%s-%s-%s-0001.ran.fits' % \
                        (version, target, cap)
        
            script = "cd %s; "%(os.environ['MKESAMPLE_DIR'])
            script += "python python/ebosscat_mocks.py %d %s %s %s %d %d"% \
                      (realiz, mask, outdir, rancat_in, do_veto, cmass) 
            print script
            qe.append(script)
    qe.commit(hard=True,submit=True)

if __name__=='__main__':

    if len(sys.argv)==7:
        realiz = int(sys.argv[1])
        mask = sys.argv[2]
        outdir = sys.argv[3]
        rancat = sys.argv[4]
        do_veto = int(sys.argv[5])
        cmass = int(sys.argv[6])
        make_mock_survey(realiz, mask, outdir, do_randoms=False, \
                         rancat_in=rancat,\
                         do_veto=do_veto, OmegaM=0.29, cmass=cmass) 

def make_official_mocks(batch=1):
    print ''
    print '=============================='
    print '===== Making mocks ==========='
    print '=============================='
    print ''

    version='1.8'
    target='LRG'
    nmocks=1000
    do_veto=1   
    with_cmass=0

    mask_north_root=os.environ['EBOSS_LSS_CATALOGS']+\
                    '/%s/mask-lrg-N-eboss_v%s_IRt'%(version, version)
    mask_south_root=os.environ['EBOSS_LSS_CATALOGS']+\
                    '/%s/mask-lrg-S-eboss_v%s_IRt'%(version, version)

    outdir='/uufs/astro.utah.edu/common/uuastro/astro_data/kdawson/bautista/'+\
           'clustering/mocks/lrgs/%s/'%version

    outdir_eboss_boss = outdir+'cmass-eboss' + ('-veto'*do_veto) + \
                        ('-noveto'*(1-do_veto))
    outdir_eboss =      outdir+'eboss' +       ('-veto'*do_veto) + \
                        ('-noveto'*(1-do_veto))
    
    if with_cmass:
        outdir = outdir_eboss_boss
    else:
        outdir = outdir_eboss

    if batch:
        make_many_mocks_batch(outdir, version=version, target=target,\
                              cmass=with_cmass, do_veto=do_veto, nmocks=nmocks)
    else:
        make_many_mocks(outdir, mask_root=mask_north_root, version=version, \
                        target=target, cap='North', cmass=with_cmass,       \
                        do_veto=do_veto, nmocks=1)
        make_many_mocks(outdir, mask_root=mask_south_root, version=version, \
                        target=target, cap='South', cmass=with_cmass,       \
                        do_veto=do_veto, nmocks=1)


def create_photometric_catalog(maskroot, do_veto=False):
    ''' Produce a data and random catalog only with targets for angular correlations
        
        Parameters:
        maskroot:  string
            name of mask in the form  PATH/mask-VERSION-TARGET-CAP (no extension)
        do_veto: True or False
            run vetos (it takes few minutes)
            
        Outputs:
            It writes catalogs as 
            PATH/catalog0-VERSION-TARGET-CAP.dat.fits
            PATH/catalog0-VERSION-TARGET-CAP.ran.fits

        Examples:
            indir = '/uufs/astro.utah.edu/common/uuastro/astro_data/bautista/clustering/catalogs/bautista/test51'
            create_photometric_catalog(indir+'/mask-test51-LRG-North')
            create_photometric_catalog(indir+'/mask-test51-LRG-South')
    '''

    masktype, version, target, cap = os.path.basename(maskroot).split('-')
    outdir = os.path.dirname(maskroot)

    cat = Catalog(collate=1)
    cat.select_targets(target)
    cat.select_galactic_cap(cap)
    cat.version = version
    if do_veto:
        cat.veto()

    mask = read_mangle_mask(maskroot)
    w = veto(cat.RA, cat.DEC, mask) 
    cat.cut(w)
    cat.IMATCH = N.ones(cat.size)
    cat.wgalaxies = cat.size

    if '0' not in os.path.basename(masktype):
        change_mangle_mask_weights(maskroot)
        maskroot = maskroot.replace(masktype, masktype+'0')
        mask = read_mangle_mask(maskroot)

    rancat = cat.create_randoms(50., maskroot, do_veto=do_veto)

    if target=='LRG':
        cat.Z = N.ones(cat.size)*0.7
        rancat.Z = N.ones(rancat.size)*0.7
    elif target=='QSO':
        cat.Z = N.ones(cat.size)*1.5
        rancat.Z = N.ones(rancat.size)*1.5

    cat.compute_area(rancat, mask)

    cat.export(outdir+'/catalog0-%s-%s-%s.dat.fits'%(version, target, cap))
    rancat.export(outdir+'/catalog0-%s-%s-%s.ran.fits'%(version, target, cap))

