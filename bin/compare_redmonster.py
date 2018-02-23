import pandas as pd
import numpy as np
import os
from astropy.table import Table, join
import matplotlib.pyplot as plt

spall_file = os.environ['BOSS_SPECTRO_REDUX']+'/v5_10_4/spAll-v5_10_4.fits'
redmonster_file =  os.environ['REDMONSTER_SPECTRO_REDUX'] +  '/v5_10_4/v1_2_0/redmonsterAll-v5_10_4.fits'

tb_spall = Table.read(spall_file)
tb_rm = Table.read(redmonster_file)
tb_rm['Z'].name = 'Z_RM'
tb_rm['ZWARNING'].name = 'ZWARNING_RM'
tb_rm['CLASS'].name = 'CLASS_RM'
tb_rm['EBOSS_TARGET0'].name = 'EBOSS_TARGET0_RM'
tb_rm['EBOSS_TARGET1'].name = 'EBOSS_TARGET1_RM'
tb_rm['BOSS_TARGET1'].name = 'BOSS_TARGET1_RM'

tb_join = join(tb_rm, tb_spall, keys=['FIBERID','PLATE','MJD'])

#numbers

#number of targets

LRGs = tb_join[ ((tb_join['EBOSS_TARGET1'] & 2**1) > 0) | ((tb_join['EBOSS_TARGET0'] & 2**2) > 0)]
print 'Number of eBOSS LRGs: ', len(LRGs)
N = len(LRGs)+0.0

#Pipeline redshift failures
print 'number of pipeline redshift failures: ', len(LRGs[LRGs['ZWARNING_NOQSO'] > 0])/N *100

print 'number of pipeline good redshifts: ', len(LRGs[LRGs['ZWARNING_NOQSO'] == 0])/N *100

print 'number of pipeline STAR classifications: ', len(LRGs[ (LRGs['CLASS_NOQSO'] == 'STAR  ')]) /N *100

print 'number of pipeline GALAXY classifications: ', len(LRGs[(LRGs['CLASS_NOQSO'] == 'GALAXY')])/N *100

print 'number of good redshifts AND galaxy: ', len(LRGs[ (LRGs['ZWARNING_NOQSO'] == 0) & (LRGs['CLASS_NOQSO'] == 'GALAXY')]) /N *100
print 'number of good redshifts AND star: ', len(LRGs[ (LRGs['ZWARNING_NOQSO'] == 0) & (LRGs['CLASS_NOQSO'] == 'STAR  ')]) /N *100
print 'number of good redshifts AND QSO: ', len(LRGs[ (LRGs['ZWARNING_NOQSO'] == 0) & (LRGs['CLASS_NOQSO'] == 'QSO   ')]) /N *100

print 'number of good redshifts AND galaxy in 0.6 < z < 1.0: ', len(LRGs[ (LRGs['ZWARNING_NOQSO'] == 0) & (LRGs['CLASS_NOQSO'] == 'GALAXY') & (LRGs['Z_NOQSO'] >= 0.6) & (LRGs['Z_NOQSO'] <= 1.0)]) /N *100
print ' '


#redmonster
print 'number of redmonster redshift failures: ', len(LRGs[LRGs['ZWARNING_RM'] > 0])/N *100

print 'number of redmonster good redshifts: ', len(LRGs[LRGs['ZWARNING_RM'] == 0])/N *100

print 'number of redmonster GALAXY classifications: ', len(LRGs[(LRGs['CLASS_RM'] == 'ssp_galaxy_glob')]) /N *100

print 'number of redmonster good redshifts AND galaxy: ', len(LRGs[ (LRGs['ZWARNING_RM'] == 0) & (LRGs['CLASS_RM'] == 'ssp_galaxy_glob')]) /N *100
print 'number of redmonster good redshifts AND galaxy AND primary: ', len(LRGs[ (LRGs['ZWARNING_RM'] == 0) & (LRGs['CLASS_RM'] == 'ssp_galaxy_glob') & (LRGs['SPECPRIMARY'] == 1)]) /N *100

print 'number of redmonster good redshifts AND star: ', len(LRGs[ (LRGs['ZWARNING_RM'] == 0) & (LRGs['CLASS_RM'] == 'CAP')]) /N *100
print 'number of redmonster good redshifts AND QSO: ', len(LRGs[ (LRGs['ZWARNING_RM'] == 0) & (LRGs['CLASS_RM'] == 'QSO')]) /N *100
print 'number of redmonster good redshifts AND galaxy in 0.6 < z < 1.0: ', len(LRGs[ (LRGs['ZWARNING_RM'] == 0) & (LRGs['CLASS_RM'] == 'ssp_galaxy_glob') & (LRGs['Z_RM'] >= 0.6) & (LRGs['Z_RM'] <= 1.0) ]) /N *100

eBOSS = LRGs[ (LRGs['ZWARNING_RM'] == 0) & (LRGs['CLASS_RM'] == 'ssp_galaxy_glob') & (LRGs['ZWARNING_NOQSO'] == 0) & (LRGs['CLASS_NOQSO'] == 'GALAXY') ]
#plt.scatter( eBOSS['Z_RM'], eBOSS['Z_RM'] - eBOSS['Z_NOQSO'], s=0.1)
#plt.show()

c = 300000.
diff = (eBOSS['Z_RM'] - eBOSS['Z_NOQSO']) * c / (1 + eBOSS['Z_RM'])
#plt.hist( diff, bins=30, range=(-100,100), histtype='step', normed=True)
#plt.ylabel('Normalised counts')
#plt.xlabel('Delta_v (km/s)')
#plt.savefig('LRG_z_hist.pdf', bbox_inches='tight')
#plt.show()

tb_join.write('redmonster_spAll_v5_10_4_v1_2_0.fits',format='fits')

