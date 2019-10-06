import redshift_failures
from astropy.table import Table

dat = Table.read('/mnt/lustre/eboss/DR16_LRG_data/v5/eBOSS_LRG_full_SGC_v5.dat.fits')
redshift_failures.get_weights_noz(dat)
redshift_failures.plot_failures(dat)





