import sys
import numpy as np
import pylab as plt
import galaxy_bao_fitter
from scipy import stats
from scipy.interpolate import interp2d

fin = sys.argv[1]
c = galaxy_bao_fitter.Chi2()
c.read_scan2d(fin)

plt.figure(figsize=(6,5))
plt.pcolormesh(c.x, c.y, c.chi2scan2d - c.chi2scan2d.min())
plt.colorbar()
c.plot_scan2d(color='w')
plt.plot(c.bestx, c.besty, 'wo')

plt.xlabel(r'$\alpha_\perp$', fontsize=14)
plt.ylabel(r'$\alpha_\parallel$', fontsize=14)
plt.tight_layout()
plt.grid()

#-- Creating 2D linear interpolator for chi2 grid
chi2int = interp2d(c.x, c.y, c.chi2scan2d-c.chi2min)

#-- What probability is Planck for 2 degrees of freedom
p_planck = stats.chi2.cdf(chi2int(1., 1.), 2)

#-- How many sigmas for 2 degrees of freedom
sigmas = np.sqrt(stats.chi2.ppf(p_planck[0], 1))
print(f'Tension with Planck is at {sigmas:.2f} sigmas')

plt.show()

