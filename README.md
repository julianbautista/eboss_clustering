# eboss_clustering

This repository contains the BAO fitter and reconstruction algorithm used in
the latest eBOSS cosmological results using the Luminous Red Galaxy sample 
(Bautista et al. 2020, https://arxiv.org/abs/2007.08993).

# Installing

Add `python/` to your `$PYTHONPATH` and `bin/` to your `$PATH`.

# Some requirements:

- numpy, scipy
- astropy 
- camb (for BAO fitter)
- iminuit (for BAO fitter)
- pyfftw (for reconstruction only)
- cython (for reconstruction only)

# BAO Fitter

The BAO fitter is the one described in Bautista et al. 2020, https://arxiv.org/abs/2007.08993 and Bautista et al. 2018 https://iopscience.iop.org/article/10.3847/1538-4357/aacea5

To test it, run in command line:

`galaxy_baofit tests/nseries_om0.31_prerec_average.ini`


# Reconstruction

This code implements in python/cython the algorithm of Burden et al. 2015 https://ui.adsabs.harvard.edu/abs/2015MNRAS.453..456B/abstract.

In order to use the reconstruction code you need to first compile the cython module. In the root folder of eboss_clustering, please run:

python setup.py build_ext --inplace

If the python/ folder is in your $PYTHONPATH, you should be ready to go. 






