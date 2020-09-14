# eboss_clustering
This repository contains the BAO fitter and reconstruction algorithm used in
the latest eBOSS cosmological results using the Luminous Red Galaxy sample 
(Bautista et al. 2020, https://arxiv.org/abs/2007.08993).

Some requirements:

- camb 
- astropy 
- iminuit (for BAO fitter only)
- pyfftw (for reconstruction only)
- cython (for reconstruction only)

In order to use the reconstruction code you need to first compile the cython module. In the root folder of eboss_clustering, please run:

python setup.py build_ext --inplace

If the python/ folder is in your $PYTHONPATH, you should be ready to go. 






