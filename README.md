# eboss_clustering
Repo containing code to:

- generate large-scale structure catalogs for eBOSS: ebosscat.py
- compute systematic weights using multi-linear regression: systematics.py
- compute spectroscopic completeness model: eff_model.py
- fit BAO on galaxy correlation function: galaxy_bao_fitter.py
- reconstruction algorithm based on Burden et al. 2015 : recon.py 

Requirements:

- camb 
- astropy 
- healpy 
- mangle 3.2 (for catalog generation only)
- iminuit (for BAO fitter only)
- pyfftw (for reconstruction only)
- cython (for reconstruction only)

In order to use the reconstruction code you need to first compile the cython module. In the root folder of eboss_clustering, please run:

python setup.py build_ext --inplace

If the python/ folder is in your $PYTHONPATH, you should be ready to go. 






