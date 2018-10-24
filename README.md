# eboss_clustering
Repo containing code to:

- generate large-scale structure catalogs for eBOSS: ebosscat.py
- compute systematic weights using multi-linear regression: systematics.py
- compute spectroscopic completeness model: eff_model.py
- fit BAO on galaxy correlation function: galaxy_bao_fitter.py
- reconstruction algorithm based on Burden et al. 2015 : recon.py 

Requirements:

- python 3.6
- numpy 1.12.1
- scipy 0.19.0
- camb 0.1.2
- astropy 3.0
- healpy 1.9.1

- mangle 3.2 (for catalog generation only)
- iminuit (for fitter only)
- pyfftw (for reconstruction only)

