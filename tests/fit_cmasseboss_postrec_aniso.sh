#!/bin/bash

export CAP=Combined
export REBIN_R=5
export SHIFT_R=2
export RMIN=30
export RMAX=180


galaxy_fit_bao \
  -i data/rec-cmassebosscat-v1.0-LRG-${CAP}-spsub-${REBIN_R}mpc-shift${SHIFT_R}.mul \
  -c data/recmock-1.8-LRG-${CAP}-average-${REBIN_R}mpc-shift${SHIFT_R}.cov \
  -o galaxy_bao_fits/rec-cmassebosscat-v1.0-LRG-${CAP}-${REBIN_R}mpc-shift${SHIFT_R}-aniso \
  --fit_beta \
  --fit_nopeak \
  --fit_quad \
  --fit_broadband \
  --rmin $RMIN \
  --rmax $RMAX \
  --nmocks 1000 \
  --fixes Sigma_par 5.5 Sigma_per 5.5 Sigma_s 0. Sigma_rec 15. \
  --limits at 0.5 1.5 ap 0.5 1.5 bias 1.0 4.0 \
  --z 0.72 \
  --scale_cov 0.9753 \
  --plotit 
  #--minos at ap #\
  #--scan2d at 0.8 1.2 40 ap 0.6 1.2 40 #\
  #--priors bias 1.0 0.3 \

