#!/bin/bash

export CAP=Combined
export REBIN_R=5
export SHIFT_R=2
export RMIN=30
export RMAX=180

galaxy_fit_bao \
  -i data/rec-cmassebosscat-v1.0-LRG-${CAP}-spsub-${REBIN_R}mpc-shift${SHIFT_R}.mul \
  -c data/recmock-1.8-LRG-${CAP}-average-${REBIN_R}mpc-shift${SHIFT_R}.cov \
  -o galaxy_bao_fits/rec-cmassebosscat-v1.0-LRG-${CAP}-${REBIN_R}mpc-shift${SHIFT_R} \
  --fit_iso \
  --fit_beta \
  --fit_nopeak \
  --fit_broadband \
  --rmin $RMIN \
  --rmax $RMAX \
  --nmocks 1000 \
  --fixes beta 0.355 Sigma_NL 5.5 Sigma_s 0. Sigma_rec 15. \
  --limits aiso 0.5 1.5 bias 1.0 3.0 \
  --z 0.72 \
  --scale_cov 0.9753 \
  --plotit
  #--minos aiso \
  #--scan aiso 0.6 1.4 100 \
  #--scan_nopeak aiso 0.6 1.4 20  
  #--priors bias 1.0 0.3 \

