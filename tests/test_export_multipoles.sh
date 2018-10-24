#!/bin/bash


python $EBOSS_CLUSTERING_DIR/bin/export_multipoles.py \
    --input $EBOSS_CLUSTERING_DIR/data/rec-cmassebosscat-v1.0-LRG-Combined-spsub.corr2drmu \
    --output $EBOSS_CLUSTERING_DIR/data/rec-cmassebosscat-v1.0-LRG-Combined-spsub-5mpc-shift0.mul \
    --rebin_r 5 --shift_r 0



