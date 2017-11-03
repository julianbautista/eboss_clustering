#!/usr/bin/env python

import sys
import configargparse
import os
import numpy as N 

from ebosscat import *

p = configargparse.ArgParser()
p.add('-c', '--my-config', required=True, is_config_file=True, help='config file path')
p.add('--collate', required=True, help='Collate file with targets')
p.add('--geometry', required=True, help='Mangle geometry file of the survey')
p.add('--vetos_dir', help='Directory containing veto masks')
p.add('--zcatalog', required=True, help='fits file containing redshifts')  
p.add('--version', required=True)
p.add('--outdir', required=True, help='Output directory')
p.add('--target', required=True, help='Type of targets')
p.add('--cap', default='Both', help='Cap: North, South or Both')
p.add('--completeness', help='Type of completeness')
p.add('--min_comp', type=float, help='Minimal value for completeness')
p.add('--apply_vetos', action='store_true', help='Apply veto masks? It is slower')
p.add('--start_over', action='store_true', help='Re-start from beginning')
p.add('--zmin', type=float, help='Minimum redshift range')
p.add('--zmax', type=float, help='Maximum redshift range')
p.add('--apply_noz', action='store_true', help='Apply nearest neighbor correction for failures')
p.add('--n_randoms', type=int, help='number of randoms = n_random * number of data')
p.add('--Omega_M', type=float, help='Value for the matter density (for FKP weights and volume)')

options = p.parse_args()

print p.format_values()


main(   outdir = options.outdir, \
        collate = options.collate, \
        geometry = options.geometry, \
        vetos_dir = options.vetos_dir, \
        zcatalog=options.zcatalog, \
        version=options.version, \
        target=options.target, \
        cap=options.cap, \
        comp=options.completeness, \
        mincomp=options.min_comp, \
        do_veto=options.apply_vetos, \
        zmin=options.zmin, \
        zmax=options.zmax, \
        noz=options.apply_noz, \
        nran=options.n_randoms, \
        fc=options.start_over)







