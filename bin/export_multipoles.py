from cf_tools import Multipoles
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', help='Input correlation function file (CUTE output)')
parser.add_argument('-o', '--output', help='Output multipoles file')
parser.add_argument('--rebin_r',
                    help='Make rebin_r bins into a single bin',
                    type=int, default=5)
parser.add_argument('--shift_r', help='Starts rebinning from bin number shift_r', \
                    type=int, default=0)
args = parser.parse_args()

m = Multipoles(args.input, rebin_r=args.rebin_r, shift_r=args.shift_r, cute=1)
m.export(args.output)




