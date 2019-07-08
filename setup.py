from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import sys
import glob
import os

ext_modules=[ Extension("fastmodules",
              ["python/fastmodules.pyx"],
              libraries=["m"],
              include_dirs=[numpy.get_include()])]
              #extra_compile_args = ["-ffast-math"],

setup(
  name = "fastmodules",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)

libfile = glob.glob('fastmodules*.so')
if len(libfile) == 0:
    print('Problem when compiling cython module: fastmodules')
    sys.exit(0)

os.rename(libfile[0], 'python/'+libfile[0])

