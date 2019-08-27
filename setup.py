#!/usr/bin/env python

"""Setup script."""


import glob
from setuptools import setup, find_packages, Extension

# Package name
name = 'forecast_SN_GW'

# Packages (subdirectories in sugar_analysis/)
packages = find_packages()

# Scripts (in scripts/)
scripts = glob.glob("scripts/*.py")

package_data = {}

setup(name=name,
      description=("forecast_SN_GW"),
      classifiers=["Topic :: Scientific :: Astronomy",
                   "Intended Audience :: Science/Research"],
      author="",
      packages=packages,
      scripts=scripts)
