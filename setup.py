#!/usr/bin/env python

import os
import sys

import numpy

from setuptools import setup
from setuptools.extension import Extension

setup(use_scm_version={'write_to': os.path.join('casa_formats_io', 'version.py')},
      ext_modules=[Extension("casa_formats_io._casa_chunking",
                             [os.path.join('casa_formats_io', '_casa_chunking.c')],
                             py_limited_api=True,
                             include_dirs=[numpy.get_include()])])
