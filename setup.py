from __future__ import absolute_import, print_function
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from numpy.distutils.misc_util import get_numpy_include_dirs
from os import path
from glob import glob

class build_ext_openmp(build_ext):
    # https://www.openmp.org/resources/openmp-compilers-tools/
    # python setup.py build_ext --help-compiler
    openmp_compile_args = {
        'msvc':  ['/openmp'],
        'intel': ['-qopenmp'],
        '*':     ['-fopenmp']
    }
    openmp_link_args = openmp_compile_args # ?

    def build_extension(self, ext):
        compiler = self.compiler.compiler_type.lower()
        if compiler.startswith('intel'):
            compiler = 'intel'
        if compiler not in self.openmp_compile_args:
            compiler = '*'

        _extra_compile_args = list(ext.extra_compile_args)
        _extra_link_args    = list(ext.extra_link_args)
        try:
            ext.extra_compile_args += self.openmp_compile_args[compiler]
            ext.extra_link_args    += self.openmp_link_args[compiler]
            super(build_ext_openmp, self).build_extension(ext)
        except:
            print('compiling with OpenMP support failed, re-trying without')
            ext.extra_compile_args = _extra_compile_args
            ext.extra_link_args    = _extra_link_args
            super(build_ext_openmp, self).build_extension(ext)


#------------------------------------------------------------------------------------



setup(
    name='normalize_fast',
    description='normalize_fast',

    cmdclass={'build_ext': build_ext_openmp},
    packages=find_packages(),

    ext_modules=[
        Extension(
            'normalize_fast.normalize_fast',
            sources=['normalize_fast/normalize_fast.cpp'],
            extra_compile_args = ['-std=c++11'],
            include_dirs=get_numpy_include_dirs(),
        ),
        Extension(
            'normalize_fast.percentile_fast',
            sources=['normalize_fast/percentile_fast.cpp'],
            extra_compile_args = ['-std=c++11'],
            include_dirs=get_numpy_include_dirs(),
        ),
    ],
    
    install_requires=[
        # 'csbdeep>=0.6.0',
        # 'numexpr',
        # 'numba',
        # 'pytest'
    ],
    
)
