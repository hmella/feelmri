#!/usr/bin/env python
import os
import sys

import setuptools
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


def read(fname):
    try:
        content = codecs.open(
            os.path.join(os.path.dirname(__file__), fname),
            encoding='utf-8'
            ).read()
    except Exception:
        content = ''
    return content


__version__ = '0.0.1'


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user
  
    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


ext_modules = [
    Pybind11Extension(
        'feelmri.MathCpp',
        sources=['feelmri/cpp/Math.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            '/usr/include/eigen3/',
        ],
        language='c++',
    ),
    Pybind11Extension(
        'feelmri.FEAssemble',
        sources=['feelmri/cpp/FEAssemble.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            '/usr/include/eigen3/',
            '/usr/include/basix/'
        ],
        language='c++',
    ),
    Pybind11Extension(
        'feelmri.BlochSimulator',
        sources=['feelmri/cpp/BlochSimulator.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            '/usr/include/eigen3/',
            'feelmri/cpp/'
        ],
        language='c++',
    ),
    Pybind11Extension(
        'feelmri.MeshRefinement',
        sources=['feelmri/cpp/MeshRefinement.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            '/usr/include/eigen3/'
        ],
        language='c++',
    ),
    Pybind11Extension(
        'feelmri.MRIEncoding',
        ['feelmri/cpp/MRIEncoding.cpp', 'feelmri/cpp/Signal.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            '/usr/include/eigen3/','feelmri/cpp/'
        ],
        language='c++',
    ),
    Pybind11Extension(
        'feelmri.Tagging',
        ['feelmri/cpp/Tagging.cpp', 'feelmri/cpp/Signal.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            '/usr/include/eigen3/','feelmri/cpp/'
        ],
        language='c++',
    ),
    Pybind11Extension(
        'feelmri.PhaseContrast',
        sources=['feelmri/cpp/PhaseContrast.cpp', 'feelmri/cpp/Signal.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            '/usr/include/eigen3/','feelmri/cpp/'
        ],
        language='c++',
    ),
]

# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17/20] compiler flag.

    The c++20 is prefered over others (when it is available).
    """
    if has_flag(compiler, '-std=c++20'):
        return '-std=c++20'
    if has_flag(compiler, '-std=c++17'):
        return '-std=c++17'
    elif has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
            if has_flag(self.compiler, '-Ofast'):
                opts.append('-Ofast')
            if has_flag(self.compiler, '-march=native'):
                opts.append('-march=native')
            if has_flag(self.compiler, '-mavx2'):
                opts.append('-mavx2')
            if has_flag(self.compiler, '-fno-math-errno'):
                opts.append('-fno-math-errno')
            if has_flag(self.compiler, '-DNDEBUG'):
                opts.append('-DNDEBUG')
            if has_flag(self.compiler, '-mfpmath=sse'):
                opts.append('-mfpmath=sse')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(name='feelmri',
      version='1.0',
      packages=['feelmri'],
      description='Finite element MRI library. For the simulation of MR images from finite element simulations.',
      long_description=read('README.md'),
      ext_modules=ext_modules,
      install_requires=['pybind11>=2.11'],
      cmdclass={'build_ext': BuildExt},
      zip_safe=False,
      author='Hernan Mella',
      author_email='hernan.mella@pucv.cl'
      )


