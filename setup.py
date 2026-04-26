from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'fd_openmp',
        ['fd_openmp.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['/O2', '/openmp'] if 'msvc' in str(pybind11.__file__) else ['-O3', '-fopenmp'],
    ),
]

setup(name='fd_openmp', ext_modules=ext_modules)