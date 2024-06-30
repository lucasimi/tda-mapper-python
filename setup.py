from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize



setup(
    ext_modules=cythonize([
        Extension("tdamapper.utils.cython.metrics", ["src/tdamapper/utils/cython/metrics.pyx"])
    ])
)