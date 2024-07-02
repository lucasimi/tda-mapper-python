from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


setup(
    ext_modules=cythonize([
        Extension(
            "tdamapper.utils._metrics",
            ["src/tdamapper/utils/_metrics.pyx"])
    ]))
