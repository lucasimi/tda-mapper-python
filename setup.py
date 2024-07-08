from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


setup(
    ext_modules=cythonize([
        Extension(
            name="tdamapper.utils._metrics",
            sources=["src/tdamapper/utils/_metrics.pyx"],
        ),
    ])
)
