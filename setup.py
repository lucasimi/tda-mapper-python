from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


ext_modules = [
    Extension(
        name='tdamapper.utils._metrics',
        sources=[
            'src/tdamapper/utils/_metrics.pyx',
        ],
    ),
]

setup(
    name='tda-mapper',
    version='0.7.0',
    ext_modules=cythonize(ext_modules),
)
