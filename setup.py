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
    ext_modules=cythonize(ext_modules),
)
