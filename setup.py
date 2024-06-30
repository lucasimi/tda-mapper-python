from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


ext0 = Extension("tdamapper.utils.cython.metrics", ["src/tdamapper/utils/cython/metrics.pyx"])
ext1 = Extension("tdamapper.utils.cython.quickselect", ["src/tdamapper/utils/cython/quickselect.pyx"])
ext2 = Extension("tdamapper.utils.cython.heap", ["src/tdamapper/utils/cython/heap.pyx"])
ext3 = Extension("tdamapper.utils.cython.vptree_flat", ["src/tdamapper/utils/cython/vptree_flat.pyx"])

setup(
    ext_modules=cythonize([
        ext0,
        ext1,
        ext2,
        ext3,
    ])
)
