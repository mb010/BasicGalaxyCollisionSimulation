#v19
#In terminal:
#	1. Return .so file ("shared object"), use function:	python3 setup.py build_ext --inplace
#	2. Run code with "galaxy2.py":	python3 galaxy2.py
# 	*To see which lines rely heavily on python run cython -a example_cy.pyx and open the .html file that is created.
#	**python -m timeit
#	3.For Microsoft Visual C++ compiler, use '/openmp' instead of '-fopenmp'.

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules=[
    Extension("example_cy",
              ["example_cy.pyx"],
              libraries=["m"],
              extra_compile_args = ["-fopenmp"],
              extra_link_args=['-fopenmp']
              )] 

setup(
	name = "example_cy",
	cmdclass = {"build_ext": build_ext},
	include_dirs = [np.get_include()], 
	ext_modules = ext_modules)


