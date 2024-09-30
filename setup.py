import os
import numpy as np
from setuptools import setup
from setuptools import find_packages
from setuptools import Extension
from pybind11.setup_helpers import Pybind11Extension
# from glob import glob
import platform



if platform.system().lower() == "windows":
    split_char = ';'
else:
    split_char = ':'

# get numpy include directory
numpy_dir = os.path.dirname(np.__file__)
numpy_include_dir = os.path.join(numpy_dir, "core", "include")
if not os.path.exists(numpy_include_dir):
    raise Exception("numpy include dir could not be found")

# OdeInt ext module
ode_int_ext = Extension(name="odeint",
                        sources=[os.path.join("japl", "Sim", "OdeInt.cpp")],
                        include_dirs=[numpy_include_dir],
                        extra_compile_args=["-std=c++14"],
                        # extra_link_args=["-shared"],
                        )

linterp_ext = Pybind11Extension("linterp", ["libs/linterp/src/linterp.cpp"])

ext_modules = [ode_int_ext, linterp_ext]

setup(
        name='japl',
        version='0.1',
        install_requires=[],
        packages=find_packages('.'),
        package_dir={'': '.'},
        ext_modules=ext_modules,
        libraries=[],
        author="nobono",
        author_email="shincdavid@gmail.com",
        license="MIT",
        classifiers=[
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3.11",
            ]
        )
