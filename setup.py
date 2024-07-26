import os
import numpy as np
from setuptools import setup
from setuptools import find_packages
from setuptools import Extension
import platform

if platform.system().lower() == "windows":
    split_char = ';'
else:
    split_char = ':'


numpy_dir = os.path.dirname(np.__file__)
numpy_include_dir = os.path.join(numpy_dir, "core", "include")
path_include_dirs = os.environ.get("PATH", "").split(split_char)
if not os.path.exists(numpy_include_dir):
    raise Exception("numpy include dir could not be found")


ode_int_ext = Extension(name="odeint",
                        sources=["./japl/Sim/OdeInt.cpp"],
                        include_dirs=[
                            numpy_include_dir,
                            *path_include_dirs,
                            ],
                        # extra_compile_args=[],
                        # extra_link_args=["-shared"],
                        )

setup(
        name='japl',
        version='0.1',
        install_requires=[],
        packages=find_packages('.'),
        package_dir={'': '.'},
        # ext_modules=[ode_int_ext],
        libraries=[],
        author="nobono",
        author_email="shincdavid@gmail.com",
        license="MIT",
        classifiers=[
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3.11",
            ]
        )

