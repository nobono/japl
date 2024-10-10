import sys
from setuptools import setup
from setuptools import find_packages
from setuptools import Extension
import platform



if platform.system().lower() == "windows":
    split_char = ';'
else:
    split_char = ':'


def get_install_requires() -> list:
    with open("./requirements.txt", 'r') as f:
        install_requires = f.read().split('\n')
    return install_requires


# Safely import numpy to get its include directory
def get_numpy_include():
    try:
        import numpy
        return numpy.get_include()
    except ImportError:
        sys.exit("Error: numpy must be installed to build this package.")


# Safely import pybind11 to get extension modules
def get_extension_modules() -> list:
    try:
        from pybind11.setup_helpers import Pybind11Extension
        linterp_ext = Pybind11Extension("linterp", ["libs/linterp/src/linterp.cpp"],
                                        extra_compile_args=['-std=c++11'],
                                        extra_link_args=['-std=c++11'])
        return [linterp_ext]
    except ImportError:
        sys.exit("Error: pybind11 must be installed to build this package.")


# # OdeInt ext module
# ode_int_ext = Extension(name="odeint",
#                         sources=[os.path.join("japl", "Sim", "OdeInt.cpp")],
#                         include_dirs=[get_numpy_include()],
#                         extra_compile_args=["-std=c++14"],
#                         # extra_link_args=["-shared"],
#                         )

setup(
        name='japl',
        version='0.1',
        install_requires=get_install_requires(),
        packages=find_packages('.'),
        package_dir={'': '.'},
        ext_modules=get_extension_modules(),
        libraries=[],
        author="nobono",
        author_email="shincdavid@gmail.com",
        license="MIT",
        classifiers=[
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3.11",
            ]
        )
