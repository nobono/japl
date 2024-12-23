import os
import sys
import shutil
import glob
from setuptools import setup
from setuptools import find_packages
from setuptools import Command
import platform



class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        shutil.rmtree('./build', ignore_errors=True)
        shutil.rmtree('./dist', ignore_errors=True)
        root_path = os.path.dirname(__file__)
        file_patterns = ["*.so", "*.dll", "*.pyd"]
        for pattern in file_patterns:
            for file in glob.iglob(os.path.join(root_path, "**", pattern), recursive=True):
                print("removing:", file)
                os.remove(file)


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
    kwargs = dict(extra_compile_args=[],
                  extra_link_args=[],
                  cxx_std=17)

    atmosphere_data_src = ["libs/atmosphere/data/_atmosphere_alts.cpp",
                           "libs/atmosphere/data/_atmosphere_density.cpp",
                           "libs/atmosphere/data/_atmosphere_grav_accel.cpp",
                           "libs/atmosphere/data/_atmosphere_pressure.cpp",
                           "libs/atmosphere/data/_atmosphere_speed_of_sound.cpp",
                           "libs/atmosphere/data/_atmosphere_temperature.cpp"]

    try:
        from pybind11.setup_helpers import Pybind11Extension
        linterp_ext = Pybind11Extension("linterp", ["libs/linterp/src/linterp.cpp"], **kwargs)

        atmosphere_ext = Pybind11Extension("atmosphere", [*atmosphere_data_src,
                                                          "libs/linterp/src/linterp.cpp",
                                                          "libs/atmosphere/atmosphere.cpp"], **kwargs)

        aerotable_ext = Pybind11Extension("aerotable", ["libs/linterp/src/linterp.cpp",
                                                        "libs/datatable/datatable.cpp",
                                                        "libs/aerotable/aerotable.cpp",
                                                        ], **kwargs)

        model_ext = Pybind11Extension("model", [*atmosphere_data_src,
                                                "libs/linterp/src/linterp.cpp",
                                                "libs/datatable/datatable.cpp",
                                                "libs/aerotable/aerotable.cpp",
                                                "libs/atmosphere/atmosphere.cpp",
                                                "libs/model/model.cpp",
                                                ], **kwargs)

        datatable_ext = Pybind11Extension("datatable", ["libs/linterp/src/linterp.cpp",
                                                        "libs/datatable/datatable.cpp"], **kwargs)

        return [linterp_ext, atmosphere_ext, aerotable_ext, model_ext, datatable_ext]
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
            ],
        cmdclass={
            "clean": CleanCommand,
            },
        entry_points={
            "console_scripts": [
                "japl = bin.japl:main"  # make 'japl' callable
                ]
            }
        )
