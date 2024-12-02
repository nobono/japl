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

    atmosphere_data_src = ["src/atmosphere_alts.cpp",
                           "src/atmosphere_density.cpp",
                           "src/atmosphere_grav_accel.cpp",
                           "src/atmosphere_pressure.cpp",
                           "src/atmosphere_speed_of_sound.cpp",
                           "src/atmosphere_temperature.cpp"]

    try:
        from pybind11.setup_helpers import Pybind11Extension
        linterp_ext = Pybind11Extension("linterp", ["src/linterp/linterp.cpp"], **kwargs)

        atmosphere_ext = Pybind11Extension("atmosphere", [*atmosphere_data_src,
                                                          "src/linterp/linterp.cpp",
                                                          "src/atmosphere.cpp"], **kwargs)

        aerotable_ext = Pybind11Extension("aerotable", ["src/linterp/linterp.cpp",
                                                        "src/datatable.cpp",
                                                        "src/aerotable.cpp",
                                                        ], **kwargs)

        model_ext = Pybind11Extension("model", [*atmosphere_data_src,
                                                "src/linterp/linterp.cpp",
                                                "src/datatable.cpp",
                                                "src/aerotable.cpp",
                                                "src/atmosphere.cpp",
                                                "src/model.cpp",
                                                ], **kwargs)

        datatable_ext = Pybind11Extension("datatable", ["src/linterp/linterp.cpp",
                                                        "src/datatable.cpp"], **kwargs)

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
