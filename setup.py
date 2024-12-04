import os
import sys
import shutil
import glob
from setuptools import setup
from setuptools import find_packages
from setuptools import Command
from setuptools.command.build_ext import build_ext
from setuptools.command.build import build
from pathlib import Path
import platform



ROOT_DIR = Path(os.path.dirname(__file__))


def copy_dir(source_dir, target_dir) -> None:
    """
    Recursively copies all directories and files from source_dir to target_dir.

    Parameters:
    -----------
        source_dir (str): The source directory to copy from.
        target_dir (str): The target directory to copy to.

    Raises:
    -------
        ValueError: If source_dir does not exist or is not a directory.
    """
    if not os.path.isdir(source_dir):
        raise ValueError(f"Source directory '{source_dir}' does not exist or is not a directory.")

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        target_item = os.path.join(target_dir, item)

        if os.path.isdir(source_item):
            # Recursively copy directories
            copy_dir(source_item, target_item)
        else:
            # Copy files
            shutil.copy2(source_item, target_item)


class BuildCommand(build):
    # user_options = build_ext.user_options + [
    #         ("build-temp=", None, "Specify a directory fro temporary build files.")
    #         ]


    # def initialize_options(self) -> None:
    #     super().initialize_options()
    #     self.build_temp = TEMP_LIBS_DIR  # default to None; can be set by user


    # def finalize_options(self) -> None:
    #     super().finalize_options()
    #     if self.build_temp:
    #         os.makedirs(self.build_temp, exist_ok=True)


    def run(self) -> None:
        return super().run()


class BuildExtCommand(build_ext):
    user_options = build_ext.user_options + [
            ("build-temp=", None, "Specify a directory fro temporary build files.")
            ]


    def initialize_options(self) -> None:
        super().initialize_options()
        self.build_temp = "libs"  # default to None; can be set by user


    def finalize_options(self) -> None:
        super().finalize_options()
        if self.build_temp:
            os.makedirs(self.build_temp, exist_ok=True)


    def build_extension(self, ext) -> None:
        # override the temporary build directory
        if self.build_temp:
            self.build_temp = os.path.abspath(self.build_temp)
            self.build_temp_dir = self.build_temp
        return super().build_extension(ext)


    def build_extensions(self) -> None:
        self.parallel = 5
        super().build_extensions()

        # Define source and destination directories
        build_temp = self.build_temp  # Temporary build directory
        libs_dest_dir = Path(self.build_lib, "libs")  # Installation directory

        include_dir = Path(ROOT_DIR, "include")
        include_dest_dir = Path(self.build_lib, "include")  # Installation directory

        # Create the destination directory if it doesn't exist
        os.makedirs(libs_dest_dir, exist_ok=True)
        os.makedirs(include_dest_dir, exist_ok=True)

        # Copy all .o files to the destination directory
        copy_dir(build_temp, libs_dest_dir)
        # copy all .hpp files to destination directory
        copy_dir(include_dir, include_dest_dir)


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
        shutil.rmtree('./libs', ignore_errors=True)
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
        packages=find_packages(),
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
            "build_ext": BuildExtCommand,
            "build": BuildCommand,
            },
        package_data={"japl": ["libs/*.o", "include/*"]},
        entry_points={
            "console_scripts": [
                "japl = bin.japl:main"  # make 'japl' callable
                ]
            }
        )
