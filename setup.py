from setuptools import setup
from setuptools import find_packages
from setuptools import Extension


ode_int_ext = Extension(name="mylib",
                        sources=["./japl/odeint.cpp"],
                        # include_dirs=["C:/Users/shindc1/Documents/boost_1_82_0"],
                        # extra_compile_args=[],
                        # extra_link_args=["-shared"],
                        )

setup(
        name='japl',
        version='0.1',
        install_requires=[],
        packages=find_packages('.'),
        package_dir={'': '.'},
        ext_modules=[ode_int_ext],
        libraries=[],
        author="nobono",
        author_email="shincdavid@gmail.com",
        license="MIT",
        classifiers=[
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3.11",
            ]
        )

