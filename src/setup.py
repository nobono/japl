import os
from setuptools import setup
from setuptools import find_packages
from setuptools import Extension


PATH = os.environ.get("PATH", "")


ode_int_ext = Extension(name="odeint",
                        sources=["./japl/Sim/OdeInt.cpp"],
                        # include_dirs=[*PATH.split(':')],
                        include_dirs=[
                            "/home/david/anaconda3/envs/control/lib/python3.11/site-packages/numpy/core/include/",
                            # *list(set(PATH.split(':'))),
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

