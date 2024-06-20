from setuptools import setup
from setuptools import find_packages



setup(
        name='japl',
        version='0.1',
        install_requires=[],
        packages=find_packages('.'),
        package_dir={'': '.'},
        author="nobono",
        author_email="shincdavid@gmail.com",
        license="MIT",
        classifiers=[
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3.11",
            ]
        )

