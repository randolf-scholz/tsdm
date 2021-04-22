import io
import os
import re
import setuptools


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(
            text_type(r':[a-z]+:`~?(.*?)`'),
            text_type(r'``\1``'),
            fd.read()
        )


setuptools.setup(
    name="tsdm",
    version="0.0.1",
    url="https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/tsdm",
    license='MIT',

    author="Randolf Scholz",
    author_email="scholz@ismll.uni-hildesheim.de",

    description="Time Series Datasets and Models",
    long_description=read("README.rst"),
    long_description_content_type='test/x-rst',
    packages=setuptools.find_packages(exclude='tests'),           # include all packages in ...
    # package_dir={"": "tsdm"},                                     # tell distutils packages are in ...
    install_requires=[
            'pyyaml',
            'pandas',
            'numpy',
            'numba',
        ],
    include_package_data=True,
    package_data={
        # If any package contains *.yaml files, include them:
        '': ['*.yaml'],
        # And include any *.dat files found in the "config_files" subdirectory
        # of the "tsdm" package, also:
        'config_files': ['*.yaml'],
    },
    # ...but exclude virtualenv.yaml from all packages
    exclude_package_data={"": ["virtualenv.yaml"]},
)
