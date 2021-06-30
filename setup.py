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
    version="0.0.8",
    url="https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/tsdm",
    license='MIT',

    author="Randolf Scholz",
    author_email="scholz@ismll.uni-hildesheim.de",

    description="Time Series Datasets and Models",
    long_description=read("README.rst"),
    long_description_content_type='test/x-rst',
    packages=setuptools.find_packages(exclude=['test']),           # include all packages in ...
    install_requires=[
            'pyyaml',
            'pandas',
            'numpy',
            'numba',
            'xarray',
            'matplotlib',
    ],
    # Files that listed in MANIFEST.in and also are in python packages,
    # i.e. contained in folders with and __init__.py, will be included.
    include_package_data=True,
    # ...but exclude virtualenv.yaml from all packages
    exclude_package_data={"": ["virtualenv.yaml"]},
)
