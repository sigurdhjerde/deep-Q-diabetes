from setuptools import find_packages
from setuptools import setup

from version import VERSION

setup(
    name = 'Utilities',
    version = VERSION,
    description = 'Utility modules',
    license = '',
    install_requires=[
          'numpy>=1.10.4', 'random', 'operator',
      ]
)
