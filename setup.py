"""

Package installation.
Controls the outputs of pip commands.

"""

from setuptools import setup, find_packages

setup(
  name="octavian",
  version="0.2.1",
  url="https://github.com/jp-duminy/octavian",
  author="JP Duminy, Jakub Szpila",
  maintainer="JP Duminy",
  author_email="jp@duminy.org",
  packages=find_packages(),
  python_requires='>=3.13',
  install_requires=[
    "numpy", "pandas", "scipy", "astropy", "unyt", "h5py",
    "joblib", "numba", 
  ],
  extras_require={"MPI": ["mpi4py",]},
  description="Simulation analysis toolkit for SPH codes."
)
