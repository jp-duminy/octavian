<div align="center">
<pre>
████████████████████████████████████
▄                                  ▄
▄ ░█▀█░█▀▀░▀█▀░█▀█░█░█░▀█▀░█░█░█▀▀ ▄
▄ ░█░█░█░░░░█░░█▀█░▀▄▀░░█░░█░█░▀▀█ ▄
▄ ░▀▀▀░▀▀▀░░▀░░▀░▀░░▀░░▀▀▀░▀▀▀░▀▀▀ ▄
▄                                  ▄
████████████████████████████████████
</pre>

[![PyPI](https://img.shields.io/pypi/v/octavius)](https://pypi.org/project/octavius/)
[![Python](https://img.shields.io/pypi/pyversions/octavius)](https://pypi.org/project/octavius/)
[![CI](https://github.com/jp-duminy/octavius/actions/workflows/ci.yml/badge.svg)](https://github.com/jp-duminy/octavius/actions)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENCE)
[![Docs](https://readthedocs.org/projects/octavius/badge/?version=latest)](https://octavius.readthedocs.io/)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Dependencies](https://img.shields.io/librariesio/release/pypi/octavius)](https://libraries.io/pypi/octavius)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

# Octavius: The Next-Generation Analysis Toolkit

Octavius is a high-performance, fully-parallelised code designed to produce analysis catalogues of SPH simulation snapshots. Written entirely in Python with minimal dependencies and all features ready out-of-the-box, it slots cleanly into analysis workflows. 

Catalogues are stored as HDF5 files containing physical properties and membership mapping for both haloes and galaxies, along with cosmological information from the snapshot.

## Install

Octavius is packaged on pip, with a planned conda-forge deployment in the near-future. It can be quickly installed from the terminal by running:

```terminal
    pip install octavius
```

The package is (currently) pinned to relatively recent dependencies and it is therefore recommended to use a tool like `uv` for version resolution. 

## Features

- Supports SWIFT and GIZMO snapshots
- Supports AHF halo catalogues (coming soon: HBT HERONS)
- Produces snapshot-agnostic catalogues
- Built-in galaxy finding with a 6D friends-of-friends algorithm
- Computes over fifty properties for haloes and galaxies (including subhaloes)
- Photometry in all FSPS-compatible bands with dust attenuation (no radiative transfer)
- User-friendly catalogues contain membership mapping
- Unit-tested analysis pipeline

## Quickstart Guide

To analyse a snapshot, you must specify parameters with a [configuration file](octavius/config.yaml). Once installed, a config YAML file for you to fill in can be generated in your current directory by running:

```terminal
    octavius init
```

The analysis can then be called either from the command line, or a Python script.

```terminal
    mpiexec -n 2 octavius analyse -c /path/to/config.yaml
```

```python
    from pathlib import Path
    from octavius import analyse_snapshot, OctaviusConfig

    config_filepath = Path("/path/to/config.yaml")
    config = OctaviusConfig.from_yaml(config_filepath)
    catalogue_path = analyse_snapshot(config)  # analyse_snapshot() is natively MPI-aware and can be run in serial too
```

It is recommended to use the command-line method for larger snapshots and more complex workflows, as it is more flexible. For more information, please run:

```terminal
    octavius analyse --help
```

An object-oriented API is provided for loading and conveniently interfacing with catalogues. This includes methods of mapping the galaxies and haloes in the catalogue to their constituent particles in the raw snapshot as well as group hierarchies (including subhaloes).

```python
    from pathlib import Path
    from octavius import load_catalogue

    catalogue_filepath = Path("/path/to/catalogue.hdf5")

    cat = load_catalogue(catalogue_filepath)

    star_mass_grams = cat.galaxies.get_dataset("mass_star", to_units="g")  # array of galaxy stellar masses in grams
```

Photometry requires a bespoke data file which requires [FSPS](https://github.com/dfm/python-fsps) to generate. The table can take a while to generate, and stores all necessary data for photometry; there is no runtime dependency on FSPS.

```python
    from pathlib import Path
    from octavius import generate_photometry_table, generate_photometry_table_from_sp

    photometry_table_filepath = Path("/path/to/table.hdf5")

    generate_photometry_table(photometry_table_filepath)  # default method with basic options 

    import fsps
    sp = fsps.StellarPopulation(...)  # full control over FSPS options
    generate_photometry_table_from_sp(photometry_table_filepath, sp)
```

Octavius is written mostly in [numba](https://numba.readthedocs.io/en/stable/), which relies on JIT compilation; this can add a minute or two of overhead when first run, but thereafter, the compiled functions are cached.

Please refer to the docs (_under construction..._) for more information.

## Dependencies

- [numba](https://numba.readthedocs.io/en/stable/)
- [numpy](https://numpy.org/doc/stable/)
- [astropy](https://docs.astropy.org/en/stable/index.html)
- [h5py](https://docs.h5py.org/en/stable/)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)

Optional:

- [fsps](https://python-fsps.readthedocs.io/en/latest/) (for photometry data tables)

## Citing

(_under construction..._)

## Beta

The package is currently in beta pre-release; please keep this in mind if using it for analysis. You are encouraged to please play around with the code and report any bugs or inaccuracies which may arise.

## Caesar

Octavius is the spiritual successor to [Caesar](https://caesar.readthedocs.io/) ([source code](https://github.com/dnarayanan/caesar)). For a thorough guide on differences to the API and bug fixes, please refer to the docs (_under construction..._)

## Licence

Octavius is licenced under the BSD 3-Clause licence.

## Contributing

Contributions and bug reports are warmly encouraged. The package is currently pinned to relatively recent versions of its dependencies, and any investigation as to whether these can be relaxed would be most welcome; otherwise, please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for more detailed information.

## Contributors

- JP Duminy, University of Edinburgh
- Jakub Szpila, Nicolaus Copernicus Astronomical Center

<small>Last updated by JP Duminy, 25/08/2026.</small>