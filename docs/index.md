<h1 style="text-align: center;">Welcome to the Octavius Documentation!</h1>

<div style="margin-bottom: 3.0em;"></div>

```{image} _static/banner.webp
:alt: Octavius
:align: center
:width: 600px
```

<div style="margin-bottom: 3.0em;"></div>

[Octavius](https://github.com/jp-duminy/octavius) is a high-performance, fully-parallelised galaxy simulation analysis toolkit written entirely in Python and designed to effortlessly slot into analysis workflows; when run on a simulation snapshot, Octavius produces HDF5 analysis catalogues containing properties and membership information for haloes and galaxies. Features include:

- Support for [SWIFT](https://swift.strw.leidenuniv.nl/) & [GIZMO](http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html) snapshots
- Support for [AHF](https://iopscience.iop.org/article/10.1088/0067-0049/182/2/608) halo catalogues (coming soon: [HBT-HERONS](https://hbt-herons.strw.leidenuniv.nl/))
- Snapshot-agnostic catalogues
- Built-in galaxy finding with a 6D friends-of-friends algorithm
- Computes over fifty properties for haloes and galaxies (including subhaloes)
- Photometry in all [FSPS](https://dfm.io/python-fsps/current/)-compatible bands with dust attenuation (no radiative transfer)
- User-friendly API for working with analysis catalogues
- Unit-tested analysis pipeline

To get started, please refer to the [installation](installation.md) guide; for a brief overview of the package, please see the [five-minute guide](five_minute_guide.md).

Octavius is the spiritual successor to [Caesar](https://caesar.readthedocs.io/). `Caesar` users should please refer to the [Caesar users guide](caesar_users_guide.md).

```{toctree}
:hidden:
:caption: Contents
installation
quickstart
five_minute_guide
configuration
catalogues
caesar_users_guide
features/galaxy_finding
features/aggregate_properties
features/photometry
features/utilities
features/parallelism
features/membership
examples/analysing_snapshots
examples/loading_catalogues
api
contributions
glossary
```