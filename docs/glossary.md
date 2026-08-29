# Glossary

This is a glossary of commonly-used terminology throughout the codebase and documentation.

## Particle Types

- `baryon`: an aggregation of gas, star, and black hole particles.

- `bh`: black hole particles and quantities.

- `dm`: dark matter particles and quantities.

- `gas`: gas particles and quantities.

- `ptype`: a specific type of particle (including aggregations).

- `star`: star particles and quantities.

- `total`: an aggregation of all particles.

## Codebase Terminology

- `config`: `OctaviusConfig` dataclass containing configuration file parameters.

- `constants`: `OctaviusConstants` dataclass containing constants, literature values, and scaling factors.

- `CSR`: compressed-sparse row, an efficient format for representing data where most entries are expected to be zero.

- `dataclass`: standard library tool in Python to create containers for data arrays; `slots=True` improves performance.

- `field halo`: a top-level halo which has no parent.

- `subhalo`: a halo which is a child of another halo.

- `GroupStore`: (usually as `haloes` or `galaxies`) dictionary-like dataclass for group-level data.

- `groups`: dictionary containing `GroupStore` objects, keyed by `group_key`.

- `Internals`: dataclass containing information parsed from `internals.yaml`, such as naming conventions and metadata.

- `HaloAssignments`: dataclass containing particle-level halo ID assignments from an external catalogue.

- `HaloSource`: class for parsing external halo ID catalogues.

- `logger`: standard library tool in Python to log messages to the terminal and output diagnostics without scattering `print()` statements across the code.

- `njit`: just-in-time (JIT) compilation in `nopython` mode, for machine-level speed with `numba`.

- `ParticleStore`: (usually as the constituent `ptype`) dictionary-like dataclass for particle-level data.

- `particles`: dictionary containing `ParticleStore` objects, keyed by `ptype`.

- `PhotometryTable`: dataclass containing SSP and filter data from FSPS, used to interpolate star properties in photometry

- `Sentinel value`: a convention for representing invalid/unassigned entries in arrays; taken to be -1.

- `SimulationAttributes`: (usually as `sim`) dataclass containing useful information from the original snapshot, such as the boxsize and cosmological information.

- `SimulationData`: dataclass containing `particles` and `groups` along with `SimulationAttributes` and `OctavianConstants`.

- `SnapshotReader`: class for reading raw snapshot information.

- `SubhaloInformation`: dataclass containing information about the subhaloes from the external catalogue.

## Smoothed Particle Hydrodynamics

- `kernel`: a normalised weighting function used to interpolate physical properties.

- `particle`: a discrete moving element sized by the resolution of the simulation which carries physical properties and interacts according to a `kernel`.

- `smoothing_length`: a maximum distance to which the kernel is defined for interpolating physical properties.

## Miscellaneous

- `AHF`: an external spherical overdensity halo finder.

- `caesar`: the predecessor analysis toolkit to Octavius.

- `FOF`: the friends-of-friends algorithm used to identify clusters of particles.

- `GIZMO`: physics framework used for simulations.

- `HBT-HERONS`: an external history-based halo finder.

- `MPI`: message passing interface, a standard for parallel computing.

- `Rank`: a unique worker process running under MPI.

- `SWIFT`: gravity and SPH solver framework used for simulations.

- `SSP`: a simple stellar population, used to look up predicted photometric properties based on physical properties.
