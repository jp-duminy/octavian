# Caesar Users Guide

[I want a quick example](#quick-example)

[My Octavius and Caesar catalogues disagree](#my-output-is-different)

## Context

The development of Octavius was necessitated by the growing big data challenge of analysing ever-larger simulations, which Caesar struggled to meet at scale. The code was developed from the ground up reimplementing, rather than directly porting, Caesar routines; this opened up opportunities for optimisation and architectural improvements; Octavius has been designed with performance at scale as an _ab initio_ concern.

The leading advantage of Octavius over its predecessor is in performance. In raw speed, memory profile, and scalability, Octavius offers significant, order-of-magnitude improvements over Caesar with fewer dependencies.

During development of Octavius, many existing bugs were discovered in Caesar and fixed in the reimplementations of routines. This is particularly relevant to galaxy finding, where the two codes can be expected to produce noticeably different results from their FOF6D algorithms owing to bugs in the Caesar code. Not only are the galaxy assignments different, but there have been numerous bug fixes and precision improvements across aggregate properties and photometry, meaning Octavius catalogues will not replicate the same results found by Caesar.

There are ensuing differences in the API and intended usage patterns between the toolkits. The primary difference is the vectorisation of catalogue quantities and membership access patterns.

## Examples

In this section we will present reimplementations of Caesar documentation examples in the Octavius API. The [examples](../examples/index.md) section contains more examples for an illustrative picture.

(quick-example)=
### Quick Example

We begin with a comparison to the [quick example](https://caesar.readthedocs.io/en/latest/usage.html#usage) in the Caesar documentation of accessing datasets and mapping groups.

:::{note}
Catalogues are loaded directly, with no need for `yt`.
:::

```python
from pathlib import Path
import octavius as oc

catalogue_path = Path("/path/to/catalogue.hdf5")
catalogue = oc.load_catalogue(catalogue_path)

galaxy_masses = catalogue.galaxies.get_dataset("mass_baryon")
galaxy_mass_units = catalogue.galaxies.get_units("mass_baryon")
galaxy_masses_converted = catalogue.galaxies.get_dataset("mass_baryon", to_units="g")
```

As we can see, we can get the default catalogue units and perform direct conversions. There are a number of convenience utilities, such as comoving/physical conversions, masks, and verbose output ([see documentation](../api/catalogues.rst)). We can also load multiple datasets using the `get_datasets()` method:

```python
galaxy_stellar_masses, galaxy_gas_masses = catalogue.galaxies.get_datasets(
    names=["mass_star", "mass_gas"],
    to_units=["Msun", "kg"],
)
```

The catalogue contains the comprehensive membership hierarchy, allowing you to map haloes, subhaloes, and galaxies. Suppose we wish to get the mass of the field haloes of each galaxy. For this we can use the `get_membership()` method to access the membership datasets, then mask the halo data accordingly:

```python
field_halo_indices = catalogue.galaxies.get_membership("field_halo_index")
parent_field_masses = catalogue.haloes.get_dataset("mass_total", mask=field_halo_indices)
```

:::{tip}
Field haloes are top-level haloes with no parent. `parent_halo_indices` accesses the immediate parent, which may be a subhalo.
:::

Accessing the central galaxies of haloes is similarly straightforward:

```python
central_galaxy_indices = catalogue.haloes.get_membership("central_galaxy_index")
central_masses = catalogue.galaxies.get_dataset("mass_baryon", mask=central_galaxy_indices)
```

We can further build on the example by involving subhaloes and masks.

```python
depths = catalogue.haloes.get_membership("depth")
subhalo_indices = depths > 0  # haloes at depth 0 are field haloes

# subhalo central galaxies
subhalo_masses = catalogue.haloes.get_dataset("mass_total", mask=subhalo_indices)
centrals_of_subhaloes = central_galaxy_indices[subhalo_indices]
subhalo_central_masses = catalogue.galaxies.get_dataset("mass_baryon", mask=centrals_of_subhaloes)

# parents of subhalos
parents_halo_indices = catalogue.haloes.get_membership("parent_halo_index")
parents_of_subhaloes = parent_halo_indices[subhalo_indices]
subhalo_parent_masses = catalogue.haloes.get_dataset("mass_total", mask=parents_of_subhaloes)

# get all the members of a halo
halo_of_interest = 22
halo_22_galaxies = catalogue.haloes.get_galaxies(halo_index=halo_of_interest)
halo_22_subhaloes = catalogue.haloes.get_subhaloes(halo_index=halo_of_interest)  # includes children of children

# get HI masses of all galaxies with > 32 stars
n_stars = catalogue.galaxies.get_dataset("n_star")
resolution_mask = n_stars > 32
HI_masses = catalogue.galaxies.get_dataset("mass_HI", mask=resolution_mask)
```

### Particle Lists

We continue with the [particle lists](https://caesar.readthedocs.io/en/latest/catalog.html#particle-lists) example where we wish to compute `Z_oxy`, the SFR-weighted oxygen metallicity. 

```python
import h5py
import numpy as np

raw_snap_path = Path("/path/to/snapshot.hdf5")

with h5py.File(raw_snap_path, "r") as snap:  # for sake of demonstration assume we can access the fields like so
    oxygen = snap["metallicity"][:]
    sfr = snap["sfr"][:]

# get indices of gas particles in snapshot
all_indices = catalogue.galaxies.get_membership("gas_indices")
# get offsets where the particles of each galaxy begin in all_indices (size is n_galaxies + 1)
all_offsets = catalogue.galaxies.get_membership("gas_offsets")
starts = all_offsets[:-1]  # ignore last element of offsets (it tells you where the last group ends)

numerator = np.add.reduceat(oxygen[all_indices] * sfr[all_indices], starts)
denominator = np.add.reduceat(sfr[all_indices], starts)
Z_oxy = numerator / denominator 
```

### Galaxy Lists

We will also examine the [halo-galaxy lists](https://caesar.readthedocs.io/en/latest/catalog.html#halo-data) example where we want to get the galaxies belonging to a specific halo. To get galaxy quantities from their haloes, you will need the indices into the data in `galaxies`. 

```python
galaxy_idx = catalogue.haloes.get_galaxies(halo_index=0)
```

This will return the indices for the specified halo at halo_index. You can then use:

```python
halo_galaxy_masses = catalogue.galaxies.get_dataset("mass_star", mask=galaxy_idx)
```

To get the quantity (or quantities, with `get_datasets()`) of interest for the galaxies belonging to that halo.

### Standalone Photometry

For standalone photometry (and other stages), you can instantiate an `OctaviusAnalyser` object with `build_analyser()`. This will let you rerun analysis on a subset of groups from the catalogue. Please refer to the {ref}`photometry-specific section <usage-photometry>` for more information.


```python
config = oc.OctaviusConfig(...)
analyser = oc.build_analyser(catalogue=catalogue, config=config)

# standalone photometry can perform rotations and keep spectra
photometry_result = analyser.compute_photometry(group_indices=galaxy_idx, orientation="face-on", keep_spectra=True)

spectra = photometry_result["spectra"]
```

## API Differences

- Octavius entirely drops the use of [yt](https://github.com/yt-project) throughout the codebase.
- Octavius uses `pathlib.Path` objects to access files (this is in the standard library).
- Octavius has no Cython dependency; performance-critical code is written in numba.
- The data manager and object access patterns in Caesar are replaced by modularised dataclasses and aligned arrays in Octavius.
- Idiomatic usage of Caesar catalogues uses list comprehension; Octavius relies on vectorised numpy.
- Dataset names follow a standardised convention and may have changed.
- Octavius does not yet support built-in progenitor matching.

## General Differences

- Caesar works in 32-bit data, whereas Octavius generally uses 64-bit to avoid overflow bugs.
- Certain Caesar dependencies such as `pygadgetreader`, `joblib`, and `synphot` are not needed in Octavius.
- Octavius verifies dimensional consistency with astropy.
- Octavius includes more guards and checks to prevent unintended behaviour.
- Functions in Octavius are unit tested and regression tests exist for catalogues.
- Several hardcoded parameters in Caesar are modifiable in Octavius through the configuration file.
- Octavius fixes many bugs in Caesar, some quite significant.

(my-output-is-different)=
## My Output is Different

If you have identified a feature present in Caesar but absent in Octavius, or have identified implausible differences in the physical output, please open an issue on the [source repository](https://github.com/jp-duminy/octavius/issues) or [contact the developer](mailto:jp@duminy.org). Some potential causes for differences may include:

- Catalogue units
- Galaxy assignments changing due to algorithm fixes (this will propagate through all galaxy quantities)
- Minor rounding differences due to dtypes
- Configuration file differences
- Bug fixes
