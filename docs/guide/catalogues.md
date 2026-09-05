# Catalogues

For rerunning analysis routines on a catalogue, see [standalone analysis.](standalone_analysis.md)

## Overview

Octavius outputs lightweight HDF5 analysis catalogues and provides a convenient object-oriented API for accessing their contents. In principle, the `OctaviusCatalogue` object returned from `load_catalogue()` covers all use cases; however, for ease-of-access, the file is structured such that it can be easily accessed with `h5py` directly.

The catalogue contains ordered arrays of group-level properties for haloes and galaxies. Care is taken in the pipeline to ensure this ordering is deterministic and invariant whether running in serial or various parallel configurations; the principal ordering is by mass, so the zeroth element corresponds to the most massive group. The groups corresponding to haloes and galaxies are named `halo_data` and `galaxy_data` respectively; in these groups, the enabled pipeline stages produce eponymous groups. Importantly, `halo_data` and `galaxy_data` contain `/membership`, which contains a host of datasets for mapping haloes and galaxies to each other, as well as their constituent particles.

Furthermore, the catalogue contains cosmological information from the snapshot in the top-level `header` group. This includes a range of various useful quantities computed with the `astropy.cosmology` module. There is also a `metadata` group which contains information including the configuration file parameters used, the Octavius version, timestamp, and git commit used to generate the catalogue.

The datasets follow a standardised naming convention: first comes the quantity name, then any identifing information, then what particle type or combination thereof it belongs to, all underscore-separated. For example, `radius_max_baryon` is the maximum radius any baryonic particle lies from the centre of a group.

## Catalogue Object

The `OctaviusCatalogue` object is designed to be easy and convenient to interface with. It provides on-demand access to datasets with unit/physical conversions, masks, and descriptions. The catalogue object can be instantiated by calling `load_catalogue()` on a `pathlib.Path` object pointing to the filepath of an Octavius HDF5 catalogue. 

```python
from pathlib import Path
from octavius import load_catalogue

catalogue_path = Path("/path/to/catalogue.hdf5")
catalogue = load_catalogue(catalogue_path)

catalogue.close()
```

Basic metadata and snapshot information can be shown by printing the catalogue object, or accessing its attributes:

```python
print(cat)  # outputs a basic summary
print(catalogue.redshift)
```

:::{tip}
The catalogue can be closed with `catalogue.close()` once you have read all desired datasets.
:::

## Accessing Properties

`halo_data` and `galaxy_data` exist as `GroupCollection` objects on the catalogue, accessible through `.haloes` and `.galaxies` respectively. We will start by examining the `get_dataset()` and `get_datasets()` methods, which are the primary access route for group-level property arrays.

:::{tip}
The `describe()` method on haloes and galaxies can be used to see the available datasets; the `verbose` flag can be set to `True` when loading data to print descriptions.
:::

```python
catalogue.galaxies.describe()  # see which datasets are available
stellar_mass = catalogue.galaxies.get_dataset("mass_star")  # raw read
stellar_mass = catalogue.galaxies.get_dataset("mass_star", verbose=True)  # print dataset descriptions when reading (and unit conversion factor if converting)
stellar_mass_grams = catalogue.galaxies.get_dataset("mass_star", to_units="g")  # unit conversions
galaxy_centres_physical = catalogue.galaxies.get_dataset("com_pos_baryon", to_physical=True)  # comoving -> physical

# get centres of galaxies with more than 1e8 solar masses of stars in physical Mpc
mass_threshold = stellar_mass > 1e8
galaxy_centres_of_interest_mpc = catalogue.galaxies.get_dataset("com_pos_baryon", mask=mass_threshold, to_units="Mpc", to_physical=True)
```

The `get_datasets()` method extends this syntax to conveniently load multiple datasets by passing a list. If unit conversions are desired, a list thereof aligned with the dataset list should be provided.


```python
galaxy_datasets = ["com_pos_baryon", "smbh_mass", "inertia_tensor_gas"]
galaxy_dataset_units = ["Mpc", "Msun", "Msun*Mpc**2"]

galaxy_centres, galaxy_smbh_masses, galaxy_gas_inertia_tensors = catalogue.galaxies.get_datasets(
    names=galaxy_datasets,
    mask=mass_threshold,
    to_units=galaxy_dataset_units,
    to_physical=True
)
```

## Accessing Membership

The `get_membership()` method provides access to membership indexing, both at group and particle level. Continuing the same example, suppose we now want to map the galaxies of interest in our analysis to the properties of their field haloes:

```python
raw_field_halo_idx = catalogue.galaxies.get_membership(name="field_halo_index")
field_halo_idx = raw_field_halo_idx[mass_threshold]
```

`field_halo_idx` is an index into `halo_data` for the field halo properties of the galaxies, aligned to the galaxy arrays. We can now access the properties of these haloes using the `mask` argument:

```python
halo_datasets = ["temperature_virial", "velocity_dispersion_total"]
halo_dataset_units = ["K", "pc/yr"]

halo_temps, halo_vel_disps = catalogue.haloes.get_datasets(
    names=halo_datasets,
    mask=field_halo_idx,
    to_units=halo_dataset_units,
)
```

This also works for extracting the child galaxies of a halo: a convenience method, `get_galaxies()`, is provided for this purpose. 

```python
galaxy_idx = catalogue.haloes.get_galaxies(group_index=22)  # get indices into galaxy_data of the galaxies which are children of halo 22
child_gal_mass_grams = catalogue.galaxies.get_dataset("mass_star", mask=galaxy_idx, to_units="g")
```

It is worth noting not all haloes have galaxies. This brings us to the important concept of a _sentinel value_, which indicates a membership mapping does not exist. The sentinel value is -1 everywhere; you should therefore take care to mask out any -1 values in membership arrays, as index -1 is valid in idiomatic numpy and will **incorrectly return the last element of an array** instead of nothing.

```python
parent_halo_idx = catalogue.haloes.get_membership("parent_halo_index")  # parent halo index (-1 for field haloes)
parent_halo_idx = parent_halo_idx[parent_halo_idx > 0]  # mask out -1 sentinel value
```

Crucially, membership mapping is extended to the constituent particles of the groups, allowing you to retrieve their data from the raw snapshot file. This is made possible by the `get_particle_indices()` method, which returns an array of indices into the original snapshot for a specified particle type of a group.

```python
halo_22_gas_idx = catalogue.haloes.get_particle_indices(group_index=22, ptype="gas")
```

This means any properties not in the catalogue can be recomputed by indexing the raw particle-level datasets directly from the snapshot.

:::{note}
`get_particle_indices()` and `get_galaxies()`(for haloes) can only be called for one group at a time.
:::

## Accessing Simulation Information

The catalogue also provides a `SimInfo` object, which contains an assortment of cosmological and snapshot-specific information such as the boxsize, redshift, critical density, mean interparticle separation, etc. This can be accessed by calling `sim_info`:

```python
boxsize = catalogue.sim_info("boxsize", to_units="Mpc", physical=True)
scale_factor = catalogue.sim_info("scale_factor")
```

## Useful Information

The catalogue arrays are designed around the numpy idiom of vectorised array operations. The IDs in the catalogue are their positional indices into their group-level data; for example, halo 22 will always live at `array[22]` for any dataset in `halo_data`. It is important to keep track of indices. 

Care is taken through the pipeline to preserve global ordering, meaning if you rerun an analysis, you can expect halo 22 will refer to the same collection of particles.

Useful numpy functions for working with both particle and group-level data include:

- [np.reduceat](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.reduceat.html)
- [np.add.at](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html)
- [np.bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html)
- [np.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
- [np.searchsorted](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html)
- [np.piecewise](https://numpy.org/doc/stable/reference/generated/numpy.piecewise.html)

Useful numpy functions for handling indices and masks include:

- [np.lexsort](https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html)
- [np.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html)
- [np.unique](https://numpy.org/doc/stable/reference/generated/numpy.unique.html)
- [np.argsort](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html)
- [np.isin](https://numpy.org/doc/stable/reference/generated/numpy.isin.html)

