"""

Advanced indexing and membership access with catalogues.

"""

from pathlib import Path
import h5py
import numpy as np
import octavius as oc

catalogue_path = Path("/path/to/catalogue.hdf5")

catalogue = oc.load_catalogue(catalogue_path=catalogue_path)

# suppose I want to analyse the field haloes of galaxies above a certain HI mass threshold
HI_mass = catalogue.galaxies.get_dataset(name="mass_HI", to_units="Msun")
HI_mask = HI_mass > 1e8
gal_stellar_mass = catalogue.galaxies.get_dataset(name="mass_star", mask=HI_mask)

# get field halo indices of these galaxies
raw_field_halo_idx = catalogue.galaxies.get_membership(
    name="field_halo_index"
)  # the get_membership method can return variable-length arrays so does not allow masking
field_halo_idx = raw_field_halo_idx[HI_mask]  # mask after loading
assert len(field_halo_idx) == len(gal_stellar_mass), "This should pass."

# and let's say I want some quantities out of their field haloes
parent_datasets_I_want = ["mass_star", "inertia_tensor_total", "temperature_virial", "rmax"]
parent_units = ["kg", "g*Mpc**2", "K", "Mpc"]
parent_star_mass, parent_I_tensor, parent_virial_temp, parent_rmax = catalogue.haloes.get_datasets(
    names=parent_datasets_I_want,
    mask=field_halo_idx,  # access only the galaxies' field haloes
    to_units=parent_units,
    to_physical=True,
)

# now what if we're going the other way? suppose I am doing an analysis and halo 23 caught my eye; I want to know its galaxies
child_gal_idx = catalogue.haloes.get_galaxies(halo_index=23)

child_metal_weighted_ages = catalogue.galaxies.get_dataset(  # I want to know their metallicity-weighted ages
    name="age_star_metal_weighted",
    to_units="Gyr",
    mask=child_gal_idx,
)

# I'm forming my analysis and want to grab some cosmological info from the cataloguealogue
print(catalogue.sim_info.keys())  # let's see what's available

Hz = catalogue.sim_info("Hz")
print(catalogue.sim_info.description("Hz"))
boxsize = catalogue.sim_info("boxsize", to_units="Mpc", physical=True)
scale_factor = catalogue.sim_info("scale_factor")
scale_factor = catalogue.scale_factor  # this one is also on the cataloguealogue object for convenience

# now we want to compute a new property, the oxygen metallicity, of the galaxies in halo 23 which are > 9Gyr old

raw_snap = Path("/path/to/snapshot.hdf5")

with h5py.File(raw_snap, "r") as snap:
    oxygen = snap["metallicity"][:]
    sfr = snap["sfr"][:]

age_mask = child_metal_weighted_ages > 9  # galaxies with metal-weighted mean ages greater than 9 Gyr
idx_of_interest = child_gal_idx[age_mask]  # their indices

Z_oxy = np.empty(shape=len(idx_of_interest), dtype=np.float64)

for i, gal_idx in enumerate(idx_of_interest):
    snap_idx = catalogue.galaxies.get_particle_indices(
        ptype="gas", group_index=gal_idx
    )  # retrieve their particle-level indices
    Z_oxy[i] = np.sum(oxygen[snap_idx] * sfr[snap_idx]) / np.sum(sfr[snap_idx])

# or, alternatively, for every galaxy
all_indices, all_offsets = (
    catalogue.galaxies.get_membership("gas_indices"),
    catalogue.galaxies.get_membership("gas_offsets"),
)
starts = all_offsets[:-1]  # ignore last element of offsets (it tells you where the last group ends)
numerator = np.add.reduceat(oxygen[all_indices] * sfr[all_indices], starts)
denominator = np.add.reduceat(sfr[all_indices], starts)

Z_oxy = numerator / denominator  # final quantity, vectorised
