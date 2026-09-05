"""

Standard ways of accessing data in the catalogues.

"""

from pathlib import Path
import octavius as oc

catalogue = Path("/path/to/catalogue.hdf5")

catalogue = oc.load_catalogue(catalogue_path=catalogue)

# shows basic catalogue info (redshift, boxsize, n_haloes, n_galaxies)
print(catalogue)
print(catalogue.timestamp)  # metadata attributes exist

# suppose I want to focus on galaxies above a certain HI mass threshold
HI_mass = catalogue.galaxies.get_dataset(name="mass_HI", to_units="Msun")
HI_mask = HI_mass > 1e8

catalogue.galaxies.describe()  # tells me what properties are available in galaxies

# we'll go with some random datasets
datasets_I_want = ["mass_star", "BoverT_baryon", "inertia_tensor_gas", "smbh_mdot", "mass_HI_30kpc"]
their_units = ["kg", "", "g*Mpc**2", "Msun/yr", "Msun"]  # I can specify their units too (we'll go for weird units)

star_mass, baryon_bovert, gas_tensor, smbh_mdot, HI_30kpc_mass = catalogue.galaxies.get_datasets(
    names=datasets_I_want,
    mask=HI_mask,  # only access galaxies above the HI mass threshold
    to_units=their_units,
    to_physical=True,  # I don't want anything comoving
    verbose=True,  # with this flag I can see what I'm loading, how many galaxies pass the mask, the dtype of the datasets, and the unit conversion factor
)

boxsize = catalogue.sim_info("boxsize", to_units="Mpc", physical=True)  # get cosmological info
