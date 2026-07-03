"""

Octavian data structures (readers, stores, dataclasses).
This is a modularised version of the old DataManager, its functionality divided amongst smaller objects.

"""

# semantic
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_management.pipeline_management import Internals

# defaults
from pathlib import Path
from dataclasses import dataclass

# others
import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM

# from the backend
from octavian.data_management.conventions import (
    OctavianConstants, DTYPES, SimulationAttributes, SnapshotReader,
    gizmo_unit_conversion_factor, derive_stellar_age, calculate_temperature,
    calculate_hydrogen_number_density, calculate_mean_interparticle_separation
)

class GizmoReader(SnapshotReader):
    """
    Gizmo (SIMBA) snapshot reader; assumes default units.
    """
    ptype_map = {"PartType0": "gas",
                 "PartType1": "dm",
                 "PartType4": "star",
                 "PartType5": "bh",
                          }
    dataset_map = {"pos":                   "Coordinates",
                   "vel":                   "Velocities",
                   "mass":                  "Masses",
                   "potential":             "Potential",
                   "internal_energy":       "InternalEnergy", # FIXME: should map properly
                   "electron_abundance":    "ElectronAbundance",
                   "rho":                   "Density",
                   "fHI":                   "NeutralHydrogenAbundance",
                   "sfr":                   "StarFormationRate",
                   "age":                   "StellarFormationTime", # NOTE: we compute age from formationtime, but using "age" is for reader agnosticity
                   "metallicity":           "Metallicity",
                   "helium_fraction":       "Metallicity", # helium fraction is metallicity[:, 1] (metallicity is nx11 array)
                   "fH2":                   "FractionH2",
                   "bhmass":                "BH_Mass",
                   "bhmdot":                "BH_Mdot",
                   "HaloID":                "HaloID", # TODO: come back to this when doing external halo finders
                   "particle_index":        "particle_index" 
                            }
    
    inverse_ptype_map = {v: k for k, v in ptype_map.items()} # for convenience

    def __init__(self, snapshot_file: Path, constants: OctavianConstants):

        self.snapshot_file = snapshot_file
        self.constants = constants

        self.read_header()
        self.unit_conversions = {
            dataset: gizmo_unit_conversion_factor(dataset, self.simulation_attributes.h, self.simulation_attributes.a)
            for dataset in self.dataset_map
            if dataset in DTYPES 
        }

    def read_header(self) -> SimulationAttributes:
        """
        Parses header attributes into a dataclass (does derived quantities too); assumes FlatLambdaCDM
        """
        with h5py.File(self.snapshot_file, "r") as f:

            header = f["Header"].attrs

            h = header["HubbleParam"]
            boxsize = header["BoxSize"] / h
            omega_matter = header["Omega0"]
            omega_lambda = header["OmegaLambda"]
            a = header["Time"]
            redshift = header["Redshift"]
            n_star, n_gas = header["NumPart_Total"][4], header["NumPart_Total"][0]

            cosmology = FlatLambdaCDM(H0=100*h, Om0=omega_matter)
            time_gyr = cosmology.age(redshift).value
            Hz = 100 * h * np.sqrt(omega_lambda + omega_matter * a**-3) * self.constants.HUBBLE_UNIT
            E_z = np.sqrt(omega_lambda + omega_matter * a**-3)
            rhocrit = (3. * Hz**2 / (8 * np.pi * self.constants.G_CGS)) * self.constants.RHO_CGS_TO_MSUN_KPC3
            omega_matter_z = (omega_matter * a**-3) / E_z**2

            self.simulation_attributes = SimulationAttributes(

                h = h,
                boxsize = boxsize,
                a = a,
                redshift = redshift,
                omega_matter = omega_matter,
                omega_lambda = omega_lambda,
                mis = calculate_mean_interparticle_separation(n_star=n_star, n_gas=n_gas, boxsize=boxsize),

                cosmology = cosmology, # perhaps slightly hacky but astropy builds all its cosmo classes on FLRW
                time_gyr = time_gyr,
                time = time_gyr * self.constants.GYR_S,
                
                Hz = Hz,
                rhocrit = rhocrit,
                rhocrit_comoving = rhocrit * a**3,
                E_z = E_z,
                omega_matter_z = omega_matter_z,
                r200_factor = (200 * 4./3. * np.pi * omega_matter_z * rhocrit * a**3)**(-1./3.)
            )

        return self.simulation_attributes

    def available_ptypes(self) -> list[str]:
        """
        Finds which Octavia-compatible ptypes are available in the snapshot.
        """
        with h5py.File(self.snapshot_file) as f:
            return [self.ptype_map[k] for k in f.keys() if k in self.ptype_map]
        
    def read_dataset(self, ptype: str, dataset: str) -> np.ndarray:
        """
        Convert a HDF5 dataset in the snapshot to a numpy array with the correct dtype (for floating point precision).
        """
        hdf5_group = self.inverse_ptype_map[ptype]
        hdf5_name = self.dataset_map[dataset]

        with h5py.File(self.snapshot_file, "r") as f:
            raw_hdf5_array = f[hdf5_group][hdf5_name][:]

        if dataset == "metallicity": # I think it's okay to have these as conditionals by way of being explicit
            raw_hdf5_array = raw_hdf5_array[:, 0]

        if dataset == "helium_fraction":
            raw_hdf5_array = raw_hdf5_array[:, 1]

        if dataset == "age":
            raw_hdf5_array = derive_stellar_age(formation_time=raw_hdf5_array, time_gyr=self.simulation_attributes.time_gyr, 
                                                cosmology=self.simulation_attributes.cosmology)
            
        conversion_factor = self.unit_conversions.get(dataset, 1.0)
        if conversion_factor != 1.0: # skip unnecessary multiplication on (potentially giant) arrays
            raw_hdf5_array = raw_hdf5_array * conversion_factor

        return raw_hdf5_array.astype(DTYPES.get(dataset, np.float64))
    
class ParticleStore:
    """
    Stores dictionaries of properties for one particle type.
    """
    __slots__ = ("columns", "n_particles", "ptype", "is_baryonic") # fixed slots

    def __init__(self, ptype: str, n_particles: int, is_baryonic: bool):

        self.ptype = ptype
        self.n_particles = n_particles
        self.is_baryonic = is_baryonic
        self.columns: dict[str, np.ndarray] = {} # O(1) lookup on a lightweight np array (preconverted units)

    def __getitem__(self, key: str) -> np.ndarray:
        """
        Use ParticleStore["key"] to access array.
        """
        return self.columns[key]
    
    def __setitem__(self, key: str, array: np.ndarray) -> None:
        """
        Use ParticleStore["key"] = array to add/modify an entry.
        """
        assert array.shape[0] == self.n_particles
        self.columns[key] = array

    def __contains__(self, key: str) -> bool:
        """
        Controls {"key" in ParticleStore} behaviour
        """
        return key in self.columns

    def __len__(self) -> int:
        """
        Allows you to use len() on the ParticleStore to find the length of its arrays, avoiding using something (I usually used mass) as a proxy.
        """
        return self.n_particles
    
    def release(self, *names: str) -> None: # chose *names as an alternative to names: list[str] for readability
        """
        Call ParticleStore.release("key1", "key2") to delete references to no-longer needed columns (like drop from old datamanager).

        Internally, this is effectively the same as the del method.
        """
        for name in names: self.columns.pop(name, None) 

def build_particle_stores(
    reader: SnapshotReader, 
    internals: Internals, 
    process_ptypes: dict[str, bool],
    constants: OctavianConstants,
) -> dict[str, ParticleStore]:
    """
    Constructs basic particle stores using information from what is available in the snapshot, and what the config specifies to process.
    """
    available = reader.available_ptypes()
    requested = [pt for pt in available if process_ptypes.get(pt, True)]

    particles: dict[str, ParticleStore] = {}

    for ptype in requested:

        halo_ids = reader.read_dataset(ptype, "HaloID")
        store = ParticleStore(ptype=ptype, n_particles=len(halo_ids), is_baryonic = ptype in internals.baryonic_ptypes)
        store["HaloID"] = halo_ids

        for dataset in ["mass", "pos", "vel"]:
            store[dataset] = reader.read_dataset(ptype, dataset)

        if ptype == "gas":

            internal_energy = reader.read_dataset(ptype, "internal_energy")

            try:
                electron_abundance = reader.read_dataset(ptype, "electron_abundance")
            except KeyError:
                electron_abundance = np.ones(store.n_particles)

            helium_fraction = reader.read_dataset(ptype, "helium_fraction")
            store["temperature"] = calculate_temperature(internal_energy=internal_energy, electron_abundance=electron_abundance,
                                                        helium_fraction=helium_fraction, constants=constants)

        store["ptype"] = np.full(len(store), ptype)
        particles[ptype] = store

    return particles

class GroupStore:
    """
    Effectively the same idea as the ParticleStore class, but storing group-level information.
    """
    def __init__(self, group_ids: np.ndarray, group_key: int, original_ids: np.ndarray | None = None): # original_ids is for external halo readers

        self.group_ids = group_ids
        self.n_groups = len(group_ids)
        self.columns: dict[str, np.ndarray] = {}
        self.group_key = group_key

        max_id = group_ids.max() if self.n_groups > 0 else 0 # TODO: add this to logger since guard should not be hit in principle
        self.id_to_idx = np.full(shape=max_id+1, fill_value=-1, dtype=DTYPES["pid"])
        self.id_to_idx[group_ids] = np.arange(self.n_groups, dtype=DTYPES["csr_offsets"])

        if original_ids is not None:
            self.columns["original_id"] = original_ids

    def __getitem__(self, key: str) -> np.ndarray:
        """
        Use GroupStore["key"] to access array.
        """
        return self.columns[key]
    
    def __setitem__(self, key: str, array: np.ndarray) -> None:
        """
        Use GroupStore["key"] = array to add/modify an entry.
        """
        assert array.shape[0] == self.n_groups
        self.columns[key] = array

    def write_batch(self, results: dict[str, np.ndarray], suffix: str = "") -> None:
        """
        Writes the data from a results dictionary into the GroupStore.
        """
        for column_name, column_data in results.items():

            store_key = f"{column_name}_{suffix}" if suffix else column_name

            if store_key in self.columns:
                raise KeyError(f"Column '{store_key}' already exists in GroupStore.")
            
            self.columns[store_key] = column_data

    def __contains__(self, key: str) -> bool:
        """
        Controls {"key" in GroupStore} behaviour
        """
        return key in self.columns
    
    def get_indexer(self, group_id: np.ndarray) -> np.ndarray:
        """
        Returns the corresponding index array from the group ID array (vectorised).
        """
        id_to_idx = np.full(len(group_id), -1, dtype=DTYPES["csr_offsets"])
        valid = (group_id >= 0) & (group_id < len(self.id_to_idx))
        id_to_idx[valid] = self.id_to_idx[group_id[valid]] # mask valid indices (-1, the sentinel, is the last array element)

        return id_to_idx
    
def build_group_store(particles: dict[str, ParticleStore], group_type: str) -> GroupStore:
    """
    Constructs GroupStore classes (for halos and galaxies) from the ParticleStores.
    """
    group_key = {"halos": "HaloID", "galaxies": "GalID"}[group_type]
    ids = [particles[ptype][group_key] for ptype in particles]
    unique_ids = np.unique(np.concatenate(ids))

    if group_type == "galaxies":
        unique_ids = unique_ids[unique_ids != -1]

    return GroupStore(group_ids=unique_ids, group_key=group_key)
    
@dataclass(slots=True) # no frozen=True as this is inherently supposed to be mutable
class SimulationData:
    """
    Object containing simulation data ready for analysis:

    - hdf5 groups converted to np.ndarrays in code units with correct datatype
    - group IDs (at instantiation, HIDs)
    - Simulation-specific attributes (boxsize, cosmological parameters, etc).
    """
    simulation:  SimulationAttributes
    constants:   OctavianConstants
    particles:   dict[str, ParticleStore]
    groups:      dict[str, GroupStore]

    @property
    def available_ptypes(self) -> list[str]:
        """
        Returns a list of available particle types (total), in Octavian-internal convention.
        """
        return list(self.particles.keys())

    @property
    def available_baryonic_ptypes(self) -> list[str]:
        """
        Returns a list of available particle types (baryonic), in Octavian-internal convention.
        """
        return [pt for pt, store in self.particles.items() if store.is_baryonic]
