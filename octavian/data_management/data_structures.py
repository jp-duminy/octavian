"""

Octavian data structures (readers, stores, dataclasses).
This is a modularised version of the old DataManager, its functionality divided amongst smaller objects.

"""

# defaults
from pathlib import Path
from dataclasses import dataclass

# others
import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM

# from the backend
from octavian.data_management.conventions import (
    CONSTANTS, DTYPES, SimulationAttributes, SnapshotReader,
    gizmo_unit_conversion_factor, derive_stellar_age,
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
    dataset_map = {"pos": "Coordinates",
                   "vel": "Velocities",
                   "mass": "Masses",
                   "potential": "Potential",
                   "temperature": "InternalEnergy", # FIXME: should map properly
                   "rho": "Density",
                   "nh": "NeutralHydrogenAbundance",
                   "sfr": "StarFormationRate",
                   "formation_time": "StellarFormationTime",
                   "metallicity": "Metallicity",
                   "fh2": "FractionH2",
                   "bhmass": "BH_Mass",
                   "bhmdot": "BH_Mdot",
                   "HaloID": "HaloID", # TODO: come back to this when doing external halo finders
                            }
    
    inverse_ptype_map = {v: k for k, v in ptype_map.items()} # for convenience

    def __init__(self, snapshot_file: Path):

        self.snapshot_file = snapshot_file

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
            omega_matter = header["Omega0"]
            omega_lambda = header["OmegaLambda"]
            a = header["Time"]
            redshift = header["Redshift"]

            cosmology = FlatLambdaCDM(H0=100*h, Om0=omega_matter)
            time_gyr = cosmology.age(redshift).value
            Hz = 100 * h * np.sqrt(omega_lambda + omega_matter * a**-3) * CONSTANTS.HUBBLE_UNIT
            E_z = np.sqrt(omega_lambda * a**-2 + omega_matter * a**-3)
            rhocrit = (3. * Hz**2 / (8 * np.pi * CONSTANTS.G_CGS)) * CONSTANTS.RHO_CGS_TO_MSUN_KPC3
            omega_matter_z = (omega_matter * a**-3) / E_z**2

            self.simulation_attributes = SimulationAttributes(

                h = h,
                boxsize = header["BoxSize"] / h,
                a = a,
                redshift = redshift,
                omega_matter = omega_matter,
                omega_lambda = omega_lambda,

                cosmology = cosmology, # perhaps slightly hacky but astropy builds all its cosmo classes on FLRW
                time_gyr = time_gyr,
                time = time_gyr * CONSTANTS.GYR_S,
                
                Hz = Hz,
                rhocrit = rhocrit,
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

        if dataset == "formation_time":
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
    __slots__ = ("columns", "n_particles", "ptype") # fixed slots

    def __init__(self, ptype: str, n_particles: int):

        self.ptype = ptype
        self.n_particles = n_particles
        self.columns: dict[str, np.ndarray] = {} # O(1) lookup on a lightweight np array (preconverted units)

    def __getitem__(self, key: str) -> np.ndarray:
        """
        Use ParticleStore["key"] to access array.
        """
        return self.columns[key]
    
    def get_columns(self, keys: str | list[str]) -> np.ndarray:
        """
        For 3D quantities (position, velocity, etc.): use ParticleStore.get_columns(["key1", "key2"])
        """
        if isinstance(keys, list):
            return np.column_stack([self.columns[k] for k in keys])
        return self.columns[keys]
    
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
    
    def release(self, *names: str) -> None: # chose *names as an alternative to names: list[str] for readability
        """
        Call ParticleStore.release("key1", "key2") to delete references to no-longer needed columns (like drop from old datamanager).

        Internally, this is effectively the same as the del method.
        """
        for name in names: self.columns.pop(name, None) 

class GroupStore:
    """
    Effectively the same idea as the ParticleStore class, but storing group-level information.
    """
    def __init__(self, group_ids: np.ndarray, original_ids: np.ndarray | None = None): # original_ids is for external halo readers

        self.group_ids = group_ids
        self.n_groups = len(group_ids)
        self.columns: dict[str, np.ndarray] = {}

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
    
    def get_columns(self, keys: str | list[str]) -> np.ndarray:
        """
        For multi-column attributes (if they pop up): use GroupStore.get_columns(["key1", "key2"])
        """
        if isinstance(keys, list):
            return np.column_stack([self.columns[k] for k in keys])
        return self.columns[keys]
    
    def __setitem__(self, key: str, array: np.ndarray) -> None:
        """
        Use GroupStore["key"] = array to add/modify an entry.
        """
        assert array.shape[0] == self.n_groups
        self.columns[key] = array

    def __contains__(self, key: str) -> bool:
        """
        Controls {"key" in GroupStore} behaviour
        """
        return key in self.columns
    
    def get_indexer(self, group_id: np.ndarray) -> np.ndarray:
        """
        Returns the corresponding index array from the group ID array (vectorised).
        """
        return self.id_to_idx[group_id]
    
@dataclass(slots=True) # no frozen=True as this is inherently supposed to be mutable
class SimulationData:
    """
    Object containing simulation data ready for analysis:

    - hdf5 groups converted to np.ndarrays in code units with correct datatype
    - group IDs (at instantiation, HIDs)
    - Simulation-specific attributes (boxsize, cosmological parameters, etc).
    """
    simulation:  SimulationAttributes
    particles:   dict[str, ParticleStore]
    groups:      dict[str, GroupStore]

# NOTE: temporary function to convert a data_manager object into a simulationdata (part of the decoupling)
from octavian.data_management.data_manager import DataManager

def convert_data_manager(data_manager: DataManager) -> SimulationData:
    """
    Converts a datamanager into a simulationdata; for the sunsetting of datamanager.
    """
    sim = data_manager.simulation
    simulation = SimulationAttributes(
        h=sim['h'],
        boxsize=sim['boxsize'] / sim['h'],
        a=sim['a'],
        redshift=sim['redshift'],
        omega_matter=sim['O0'],
        omega_lambda=sim['Ol'],
        cosmology=data_manager.cosmology,
        time_gyr=sim['time_gyr'],
        time=sim['time'],
        Hz=sim['Hz'],
        rhocrit=sim['rhocrit'],
        E_z=sim['E_z'],
        omega_matter_z=sim['Om_z'],
        r200_factor=sim['r200_factor'],
    )

    particles = {}
    for ptype in data_manager.config['ptypes']:
        df = data_manager.data[ptype]
        store = ParticleStore(ptype=ptype, n_particles=len(df))
        for col in df.columns:
            arr = df[col].to_numpy()
            if arr.dtype == 'category':
                arr = arr.astype(DTYPES.get(col, np.int64))
            store[col] = arr
        particles[ptype] = store

    groups = {}
    if hasattr(data_manager, 'group_data'):
        for group_name, df in data_manager.group_data.items():
            group_ids = df.index.to_numpy().astype(np.int64)
            gstore = GroupStore(group_ids=group_ids)
            for col in df.columns:
                gstore[col] = df[col].to_numpy()
            groups[group_name] = gstore

    return SimulationData(
        simulation=simulation,
        particles=particles,
        groups=groups,
    )