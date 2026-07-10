"""

Octavian data structures (readers, stores, dataclasses).
This is a modularised version of the old DataManager, its functionality divided amongst smaller objects.

"""

# semantic
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_management.pipeline_management import Internals
    from octavian.data_management.conventions import OctavianConstants
from octavian.log import get_logger

# defaults
from pathlib import Path
from dataclasses import dataclass

# others
import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM

# from the backend
from octavian.data_management.conventions import (
    DTYPES,
    SimulationAttributes,
    SnapshotReader,
    gizmo_unit_conversion_factor,
)

from octavian.data_management.physics import (
    derive_stellar_age,
    gizmo_temperature,
    derive_simulation_attributes,
)

logger = get_logger()


class GizmoReader(SnapshotReader):
    """
    Gizmo (SIMBA) snapshot reader; assumes default units.
    """

    ptype_map = {
        "PartType0": "gas",
        "PartType1": "dm",
        "PartType4": "star",
        "PartType5": "bh",
    }
    dataset_map = {
        "pos": "Coordinates",
        "vel": "Velocities",
        "mass": "Masses",
        "potential": "Potential",
        "internal_energy": "InternalEnergy",
        "electron_abundance": "ElectronAbundance",
        "rho": "Density",
        "fHI": "NeutralHydrogenAbundance",
        "sfr": "StarFormationRate",
        "age": "StellarFormationTime",  # NOTE: we compute age from formationtime, but using "age" is for reader agnosticity
        "metallicity": "Metallicity",
        "helium_fraction": "Metallicity",  # helium fraction is metallicity[:, 1] (metallicity is nx11 array)
        "fH2": "FractionH2",
        "bhmass": "BH_Mass",
        "bhmdot": "BH_Mdot",
        "particle_index": "particle_index",
    }

    inverse_ptype_map = {v: k for k, v in ptype_map.items()}  # for convenience

    def __init__(self, snapshot_path: Path, constants: OctavianConstants):

        logger.info("Using GIZMO reader.")

        self.snapshot_path = snapshot_path
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
        with h5py.File(self.snapshot_path, "r") as f:
            header = f["Header"].attrs

            h = header["HubbleParam"]
            boxsize = header["BoxSize"] / h
            omega_matter = header["Omega0"]
            omega_lambda = header["OmegaLambda"]
            a = header["Time"]
            redshift = header["Redshift"]
            n_star, n_gas = header["NumPart_Total"][4], header["NumPart_Total"][0]

        flat_lambda_cdm = FlatLambdaCDM(H0=100 * h, Om0=omega_matter)  # always flatlambdacdm for gizmo

        self.simulation_attributes = derive_simulation_attributes(
            cosmology=flat_lambda_cdm,
            h=h,
            a=a,
            redshift=redshift,
            omega_matter=omega_matter,
            omega_lambda=omega_lambda,
            boxsize=boxsize,
            n_star=n_star,
            n_gas=n_gas,
            constants=self.constants,
        )

        return self.simulation_attributes

    def available_ptypes(self) -> list[str]:
        """
        Finds which Octavia-compatible ptypes are available in the snapshot.
        """
        with h5py.File(self.snapshot_path) as f:
            return [self.ptype_map[k] for k in f.keys() if k in self.ptype_map]

    def read_dataset(self, ptype: str, dataset: str) -> np.ndarray:
        """
        Convert a HDF5 dataset in the snapshot to a numpy array with the correct dtype (for floating point precision).
        """
        hdf5_group = self.inverse_ptype_map[ptype]
        hdf5_name = self.dataset_map[dataset]

        with h5py.File(self.snapshot_path, "r") as f:
            raw_hdf5_array = f[hdf5_group][hdf5_name][:]

        if dataset == "metallicity":  # I think it's okay to have these as conditionals by way of being explicit
            raw_hdf5_array = raw_hdf5_array[:, 0]

        if dataset == "helium_fraction":
            raw_hdf5_array = raw_hdf5_array[:, 1]

        if dataset == "formation_time":
            raw_hdf5_array = derive_stellar_age(
                formation_time=raw_hdf5_array,
                time_gyr=self.simulation_attributes.time_gyr,
                cosmology=self.simulation_attributes.cosmology,
            )
            return raw_hdf5_array.astype(DTYPES.get(dataset, np.float64))

        conversion_factor = self.unit_conversions.get(dataset, 1.0)
        if conversion_factor != 1.0:  # skip unnecessary multiplication on (potentially giant) arrays
            raw_hdf5_array = raw_hdf5_array * conversion_factor

        return raw_hdf5_array.astype(DTYPES.get(dataset, np.float64))

    def read_halo_ids(self, ptype: str) -> np.ndarray:
        """
        Reads snapshot-sourced HaloIDs. GIZMO uses 0 as the sentinel value; we map to Octavian's -1.
        """
        hdf5_group = self.inverse_ptype_map[ptype]

        with h5py.File(self.snapshot_path, "r") as f:
            halo_ids = f[hdf5_group]["HaloID"][:].astype(
                DTYPES.get("HaloID", np.int64)
            )  # change dtype here otherwise you get int overflow

        halo_ids -= 1  # shift IDs left to compensate with Octavian sentinel

        return halo_ids

    def read_temperature(self, ptype: str = "gas") -> np.ndarray:
        """
        Reads data to calculate temperature according to method in http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html
        """
        assert ptype == "gas", f"Temperature is configured to be computed from gas, not {ptype}."

        internal_energy = self.read_dataset(ptype, "internal_energy")

        try:
            electron_abundance = self.read_dataset(ptype, "electron_abundance")
        except KeyError:
            electron_abundance = np.ones(shape=len(internal_energy))

        helium_fraction = self.read_dataset(ptype, "helium_fraction")
        temperature = gizmo_temperature(
            internal_energy=internal_energy,
            electron_abundance=electron_abundance,
            helium_fraction=helium_fraction,
            constants=self.constants,
        )

        return temperature


class SwiftReader(SnapshotReader):
    """
    Swift (SWIMBA/KIARA) snapshot reader.
    """

    ptype_map = {
        "PartType0": "gas",
        "PartType1": "dm",
        "PartType4": "star",
        "PartType5": "bh",
    }
    dataset_map = {
        "pos": "Coordinates",
        "vel": "Velocities",
        "mass": "Masses",
        "potential": "Potentials",
        "internal_energy": "InternalEnergies",
        "electron_abundance": "ElectronNumberDensities",
        "rho": "Densities",
        "fHI": "AtomicHydrogenFractions",
        "sfr": "StarFormationRates",
        "age": "BirthScaleFactors",
        "metallicity": "MetalMassFractions",
        "helium_fraction": "ElementMassFractions",  # assuming [Z, He...] GIZMO convention
        "fH2": "MolecularHydrogenFractions",
        "bhmass": "SubgridMasses",
        "bhmdot": "AccretionRates",
        "particle_index": "particle_index",
    }

    inverse_ptype_map = {v: k for k, v in ptype_map.items()}  # for convenience

    def __init__(self, snapshot_path: Path, constants: OctavianConstants):

        logger.info("Using SWIFT reader.")

        self.snapshot_path = snapshot_path
        self.constants = constants

        self.read_header()


class ParticleStore:
    """
    Stores dictionaries of properties for one particle type.
    """

    __slots__ = ("columns", "n_particles", "ptype", "is_baryonic")  # fixed slots

    def __init__(self, ptype: str, n_particles: int, is_baryonic: bool):

        self.ptype = ptype
        self.n_particles = n_particles
        self.is_baryonic = is_baryonic
        self.columns: dict[str, np.ndarray] = {}  # O(1) lookup on a lightweight np array (preconverted units)

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

    def release(self, *names: str) -> None:  # chose *names as an alternative to names: list[str] for readability
        """
        Call ParticleStore.release("key1", "key2") to delete references to no-longer needed columns (like drop from old datamanager).

        Internally, this is effectively the same as the del method.
        """
        for name in names:
            self.columns.pop(name, None)


def build_particle_stores(
    reader: SnapshotReader,
    internals: Internals,
    process_ptypes: dict[str, bool],
) -> dict[str, ParticleStore]:
    """
    Constructs basic particle stores using information from what is available in the snapshot, and what the config specifies to process.
    """
    available = reader.available_ptypes()
    requested = [pt for pt in available if process_ptypes.get(pt, True)]

    particles: dict[str, ParticleStore] = {}

    for ptype in requested:
        halo_ids = reader.read_halo_ids(ptype=ptype)
        store = ParticleStore(ptype=ptype, n_particles=len(halo_ids), is_baryonic=ptype in internals.baryonic_ptypes)
        store["HaloID"] = halo_ids

        for dataset in ["mass", "pos", "vel"]:
            store[dataset] = reader.read_dataset(ptype, dataset)

        if ptype == "gas":
            store["temperature"] = reader.read_temperature(ptype=ptype)

        store["ptype"] = np.full(len(store), ptype)
        particles[ptype] = store

    return particles


class GroupStore:
    """
    Effectively the same idea as the ParticleStore class, but storing group-level information.
    """

    def __init__(
        self, group_ids: np.ndarray, group_key: int, original_ids: np.ndarray | None = None
    ):  # original_ids is for external halo readers

        self.group_ids = group_ids
        self.n_groups = len(group_ids)
        self.columns: dict[str, np.ndarray] = {}
        self.group_key = group_key

        max_id = group_ids.max() if self.n_groups > 0 else 0
        logger.debug(f"Max ID guard hit in group store for {group_key}")
        self.id_to_idx = np.full(shape=max_id + 1, fill_value=-1, dtype=DTYPES["pid"])
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
        id_to_idx[valid] = self.id_to_idx[
            group_id[valid]
        ]  # mask valid indices (-1, the sentinel, is the last array element)

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


@dataclass(slots=True)  # no frozen=True as this is inherently supposed to be mutable
class SimulationData:
    """
    Object containing simulation data ready for analysis:

    - hdf5 groups converted to np.ndarrays in code units with correct datatype
    - group IDs (at instantiation, HIDs)
    - Simulation-specific attributes (boxsize, cosmological parameters, etc).
    """

    simulation: SimulationAttributes
    constants: OctavianConstants
    particles: dict[str, ParticleStore]
    groups: dict[str, GroupStore]

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
