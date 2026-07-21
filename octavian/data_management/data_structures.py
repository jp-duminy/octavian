"""

Octavian data structures (readers, stores, dataclasses).
This is a modularised version of the old DataManager, its functionality divided amongst smaller objects.

"""

# semantic
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_management.pipeline_management import Internals
    from octavian.data_management.conventions import OctavianConstants, OctavianConfig
    from octavian.external_halo_sources import HaloAssignments, SubhaloInformation
from octavian.log import get_logger

# defaults
from pathlib import Path
from dataclasses import dataclass

# others
import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
import astropy.units as u

# from the backend
from octavian.data_management.conventions import (
    DTYPES,
    CODE_UNITS,
    SimulationAttributes,
    SnapshotReader,
    gizmo_unit_conversion_factor,
)

from .csr import (
    build_group_csr,
    propagate_membership_csr,
)

from octavian.data_management.physics import (
    derive_stellar_age,
    calculate_temperature,
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
    }

    inverse_ptype_map = {v: k for k, v in ptype_map.items()}  # for convenience

    def __init__(self, snapshot_path: Path, constants: OctavianConstants):

        logger.info("Using GIZMO reader.")

        self.snapshot_path = snapshot_path
        self.constants = constants
        self.indices: dict[str, np.ndarray] | None = None

        self.read_header()
        self.unit_conversions = {
            dataset: gizmo_unit_conversion_factor(dataset, self.simulation_attributes.h, self.simulation_attributes.a)
            for dataset in self.dataset_map
            if dataset in DTYPES
        }

    def set_indices(self, indices: dict[str, np.ndarray]) -> None:
        """
        Stores the indice mask which allows a rank to access its assigned portion of the snapshot.
        """
        self.indices = indices  # avoids passing "indices=" into functions

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
        Reads a HDF5 dataset from the file. Returns an ndarray of the chosen dataset in Octavian code units with the correct dtype applied, masked to the rank's allocation.
        """
        hdf5_group = self.inverse_ptype_map[ptype]
        hdf5_name = self.dataset_map[dataset]

        with h5py.File(self.snapshot_path, "r") as f:
            raw_hdf5_array = f[hdf5_group][hdf5_name][:]

        if dataset == "metallicity":  # I think it's okay to have these as conditionals by way of being explicit
            raw_hdf5_array = raw_hdf5_array[:, 0]

        if dataset == "helium_fraction":
            raw_hdf5_array = raw_hdf5_array[:, 1]

        if dataset == "age":
            raw_hdf5_array = derive_stellar_age(
                formation_time=raw_hdf5_array,
                time_gyr=self.simulation_attributes.time_gyr,
                cosmology=self.simulation_attributes.cosmology,
            )
            if self.indices is not None:
                raw_hdf5_array = raw_hdf5_array[self.indices[ptype]]

            return raw_hdf5_array.astype(DTYPES.get(dataset, np.float64))

        conversion_factor = self.unit_conversions.get(dataset, 1.0)
        if conversion_factor != 1.0:  # skip unnecessary multiplication on (potentially giant) arrays
            raw_hdf5_array = raw_hdf5_array * conversion_factor

        if self.indices is not None:
            raw_hdf5_array = raw_hdf5_array[self.indices[ptype]]

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

            if self.indices is not None:
                halo_ids = halo_ids[self.indices[ptype]]

        halo_ids -= 1  # shift IDs left to compensate with Octavian sentinel

        return halo_ids

    def read_particle_ids(self, ptype: str) -> np.ndarray:
        """
        Reads GIZMO snapshot PIDs in int64.
        """
        hdf5_group = self.inverse_ptype_map[ptype]

        with h5py.File(self.snapshot_path, "r") as f:
            particle_ids = f[hdf5_group]["ParticleIDs"][:].astype(DTYPES.get("pid", np.int64))

            if self.indices is not None:
                particle_ids = particle_ids[self.indices[ptype]]

        return particle_ids

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
        temperature = calculate_temperature(
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
        "rho": "Densities",
        "fHI": "AtomicHydrogenMasses",  # placeholder since not stored
        "sfr": "StarFormationRates",
        "age": "BirthScaleFactors",
        "metallicity": "MetalMassFractions",
        "helium_fraction": "ElementMassFractions",  # assuming [Z, He...] GIZMO convention
        "fH2": "MolecularHydrogenFractions",
        "bhmass": "SubgridMasses",
        "bhmdot": "AccretionRates",
    }

    dataset_map_overrides: dict[tuple[str, str], str] = {  # this is for the dynamical vs subgrid bh mass
        ("bh", "mass"): "DynamicalMasses",
    }

    inverse_ptype_map = {v: k for k, v in ptype_map.items()}  # for convenience

    def __init__(self, snapshot_path: Path, constants: OctavianConstants):

        logger.info("Using SWIFT reader.")

        self.snapshot_path = snapshot_path
        self.constants = constants
        self.indices: dict[str, np.ndarray] | None = None

        self.read_header()

    def set_indices(self, indices: dict[str, np.ndarray]) -> None:
        """
        Stores the indice mask which allows a rank to access its assigned portion of the snapshot.
        """
        self.indices = indices  # avoids passing "indices=" into functions

    def available_ptypes(self) -> list[str]:
        """
        Finds which Octavia-compatible ptypes are available in the snapshot.
        """
        with h5py.File(self.snapshot_path, "r") as f:
            return [pt for raw, pt in self.ptype_map.items() if raw in f and len(f[raw]) > 0]

    def read_header(self) -> SimulationAttributes:
        """
        Parses header attributes into a dataclass (does derived quantities too). SWIFT can do simulations with evolving dark energy, so we use flat w0wa cdm cosmology, which reduces to flat lambda cdm when wa = 0; SWIFT also splits the header into cosmology and header fields.
        """
        with h5py.File(self.snapshot_path, "r") as f:
            cosmo = f["Cosmology"].attrs
            header = f["Header"].attrs

            boxsize_vec = header["BoxSize"]  # usually stored as (x, y, z) in Mpc comoving
            boxsize_raw = boxsize_vec[0]
            assert np.allclose(boxsize_vec, boxsize_raw), "Octavian does not presently support non-cubic boxes."

            unit_length_cgs = f["Units"].attrs["Unit length in cgs (U_L)"].item()
            boxsize_cgs = boxsize_raw * unit_length_cgs
            boxsize_kpc = boxsize_cgs / u.kpc.to(u.cm)

            n_star = header["NumPart_Total"][4]
            n_gas = header["NumPart_Total"][0]

            h = cosmo["h"].item()
            a = cosmo["Scale-factor"].item()
            w_0 = cosmo["w_0"].item()
            w_a = cosmo["w_a"].item()
            T_cmb_0 = cosmo["T_CMB_0 [K]"].item()  # I am not sure if this is strictly necessary but I put it in anyway
            redshift = cosmo["Redshift"].item()
            omega_matter = cosmo["Omega_m"].item()
            omega_lambda = cosmo["Omega_lambda"].item()

        flat_w0wa_cdm = Flatw0waCDM(
            H0=100 * h, Om0=omega_matter, Tcmb0=T_cmb_0, w0=w_0, wa=w_a
        )  # reduces to lambdacdm if wa=0

        self.simulation_attributes = derive_simulation_attributes(
            cosmology=flat_w0wa_cdm,
            h=h,
            a=a,
            redshift=redshift,
            omega_matter=omega_matter,
            omega_lambda=omega_lambda,
            boxsize=boxsize_kpc,
            n_star=n_star,
            n_gas=n_gas,
            constants=self.constants,
        )

        return self.simulation_attributes

    def read_dataset(self, ptype: str, dataset: str) -> np.ndarray:
        """
        Convert a HDF5 dataset in the snapshot to a numpy array with the correct dtype (for floating point precision); auto-applies SWIFT attribute conversions to Octavian code units.
        """
        hdf5_group = self.inverse_ptype_map[ptype]
        hdf5_name = self.dataset_map_overrides.get((ptype, dataset), self.dataset_map[dataset])

        with h5py.File(self.snapshot_path, "r") as f:
            hdf5_dataset = f[hdf5_group][hdf5_name]

            if dataset == "fHI":
                masses = f[hdf5_group]["Masses"][:]
                HI_masses = f[hdf5_group]["AtomicHydrogenMasses"][:]

                if self.indices is not None:
                    masses = masses[self.indices[ptype]]
                    HI_masses = HI_masses[self.indices[ptype]]

                return (HI_masses / masses).astype(DTYPES.get(dataset, np.float64))

            else:
                raw_hdf5_array = hdf5_dataset[:]
                a_exp, h_exp = hdf5_dataset.attrs["a-scale exponent"], hdf5_dataset.attrs["h-scale exponent"]
                cgs_factor = hdf5_dataset.attrs["Conversion factor to CGS (not including cosmological corrections)"]

        if dataset == "helium_fraction":
            raw_hdf5_array = raw_hdf5_array[:, 1]

        if dataset == "age":
            raw_hdf5_array = derive_stellar_age(
                formation_time=raw_hdf5_array,
                time_gyr=self.simulation_attributes.time_gyr,
                cosmology=self.simulation_attributes.cosmology,
            )
            if self.indices is not None:
                raw_hdf5_array = raw_hdf5_array[self.indices[ptype]]

            return raw_hdf5_array.astype(DTYPES.get(dataset, np.float64))

        if self.indices is not None:
            raw_hdf5_array = raw_hdf5_array[self.indices[ptype]]

        target_units = CODE_UNITS[dataset]
        target_cgs_units = (1.0 * target_units.unit).cgs.value
        a_correction = self.simulation_attributes.a ** (a_exp - target_units.a_exponent)
        h_correction = self.simulation_attributes.h**h_exp  # code units do not carry h

        unit_factor = (a_correction * h_correction) * (cgs_factor / target_cgs_units)

        result = raw_hdf5_array * unit_factor

        return result.astype(DTYPES.get(dataset, np.float64))

    def read_halo_ids(self, ptype: str):
        """
        Reads (placeholder) FOFGroupIDs as HaloIDs if the external doesn't exist. SWIFT sentinel value is the uint32 max.
        """
        hdf5_group = self.inverse_ptype_map[ptype]

        with h5py.File(self.snapshot_path, "r") as f:
            halo_ids = f[hdf5_group]["FOFGroupIDs"][:].astype(DTYPES.get("HaloID", np.int64))
            if self.indices is not None:
                halo_ids = halo_ids[self.indices[ptype]]

        sentinel_mask = halo_ids == 2147483647  # uint32 max value (as for why they do this? I have no idea)
        halo_ids -= 1  # SWIFT is also 1-indexed
        halo_ids[sentinel_mask] = -1

        return halo_ids

    def read_particle_ids(self, ptype: str) -> np.ndarray:
        """
        Reads SWIFT snapshot PIDs in int64.
        """
        hdf5_group = self.inverse_ptype_map[ptype]

        with h5py.File(self.snapshot_path, "r") as f:
            particle_ids = f[hdf5_group]["ParticleIDs"][:].astype(DTYPES.get("pid", np.int64))

            if self.indices is not None:
                particle_ids = particle_ids[self.indices[ptype]]

        return particle_ids

    def read_temperature(self, ptype: str = "gas"):
        """
        Computes the per-particle temperatures from composition & internal energy (method described in GIZMO docs). SWIFT does not directly store electron abundance but this is trivial to compute.
        """
        internal_energy = self.read_dataset(ptype=ptype, dataset="internal_energy")
        helium_frac = self.read_dataset(ptype=ptype, dataset="helium_fraction")
        y_helium = helium_frac / (4 * (1 - helium_frac))
        electron_abundance = (1 + 2 * y_helium) / (1 + 4 * y_helium)

        temperature = calculate_temperature(
            internal_energy=internal_energy,
            electron_abundance=electron_abundance,
            helium_fraction=helium_frac,
            constants=self.constants,
        )

        return temperature


def build_reader(snapshot_path: Path, constants: OctavianConstants, config: OctavianConfig) -> SnapshotReader:
    """
    Builds a GIZMO/SWIFT reader class depending on what was specified in the config.
    """
    if config.simulation_type == "GIZMO":
        return GizmoReader(snapshot_path=snapshot_path, constants=constants)
    elif config.simulation_type == "SWIFT":
        return SwiftReader(snapshot_path=snapshot_path, constants=constants)
    else:
        raise ValueError(f"Unknown simulation ({config.simulation_type}), please put GIZMO/SWIFT!")


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
    halo_assignments: HaloAssignments,
    process_ptypes: dict[str, bool],
) -> dict[str, ParticleStore]:
    """
    Constructs basic particle stores using information from what is available in the snapshot, and what the config specifies to process.
    """
    available = reader.available_ptypes()
    requested = [pt for pt in available if process_ptypes.get(pt, True)]

    particles: dict[str, ParticleStore] = {}

    for ptype in requested:
        halo_ids = halo_assignments.halo_ids[ptype]
        store = ParticleStore(ptype=ptype, n_particles=len(halo_ids), is_baryonic=ptype in internals.baryonic_ptypes)
        store["HaloID"] = halo_ids
        if halo_assignments.subhalo_ids is not None:
            store["SubhaloID"] = halo_assignments.subhalo_ids[ptype]

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
        self, group_ids: np.ndarray, group_key: int, kind: str, original_ids: np.ndarray | None = None
    ):  # original_ids is for external halo readers

        self.group_ids = group_ids
        self.n_groups = len(group_ids)
        self.columns: dict[str, np.ndarray] = {}
        self.csr_membership: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.group_key = group_key
        self.kind = kind

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

    def get_particle_csr(self, ptype: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a tuple of (offsets, sorted_indices) into the ParticleStore of the ptype.
        """
        return self.csr_membership[ptype]


def build_galaxy_store(
    particles: dict[str, ParticleStore],
    galaxy_key: str,  # NOTE: this doesn't need to be an argument but I prefer it for explicit purposes
    group_kind: str,
) -> GroupStore:
    """
    Constructs the galaxy GroupStore.
    """
    ids = [particles[ptype][galaxy_key] for ptype in particles]
    unique_ids = np.unique(np.concatenate(ids))

    unique_ids = unique_ids[unique_ids != -1]

    store = GroupStore(group_ids=unique_ids, group_key=galaxy_key, kind=group_kind)

    for ptype in particles:
        offsets, sorted_indices = build_group_csr(
            group_idx=store.get_indexer(group_id=particles[ptype][galaxy_key]), n_groups=store.n_groups
        )
        store.csr_membership[ptype] = (offsets, sorted_indices)

    return store


def build_halo_store(
    particles: dict[str, ParticleStore],
    halo_key: str = "HaloID",
    subhalo_key: str | None = None,
    group_kind: str = "halo",
    subhalo_info: SubhaloInformation | None = None,
) -> GroupStore:
    """
    Constructs a halo GroupStore.
    """
    all_halo_ids = [particles[ptype][halo_key] for ptype in particles]
    unique_hids = np.unique(np.concatenate(all_halo_ids))
    unique_hids = unique_hids[unique_hids != -1]
    n_halos = len(unique_hids)

    if subhalo_info is not None:
        subhalo_ids = [particles[ptype][subhalo_key] for ptype in particles]
        unique_subhids = np.unique(np.concatenate(subhalo_ids))
        unique_subhids = unique_subhids[unique_subhids != -1]
        n_subhalos = len(unique_subhids)

        field_to_row = np.full(shape=(unique_hids.max() + 1), fill_value=-1, dtype=np.int64)
        field_to_row[unique_hids] = np.arange(n_halos)

        combined_ids = np.concatenate([unique_hids, unique_subhids])
        store = GroupStore(group_ids=combined_ids, group_key=halo_key, kind=group_kind)

        for ptype in particles:
            halo_ids = particles[ptype][halo_key]
            sub_ids = particles[ptype][subhalo_key]
            group_idx = field_to_row[halo_ids]

            mask = sub_ids != -1
            group_idx[mask] = sub_ids[mask] + n_halos

            offsets, sorted_indices = build_group_csr(group_idx=group_idx, n_groups=store.n_groups)
            store.csr_membership[ptype] = (offsets, sorted_indices)

        parent_rows = np.full(n_halos + n_subhalos, -1, dtype=np.int64)
        depth_1_mask = subhalo_info.depth == 1
        deeper_mask = subhalo_info.depth > 1
        parent_rows[n_halos:][depth_1_mask] = field_to_row[subhalo_info.parent_index[depth_1_mask]]
        parent_rows[n_halos:][deeper_mask] = subhalo_info.parent_index[deeper_mask] + n_halos

        for ptype in particles:
            exclusive_offsets, exclusive_sorted = store.csr_membership[ptype]
            inclusive_offsets, inclusive_sorted = propagate_membership_csr(
                offsets=exclusive_offsets,
                sorted_indices=exclusive_sorted,
                parent=parent_rows,
                n_groups=store.n_groups,
            )
            store.csr_membership[ptype] = (inclusive_offsets, inclusive_sorted)

        store["_parent"] = parent_rows
        store["_depth"] = np.concatenate([np.zeros(n_halos, dtype=np.int64), subhalo_info.depth])

    else:
        store = GroupStore(group_ids=unique_hids, group_key=halo_key, kind=group_kind)
        for ptype in particles:
            offsets, sorted_indices = build_group_csr(
                group_idx=store.get_indexer(group_id=particles[ptype][halo_key]), n_groups=store.n_groups
            )
            store.csr_membership[ptype] = (offsets, sorted_indices)

    return store


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
