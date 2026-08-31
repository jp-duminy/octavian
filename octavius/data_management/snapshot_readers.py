"""

h5py-backed, MPI-native raw snapshot readers. Parse raw simulation output into Octavius analysis, converting
snapshot-specific terminology into an agnostic interface for the data structures.

"""

# type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conventions import OctaviusConstants, OctaviusConfig
    from .parallel_reading import RedistributionMap
    from mpi4py.MPI import Comm

# default libraries
from pathlib import Path

# other packages
import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
import astropy.units as u

# internal imports
from .parallel_reading import redistribute_data, split_slab
from .physics import (
    derive_stellar_age,
    calculate_temperature,
    derive_simulation_attributes,
)
from .conventions import (
    DTYPES,
    CODE_UNITS,
    SimulationAttributes,
    gizmo_unit_conversion_factor,
)
from ..log import get_logger

logger = get_logger()


class SnapshotReader:
    """
    Base reader class (for inheritance). I tucked it away here.
    """

    inverse_ptype_map: dict[str, str] = NotImplemented
    dataset_map: dict[str, dict[str, str]] = NotImplemented
    simulation_attributes: SimulationAttributes = NotImplemented
    global_indices: dict[str, np.ndarray] | None = NotImplemented
    particle_counts: dict[str, int] | None = NotImplemented

    def set_maps(
        self,
        slabs: dict[str, slice],
        masks: dict[str, np.ndarray],
        maps: dict[str, RedistributionMap],
        comm: Comm | None,
    ) -> None:
        """
        Sets the per-rank slabs; global particle redistribution map; corresponding halo threshold masks; and  and comm for MPI reading.
        """
        raise NotImplementedError

    def read_header(self) -> SimulationAttributes:
        """
        Read header attributes and, where necessary, convert units.
        """
        raise NotImplementedError

    def has_dataset(self, ptype: str, dataset: str) -> bool:
        """
        Checks whether a dataset exists in the snapshot.
        """
        raise NotImplementedError

    def read_dataset(self, ptype: str, dataset: str) -> np.ndarray:
        """
        Returns array in Octavius code units with the correct dtype.
        """
        raise NotImplementedError

    def available_ptypes(self) -> list[str]:
        """
        List of available ptypes in Octavius convention (gas, star, etc.)
        """
        raise NotImplementedError

    def read_halo_ids(self, ptype: str, slab: slice = slice(None)) -> np.ndarray:
        """
        Reads snapshot-assigned HaloIDs and maps them to a continuous 0-indexed array with a sentinel value of -1.
        """
        raise NotImplementedError

    def read_particle_ids(self, ptype: str) -> np.ndarray:
        """
        Reads snapshot-assigned particle IDs for a specified ptype.
        """
        raise NotImplementedError

    def read_temperature(self, ptype: str) -> np.ndarray:
        """
        Temperature is usually computed from multiple datasets, and how it can be computed differs between readers.
        """
        raise NotImplementedError


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
        "dust_mass": "Dust_Masses",
        "smoothing_length": "SmoothingLength",
    }

    inverse_ptype_map = {v: k for k, v in ptype_map.items()}  # for convenience

    def __init__(self, snapshot_path: Path, constants: OctaviusConstants, n_io_chunks: int):

        self.snapshot_path = snapshot_path
        self.constants = constants
        self.n_io_chunks = n_io_chunks
        self.global_indices: dict[str, np.ndarray] | None = None
        self.maps: dict[str, np.ndarray] | None = None

        self.read_header()
        self.unit_conversions = {
            dataset: gizmo_unit_conversion_factor(
                dataset, self.simulation_attributes.h, self.simulation_attributes.scale_factor
            )
            for dataset in self.dataset_map
            if dataset in DTYPES
        }

    def set_maps(
        self,
        slabs: dict[str, slice],
        masks: dict[str, np.ndarray],
        maps: dict[str, RedistributionMap],
        comm: Comm | None,
    ) -> None:
        """
        Stores the global particle redistribution map and comm for MPI file reads.
        """
        self.slabs = slabs
        self.masks = masks
        self.maps = maps
        self.comm = comm
        self.global_indices: dict[str, np.ndarray] = {}

        assert slabs.keys() == masks.keys()

        for ptype in masks:
            slab = slabs[ptype]
            global_indices = np.arange(slab.start, slab.stop, dtype=np.int64)[masks[ptype]]
            self.global_indices[ptype] = redistribute_data(
                local_data=global_indices, redistribution_map=maps[ptype], comm=comm
            )

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

            self.n_particles_total = header["NumPart_Total"]
            self.particle_counts = {
                ptype_name: int(self.n_particles_total[int(hdf5_key[-1])])
                for hdf5_key, ptype_name in self.ptype_map.items()
            }

        flat_lambda_cdm = FlatLambdaCDM(H0=100 * h, Om0=omega_matter)  # always flatlambdacdm for gizmo

        self.simulation_attributes = derive_simulation_attributes(
            cosmology=flat_lambda_cdm,
            h=h,
            a=a,
            w_0=-1,  # always flatlambdacdm
            w_a=0,
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

    def has_dataset(self, ptype: str, dataset: str) -> bool:
        """
        Checks whether a dataset exists in the snapshot.
        """
        hdf5_group = self.inverse_ptype_map[ptype]
        hdf5_name = self.dataset_map.get(dataset)

        if hdf5_name is None:
            logger.warning(f"{dataset} is not defined to the reader.")
            return False
        with h5py.File(self.snapshot_path, "r") as f:
            return hdf5_name in f.get(hdf5_group, {})

    def read_dataset(self, ptype: str, dataset: str) -> np.ndarray:
        """
        Reads a HDF5 dataset from the file. Returns an ndarray of the chosen dataset in Octavius code units with the correct dtype applied, masked to the rank's allocation.
        """
        hdf5_group = self.inverse_ptype_map[ptype]
        hdf5_name = self.dataset_map[dataset]
        slab = self.slabs[ptype]
        slab_length = slab.stop - slab.start

        # read the rank's slab
        with h5py.File(self.snapshot_path, "r") as f:
            hdf5_dataset = f[hdf5_group][hdf5_name]

            if dataset == "metallicity":  # need first column for the total metal fraction
                raw_array = np.empty(slab_length, dtype=hdf5_dataset.dtype)

                for chunk in split_slab(slab, self.n_io_chunks):
                    offset = chunk.start - slab.start
                    chunk_length = chunk.stop - chunk.start
                    raw_array[offset : offset + chunk_length] = hdf5_dataset[chunk, 0]

            elif dataset == "helium_fraction":  # second column for the helium fraction
                raw_array = np.empty(slab_length, dtype=hdf5_dataset.dtype)

                for chunk in split_slab(slab, self.n_io_chunks):
                    offset = chunk.start - slab.start
                    chunk_length = chunk.stop - chunk.start
                    raw_array[offset : offset + chunk_length] = hdf5_dataset[chunk, 1]

            else:  # read flat for all others
                full_shape = (slab_length,) + hdf5_dataset.shape[
                    1:
                ]  # this returns () if flat and tuple addition handles it (shape needed for 3d columns)
                raw_array = np.empty(full_shape, dtype=hdf5_dataset.dtype)

                for chunk in split_slab(slab, self.n_io_chunks):
                    offset = chunk.start - slab.start
                    chunk_length = chunk.stop - chunk.start
                    raw_array[offset : offset + chunk_length] = hdf5_dataset[chunk]

            filtered_array = raw_array[self.masks[ptype]]

        # close file, do unit/dtype conversions and MPI redistribution
        if (
            dataset == "age"
        ):  # age has an early return because its unit conversion is handled by the function internally (this dataset is different)
            filtered_array = derive_stellar_age(
                formation_time=filtered_array,
                time_gyr=self.simulation_attributes.time_gyr,
                cosmology=self.simulation_attributes.cosmology,
            )

            if self.maps is not None:
                result = redistribute_data(
                    local_data=filtered_array, redistribution_map=self.maps[ptype], comm=self.comm
                )
            else:
                result = filtered_array

            return result.astype(DTYPES.get(dataset, np.float64), copy=False)

        # apply unit conversion factor (after data is filtered and masked, so the operation is cheaper)
        conversion_factor = self.unit_conversions.get(dataset, 1.0)
        if conversion_factor != 1.0:  # skip unit conversion multiplication if unnecessary
            filtered_array *= conversion_factor

        # mpi vs serial
        if self.maps is not None:
            result = redistribute_data(local_data=filtered_array, redistribution_map=self.maps[ptype], comm=self.comm)
        else:
            result = filtered_array

        return result.astype(DTYPES.get(dataset, np.float64), copy=False)  # change dtype at the end

    def read_halo_ids(self, ptype: str, slab: slice = slice(None)) -> np.ndarray:
        """
        Reads snapshot-sourced HaloIDs. GIZMO uses 0 as the sentinel value; we map to Octavius's -1.
        """
        hdf5_group = self.inverse_ptype_map[ptype]

        if slab.start is None:  # SnapshotHaloSource calls this without slab arg
            slab = slice(0, self.particle_counts[ptype])

        slab_length = slab.stop - slab.start

        with h5py.File(self.snapshot_path, "r") as f:
            halo_hdf5_dataset = f[hdf5_group]["HaloID"]
            raw_halo_ids = np.empty(shape=slab_length, dtype=halo_hdf5_dataset.dtype)

            for chunk in split_slab(slab, self.n_io_chunks):
                offset = chunk.start - slab.start
                chunk_length = chunk.stop - chunk.start
                raw_halo_ids[offset : offset + chunk_length] = halo_hdf5_dataset[chunk]

            raw_halo_ids = raw_halo_ids.astype(
                DTYPES.get("HaloID", np.int64), copy=False
            )  # change dtype here otherwise you get int overflow

        raw_halo_ids -= 1  # shift IDs left to compensate with Octavius sentinel

        return raw_halo_ids

    def read_particle_ids(self, ptype: str) -> np.ndarray:
        """
        Reads GIZMO snapshot PIDs in int64.
        """
        hdf5_group = self.inverse_ptype_map[ptype]

        with h5py.File(self.snapshot_path, "r") as f:
            hdf5_dataset = f[hdf5_group]["ParticleIDs"]

            if self.maps is None:
                total_length = hdf5_dataset.shape[0]
                particle_ids = np.empty(total_length, dtype=hdf5_dataset.dtype)

                for chunk in split_slab(slice(0, total_length), self.n_io_chunks):
                    offset = chunk.start
                    chunk_length = chunk.stop - chunk.start
                    particle_ids[offset : offset + chunk_length] = hdf5_dataset[chunk]

                result = particle_ids

            else:
                slab = self.slabs[ptype]
                slab_length = slab.stop - slab.start
                particle_ids = np.empty(slab_length, dtype=hdf5_dataset.dtype)

                for chunk in split_slab(slab, self.n_io_chunks):
                    offset = chunk.start - slab.start
                    chunk_length = chunk.stop - chunk.start
                    particle_ids[offset : offset + chunk_length] = hdf5_dataset[chunk]

                filtered_particle_ids = particle_ids[self.masks[ptype]]
                result = redistribute_data(
                    local_data=filtered_particle_ids, redistribution_map=self.maps[ptype], comm=self.comm
                )

        return result.astype(DTYPES.get("pid", np.int64))

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
        "dust_mass": "DustMasses",
        "smoothing_length": "SmoothingLengths",
    }

    dataset_map_overrides: dict[tuple[str, str], str] = {  # this is for the dynamical vs subgrid bh mass
        ("bh", "mass"): "DynamicalMasses",
    }

    inverse_ptype_map = {v: k for k, v in ptype_map.items()}  # for convenience

    def __init__(self, snapshot_path: Path, constants: OctaviusConstants, n_io_chunks: int):

        self.snapshot_path = snapshot_path
        self.constants = constants
        self.n_io_chunks = n_io_chunks
        self.global_indices: dict[str, np.ndarray] | None = None
        self.maps: dict[str, RedistributionMap] | None = None

        self.read_header()

    def set_maps(
        self,
        slabs: dict[str, slice],
        masks: dict[str, np.ndarray],
        maps: dict[str, RedistributionMap],
        comm: Comm | None,
    ) -> None:
        """
        Stores the global particle redistribution map and comm for MPI file reads.
        """
        self.slabs = slabs
        self.masks = masks
        self.maps = maps
        self.comm = comm
        self.global_indices: dict[str, np.ndarray] = {}

        for ptype in masks:
            slab = slabs[ptype]
            global_indices = np.arange(slab.start, slab.stop, dtype=np.int64)[masks[ptype]]
            self.global_indices[ptype] = redistribute_data(
                local_data=global_indices, redistribution_map=maps[ptype], comm=comm
            )

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
            assert np.allclose(boxsize_vec, boxsize_raw), "Octavius does not presently support non-cubic boxes."

            unit_length_cgs = f["Units"].attrs["Unit length in cgs (U_L)"].item()
            boxsize_cgs = boxsize_raw * unit_length_cgs
            boxsize_kpc = boxsize_cgs / u.kpc.to(u.cm)

            n_star = header["NumPart_Total"][4]
            n_gas = header["NumPart_Total"][0]

            self.n_particles_total = header["NumPart_Total"]
            self.particle_counts = {
                ptype_name: int(self.n_particles_total[int(hdf5_key[-1])])
                for hdf5_key, ptype_name in self.ptype_map.items()
            }

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
            w_0=w_0,
            w_a=w_a,
            redshift=redshift,
            omega_matter=omega_matter,
            omega_lambda=omega_lambda,
            boxsize=boxsize_kpc,
            n_star=n_star,
            n_gas=n_gas,
            constants=self.constants,
        )

        return self.simulation_attributes

    def has_dataset(self, ptype: str, dataset: str) -> bool:
        """
        Checks whether a dataset exists in the snapshot.
        """
        hdf5_group = self.inverse_ptype_map[ptype]
        hdf5_name = self.dataset_map_overrides.get((ptype, dataset), self.dataset_map.get(dataset))

        if hdf5_name is None:
            logger.warning(f"{dataset} is not defined to the reader.")
            return False
        with h5py.File(self.snapshot_path, "r") as f:
            return hdf5_name in f.get(hdf5_group, {})

    def read_dataset(self, ptype: str, dataset: str) -> np.ndarray:
        """
        Convert a HDF5 dataset in the snapshot to a numpy array with the correct dtype (for floating point precision); auto-applies SWIFT attribute conversions to Octavius code units.
        """
        hdf5_group = self.inverse_ptype_map[ptype]
        hdf5_name = self.dataset_map_overrides.get((ptype, dataset), self.dataset_map[dataset])
        slab = self.slabs[ptype]
        slab_length = slab.stop - slab.start

        with h5py.File(self.snapshot_path, "r") as f:
            if dataset == "fHI":  # separate return path here because fHI doesn't directly exist in kiara
                hdf5_mass = f[hdf5_group]["Masses"]
                hdf5_HI_mass = f[hdf5_group]["AtomicHydrogenMasses"]

                raw_masses = np.empty(shape=slab_length, dtype=hdf5_mass.dtype)
                raw_HI_masses = np.empty(shape=slab_length, dtype=hdf5_HI_mass.dtype)

                for chunk in split_slab(slab, self.n_io_chunks):
                    offset = chunk.start - slab.start
                    chunk_length = chunk.stop - chunk.start
                    raw_masses[offset : offset + chunk_length] = hdf5_mass[chunk]
                    raw_HI_masses[offset : offset + chunk_length] = hdf5_HI_mass[chunk]

                filtered_masses = raw_masses[self.masks[ptype]]
                filtered_HI_masses = raw_HI_masses[self.masks[ptype]]

                if self.maps is not None:
                    result_HI_mass = redistribute_data(
                        local_data=filtered_HI_masses, redistribution_map=self.maps[ptype], comm=self.comm
                    )
                    result_mass = redistribute_data(
                        local_data=filtered_masses, redistribution_map=self.maps[ptype], comm=self.comm
                    )
                else:
                    result_HI_mass = filtered_HI_masses
                    result_mass = filtered_masses

                return (result_HI_mass / result_mass).astype(DTYPES.get(dataset, np.float64), copy=False)

            else:
                hdf5_dataset = f[hdf5_group][hdf5_name]

                if dataset == "helium_fraction":
                    raw_array = np.empty(slab_length, dtype=hdf5_dataset.dtype)

                    for chunk in split_slab(slab, self.n_io_chunks):
                        offset = chunk.start - slab.start
                        chunk_length = chunk.stop - chunk.start
                        raw_array[offset : offset + chunk_length] = hdf5_dataset[chunk, 1]

                else:
                    full_shape = (slab_length,) + hdf5_dataset.shape[
                        1:
                    ]  # this returns () if flat and tuple addition handles it (shape needed for 3d columns)
                    raw_array = np.empty(full_shape, dtype=hdf5_dataset.dtype)

                    for chunk in split_slab(slab, self.n_io_chunks):
                        offset = chunk.start - slab.start
                        chunk_length = chunk.stop - chunk.start
                        raw_array[offset : offset + chunk_length] = hdf5_dataset[chunk]

                filtered_array = raw_array[self.masks[ptype]]
                a_exp, h_exp = hdf5_dataset.attrs["a-scale exponent"], hdf5_dataset.attrs["h-scale exponent"]
                cgs_factor = hdf5_dataset.attrs["Conversion factor to CGS (not including cosmological corrections)"]

        if dataset == "age":
            filtered_array = derive_stellar_age(
                formation_time=filtered_array,
                time_gyr=self.simulation_attributes.time_gyr,
                cosmology=self.simulation_attributes.cosmology,
            )

            if self.maps is not None:
                result = redistribute_data(
                    local_data=filtered_array, redistribution_map=self.maps[ptype], comm=self.comm
                )
            else:
                result = filtered_array

            return result.astype(DTYPES.get(dataset, np.float64))

        target_units = CODE_UNITS[dataset]
        target_cgs_units = (1.0 * target_units.unit).cgs.value
        a_correction = self.simulation_attributes.scale_factor ** (a_exp - target_units.a_exponent)
        h_correction = self.simulation_attributes.h**h_exp  # code units do not carry h

        unit_factor = (a_correction * h_correction) * (cgs_factor / target_cgs_units)

        if unit_factor != 1.0:
            filtered_array *= unit_factor

        # mpi vs serial
        if self.maps is not None:
            result = redistribute_data(local_data=filtered_array, redistribution_map=self.maps[ptype], comm=self.comm)
        else:
            result = filtered_array

        return result.astype(DTYPES.get(dataset, np.float64))

    def read_halo_ids(self, ptype: str, slab: slice = slice(None)) -> np.ndarray:
        """
        Reads (placeholder) FOFGroupIDs as HaloIDs if the external doesn't exist. SWIFT sentinel value is the uint32 max.
        """
        hdf5_group = self.inverse_ptype_map[ptype]

        if slab.start is None:  # SnapshotHaloSource calls this without slab arg
            slab = slice(0, self.particle_counts[ptype])

        slab_length = slab.stop - slab.start

        with h5py.File(self.snapshot_path, "r") as f:
            halo_hdf5_dataset = f[hdf5_group]["FOFGroupIDs"]
            raw_halo_ids = np.empty(shape=slab_length, dtype=halo_hdf5_dataset.dtype)

            for chunk in split_slab(slab, self.n_io_chunks):
                offset = chunk.start - slab.start
                chunk_length = chunk.stop - chunk.start
                raw_halo_ids[offset : offset + chunk_length] = halo_hdf5_dataset[chunk]

            raw_halo_ids = raw_halo_ids.astype(DTYPES.get("HaloID", np.int64), copy=False)

        sentinel_mask = raw_halo_ids == 2147483647  # uint32 max value (as for why they do this? I have no idea)
        raw_halo_ids -= 1  # SWIFT is also 1-indexed
        raw_halo_ids[sentinel_mask] = -1

        return raw_halo_ids

    def read_particle_ids(self, ptype: str) -> np.ndarray:
        """
        Reads SWIFT snapshot PIDs in int64.
        """
        hdf5_group = self.inverse_ptype_map[ptype]

        with h5py.File(self.snapshot_path, "r") as f:
            hdf5_dataset = f[hdf5_group]["ParticleIDs"]

            if self.maps is None:
                total_length = hdf5_dataset.shape[0]
                particle_ids = np.empty(total_length, dtype=hdf5_dataset.dtype)

                for chunk in split_slab(slice(0, total_length), self.n_io_chunks):
                    offset = chunk.start
                    chunk_length = chunk.stop - chunk.start
                    particle_ids[offset : offset + chunk_length] = hdf5_dataset[chunk]

                result = particle_ids

            else:
                slab = self.slabs[ptype]
                slab_length = slab.stop - slab.start
                particle_ids = np.empty(slab_length, dtype=hdf5_dataset.dtype)

                for chunk in split_slab(slab, self.n_io_chunks):
                    offset = chunk.start - slab.start
                    chunk_length = chunk.stop - chunk.start
                    particle_ids[offset : offset + chunk_length] = hdf5_dataset[chunk]

                filtered_particle_ids = particle_ids[self.masks[ptype]]
                result = redistribute_data(
                    local_data=filtered_particle_ids, redistribution_map=self.maps[ptype], comm=self.comm
                )

        return result.astype(DTYPES.get("pid", np.int64))

    def read_temperature(self, ptype: str = "gas"):
        """
        Computes the per-particle temperatures from composition & internal energy (method described in GIZMO docs). SWIFT does not directly store electron abundance but this is trivial to compute.
        """
        internal_energy = self.read_dataset(ptype=ptype, dataset="internal_energy")
        helium_frac = self.read_dataset(ptype=ptype, dataset="helium_fraction")
        y_helium = helium_frac / (4 * (1 - helium_frac))
        electron_abundance = 1 + 2 * y_helium

        temperature = calculate_temperature(
            internal_energy=internal_energy,
            electron_abundance=electron_abundance,
            helium_fraction=helium_frac,
            constants=self.constants,
        )

        return temperature


def build_reader(snapshot_path: Path, constants: OctaviusConstants, config: OctaviusConfig) -> SnapshotReader:
    """
    Builds a GIZMO/SWIFT reader class depending on what was specified in the config.
    """
    if config.simulation_type == "GIZMO":
        logger.info("Using GIZMO reader.")
        return GizmoReader(snapshot_path=snapshot_path, constants=constants, n_io_chunks=config.n_io_chunks)
    elif config.simulation_type == "SWIFT":
        logger.info("Using SWIFT reader.")
        return SwiftReader(snapshot_path=snapshot_path, constants=constants, n_io_chunks=config.n_io_chunks)
    else:
        raise ValueError(f"Unknown simulation ({config.simulation_type}), please put GIZMO/SWIFT!")
