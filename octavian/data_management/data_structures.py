"""

Octavian data structures (readers, stores, dataclasses).
This is a modularised version of the old DataManager, its functionality divided amongst smaller objects.

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline_management import Internals
    from .conventions import OctavianConstants, OctavianConfig
    from ..external_halo_sources import HaloAssignments, SubhaloInformation
    from .parallel_reading import RedistributionMap
    from mpi4py.MPI import Comm

# defaults
from pathlib import Path
from dataclasses import dataclass

# others
import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
import astropy.units as u

# internal imports
from .conventions import (
    DTYPES,
    CODE_UNITS,
    SimulationAttributes,
    SnapshotReader,
    gizmo_unit_conversion_factor,
)

from .parallel_reading import redistribute_data, split_slab

from .csr import (
    build_group_csr,
    propagate_membership_csr,
)

from .physics import (
    derive_stellar_age,
    calculate_temperature,
    derive_simulation_attributes,
)

from ..log import get_logger

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
        "dust_mass": "Dust_Masses",
        "smoothing_length": "SmoothingLength",
    }

    inverse_ptype_map = {v: k for k, v in ptype_map.items()}  # for convenience

    def __init__(self, snapshot_path: Path, constants: OctavianConstants, n_chunks: int):

        self.snapshot_path = snapshot_path
        self.constants = constants
        self.n_chunks = n_chunks
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
        Reads a HDF5 dataset from the file. Returns an ndarray of the chosen dataset in Octavian code units with the correct dtype applied, masked to the rank's allocation.
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

                for chunk in split_slab(slab, self.n_chunks):
                    offset = chunk.start - slab.start
                    chunk_length = chunk.stop - chunk.start
                    raw_array[offset : offset + chunk_length] = hdf5_dataset[chunk, 0]

            elif dataset == "helium_fraction":  # second column for the helium fraction
                raw_array = np.empty(slab_length, dtype=hdf5_dataset.dtype)

                for chunk in split_slab(slab, self.n_chunks):
                    offset = chunk.start - slab.start
                    chunk_length = chunk.stop - chunk.start
                    raw_array[offset : offset + chunk_length] = hdf5_dataset[chunk, 1]

            else:  # read flat for all others
                full_shape = (slab_length,) + hdf5_dataset.shape[
                    1:
                ]  # this returns () if flat and tuple addition handles it (shape needed for 3d columns)
                raw_array = np.empty(full_shape, dtype=hdf5_dataset.dtype)

                for chunk in split_slab(slab, self.n_chunks):
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
        Reads snapshot-sourced HaloIDs. GIZMO uses 0 as the sentinel value; we map to Octavian's -1.
        """
        hdf5_group = self.inverse_ptype_map[ptype]

        if slab.start is None:  # SnapshotHaloSource calls this without slab arg
            slab = slice(0, self.particle_counts[ptype])

        slab_length = slab.stop - slab.start

        with h5py.File(self.snapshot_path, "r") as f:
            halo_hdf5_dataset = f[hdf5_group]["HaloID"]
            raw_halo_ids = np.empty(shape=slab_length, dtype=halo_hdf5_dataset.dtype)

            for chunk in split_slab(slab, self.n_chunks):
                offset = chunk.start - slab.start
                chunk_length = chunk.stop - chunk.start
                raw_halo_ids[offset : offset + chunk_length] = halo_hdf5_dataset[chunk]

            raw_halo_ids = raw_halo_ids.astype(
                DTYPES.get("HaloID", np.int64), copy=False
            )  # change dtype here otherwise you get int overflow

        raw_halo_ids -= 1  # shift IDs left to compensate with Octavian sentinel

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

                for chunk in split_slab(slice(0, total_length), self.n_chunks):
                    offset = chunk.start
                    chunk_length = chunk.stop - chunk.start
                    particle_ids[offset : offset + chunk_length] = hdf5_dataset[chunk]

                result = particle_ids

            else:
                slab = self.slabs[ptype]
                slab_length = slab.stop - slab.start
                particle_ids = np.empty(slab_length, dtype=hdf5_dataset.dtype)

                for chunk in split_slab(slab, self.n_chunks):
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

    def __init__(self, snapshot_path: Path, constants: OctavianConstants, n_chunks: int):

        self.snapshot_path = snapshot_path
        self.constants = constants
        self.n_chunks = n_chunks
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
            assert np.allclose(boxsize_vec, boxsize_raw), "Octavian does not presently support non-cubic boxes."

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
        Convert a HDF5 dataset in the snapshot to a numpy array with the correct dtype (for floating point precision); auto-applies SWIFT attribute conversions to Octavian code units.
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

                for chunk in split_slab(slab, self.n_chunks):
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

                    for chunk in split_slab(slab, self.n_chunks):
                        offset = chunk.start - slab.start
                        chunk_length = chunk.stop - chunk.start
                        raw_array[offset : offset + chunk_length] = hdf5_dataset[chunk, 1]

                else:
                    full_shape = (slab_length,) + hdf5_dataset.shape[
                        1:
                    ]  # this returns () if flat and tuple addition handles it (shape needed for 3d columns)
                    raw_array = np.empty(full_shape, dtype=hdf5_dataset.dtype)

                    for chunk in split_slab(slab, self.n_chunks):
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

            for chunk in split_slab(slab, self.n_chunks):
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

                for chunk in split_slab(slice(0, total_length), self.n_chunks):
                    offset = chunk.start
                    chunk_length = chunk.stop - chunk.start
                    particle_ids[offset : offset + chunk_length] = hdf5_dataset[chunk]

                result = particle_ids

            else:
                slab = self.slabs[ptype]
                slab_length = slab.stop - slab.start
                particle_ids = np.empty(slab_length, dtype=hdf5_dataset.dtype)

                for chunk in split_slab(slab, self.n_chunks):
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


def build_reader(snapshot_path: Path, constants: OctavianConstants, config: OctavianConfig) -> SnapshotReader:
    """
    Builds a GIZMO/SWIFT reader class depending on what was specified in the config.
    """
    if config.simulation_type == "GIZMO":
        logger.info("Using GIZMO reader.")
        return GizmoReader(snapshot_path=snapshot_path, constants=constants, n_chunks=config.n_chunks)
    elif config.simulation_type == "SWIFT":
        logger.info("Using SWIFT reader.")
        return SwiftReader(snapshot_path=snapshot_path, constants=constants, n_chunks=config.n_chunks)
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
        if max_id == 0:
            logger.debug(f"Empty GroupStore for {group_key}")

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
        For indexing particle arrays in the CSR format. Returns a tuple of:

        - offsets: CSR offsets
        - idx_sorted: indices into ParticleStore aligned to GroupStore order
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
        n_subhalos = len(subhalo_info.depth)

        field_to_row = np.full(shape=(unique_hids.max() + 1), fill_value=-1, dtype=np.int64)
        field_to_row[unique_hids] = np.arange(n_halos)

        combined_ids = np.concatenate([unique_hids, subhalo_info.global_index], dtype=np.int64)
        store = GroupStore(group_ids=combined_ids, group_key=halo_key, kind=group_kind)

        for ptype in particles:
            halo_ids = particles[ptype][halo_key]
            sub_ids = particles[ptype][subhalo_key]
            group_idx = np.where(halo_ids == -1, -1, field_to_row[halo_ids])

            mask = sub_ids != -1
            group_idx[mask] = sub_ids[mask] + n_halos

            offsets, sorted_indices = build_group_csr(group_idx=group_idx, n_groups=store.n_groups)
            store.csr_membership[ptype] = (offsets, sorted_indices)

        parent_rows = np.full(n_halos + n_subhalos, -1, dtype=np.int64)
        depth_1_mask = subhalo_info.depth == 1
        deeper_mask = subhalo_info.depth > 1
        parent_rows[n_halos:][depth_1_mask] = field_to_row[subhalo_info.host_halo_ids[depth_1_mask]]
        parent_rows[n_halos:][deeper_mask] = subhalo_info.parent_index[deeper_mask] + n_halos

        for ptype in particles:
            exclusive_offsets, exclusive_sorted = store.csr_membership[ptype]
            inclusive_offsets, inclusive_sorted = propagate_membership_csr(
                offsets=exclusive_offsets,
                sorted_indices=exclusive_sorted,
                parent_ids=parent_rows,
                n_groups=store.n_groups,
            )
            store.csr_membership[ptype] = (inclusive_offsets, inclusive_sorted)

        store["parent"] = parent_rows
        store["depth"] = np.concatenate([np.zeros(n_halos, dtype=np.int64), subhalo_info.depth])

    else:
        store = GroupStore(group_ids=unique_hids, group_key=halo_key, kind=group_kind)
        for ptype in particles:
            offsets, sorted_indices = build_group_csr(
                group_idx=store.get_indexer(group_id=particles[ptype][halo_key]), n_groups=store.n_groups
            )
            store.csr_membership[ptype] = (offsets, sorted_indices)

        store["parent"] = np.full(
            n_halos, -1, dtype=np.int64
        )  # NOTE: I know this is inefficient but otherwise tests break (catalogue inconsistency)
        store["depth"] = np.zeros(n_halos, dtype=np.int64)

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
