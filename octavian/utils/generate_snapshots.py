"""

Procedurally generate SWIFT/GIZMO snapshots with junk data for pipeline testing and producing synthetic test catalogues.

"""

import h5py
import numpy as np
from pathlib import Path

SEED = 2371434
rng = np.random.default_rng(seed=SEED)

SWIFT_DATASET_ATTRS: dict[str, dict[str, float]] = {
    "Coordinates": {"a-scale exponent": 1, "h-scale exponent": 0, "cgs_factor": 3.08567758e24},
    "Velocities": {"a-scale exponent": 0, "h-scale exponent": 0, "cgs_factor": 100000},
    "Masses": {"a-scale exponent": 0, "h-scale exponent": 0, "cgs_factor": 1.98841e43},
    "Potentials": {"a-scale exponent": -1, "h-scale exponent": 0, "cgs_factor": 9999999999.999998},
    "InternalEnergies": {"a-scale exponent": -2, "h-scale exponent": 0, "cgs_factor": 9999999999.999998},
    "Densities": {"a-scale exponent": -3, "h-scale exponent": 0, "cgs_factor": 6.767905773162602e-31},
    "AtomicHydrogenMasses": {"a-scale exponent": 0, "h-scale exponent": 0, "cgs_factor": 1.98841e43},
    "StarFormationRates": {"a-scale exponent": 0, "h-scale exponent": 0, "cgs_factor": 6.443997950038578e23},
    "MetalMassFractions": {"a-scale exponent": 0, "h-scale exponent": 0, "cgs_factor": 1},
    "ElementMassFractions": {"a-scale exponent": 0, "h-scale exponent": 0, "cgs_factor": 1},
    "MolecularHydrogenFractions": {"a-scale exponent": 0, "h-scale exponent": 0, "cgs_factor": 1},
    "BirthScaleFactors": {"a-scale exponent": 0, "h-scale exponent": 0, "cgs_factor": 1},
    "DynamicalMasses": {"a-scale exponent": 0, "h-scale exponent": 0, "cgs_factor": 1.98841e43},
    "SubgridMasses": {"a-scale exponent": 0, "h-scale exponent": 0, "cgs_factor": 1.98841e43},
    "AccretionRates": {"a-scale exponent": 0, "h-scale exponent": 0, "cgs_factor": 6.443997950038578e23},
}

# cosmology
h = 0.68
scale_factor = 0.25
w_0 = -1
w_a = 0
T_cmb_0 = 2.73
redshift = 3.0
omega_matter = 0.3
omega_lambda = 0.7
boxsize = 500  # ckpc

# particles
N_GAS = 240
N_DM = 250
N_STAR = 50
N_BH = 2

# haloes
N_HALOES = 3
FRAC_INTERLOPERS = 0.15  # fraction of particles which interlope about the snapshot (sentinel HaloIDs)
halo_centres = rng.uniform(100, boxsize - 100, size=(N_HALOES, 3))
halo_velocities = rng.normal(loc=0, scale=100, size=(N_HALOES, 3))


def generate_gizmo_snapshot(path: Path) -> None:
    """
    Procedurally generates a GIZMO snapshot filled with junk data. The catalogue is self-consistent, but its data only exists to be accessed for file validation, not any sort of physics checks.

    Parameters
    ----------
    path: pathlib.Path
        Path object pointing to where the GIZMO snapshot should be written.
    """
    # generate the junk data
    gas_base = _generate_base_datasets(n_particles=N_GAS, halo_centres=halo_centres, halo_velocities=halo_velocities)
    dm_base = _generate_base_datasets(n_particles=N_DM, halo_centres=halo_centres, halo_velocities=halo_velocities)
    star_base = _generate_base_datasets(n_particles=N_STAR, halo_centres=halo_centres, halo_velocities=halo_velocities)
    bh_base = _generate_base_datasets(n_particles=N_BH, halo_centres=halo_centres, halo_velocities=halo_velocities)

    gas_specific = _generate_gas_datasets(n_gas=N_GAS)
    star_specific = _generate_star_datasets(n_star=N_STAR)
    bh_specific = _generate_bh_datasets(n_bh=N_BH)

    # create hdf5 file
    with h5py.File(path, "w") as f:
        # header
        header = f.create_group("Header")
        header.attrs["HubbleParam"] = h
        header.attrs["BoxSize"] = boxsize * h  # reader divides by h
        header.attrs["Omega0"] = omega_matter
        header.attrs["OmegaLambda"] = omega_lambda
        header.attrs["Time"] = scale_factor
        header.attrs["Redshift"] = redshift
        header.attrs["NumPart_Total"] = np.array([N_GAS, N_DM, 0, 0, N_STAR, N_BH], dtype=np.uint32)

        # gas datasets
        gas = f.create_group("PartType0")
        gas.create_dataset("Coordinates", data=gas_base["pos"] * h)
        gas.create_dataset("Velocities", data=gas_base["vel"])
        gas.create_dataset("Masses", data=gas_base["mass"] * h)
        gas.create_dataset("Potential", data=gas_base["potential"])
        gas.create_dataset("HaloID", data=gas_base["HaloID"] + 1)  # gizmo has 0 as its sentinel

        gas.create_dataset("InternalEnergy", data=gas_specific["internal_energy"])
        gas.create_dataset("ElectronAbundance", data=gas_specific["electron_abundance"])
        gas.create_dataset("Density", data=gas_specific["rho"] / h**2)
        gas.create_dataset("NeutralHydrogenAbundance", data=gas_specific["fHI"])
        gas.create_dataset("StarFormationRate", data=gas_specific["sfr"])
        gas.create_dataset("FractionH2", data=gas_specific["fH2"])
        gas.create_dataset("Metallicity", data=gas_specific["metallicity"])

        # dm datasets
        dm = f.create_group("PartType1")
        dm.create_dataset("Coordinates", data=dm_base["pos"] * h)
        dm.create_dataset("Velocities", data=dm_base["vel"])
        dm.create_dataset("Masses", data=dm_base["mass"] * h)
        dm.create_dataset("Potential", data=dm_base["potential"])
        dm.create_dataset("HaloID", data=dm_base["HaloID"] + 1)  # gizmo has 0 as its sentinel

        # star datasets
        star = f.create_group("PartType4")
        star.create_dataset("Coordinates", data=star_base["pos"] * h)
        star.create_dataset("Velocities", data=star_base["vel"])
        star.create_dataset("Masses", data=star_base["mass"] * h)
        star.create_dataset("Potential", data=star_base["potential"])
        star.create_dataset("HaloID", data=star_base["HaloID"] + 1)  # gizmo has 0 as its sentinel

        star.create_dataset("StellarFormationTime", data=star_specific["age"])
        star.create_dataset("Metallicity", data=star_specific["metallicity"])

        # bh datasets
        bh = f.create_group("PartType5")
        bh.create_dataset("Coordinates", data=bh_base["pos"] * h)
        bh.create_dataset("Velocities", data=bh_base["vel"])
        bh.create_dataset("Masses", data=bh_base["mass"])
        bh.create_dataset("Potential", data=bh_base["potential"])
        bh.create_dataset("HaloID", data=bh_base["HaloID"] + 1)  # gizmo has 0 as its sentinel

        bh.create_dataset("BH_Mass", data=bh_specific["bhmass"] * h)
        bh.create_dataset("BH_Mdot", data=bh_specific["bhmdot"] * h)


def generate_swift_snapshot(path: Path) -> None:
    """
    Procedurally generates a SWIFT snapshot filled with junk data. The catalogue is self-consistent, but its data only exists to be accessed for file validation, not any sort of physics checks.

    Parameters
    ----------
    path: pathlib.Path
        Path object pointing to where the SWIFT snapshot should be written.
    """

    gas_base = _generate_base_datasets(n_particles=N_GAS, halo_centres=halo_centres, halo_velocities=halo_velocities)
    dm_base = _generate_base_datasets(n_particles=N_DM, halo_centres=halo_centres, halo_velocities=halo_velocities)
    star_base = _generate_base_datasets(n_particles=N_STAR, halo_centres=halo_centres, halo_velocities=halo_velocities)
    bh_base = _generate_base_datasets(n_particles=N_BH, halo_centres=halo_centres, halo_velocities=halo_velocities)

    gas_specific = _generate_gas_datasets(n_gas=N_GAS)
    star_specific = _generate_star_datasets(n_star=N_STAR)
    bh_specific = _generate_bh_datasets(n_bh=N_BH)

    with h5py.File(path, "w") as f:
        # header
        header = f.create_group("Header")
        header.attrs["BoxSize"] = np.array([boxsize, boxsize, boxsize], dtype=np.float64)
        header.attrs["NumPart_Total"] = np.array([N_GAS, N_DM, 0, 0, N_STAR, N_BH], dtype=np.uint32)

        # cosmology
        cosmo = f.create_group("Cosmology")
        cosmo.attrs["h"] = h
        cosmo.attrs["Scale-factor"] = scale_factor
        cosmo.attrs["w_0"] = w_0
        cosmo.attrs["w_a"] = w_a
        cosmo.attrs["T_CMB_0 [K]"] = T_cmb_0
        cosmo.attrs["Redshift"] = redshift
        cosmo.attrs["Omega_m"] = omega_matter
        cosmo.attrs["Omega_lambda"] = omega_lambda

        # units
        units = f.create_group("Units")
        units.attrs["Unit length in cgs (U_L)"] = 3.08567758e24  # 1 Mpc in cm
        units.attrs["Unit mass in cgs (U_M)"] = 1.98841e43  # 10^10 Msun in g
        units.attrs["Unit time in cgs (U_t)"] = 3.08567758e19  # Mpc / km/s in s
        units.attrs["Unit current in cgs (U_I)"] = 1.0
        units.attrs["Unit temperature in cgs (U_T)"] = 1.0

        # gas
        gas = f.create_group("PartType0")
        _create_swift_dataset(gas, "Coordinates", gas_base["pos"])
        _create_swift_dataset(gas, "Velocities", gas_base["vel"])
        _create_swift_dataset(gas, "Masses", gas_base["mass"])
        _create_swift_dataset(gas, "Potentials", gas_base["potential"])
        _create_swift_halo_ids(gas, gas_base["HaloID"])

        _create_swift_dataset(gas, "InternalEnergies", gas_specific["internal_energy"])
        _create_swift_dataset(gas, "Densities", gas_specific["rho"])
        _create_swift_dataset(gas, "AtomicHydrogenMasses", gas_specific["fHI"] * gas_base["mass"])
        _create_swift_dataset(gas, "StarFormationRates", gas_specific["sfr"])
        _create_swift_dataset(gas, "MolecularHydrogenFractions", gas_specific["fH2"])
        _create_swift_dataset(gas, "MetalMassFractions", gas_specific["metallicity"][:, 0])
        _create_swift_dataset(gas, "ElementMassFractions", gas_specific["metallicity"])

        # dm
        dm = f.create_group("PartType1")
        _create_swift_dataset(dm, "Coordinates", dm_base["pos"])
        _create_swift_dataset(dm, "Velocities", dm_base["vel"])
        _create_swift_dataset(dm, "Masses", dm_base["mass"])
        _create_swift_dataset(dm, "Potentials", dm_base["potential"])
        _create_swift_halo_ids(dm, dm_base["HaloID"])

        # star
        star = f.create_group("PartType4")
        _create_swift_dataset(star, "Coordinates", star_base["pos"])
        _create_swift_dataset(star, "Velocities", star_base["vel"])
        _create_swift_dataset(star, "Masses", star_base["mass"])
        _create_swift_dataset(star, "Potentials", star_base["potential"])
        _create_swift_halo_ids(star, star_base["HaloID"])

        _create_swift_dataset(star, "BirthScaleFactors", star_specific["age"])
        _create_swift_dataset(star, "MetalMassFractions", star_specific["metallicity"][:, 0])
        _create_swift_dataset(star, "ElementMassFractions", star_specific["metallicity"])

        # bh
        bh = f.create_group("PartType5")
        _create_swift_dataset(bh, "Coordinates", bh_base["pos"])
        _create_swift_dataset(bh, "Velocities", bh_base["vel"])
        _create_swift_dataset(bh, "DynamicalMasses", bh_base["mass"])
        _create_swift_dataset(bh, "Potentials", bh_base["potential"])
        _create_swift_halo_ids(bh, bh_base["HaloID"])

        _create_swift_dataset(bh, "SubgridMasses", bh_specific["bhmass"])
        _create_swift_dataset(bh, "AccretionRates", bh_specific["bhmdot"])


def _generate_base_datasets(
    n_particles: int,
    halo_centres: np.ndarray,
    halo_velocities: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Generates positions, velocities, masses, potential, and halo IDs
    for n_particles distributed across N_HALOS clusters plus a field population.
    """
    n_interlopers = int(n_particles * FRAC_INTERLOPERS)
    n_bound = n_particles - n_interlopers

    assignments = rng.integers(0, N_HALOES, size=n_bound)  # assign particles to haloes in a roughly even split
    positions = halo_centres[assignments] + rng.normal(scale=0.25, size=(n_bound, 3))
    velocities = halo_velocities[assignments] + rng.normal(scale=1.5, size=(n_bound, 3))

    # interlopers get randomly distributed about the box
    interloper_pos = rng.uniform(0, boxsize, size=(n_interlopers, 3))
    interloper_vel = rng.normal(scale=200, size=(n_interlopers, 3))
    interloper_ids = np.full(n_interlopers, -1, dtype=np.int64)  # sentinel value

    positions = np.vstack([positions, interloper_pos]) % boxsize
    velocities = np.vstack([velocities, interloper_vel])
    halo_ids = np.concatenate([assignments, interloper_ids])

    return {
        "pos": positions.astype(np.float64),
        "vel": velocities.astype(np.float64),
        "mass": rng.uniform(1e-4, 1e-2, size=n_particles).astype(np.float64),
        "potential": rng.uniform(-1e4, -1, size=n_particles).astype(np.float64),
        "HaloID": halo_ids,
    }


def _generate_gas_datasets(n_gas: int) -> dict[str, np.ndarray]:
    """
    Generates gas-specific datasets.
    """
    return {
        "internal_energy": rng.uniform(1e3, 1e5, size=n_gas).astype(np.float64),
        "electron_abundance": rng.uniform(0.0, 1.16, size=n_gas).astype(np.float64),
        "rho": rng.uniform(1e-6, 1e-2, size=n_gas).astype(np.float64),
        "fHI": rng.uniform(0.0, 1.0, size=n_gas).astype(np.float64),
        "sfr": rng.uniform(0.0, 10.0, size=n_gas).astype(np.float64),
        "metallicity": rng.uniform(0.0, 0.05, size=(n_gas, 11)).astype(np.float64),
        "fH2": rng.uniform(0.0, 1.0, size=n_gas).astype(np.float64),
    }


def _generate_star_datasets(n_star: int) -> dict[str, np.ndarray]:
    """
    Generates star-specific datasets.
    """
    return {
        "age": rng.uniform(0.05, scale_factor, size=n_star).astype(np.float64),
        "metallicity": rng.uniform(0.0, 0.05, size=(n_star, 11)).astype(np.float64),
    }


def _generate_bh_datasets(n_bh: int) -> dict[str, np.ndarray]:
    """
    Generates bh-specific datasets.
    """
    return {
        "bhmass": rng.uniform(1e-6, 1e-3, size=n_bh).astype(np.float64),
        "bhmdot": rng.uniform(0.0, 1e-4, size=n_bh).astype(np.float64),
    }


def _create_swift_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    """
    Creates a dataset and stamps the predefined SWIFT unit attributes.
    """
    dataset = group.create_dataset(name, data=data)
    attrs = SWIFT_DATASET_ATTRS[name]
    dataset.attrs["a-scale exponent"] = attrs["a-scale exponent"]
    dataset.attrs["h-scale exponent"] = attrs["h-scale exponent"]
    dataset.attrs["Conversion factor to CGS (not including cosmological corrections)"] = attrs["cgs_factor"]


def _create_swift_halo_ids(group: h5py.Group, halo_ids: np.ndarray) -> None:
    """
    Moves the SWIFT IDs to the sentinel value in their snapshots.
    """
    swift_ids = halo_ids.copy() + 1  # shift to 1-indexed
    swift_ids[halo_ids == -1] = 2147483647
    group.create_dataset("FOFGroupIDs", data=swift_ids.astype(np.int32))
