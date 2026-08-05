"""

Simple physics calculations which need to be done on read-in.

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astropy.cosmology import FLRW
    from octavian.data_management.conventions import OctavianConstants

# other packages
import numpy as np
import astropy.units as u

# internal imports
from octavian.data_management.data_structures import SimulationAttributes
from octavian.log import get_logger

logger = get_logger()


def derive_stellar_age(formation_time: np.ndarray, time_gyr: float, cosmology: FLRW) -> np.ndarray:
    """
    Converts stellar formation time (stored as a scale factor by gizmo/swift) into stellar age in GYr.
    """
    redshifts = 1.0 / formation_time - 1.0
    return time_gyr - cosmology.age(redshifts).to_value(u.Gyr)  # see astropy for integration details


def calculate_hydrogen_number_density(rho_cgs: np.ndarray, constants: OctavianConstants, XH: float) -> np.ndarray:
    """
    Calculates nH from the simulation parameters and user config.yaml.
    """
    return rho_cgs * XH / constants.PROTON_MASS_G


def calculate_temperature(
    internal_energy: np.ndarray,
    electron_abundance: np.ndarray,
    helium_fraction: np.ndarray,
    constants: OctavianConstants,
    gamma: float = 5 / 3,
) -> np.ndarray:
    """
    Calculates temperature from internal energy and electron abundance.
    """
    y_helium = helium_fraction / (4 * (1 - helium_fraction))
    mu = (1 + 4 * y_helium) / (1 + y_helium + electron_abundance)

    mean_molecular_weight = mu * constants.PROTON_MASS_G

    temperature = mean_molecular_weight * (gamma - 1) * internal_energy / constants.BOLTZMANN_CGS

    return temperature


def calculate_mean_interparticle_separation(
    n_star: int,
    n_gas: int,
    boxsize: float,
) -> float:
    """
    Computes the baryonic mean interparticle separation from the number of star and gas particles; black holes make up a small-enough subset of the box to be safely disregarded.

    Returns the mean interparticle separation.
    """
    mis = boxsize / (n_star + n_gas) ** (1.0 / 3.0)
    logger.debug(f"Mean baryonic interparticle separation: {mis:.2f}")

    return mis


def derive_simulation_attributes(
    cosmology: FLRW,
    h: float,
    a: float,
    redshift: float,
    omega_matter: float,
    omega_lambda: float,
    w_0: float,
    w_a: float,
    boxsize: float,
    n_star: int,
    n_gas: int,
    constants: OctavianConstants,
) -> SimulationAttributes:
    """
    Derives cosmology and calculates mean interparticle separation.

    Returns a SimulationAttributes dataclass.
    """
    time_gyr = cosmology.age(redshift).value
    Hz = cosmology.H(redshift).to(1 / u.s).value
    E_z = cosmology.efunc(redshift)
    rhocrit = cosmology.critical_density(redshift).to(u.M_sun / u.kpc**3).value
    omega_matter_z = cosmology.Om(redshift)

    r200_factor = (200 * 4.0 / 3.0 * np.pi * omega_matter_z * rhocrit * a**3) ** (-1.0 / 3.0)
    mis = calculate_mean_interparticle_separation(n_star=n_star, n_gas=n_gas, boxsize=boxsize)

    return SimulationAttributes(
        h=h,
        boxsize=boxsize,
        scale_factor=a,
        redshift=redshift,
        w_0=w_0,
        w_a=w_a,
        omega_matter=omega_matter,
        omega_lambda=omega_lambda,
        mis=mis,
        cosmology=cosmology,
        time_gyr=time_gyr,
        time=time_gyr * constants.GYR_S,
        Hz=Hz,
        rhocrit=rhocrit,
        rhocrit_comoving=rhocrit * a**3,
        E_z=E_z,
        omega_matter_z=omega_matter_z,
        r200_factor=r200_factor,
    )
