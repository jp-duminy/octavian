"""

Simple physics calculations which need to be done on read-in.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astropy.cosmology import FLRW
    from octavian.data_management.conventions import OctavianConstants

import numpy as np
from octavian.log import get_logger

logger = get_logger()


def derive_stellar_age(formation_time: np.ndarray, time_gyr: float, cosmology: FLRW) -> np.ndarray:
    """
    Converts GIZMO stellar formation time into stellar age in GYr.
    """
    redshifts = 1.0 / formation_time - 1.0
    return time_gyr - cosmology.age(redshifts).value  # see astropy for integration details


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
