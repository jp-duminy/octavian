"""

Standard attenuation curve formulae which compute the optical depth tau as a function of wavelength.

"""

# default library imports

# other packages
import numpy as np
from numba import njit

# internal imports

NORMALISATION = 5500  # angstrom


@njit(cache=True)
def atten_power_law(
    wavelengths: np.ndarray,
    alpha: float = 1.0,
    normalisation: float = 5500.0,
) -> np.ndarray:
    """
    Attenuate based on a simple power law in -(alpha), normalised by default to 5,500 angstrom. Returns:

    - taus: the optical depth at each wavelength.
    """
    taus = (wavelengths / NORMALISATION) ** (-alpha)

    return taus


@njit(cache=True)
def _calzetti_ir(
    wavelength: float,
    R_v: float = 4.05,
) -> float:
    """
    Evaluates the Calzetti IR attenuation law (defined between 6.3um and 22um) at the input wavelength (in angstrom). Returns:

    - k_lambda: k(lambda) at the input wavelength.

    Daniela Calzetti et al. 2000 ApJ 533 682 (doi: 10.1086/308692) Eq. 4
    """
    wavelength_um = wavelength / 1e4
    k_lambda = 2.659 * (-1.857 + (1.040 / wavelength_um)) + R_v

    return k_lambda


@njit(cache=True)
def _calzetti_uv(
    wavelength: float,
    R_v: float = 4.05,
) -> float:
    """
    Evaluates the Calzetti UV attenuation law (defined between 0.12um and 6.3um) at the input wavelength (in angstrom). Returns:

    - k_lambda: k(lambda) at the input wavelength.

    Daniela Calzetti et al. 2000 ApJ 533 682 (doi: 10.1086/308692) Eq. 4
    """
    wavelength_um = wavelength / 1e4
    k_lambda = (
        2.659 * (-2.156 + (1.509 / wavelength_um) - (0.198 / wavelength_um**2) + (0.011 / wavelength_um**3)) + R_v
    )

    return k_lambda


@njit(cache=True)
def atten_calzetti(wavelengths: np.ndarray) -> np.ndarray:
    """
    Attenuate based on the starburst curve described by Calzetti et al. (2000). The law is defined between 0.12um and 22um, so extrapolations to the far-UV and near-IR are made using the slope at the edge of either domain. Returns:

    - taus: the optical depth at each wavelength.

    Daniela Calzetti et al. 2000 ApJ 533 682 (doi: 10.1086/308692)
    """
    calzetti_rv = 4.05  # +/- 0.8

    # UV extrapolation: evaluate the UV power law at 1,100 and 1,200 angstrom, then use its slope.
    k_1100 = _calzetti_uv(wavelength=1100, R_v=calzetti_rv)
    k_1200 = _calzetti_uv(wavelength=1200, R_v=calzetti_rv)
    uv_slope = (k_1200 - k_1100) / 100

    # IR extrapolation: evaluate the IR power law at 21,000 and 22,000 angstrom
    k_21900 = _calzetti_ir(wavelength=21900, R_v=calzetti_rv)
    k_22000 = _calzetti_ir(wavelength=22000, R_v=calzetti_rv)
    ir_slope = (k_22000 - k_21900) / 100

    # normalisation
    normalisation = _calzetti_ir(wavelength=NORMALISATION, R_v=calzetti_rv) / calzetti_rv

    n_wave = wavelengths.shape[0]
    taus = np.empty(shape=n_wave, dtype=np.float64)

    for i in range(n_wave):
        wave = wavelengths[i]

        if wave < 1200.0:
            k = k_1100 + (wave - 1100.0) * uv_slope

        elif wave < 6300.0:
            k = _calzetti_uv(wavelength=wave, R_v=calzetti_rv)

        elif wave <= 22000.0:
            k = _calzetti_ir(wavelength=wave, R_v=calzetti_rv)

        else:
            k = k_22000 + (wave - 22000.0) * ir_slope

        if k < 0.0:
            k = 0.0

        taus[i] = k / calzetti_rv / normalisation

    return taus
