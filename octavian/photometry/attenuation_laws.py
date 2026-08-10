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
    Returns k(lambda) from the Calzetti IR attenuation law (defined between 6.3um and 22um) evaluated at the input wavelength (in angstrom).
    """
    wavelength_um = 1e4 / wavelength
    k_lambda = 2.659 * (-1.857 + (1.040 * wavelength_um)) + R_v

    return k_lambda


@njit(cache=True)
def _calzetti_uv(
    wavelength: float,
    R_v: float = 4.05,
) -> float:
    """
    Returns k(lambda) from the Calzetti UV attenuation law (defined between 0.12um and 6.3um) evaluated at the input wavelength (in angstrom).
    """
    wavelength_um = 1e4 / wavelength
    k_lambda = (
        2.659 * (-2.156 + (1.509 * wavelength_um) - (0.198 * wavelength_um**2) + (0.011 * wavelength_um**3)) + R_v
    )

    return k_lambda


@njit(cache=True)
def atten_calzetti(wavelengths: np.ndarray) -> np.ndarray:
    """
    Attenuate based on the starburst curve described by Calzetti et al. (2000). The law is defined between 0.12um and 22um, so extrapolations to the far-UV and near-IR are made using the slope at the edge of either domain. Returns:

    - taus: the optical depth at each wavelength.

    Daniela Calzetti et al. 2000 ApJ 533 682 (doi: 10.1086/308692) Eq. 4
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
        wavelength = wavelengths[i]

        if wavelength < 1200.0:
            k = k_1100 + (wavelength - 1100.0) * uv_slope

        elif wavelength < 6300.0:
            k = _calzetti_uv(wavelength=wavelength, R_v=calzetti_rv)

        elif wavelength <= 22000.0:
            k = _calzetti_ir(wavelength=wavelength, R_v=calzetti_rv)

        else:
            k = k_22000 + (wavelength - 22000.0) * ir_slope

        if k < 0.0:
            k = 0.0

        taus[i] = k / calzetti_rv / normalisation

    return taus


@njit(cache=True)
def _conroy_ir(
    x: float,
    R_v: float = 3.1,
) -> float:
    """
    Returns a(lambda) + b(lambda)/R_v from the Conroy IR attenuation law (defined between 0.3um^-1 and 1.1um^-1) evaluated at the input wavelength (in angstroms).
    """
    a = 0.574 * (x**1.61)
    b = -0.527 * (x**1.61)
    result = a + b / R_v

    return result


@njit(cache=True)
def _conroy_optical(
    x: float,
    R_v: float = 3.1,
) -> float:
    """
    Returns a(lambda) + b(lambda)/R_v from the Conroy optical/near-IR attenuation law (defined between 1.1um^-1 and 3.3um^-1) evaluated at the input wavelength (in angstroms).
    """
    y = x - 1.82  # NOTE: in the paper they define y = x - 1.82 where x is in um^-1

    a = (
        1
        + (0.177 * y)
        - (0.504 * y**2)
        - (0.0243 * y**3)
        + (0.721 * y**4)
        + (0.0198 * y**5)
        - (0.775 * y**6)
        + (0.330 * y**7)
    )
    b = (
        (1.413 * y)
        + (2.283 * y**2)
        + (1.072 * y**3)
        - (5.384 * y**4)
        - (0.622 * y**5)
        + (5.303 * y**6)
        - (2.090 * y**7)
    )

    result = a + b / R_v

    return result


@njit(cache=True)
def _conroy_mid_uv(
    x: float,
    f_bump: float,
    R_v: float = 3.1,
) -> float:
    """
    Returns a(lambda) + b(lambda)/R_v from the Conroy near/mid UV attenuation law (defined between 3.3um^-1 and 5.9um^-1) evaluated at the input wavelength (in angstroms).
    """
    fa = (3.3 / x) ** 6 * (
        -0.0370 + (0.0469 * f_bump) - (0.601 * f_bump / R_v) + (0.542 / R_v)
    )  # this is the paper variable name

    a = 1.752 - (0.316 * x) - ((0.104 * f_bump) / ((x - 4.67) ** 2 + 0.341)) + fa
    b = -3.09 + (1.825 * x) + ((1.206 * f_bump) / ((x - 4.62) ** 2 + 0.263))

    result = a + b / R_v

    return result


@njit(cache=True)
def _conroy_far_uv(
    x: float,
    f_bump: float,
    R_v: float = 3.1,
) -> float:
    """
    Returns a(lambda) + b(lambda)/R_v from the Conroy far-UV attenuation law (defined between 3.3um^-1 and 5.9um^-1) evaluated at the input wavelength (in angstrom).
    """
    fa = -0.0447 * (x - 5.9) ** 2 - 0.00978 * (x - 5.9) ** 3  # this is the paper variable name
    fb = 0.213 * (x - 5.9) ** 2 + 0.121 * (x - 5.9) ** 3  # this is the paper variable name

    a = 1.752 - (0.316 * x) - ((0.104 * f_bump) / ((x - 4.67) ** 2 + 0.341)) + fa
    b = -3.09 + (1.825 * x) + ((1.206 * f_bump) / ((x - 4.62) ** 2 + 0.263)) + fb

    result = a + b / R_v

    return result


@njit(cache=True)
def atten_conroy(wavelengths: np.ndarray, f_bump: float = 0.6) -> np.ndarray:
    """
    Attenuate based on the Milky Way extinction curve with arbitrary UV bump (parametrised by f_bump) described by Conroy et al. (2010). This is defined from 0.3um to 8.0um. At f_bump = 0, Cardelli is recovered; 1.0 returns the standard UV bump. Returns:

    - taus: the optical depth at each wavelength.

    Charlie Conroy et al 2010 ApJ 718 184 (doi: 10.1088/0004-637X/718/1/184)
    """
    conroy_rv = 3.1

    wavelengths_inverse_um = 1e4 / wavelengths  # angstrom -> um^-1
    n_wave = wavelengths_inverse_um.shape[0]
    taus = np.empty(shape=n_wave, dtype=np.float64)

    upper_value = 8.0  # where Conroy ends
    end_of_domain = _conroy_far_uv(x=upper_value, f_bump=f_bump, R_v=conroy_rv)  # tau at the value where Conroy ends

    for i in range(n_wave):
        x = wavelengths_inverse_um[
            i
        ]  # wavelength in inverse microns (what the paper calls it, and why the loop looks slightly weird))

        if x > upper_value:  # X-UV extrapolation
            tau = (upper_value / x) ** -1.3 * end_of_domain  # power law taken from pyloser in Caesar (undocumented)

        elif x > 5.9:
            tau = _conroy_far_uv(x=x, f_bump=f_bump, R_v=conroy_rv)

        elif x > 3.3:
            tau = _conroy_mid_uv(x=x, f_bump=f_bump, R_v=conroy_rv)

        elif x > 1.1:
            tau = _conroy_optical(x=x, R_v=conroy_rv)

        else:
            tau = _conroy_ir(x=x, R_v=conroy_rv)

        taus[i] = tau

    return taus
