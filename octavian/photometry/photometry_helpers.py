"""

Helper functions for the photometry pipeline.

# NOTE: https://doi.org/10.1093/mnras/stu936 (another IGM attenuation formula)

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_tables import PhotometryTable

# other packages
import numpy as np

# internal imports
from ..log import get_logger

logger = get_logger()


def extinct_madau(
    wavelengths: np.ndarray,
    redshift: float,
) -> np.ndarray:
    """
    Applies IGM extinction for redshifted wavelengths according to the formula described by Madau (1995). The form used here is the approximation described in the footnote of p21, which is claimed to be accurate to within 5% and is used as the standard formula in codes such as synphot. Returns:

    - transmission: exp(-tau) lying between [0, 1]

    Madau (1995) ApJ 441 18 (doi: 10.1086/175332).
    """
    if redshift < 1e-10:  # in principle this won't be asked for from the config but this guards in case it is
        logger.debug(
            f"Requested Madau (1995) extinction but snapshot is at redshift {redshift}; Madau terms are all ones."
        )
        return np.ones_like(wavelengths, dtype=np.float64)

    logger.debug("Applying Madau (1995) IGM extinction.")
    lyman_limit = 912.0
    lyman_rest_wavelengths = np.array([1216, 1026, 973, 950])  # p21, after eq. 15
    a_vals = np.array([3.6e-3, 1.7e-3, 1.2e-3, 9.3e-4])  # p21 after eq. 15

    # integral bounds
    x_e = 1 + redshift
    in_bounds = wavelengths < lyman_limit * x_e
    x_c = wavelengths[in_bounds] / lyman_limit

    tau = np.zeros_like(wavelengths, dtype=np.float64)

    for rest_wavelength, coeff in zip(lyman_rest_wavelengths, a_vals):
        mask = wavelengths <= rest_wavelength * x_e
        tau[mask] += coeff * (wavelengths[mask] / rest_wavelength) ** 3.46

    # footnote equation
    tau[in_bounds] += (
        0.25 * (x_c**3.0) * (x_e**0.46 - x_c**0.46)
        + 9.4 * (x_c**1.5) * (x_e**0.18 - x_c**0.18)
        - 0.7 * (x_c**3.0) * (x_c**-1.32 - x_e**-1.32)
        - 0.023 * (x_e**1.68 - x_c**1.68)
    )

    transmission = np.clip(np.exp(-tau), 0.0, 1.0)  # this clip is more a guard and shouldn't be hit in practice

    return transmission


def resolve_band_names(
    requested: list[str],
    table: PhotometryTable,
) -> list[str]:
    """
    Matches and expands the user-requested bands in the photometry section of the config. This means users can do 'sdss' and get all sdss bands, for example; engineered against fsps.list_filters(), all of which are present in our photometry table. Returns:

    - filters_to_process: resolved list of filters to pull out of PhotometryTable, automatically deduplicated
    """
    available_filters = list(table.filters.keys())
    effective_wavelengths = np.array([f.lambda_eff for f in table.filters.values()])
    filters_to_process: list[str] = []

    for filter_name in requested:
        if filter_name == "all":
            return available_filters

        elif filter_name == "uvoir":  # all filters bluewards of 5 microns
            filters_to_process += [
                f for f, lambda_eff in zip(available_filters, effective_wavelengths) if lambda_eff < 50000.0
            ]

        elif filter_name in available_filters:
            filters_to_process.append(filter_name)

        else:  # filter_name is the instrument name usually followed by {_band} so we can grab all the bands associated with a filter
            matches = [f for f in available_filters if f.startswith(filter_name + "_")]

            if not matches:
                raise KeyError(
                    f"{filter_name} is not in the list of available filters: check typo? (run fsps.list_filters() for all filters)"
                )

            filters_to_process += matches

    filters_to_process = list(
        dict.fromkeys(filters_to_process)
    )  # HACK: deduplicate any filters by converting to a dict then back to a list

    if "v" not in filters_to_process:  # v must always be processed to get A_v
        filters_to_process.insert(0, "v")  # and put it first

    return filters_to_process
