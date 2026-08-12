"""

Tests whether the photometry functions correctly recover known values.

"""

# other packages
import numpy as np

# internal imports
from octavian.photometry.photometry_helpers import extinct_madau


def test_madau() -> None:
    """
    Tests photometry helper extinct_madau by verifying its output against the Caesar equivalent provided by synphot. Also verifies its result is physically sensible (at redshift zero, Madau term is ones).
    """
    test_redshift = 3
    test_wavelengths = np.linspace(500, 7000, 30)  # in observer frame

    # print(repr(vals)) where vals is the output of synphot's ExtinctionCurve.value from its etau_madau method on test_wavelengths with test_redshift
    synphot_vals = np.array(
        [
            0.34793921,
            0.1935625,
            0.11392457,
            0.07147155,
            0.04805771,
            0.03485828,
            0.02750745,
            0.02387668,
            0.02311123,
            0.02535915,
            0.03215376,
            0.04815082,
            0.08727324,
            0.19670177,
            0.56782333,
            0.60376266,
            0.64316617,
            0.75053057,
            0.71035254,
            0.66755994,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    )

    result = extinct_madau(wavelengths=test_wavelengths, redshift=test_redshift)
    np.testing.assert_allclose(result, synphot_vals, rtol=1e-6, err_msg="test_madau failed.")

    at_redshift_zero = extinct_madau(wavelengths=test_wavelengths, redshift=0)
    np.testing.assert_array_equal(at_redshift_zero, np.ones_like(test_wavelengths), err_msg="test_madau failed.")
