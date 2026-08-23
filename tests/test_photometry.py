"""

Tests whether the photometry functions correctly recover known values.

"""

# other packages
import numpy as np

# internal imports
from octavian.photometry.photometry_helpers import extinct_madau
from octavian.photometry.dust_curves import (
    atten_calzetti,
    atten_conroy,
    atten_power_law,
    extinct_cardelli,
    extinct_lmc,
    extinct_smc,
    NORMALISATION,
)
from octavian.photometry.photometry_helpers import (
    build_interpolation_table,
    interpolate_ssp,
    _cubic_kernel,
    _quintic_kernel,
    DUST_CURVE_IDX,
)
from octavian.photometry.photometry_computations import compute_metal_column_densities, apply_extinction_law

WAVELENGTHS = np.linspace(900, 30000, 500)


def test_dust_curves() -> None:
    """
    Tests the extinction laws from dust_curves.py
    """
    curves = {
        "calzetti": atten_calzetti(wavelengths=WAVELENGTHS),
        "conroy": atten_conroy(wavelengths=WAVELENGTHS),
        "power_law": atten_power_law(wavelengths=WAVELENGTHS),
        "cardelli": extinct_cardelli(wavelengths=WAVELENGTHS),
        "smc": extinct_smc(wavelengths=WAVELENGTHS),
        "lmc": extinct_lmc(wavelengths=WAVELENGTHS),
    }

    for name, tau in curves.items():
        assert np.all(np.isfinite(tau)), f"test_dust_curves failed: non-finite values in dust curve {name}."
        assert np.all(tau >= 0.0), f"test_dust_curves failed: negative optical depth in dust curve {name}."

    # quick check on the normalisation, NOTE: I hand checked these before reducing the tolerance
    for name, func in [
        ("calzetti", atten_calzetti),
        ("conroy", atten_conroy),
        ("power_law", atten_power_law),
    ]:  # alpha is defaulted to 1.0 so we get away with the generic
        tau_norm = func(wavelengths=np.array([NORMALISATION]))
        np.testing.assert_allclose(
            tau_norm, 1.0, rtol=2e-3, err_msg=f"test_dust_curves failed: {name} is not properly normalised."
        )


def test_kernel_table() -> None:
    """
    Verify the kernel values are in increasing order, zero at the edge, and their integrands sum to one.
    """
    for kernel_type in [0, 1]:  # cubic, quintic
        n_bins = 1000
        table = build_interpolation_table(n_bins=n_bins, kernel_type=kernel_type)

        label = "cubic" if kernel_type == 0 else "quintic"

        assert np.all(table >= 0.0), f"test_kernel_table failed: {label} has negative kernel weight."
        assert table[-1] == 0.0, f"test_kernel_table failed: {label} is not zero at its edge."
        assert np.all(np.diff(table) <= 0.0), f"test_kernel_table failed: {label} is not in increasing order."

        # verify our numba trapezoid rule agrees with np.trapezoid
        bin_width = 1.0 / n_bins
        b_values = np.arange(n_bins + 1, dtype=np.float64) * bin_width
        volume_integral = 2.0 * np.pi * np.trapezoid(b_values * table, b_values)

        q_vals = np.linspace(0, 1, 10000)
        if kernel_type == 0:
            w_vals = np.array([_cubic_kernel(q) for q in q_vals])
        else:
            w_vals = np.array([_quintic_kernel(q) for q in q_vals])
        volume_integral_3d = 4.0 * np.pi * np.trapezoid(q_vals**2 * w_vals, q_vals)

        np.testing.assert_allclose(
            volume_integral,
            volume_integral_3d,
            rtol=1e-2,
            err_msg=f"test_kernel_table failed: {label} projected and volume integrals disagree.",
        )


def test_extinction_law_selection() -> None:
    """
    Verifies apply_extinction_law selects the correct dust curve based on sSFR and metallicity.
    """

    # give each dust curve constant values (for identification purposes)
    dust_curves = np.empty(shape=(len(DUST_CURVE_IDX), len(WAVELENGTHS)), dtype=np.float64)
    for idx in DUST_CURVE_IDX.values():
        dust_curves[idx] = float(idx + 1)

    out_curve = np.empty(len(WAVELENGTHS), dtype=np.float64)

    # log ssfr > 0 is calzetti regime
    apply_extinction_law(
        out_curve=out_curve,
        log_ssfr=0.1,
        log_Z_solar=-1.0,
        dust_curves=dust_curves,
        dust_law=DUST_CURVE_IDX["composite"],
    )
    expected_calzetti = float(DUST_CURVE_IDX["calzetti"] + 1)
    assert np.allclose(out_curve, expected_calzetti), "test_extinction_law_selection failed: expected calzetti branch."

    # log ssfr < -1 is MW regime (expect cardelli)
    apply_extinction_law(
        out_curve=out_curve,
        log_ssfr=-11.5,
        log_Z_solar=0.0,
        dust_curves=dust_curves,
        dust_law=DUST_CURVE_IDX["composite"],
    )
    expected_cardelli = float(DUST_CURVE_IDX["cardelli"] + 1)
    assert np.allclose(out_curve, expected_cardelli), "test_extinction_law_selection failed: expected cardelli branch."

    # for a single law we should have no mixing
    for law_name, law_idx in DUST_CURVE_IDX.items():
        if law_name in ("mix_calz_mw", "composite"):
            continue
        apply_extinction_law(
            out_curve=out_curve,
            log_ssfr=-10.0,
            log_Z_solar=0.0,
            dust_curves=dust_curves,
            dust_law=law_idx,
        )
        expected = float(law_idx + 1)
        assert np.allclose(out_curve, expected), (
            "test_extinction_law_selection failed: did not correctly select a single law."
        )

    # request composite but add low metallicity, should get some SMC mixing in there
    apply_extinction_law(
        out_curve=out_curve,
        log_ssfr=-11.5,
        log_Z_solar=-3.0,  # well below -2, so Z_weight = 0 -> pure SMC
        dust_curves=dust_curves,
        dust_law=DUST_CURVE_IDX["composite"],
    )
    expected_smc = float(DUST_CURVE_IDX["smc"] + 1)
    assert np.allclose(out_curve, expected_smc), (
        "test_extinction_law_selection failed: expected SMC mixing for metal-poor galaxies."
    )

    # by contrast the calzetti/mw mixing should not apply any SMC mixing
    apply_extinction_law(
        out_curve=out_curve,
        log_ssfr=0.1,
        log_Z_solar=-3.0,
        dust_curves=dust_curves,
        dust_law=DUST_CURVE_IDX["mix_calz_mw"],
    )
    assert np.allclose(out_curve, expected_calzetti), (
        "test_extinction_law_selection failed: mix_calz_mw should not do any SMC mixing."
    )


def test_ssp_interpolation() -> None:
    """
    Tests photometry helper interpolate_ssp().
    """
    # generate a synthetic ssp table (so we don't do the big fsps call)
    n_Z, n_age, n_wave = 3, 4, 5
    age_grid = np.array([6.0, 7.0, 8.0, 9.0])
    Z_grid = np.array([-4.0, -3.0, -2.0])
    spectra = np.arange(n_Z * n_age * n_wave, dtype=np.float64).reshape(n_Z, n_age, n_wave)
    mass_remaining = np.linspace(0.1, 1.0, n_Z * n_age).reshape(n_Z, n_age)

    out_spectrum = np.empty(n_wave, dtype=np.float64)

    # verify the interpolation is exact if you put in the actual grid points
    for i_Z in range(n_Z):
        for i_age in range(n_age):
            m_rem = interpolate_ssp(
                log_age=age_grid[i_age],
                log_Z=Z_grid[i_Z],
                age_grid=age_grid,
                Z_grid=Z_grid,
                spectra=spectra,
                mass_remaining=mass_remaining,
                out_spectrum=out_spectrum,
            )
            np.testing.assert_allclose(
                out_spectrum,
                spectra[i_Z, i_age],
                rtol=1e-12,
                err_msg="test_ssp_interpolation failed: does not recover exact spectra on grid points.",
            )
            np.testing.assert_allclose(
                m_rem,
                mass_remaining[i_Z, i_age],
                rtol=1e-12,
                err_msg="test_ssp_interpolation failed: does not recover exact mass remaining on grid points.",
            )

    # verify interpolated points actually fall between the two values between which they are interpolated
    mid_age = 0.5 * (age_grid[1] + age_grid[2])
    mid_Z = 0.5 * (Z_grid[0] + Z_grid[1])
    interpolate_ssp(
        log_age=mid_age,
        log_Z=mid_Z,
        age_grid=age_grid,
        Z_grid=Z_grid,
        spectra=spectra,
        mass_remaining=mass_remaining,
        out_spectrum=out_spectrum,
    )
    edges = spectra[:2, 1:3, :]  # edges within which we are interpolating. NOTE: hardcoded to the 3, 4, 5 above
    assert np.all(out_spectrum >= edges.min(axis=(0, 1))), (
        "test_ssp_interpolation failed: spectrum values are below interpolation range."
    )
    assert np.all(out_spectrum <= edges.max(axis=(0, 1))), (
        "test_ssp_interpolation failed: spectrum values are above interpolation range."
    )


def test_metal_column_density() -> None:
    """
    Verifies metal column density: analytic b=0 case, zero behind star, zero beyond smoothing length.
    """
    kernel_table = build_interpolation_table(n_bins=1000, kernel_type=0)
    los_axis = 2
    boxsize = 1000.0
    gal_centre = np.array([500.0, 500.0, 500.0])
    h = 10.0
    neighbour_offsets = np.array([(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)], dtype=np.int64)

    gas_mass = np.array([1.0])
    gas_metallicity = np.array([0.02])  # the galaxies and cosmology course solar metallicity
    smoothing_lengths = np.array([h])

    # gas directly in front of star so the analytic formula is nice and easy
    star_pos = np.array([[500.0, 500.0, 500.0]])  # remember observer is at -infinity
    gas_pos = np.array([[500.0, 500.0, 495.0]])

    Z_col = compute_metal_column_densities(
        star_pos=star_pos,
        gas_pos=gas_pos,
        gas_mass=gas_mass,
        gas_metallicity=gas_metallicity,
        smoothing_lengths=smoothing_lengths,
        neighbour_offsets=neighbour_offsets,
        gal_centre=gal_centre,
        kernel_table=kernel_table,
        los_axis=los_axis,
        boxsize=boxsize,
    )
    expected = gas_mass[0] * gas_metallicity[0] * kernel_table[0] / h**2
    np.testing.assert_allclose(
        Z_col[0], expected, rtol=1e-10, err_msg="test_metal_column_density failed: analytic result does not match."
    )

    # put some gas particles behind the star so should not attenuate it
    gas_pos_behind = np.array([[500.0, 500.0, 505.0]])
    Z_col_behind = compute_metal_column_densities(
        star_pos=star_pos,
        gas_pos=gas_pos_behind,
        gas_mass=gas_mass,
        gas_metallicity=gas_metallicity,
        smoothing_lengths=smoothing_lengths,
        neighbour_offsets=neighbour_offsets,
        gal_centre=gal_centre,
        kernel_table=kernel_table,
        los_axis=los_axis,
        boxsize=boxsize,
    )
    assert Z_col_behind[0] == 0.0, "test_metal_column_density failed: gas behind star should not attenuate its light"

    # gas beyond one smoothing length should be kernel weighted to zero
    gas_pos_far = np.array([[500.0 + h + 1.0, 500.0, 495.0]])
    Z_col_far = compute_metal_column_densities(
        star_pos=star_pos,
        gas_pos=gas_pos_far,
        gas_mass=gas_mass,
        gas_metallicity=gas_metallicity,
        smoothing_lengths=smoothing_lengths,
        neighbour_offsets=neighbour_offsets,
        gal_centre=gal_centre,
        kernel_table=kernel_table,
        los_axis=los_axis,
        boxsize=boxsize,
    )
    assert Z_col_far[0] == 0.0, (
        "test_metal_column_density failed: gas beyond smoothing length should contribute zero via kernel weighting."
    )


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
