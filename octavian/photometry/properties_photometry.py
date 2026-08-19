"""

Functions to handle executing the photometry pipeline (physics in photometry_computations.py). This is from where the pipeline imports.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import SimulationData, OctavianConfig

# other packages
import numpy as np

# internal imports
from .dust_curves import (
    atten_calzetti,
    atten_conroy,
    atten_power_law,
    extinct_cardelli,
    extinct_lmc,
    extinct_smc,
)
from .photometry_helpers import (
    extinct_madau,
    build_interpolation_table,
    StarData,
    GasData,
    SSPData,
    DustData,
    FilterData,
    PhotometryConstants,
    DUST_CURVE_IDX,
    LOS_AXIS_MAP,
)
from .photometry_tables import read_photometry_table
from .photometry_computations import compute_photometric_properties
from ..log import get_logger

logger = get_logger()


def run_photometry(simulation_data: SimulationData, config: OctavianConfig) -> None:
    """
    Top-level executor for photometry.
    """
    logger.info("Preparing photometry data.")
    photometry_table = read_photometry_table(table_path=config.table_filepath)
    constants = simulation_data.constants
    sim = simulation_data.simulation
    kernel_table = build_interpolation_table(n_bins=config.interpolation_bins, kernel_type=config.kernel_type)
    los_axis = LOS_AXIS_MAP[config.viewing_dir]

    # construct dust curves
    wavelengths = photometry_table.wavelengths
    n_wave = len(wavelengths)
    calzetti = atten_calzetti(wavelengths=wavelengths)
    conroy = atten_conroy(wavelengths=wavelengths)
    power_law = atten_power_law(wavelengths=wavelengths, alpha=config.power_law_alpha)
    cardelli = extinct_cardelli(wavelengths=wavelengths)
    smc = extinct_smc(wavelengths=wavelengths)
    lmc = extinct_lmc(wavelengths=wavelengths)

    # pack dust curves into the expected 2D array form
    dust_curves = np.empty(shape=(len(DUST_CURVE_IDX), n_wave), dtype=np.float64)
    dust_curves[DUST_CURVE_IDX["calzetti"]] = calzetti
    dust_curves[DUST_CURVE_IDX["conroy"]] = conroy
    dust_curves[DUST_CURVE_IDX["cardelli"]] = cardelli
    dust_curves[DUST_CURVE_IDX["smc"]] = smc
    dust_curves[DUST_CURVE_IDX["lmc"]] = lmc
    dust_curves[DUST_CURVE_IDX["power_law"]] = power_law

    # construct delta_nu (c/lambda**2 * bin width)
    wavelengths_cm = wavelengths * 1e-8  # angstrom -> cm (CGS)
    delta_lambda = np.gradient(wavelengths_cm)
    delta_nu = (constants.C_CGS / wavelengths_cm**2) * delta_lambda

    # metal column density -> stellar A_v conversion factor
    Z_col_to_A_v = (
        constants.M_SUN_G
        * constants.X_H
        * (1 + sim.redshift) ** 2
        / (constants.KPC_CM**2 * constants.PROTON_MASS_G)
        * constants.MW_DUST_TO_METAL
        / (constants.AV_TO_NH * constants.Z_SUN_WATSON)
    )

    # precompute cosmic extinction (Madau 1995): ssp table is in rest frame so shift
    madau_transmission = extinct_madau(wavelengths=wavelengths * (1.0 + sim.redshift), redshift=sim.redshift)

    # prefactor for absolute magnitude flux (10pc)
    flux_factor_abs = constants.L_SUN_CGS / (4.0 * np.pi * (10.0 * constants.PC_CM) ** 2)
    # prefactor for apparent magnitude flux (luminosity distance)
    flux_factor_app = constants.L_SUN_CGS / (
        4.0 * np.pi * sim.cosmology.luminosity_distance(sim.redshift).to("cm").value ** 2
    )

    # UV bounds
    uv_start_idx = int(np.searchsorted(wavelengths, 1500.0))
    uv_end_idx = int(np.searchsorted(wavelengths, 3000.0))

    # galaxy data
    galaxies = simulation_data.groups["galaxies"]
    field_halo_idx = galaxies["field_halo_index"]
    ssfr = galaxies["sfr"] / galaxies["mass_star"]
    ssfr[ssfr <= 0.0] = 1e-30  # prevent NaN/inf
    gal_ssfr = np.log10(ssfr)
    Z_ratio = (
        galaxies["metallicity_sfr_weighted"] / constants.Z_SUN_ASPLUND
    )  # REVIEW: asplund metallicity is inherited convention (might want to move to config?)
    Z_ratio[Z_ratio <= 0.0] = 1e-30  # prevent NaN/inf
    gal_Z = np.log10(Z_ratio)
    star_offsets, star_idx = galaxies.get_particle_csr(ptype="star")

    # halo data
    haloes = simulation_data.groups["halos"]
    gas_offsets, gas_idx = haloes.get_particle_csr(ptype="gas")

    # filter data transmission coefficients [0, 1] + normalisations for apparent and absolute magnitudes
    n_bands = len(config.bands)
    transmission_weighted_abs = np.zeros((n_bands, n_wave), dtype=np.float64)
    transmission_norm_abs = np.empty(n_bands, dtype=np.float64)
    transmission_weighted_app = np.zeros((n_bands, n_wave), dtype=np.float64)
    transmission_norm_app = np.empty(n_bands, dtype=np.float64)

    for band_idx, band_name in enumerate(config.bands):
        curve = photometry_table.filters[band_name]

        # absolute magnitude: rest frame (what ssp table outputs)
        T_abs = np.interp(
            wavelengths, curve.wavelength, curve.transmission, left=0.0, right=0.0
        )  # set to 0 outside valid range
        transmission_weighted_abs[band_idx] = T_abs * delta_nu
        transmission_norm_abs[band_idx] = np.sum(T_abs * delta_nu)

        # apparent magnitude: observer frame, shift ssp output
        T_app = np.interp(
            wavelengths, curve.wavelength / (1.0 + sim.redshift), curve.transmission, left=0.0, right=0.0
        )  # set to 0 outside valid range
        transmission_weighted_app[band_idx] = T_app * delta_nu
        transmission_norm_app[band_idx] = np.sum(T_app * delta_nu)

    # build namedtuples
    stars = simulation_data.particles["star"]
    star_data = StarData(
        pos=stars["pos"],
        vel_los=stars["vel"][:, los_axis],
        mass=stars["mass"],
        age=stars["age"],
        metallicity=stars["metallicity"],
        offsets=star_offsets,
        idx_sorted=star_idx,
    )

    # ascertain whether to include dust
    gas = simulation_data.particles["gas"]
    effective_use_dust = config.dust and ("dust_mass" in gas.columns)  # both flags have to pass

    if effective_use_dust:
        dust_mass = gas["dust_mass"]
        dust_metallicity = np.ones(shape=len(gas), dtype=np.float64)
        logger.debug("Using dust from snapshot.")
    else:
        dust_mass = gas["mass"]
        dust_metallicity = gas["metallicity"]
        logger.debug("Computing dust from Li (2019) relation.")

    gas_data = GasData(
        pos=gas["pos"],
        smoothing_lengths=gas["smoothing_length"],
        dust_mass=dust_mass,
        metallicity=dust_metallicity,
        offsets=gas_offsets,
        idx_sorted=gas_idx,
    )

    ssp_data = SSPData(
        spectra=photometry_table.spectra,
        mass_remaining=photometry_table.mass_remaining,
        ages=photometry_table.ages,
        metallicities=photometry_table.metallicities,
    )

    phot_constants = PhotometryConstants(
        split_age=config.split_age,
        los_axis=los_axis,
        c_kms=constants.C_KMS,
        boxsize=sim.boxsize,
        redshift=sim.redshift,
        mw_dust_to_metal=constants.MW_DUST_TO_METAL,
        Z_sun=constants.Z_SUN_ASPLUND,  # 0.0134
        Z_col_to_A_v=Z_col_to_A_v,
        use_cosmic_ext=config.cosmic_extinction,
        use_dust=config.dust,
    )

    dust_data = DustData(
        dust_curves=dust_curves,
        dust_law_idx=DUST_CURVE_IDX[config.extinction_law],
        gal_ssfr=gal_ssfr,
        gal_Z=gal_Z,
    )

    filter_data = FilterData(
        transmission_weighted_abs=transmission_weighted_abs,
        transmission_norm_abs=transmission_norm_abs,
        transmission_weighted_app=transmission_weighted_app,
        transmission_norm_app=transmission_norm_app,
        flux_factor_abs=flux_factor_abs,
        flux_factor_app=flux_factor_app,
    )

    logger.info("Computing photometric properties for galaxies.")
    mag_abs, mag_abs_nodust, mag_app, mag_app_nodust, luminosity_fir, beta, beta_nodust = (
        compute_photometric_properties(
            star_data=star_data,
            gas_data=gas_data,
            ssp_data=ssp_data,
            filter_data=filter_data,
            dust_data=dust_data,
            phot_constants=phot_constants,
            field_halo_idx=field_halo_idx,
            n_galaxies=galaxies.n_groups,
            delta_nu=delta_nu,
            wavelengths=wavelengths,
            uv_start_idx=uv_start_idx,
            uv_end_idx=uv_end_idx,
            madau_transmission=madau_transmission,
            kernel_table=kernel_table,
        )
    )

    # photometry is a bit of a maverick in how it runs so doesn't obey the group_store.write_batch() with results dict convention
    for band_idx, band_name in enumerate(config.bands):
        galaxies[f"mag_abs_{band_name}"] = mag_abs[:, band_idx]
        galaxies[f"mag_abs_nodust_{band_name}"] = mag_abs_nodust[:, band_idx]
        galaxies[f"mag_app_{band_name}"] = mag_app[:, band_idx]
        galaxies[f"mag_app_nodust_{band_name}"] = mag_app_nodust[:, band_idx]

    galaxies["luminosity_fir"] = luminosity_fir
    galaxies["beta"] = beta
    galaxies["beta_nodust"] = beta_nodust

    logger.info(f"Successfully computed photometric properties for {galaxies.n_groups} galaxies.")
