"""

Particle type-specific aggregate properties. For example:

- Hydrogen mass fractions (gas)
- Star formation histories (stars)
- Eddington fractions (black holes)

"""

# semantic

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_management import SimulationData, OctavianConfig

# others
import numpy as np

from octavian.aggregate_properties.aggregate_helpers import (
    sum_per_group,
    max_idx_per_group,
    guarded_divide,
)
from octavian.log import get_logger

logger = get_logger()


def run_ptype_specific_properties(simulation_data: SimulationData, config: OctavianConfig) -> None:
    """
    Top-level executor for the ptype-specific aggregate properties.
    """
    particles = simulation_data.particles
    constants = simulation_data.constants

    logger.info("Hydrogen fractions prepared.")

    for group_type in simulation_data.groups:
        logger.info(
            f"Running ptype-specific properties for {group_type}: {simulation_data.groups[group_type].n_groups} members"
        )

        group_store = simulation_data.groups[group_type]
        n_groups = group_store.n_groups

        # gas
        if "gas" in particles:
            gas = particles["gas"]
            gas_offsets, gas_idx = group_store.get_particle_csr(ptype="gas")

            nH, fHI, fH2 = _prepare_hydrogen_fractions(
                rho=gas["rho"],
                fHI=gas["fHI"],
                fH2=gas["fH2"],
                gas_Y=gas["helium_fraction"],
                gas_Z=gas["metallicity"],
                proton_mass=constants.PROTON_MASS_G,
            )
            mass_HI = gas["mass"] * fHI
            mass_H2 = gas["mass"] * fH2
            gas["mass_HI"] = mass_HI  # these need to go on ParticleStore for local environment properties
            gas["mass_H2"] = mass_H2

            gas_results = compute_gas_properties(
                masses=gas["mass"],
                masses_HI=mass_HI,
                masses_H2=mass_H2,
                metallicities=gas["metallicity"],
                temperatures=gas["temperature"],
                sfrs=gas["sfr"],
                gas_mass=group_store["mass_gas"],
                offsets=gas_offsets,
                idx_sorted=gas_idx,
                n_groups=n_groups,
                Tlim=config.T_lim,
            )
            group_store.write_batch(results=gas_results)

            if group_store.kind == "halo":
                cgm_results = compute_cgm_properties(
                    masses=gas["mass"],
                    metallicities=gas["metallicity"],
                    temperatures=gas["temperature"],
                    nH=nH,
                    offsets=gas_offsets,
                    idx_sorted=gas_idx,
                    n_groups=n_groups,
                    nHlim=config.nH_lim,
                )
                group_store.write_batch(results=cgm_results)

        # stars
        if "star" in particles:
            star = particles["star"]
            star_offsets, star_idx = group_store.get_particle_csr(ptype="star")
            star_results = compute_star_properties(
                masses=star["mass"],
                metallicities=star["metallicity"],
                ages=star["age"],
                star_mass=group_store["mass_star"],
                offsets=star_offsets,
                idx_sorted=star_idx,
                n_groups=n_groups,
            )
            group_store.write_batch(results=star_results)

        # black holes
        if "bh" in particles:
            bh = particles["bh"]
            bh_offsets, bh_idx = group_store.get_particle_csr(ptype="bh")
            bh_results = compute_bh_properties(
                masses=bh["bhmass"],
                bhmdots=bh["bhmdot"],
                offsets=bh_offsets,
                idx_sorted=bh_idx,
                n_groups=n_groups,
                edd_factor=constants.EDD_FACTOR,
            )
            group_store.write_batch(results=bh_results)

        logger.info(f"Computed ptype-specific properties for {group_type}.")


def _prepare_hydrogen_fractions(
    rho: np.ndarray, fHI: np.ndarray, fH2: np.ndarray, gas_Z: np.ndarray, gas_Y: np.ndarray, proton_mass: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enforces hydrogen fraction conservation and computes and hydrogen abundance, returning a tuple of:

    - nH: hydrogen abundance
    - fHI: fraction of ionised hydrogen
    - fH2: fraction of molecular hydrogen

    Necessary for (cgm) gas properties.
    """
    not_conserving = (fHI + fH2) > 1.0  # enforce mass conservation: fHI + fH2 <= 1
    logger.debug(f"{not_conserving.sum()} particles not conserving hydrogen mass.")
    fHI = fHI.copy()
    fHI[not_conserving] = 1.0 - fH2[not_conserving]  # fix relative to fH2 (this is an inherited convention)

    nH = rho * (1.0 - gas_Z - gas_Y) / proton_mass

    return nH, fHI, fH2


def compute_gas_properties(
    masses: np.ndarray,
    masses_HI: np.ndarray,
    masses_H2: np.ndarray,
    metallicities: np.ndarray,
    temperatures: np.ndarray,
    sfrs: np.ndarray,
    gas_mass: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    Tlim: float,
) -> dict[str, np.ndarray]:
    """
    Computes gas-specific properties, returning a dict of:

    - mass_HI, mass_H2
    - sfr
    - metallicity_{mass/sfr}_weighted
    - temp_mass_weighted
    - mass_hot: gas mass above Tlim
    - mass_cold: gas mass below Tlim

    And writes HI/H2 masses back to ParticleStore for local environment properties.
    """
    results: dict[str, np.ndarray] = {}

    mass_HI = sum_per_group(values=masses_HI, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)
    mass_H2 = sum_per_group(values=masses_H2, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)
    sfr = sum_per_group(values=sfrs, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)
    metal_mass = sum_per_group(
        values=(metallicities * masses), offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups
    )
    metal_sfr = sum_per_group(values=(metallicities * sfrs), offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)
    temp_mass = sum_per_group(values=(temperatures * masses), offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)

    cold_masses = np.where(temperatures < Tlim, masses, 0.0)
    mass_cold = sum_per_group(values=cold_masses, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)
    hot_masses = np.where(temperatures >= Tlim, masses, 0.0)
    mass_hot = sum_per_group(values=hot_masses, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)

    results["mass_HI"] = mass_HI
    results["mass_H2"] = mass_H2
    results["mass_cold"] = mass_cold
    results["mass_hot"] = mass_hot
    results["sfr"] = sfr
    results["metallicity_mass_weighted"] = guarded_divide(numerator=metal_mass, denominator=gas_mass)
    results["metallicity_sfr_weighted"] = guarded_divide(numerator=metal_sfr, denominator=sfr)
    results["temp_mass_weighted"] = guarded_divide(numerator=temp_mass, denominator=gas_mass)

    return results


def compute_cgm_properties(
    masses: np.ndarray,
    temperatures: np.ndarray,
    metallicities: np.ndarray,
    nH: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    nHlim: float,
) -> dict[str, np.ndarray]:
    """
    Computes gas CGM-specific quantities (which are only well-defined for halos), returning a dict of:

    - mass_cgm
    - temp_{mass/metal}_weighted_cgm
    - metallicity_{mass/temp}_weighted_cgm
    """
    results: dict[str, np.ndarray] = {}

    cgm_criterion = nH < nHlim
    cgm_masses = np.where(cgm_criterion, masses, 0.0)  # cannot mask directly on inclusive csr: must use np.where
    cgm_temperatures = np.where(cgm_criterion, temperatures, 0.0)
    cgm_metallicities = np.where(cgm_criterion, metallicities, 0.0)

    cgm_mass = sum_per_group(values=cgm_masses, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)
    cgm_temp_mass = sum_per_group(
        values=(cgm_temperatures * cgm_masses), offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups
    )
    cgm_temp_metal = sum_per_group(
        values=(cgm_temperatures * cgm_masses * cgm_metallicities),
        offsets=offsets,
        idx_sorted=idx_sorted,
        n_groups=n_groups,
    )
    cgm_metal_mass = sum_per_group(
        values=(cgm_masses * cgm_metallicities), offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups
    )

    results["mass_cgm"] = cgm_mass
    results["temp_mass_weighted_cgm"] = guarded_divide(numerator=cgm_temp_mass, denominator=cgm_mass)
    results["temp_metal_weighted_cgm"] = guarded_divide(numerator=cgm_temp_metal, denominator=cgm_metal_mass)
    results["metallicity_mass_weighted_cgm"] = guarded_divide(numerator=cgm_metal_mass, denominator=cgm_mass)
    results["metallicity_temp_weighted_cgm"] = guarded_divide(numerator=cgm_temp_metal, denominator=cgm_temp_mass)

    return results


def compute_star_properties(
    masses: np.ndarray,
    metallicities: np.ndarray,
    ages: np.ndarray,
    star_mass: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
) -> dict[str, np.ndarray]:
    """
    Computes star-specific properties, returning a dict of:

    - metallicity_stellar
    - age_{mass/metal}_weighted
    """
    results: dict[str, np.ndarray] = {}

    metal_mass = sum_per_group(
        values=(masses * metallicities), offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups
    )
    age_mass = sum_per_group(values=(ages * masses), offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)
    age_metal = sum_per_group(
        values=(ages * masses * metallicities), offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups
    )

    results["metallicity_stellar"] = guarded_divide(numerator=metal_mass, denominator=star_mass)
    results["age_mass_weighted"] = guarded_divide(numerator=age_mass, denominator=star_mass)
    results["age_metal_weighted"] = guarded_divide(numerator=age_metal, denominator=metal_mass)

    return results


def compute_bh_properties(
    masses: np.ndarray,
    bhmdots: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    edd_factor: float,
) -> dict[str, np.ndarray]:
    """
    Computes (supermassive) black-hole specific properties, returning a dict of:

    - smbh_mdot: mass accretion rate of SMBH
    - smbh_fedd: Eddington fraction of SMBH
    - smbh_mass: mass of SMBH
    """
    results: dict[str, np.ndarray] = {}

    smbh_idx = max_idx_per_group(
        values=masses, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups
    )  # also assigns -1 as sentinel
    with_bh = smbh_idx >= 0

    smbh_mass = np.full(shape=n_groups, fill_value=np.nan)
    smbh_mdot = np.full(shape=n_groups, fill_value=np.nan)

    smbh_mass[with_bh] = masses[smbh_idx[with_bh]]
    smbh_mdot[with_bh] = bhmdots[smbh_idx[with_bh]]
    smbh_fedd = guarded_divide(numerator=smbh_mdot, denominator=(edd_factor * smbh_mass))

    results["smbh_mdot"] = smbh_mdot
    results["smbh_fedd"] = smbh_fedd
    results["smbh_mass"] = smbh_mass

    return results
