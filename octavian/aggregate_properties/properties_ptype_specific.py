"""

Particle type-specific aggregate properties. For example:

- Hydrogen mass fractions (gas)
- Star formation histories (stars)
- Eddington fractions (black holes)

"""

# semantic
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_management import ParticleStore, SimulationData

# others
import numpy as np

from octavian.aggregate_properties.aggregate_helpers import (
    sum_per_group,
    max_value_per_group,
    max_idx_per_group,
    guarded_divide,
)
from octavian.data_management import get_logger

logger = get_logger()


def run_ptype_specific_properties(simulation_data: SimulationData, config: dict) -> None:
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
        group_key = group_store.group_key
        n_groups = group_store.n_groups

        # gas
        if "gas" in particles:
            gas = particles["gas"]
            gas_group_idx = group_store.get_indexer(group_id=gas[group_key])

            nH, fHI, fH2 = _prepare_hydrogen_fractions(
                rho=gas["rho"], fHI=gas["fHI"], fH2=gas["fH2"], XH=config["XH"], proton_mass=constants.PROTON_MASS_G
            )

            gas_results = compute_gas_properties(
                gas=gas, gas_mass=group_store["mass_gas"], fHI=fHI, fH2=fH2, group_idx=gas_group_idx, n_groups=n_groups
            )
            group_store.write_batch(results=gas_results)

            if group_type == "halos":
                cgm_results = compute_cgm_properties(
                    gas=gas, nH=nH, group_idx=gas_group_idx, n_groups=n_groups, nHlim=config["nHlim"]
                )
                group_store.write_batch(results=cgm_results)

        # stars
        if "star" in particles:
            star = particles["star"]
            star_group_idx = group_store.get_indexer(group_id=star[group_key])
            star_results = compute_star_properties(
                star=star, star_mass=group_store["mass_star"], group_idx=star_group_idx, n_groups=n_groups
            )
            group_store.write_batch(results=star_results)

        # black holes
        if "bh" in particles:
            bh = particles["bh"]
            bh_group_idx = group_store.get_indexer(group_id=bh[group_key])
            bh_results = compute_bh_properties(
                bh=bh, group_idx=bh_group_idx, n_groups=n_groups, edd_factor=constants.EDD_FACTOR
            )
            group_store.write_batch(results=bh_results)

        logger.info(f"Computed ptype-specific properties for {group_type}.")


def _prepare_hydrogen_fractions(
    rho: np.ndarray, fHI: np.ndarray, fH2: np.ndarray, XH: float, proton_mass: float
) -> tuple[np.ndarray, ...]:
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

    nH = rho * XH / proton_mass

    return nH, fHI, fH2


def compute_gas_properties(
    gas: ParticleStore,
    gas_mass: np.ndarray,
    fHI: np.ndarray,
    fH2: np.ndarray,
    group_idx: np.ndarray,
    n_groups: int,
) -> dict[str, np.ndarray]:
    """
    Computes gas-specific properties, returning a dict of:

    - mass_HI, mass_H2
    - sfr
    - metallicity_{mass/sfr}_weighted
    - temp_mass_weighted

    And writes HI/H2 masses back to ParticleStore for local environment properties.
    """
    results: dict[str, np.ndarray] = {}
    valid = group_idx >= 0  # indexer assigns -1 to particles not in groups
    group_idx = group_idx[valid]

    temperatures = gas["temperature"][valid]
    metallicities = gas["metallicity"][valid]
    sfrs = gas["sfr"][valid]
    masses = gas["mass"][valid]  # particle-level

    masses_HI = fHI * gas["mass"]  # also store these on ParticleStore for local environment
    masses_H2 = fH2 * gas["mass"]
    gas["mass_HI"] = masses_HI
    gas["mass_H2"] = masses_H2

    mass_HI = sum_per_group(values=masses_HI[valid], group_idx=group_idx, n_groups=n_groups)
    mass_H2 = sum_per_group(values=masses_H2[valid], group_idx=group_idx, n_groups=n_groups)
    sfr = sum_per_group(values=sfrs, group_idx=group_idx, n_groups=n_groups)
    metal_mass = sum_per_group(values=(metallicities * masses), group_idx=group_idx, n_groups=n_groups)
    metal_sfr = sum_per_group(values=(metallicities * sfrs), group_idx=group_idx, n_groups=n_groups)
    temp_mass = sum_per_group(values=(temperatures * masses), group_idx=group_idx, n_groups=n_groups)

    results["mass_HI"] = mass_HI
    results["mass_H2"] = mass_H2
    results["sfr"] = sfr
    results["metallicity_mass_weighted"] = guarded_divide(numerator=metal_mass, denominator=gas_mass)
    results["metallicity_sfr_weighted"] = guarded_divide(numerator=metal_sfr, denominator=sfr)
    results["temp_mass_weighted"] = guarded_divide(numerator=temp_mass, denominator=gas_mass)

    return results


def compute_cgm_properties(
    gas: ParticleStore,
    group_idx: np.ndarray,
    nH: np.ndarray,
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
    valid = group_idx >= 0  # indexer assigns -1 to particles not in groups
    group_idx = group_idx[valid]

    temperatures = gas["temperature"][valid]
    metallicities = gas["metallicity"][valid]
    masses = gas["mass"][valid]  # particle-level

    nH = nH[valid]
    cgm_criterion = nH < nHlim
    cgm_idx = group_idx[cgm_criterion]
    cgm_masses = masses[cgm_criterion]
    cgm_temperatures = temperatures[cgm_criterion]
    cgm_metallicities = metallicities[cgm_criterion]

    cgm_mass = sum_per_group(values=cgm_masses, group_idx=cgm_idx, n_groups=n_groups)
    cgm_temp_mass = sum_per_group(values=(cgm_temperatures * cgm_masses), group_idx=cgm_idx, n_groups=n_groups)
    cgm_temp_metal = sum_per_group(
        values=(cgm_temperatures * cgm_masses * cgm_metallicities), group_idx=cgm_idx, n_groups=n_groups
    )
    cgm_metal_mass = sum_per_group(values=(cgm_masses * cgm_metallicities), group_idx=cgm_idx, n_groups=n_groups)

    results["temp_mass_weighted_cgm"] = guarded_divide(numerator=cgm_temp_mass, denominator=cgm_mass)
    results["temp_metal_weighted_cgm"] = guarded_divide(numerator=cgm_temp_metal, denominator=cgm_metal_mass)
    results["metallicity_mass_weighted_cgm"] = guarded_divide(numerator=cgm_metal_mass, denominator=cgm_mass)
    results["metallicity_temp_weighted_cgm"] = guarded_divide(numerator=cgm_temp_metal, denominator=cgm_temp_mass)

    return results


def compute_star_properties(
    star: ParticleStore, star_mass: np.ndarray, group_idx: np.ndarray, n_groups: int
) -> dict[str, np.ndarray]:
    """
    Computes star-specific properties, returning a dict of:

    - metallicity_stellar
    - age_{mass/metal}_weighted
    """
    results: dict[str, np.ndarray] = {}
    valid = group_idx >= 0  # indexer assigns -1 to particles not in groups
    group_idx = group_idx[valid]

    metallicities = star["metallicity"][valid]
    ages = star["age"][valid]
    masses = star["mass"][valid]  # particle-level

    metal_mass = sum_per_group(values=(masses * metallicities), group_idx=group_idx, n_groups=n_groups)
    age_mass = sum_per_group(values=(ages * masses), group_idx=group_idx, n_groups=n_groups)
    age_metal = sum_per_group(values=(ages * masses * metallicities), group_idx=group_idx, n_groups=n_groups)

    results["metallicity_stellar"] = guarded_divide(numerator=metal_mass, denominator=star_mass)
    results["age_mass_weighted"] = guarded_divide(numerator=age_mass, denominator=star_mass)
    results["age_metal_weighted"] = guarded_divide(numerator=age_metal, denominator=metal_mass)

    return results


def compute_bh_properties(
    bh: ParticleStore, group_idx: np.ndarray, n_groups: int, edd_factor: float
) -> dict[str, np.ndarray]:
    """
    Computes black-hole specific properties, returning a dict of:

    - bhmdot: mass accretion rate
    - bh_fedd: Eddington fraction
    - bh_mass_max: largest black hole mass in each group
    """
    results: dict[str, np.ndarray] = {}
    valid = group_idx >= 0  # indexer assigns -1 to particles not in groups
    group_idx = group_idx[valid]

    masses = bh["mass"][valid]
    bhmdots = bh["bhmdot"][valid]

    max_idx = max_idx_per_group(values=masses, group_idx=group_idx, n_groups=n_groups)  # also assigns -1 as sentinel
    with_bh = max_idx >= 0

    mass = np.full(shape=n_groups, fill_value=np.nan)  # split across line for when more properties are added
    bhmdot = np.full(shape=n_groups, fill_value=np.nan)

    max_mass = max_value_per_group(values=masses, group_idx=group_idx, n_groups=n_groups)
    max_mass = np.where(np.isfinite(max_mass), max_mass, 0.0)  # mask out -inf for no-bh groups

    mass[with_bh] = masses[max_idx[with_bh]]
    bhmdot[with_bh] = bhmdots[max_idx[with_bh]]

    results["bhmdot"] = bhmdot
    results["bh_fedd"] = guarded_divide(numerator=bhmdot, denominator=(edd_factor * mass))
    results["bh_mass_max"] = max_mass

    return results
