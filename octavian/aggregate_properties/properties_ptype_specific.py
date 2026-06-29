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
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # suppresses expected warnings for NaN (empty) groups.

# others
import numpy as np

from octavian.aggregate_properties.aggregate_internals import write_results
from octavian.data_management.conventions import CONSTANTS, DTYPES

from octavian.aggregate_properties.group_helpers import (
    sum_per_group,
    max_value_per_group,
    max_idx_per_group,
)

PTYPES = ["star", "gas", "bh", "dm"]
BARYONIC_PTYPES = ["star", "gas", "bh"]
GROUP_KEYS = {"halos": "HaloID", "galaxies": "GalID"}
GROUPS = ["halos", "galaxies"]

def run_ptype_specific_properties(simulation_data: SimulationData, config: dict) -> None:
    """
    Top-level executor for the ptype-specific aggregate properties.
    """
    particles = simulation_data.particles

    _prepare_hydrogen_fractions(gas=particles["gas"], XH=config["XH"])

    for group_type in ["halos", "galaxies"]:

        group_store = simulation_data.groups[group_type]
        group_key = GROUP_KEYS[group_type]
        n_groups = group_store.n_groups

        # gas
        gas = particles["gas"]
        gas_group_idx = gas[group_key]
        gas_results = compute_gas_properties(
            gas=gas, gas_mass=group_store["mass_gas"],
            group_idx=gas_group_idx, n_groups=n_groups
        )
        write_results(group_store=group_store, results=gas_results)

        # stars
        star = particles["star"]
        star_group_idx = star[group_key]
        star_results = compute_star_properties(
            star=star, star_mass=group_store["mass_star"],
            group_idx=star_group_idx, n_groups=n_groups
        )
        write_results(group_store=group_store, results=star_results)

        # black holes
        bh = particles["bh"]
        bh_group_idx = bh[group_key]
        bh_results = compute_bh_properties(
            bh=bh, group_idx=bh_group_idx,
            n_groups=n_groups, edd_factor=config["edd_factor"]
        )
        write_results(group_store=group_store, results=bh_results)

        if group_type == "halos":
            cgm_results = compute_cgm_properties(
                gas=gas, group_idx=gas_group_idx,
                n_groups=n_groups, nHlim=config["nHlim"]
            )
            write_results(group_store=group_store, results=cgm_results)

def _prepare_hydrogen_fractions(gas: ParticleStore, XH: float) -> None:
    """
    Derive HI/H2 masses from snapshot information, mutates the gas ParticleStore.

    Must be run before compute_gas_properties().
    """
    fHI = gas["nh"].copy()
    fH2 = gas["fH2"]
    gas["nH"] = gas["rho"] * XH / CONSTANTS.PROTON_MASS_G # neutral hydrogen abundance

    # enforce mass conservation: fHI + fH2 <= 1
    not_conserving = (fHI + fH2) > 1.0
    fHI[not_conserving] = 1.0 - fH2[not_conserving]

    mass = gas["mass"]
    gas["fHI"] = fHI
    gas["mass_HI"] = XH * fHI * mass
    gas["mass_H2"] = XH * fH2 * mass

def compute_gas_properties(
    gas: ParticleStore, 
    gas_mass: np.ndarray, 
    group_idx: np.ndarray, 
    n_groups: int, 
) -> dict[str, np.ndarray]:
    """
    Run _prepare_gas_fractions() first!

    Computes gas-specific properties, returning a dict of:

    - mass_HI, mass_H2
    - sfr
    - metallicity_{mass/sfr}_weighted
    - temp_mass_weighted
    """
    results: dict[str, np.ndarray] = {}
    valid = group_idx >= 0 # indexer assigns -1 to particles not in groups
    group_idx = group_idx[valid]

    temperatures = gas["temperature"][valid]
    metallicities = gas["metallicity"][valid]
    sfrs = gas["sfr"][valid]
    masses = gas["mass"][valid] # particle-level

    mass_HI = sum_per_group(values=gas["mass_HI"][valid], group_idx=group_idx, n_groups=n_groups)
    mass_H2 = sum_per_group(values=gas["mass_H2"][valid], group_idx=group_idx, n_groups=n_groups)
    sfr = sum_per_group(values=sfrs, group_idx=group_idx, n_groups=n_groups)
    metal_mass = sum_per_group(values=(metallicities * masses), group_idx=group_idx, n_groups=n_groups)
    metal_sfr = sum_per_group(values=(metallicities * sfrs), group_idx=group_idx, n_groups=n_groups)
    temp_mass = sum_per_group(values=(temperatures * masses), group_idx=group_idx, n_groups=n_groups)

    results["mass_HI"] = mass_HI
    results["mass_H2"] = mass_H2
    results["sfr"] = sfr
    results["metallicity_mass_weighted"] =  metal_mass / gas_mass
    results["metallicity_sfr_weighted"] =  metal_sfr / sfr
    results["temp_mass_weighted"] = temp_mass / gas_mass

    return results

def compute_cgm_properties(
    gas: ParticleStore, 
    group_idx: np.ndarray, 
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
    valid = group_idx >= 0 # indexer assigns -1 to particles not in groups
    group_idx = group_idx[valid]

    temperatures = gas["temperature"][valid]
    metallicities = gas["metallicity"][valid]
    masses = gas["mass"][valid] # particle-level

    nH = gas["nH"][valid]
    cgm_criterion = nH < nHlim
    cgm_idx = group_idx[cgm_criterion]
    cgm_masses = masses[cgm_criterion]
    cgm_temperatures = temperatures[cgm_criterion]
    cgm_metallicities = metallicities[cgm_criterion]

    cgm_mass = sum_per_group(values=cgm_masses, group_idx=cgm_idx, n_groups=n_groups)
    cgm_temp_mass = sum_per_group(values=(cgm_temperatures * cgm_masses), group_idx=cgm_idx, n_groups=n_groups)
    cgm_temp_metal = sum_per_group(values=(cgm_temperatures * cgm_masses * cgm_metallicities), group_idx=cgm_idx, n_groups=n_groups)
    cgm_metal_mass = sum_per_group(values=(cgm_masses * cgm_metallicities), group_idx=cgm_idx, n_groups=n_groups)

    results["mass_cgm"] = cgm_mass
    results["temp_mass_weighted_cgm"] = cgm_temp_mass / cgm_mass
    results["temp_metal_weighted_cgm"] = cgm_temp_metal / cgm_temp_mass
    results["metallicity_mass_weighted_cgm"] = cgm_metal_mass / cgm_mass
    results["metallicity_temp_weighted_cgm"] = cgm_temp_metal / cgm_metal_mass

    return results

def compute_star_properties(
    star: ParticleStore, 
    star_mass: np.ndarray, 
    group_idx: np.ndarray, 
    n_groups: int
) -> dict[str, np.ndarray]:
    """
    Computes star-specific properties, returning a dict of:

    - metallicity_stellar
    - age_{mass/metal}_weighted
    """
    results: dict[str, np.ndarray] = {}
    valid = group_idx >= 0 # indexer assigns -1 to particles not in groups
    group_idx = group_idx[valid]

    metallicities = star["metallicity"][valid]
    ages = star["age"][valid]
    masses = star["mass"][valid] # particle-level

    metal_mass = sum_per_group(values=(masses * metallicities), group_idx=group_idx, n_groups=n_groups)
    age_mass = sum_per_group(values=(ages * masses), group_idx=group_idx, n_groups=n_groups)
    age_metal = sum_per_group(values=(ages * masses * metallicities), group_idx=group_idx, n_groups=n_groups)

    results["metallicity_stellar"] = metal_mass / star_mass
    results["age_mass_weighted"] = age_mass / star_mass
    results["age_metal_weighted"] = age_metal / metal_mass

    return results

def compute_bh_properties(
    bh: ParticleStore, 
    group_idx: np.ndarray, 
    n_groups: int, 
    edd_factor: float
) -> dict[str, np.ndarray]:
    """
    Computes black-hole specific properties, returning a dict of:

    - bhmdot: mass accretion rate
    - bh_fedd: Eddington fraction
    - bh_mass_max: largest black hole mass in each group
    """
    results: dict[str, np.ndarray] = {}
    valid = group_idx >= 0 # indexer assigns -1 to particles not in groups
    group_idx = group_idx[valid]

    masses = bh["mass"][valid]
    bhmdots = bh["bhmdot"][valid]

    max_idx = max_idx_per_group(values=masses, group_idx=group_idx, n_groups=n_groups) # also assigns -1 as sentinel
    with_bh = max_idx >= 0

    mass = np.full(shape=n_groups, fill_value=np.nan) # split across line for when more properties are added
    bhmdot = np.full(shape=n_groups, fill_value=np.nan)

    max_mass = max_value_per_group(values=masses, group_idx=group_idx, n_groups=n_groups) 
    max_mass = np.where(np.isfinite(max_mass), max_mass, 0.0) # mask out -inf for no-bh groups

    mass[with_bh] = masses[max_idx[with_bh]]
    bhmdot[with_bh] = bhmdots[max_idx[with_bh]]

    results["bhmdot"] = bhmdot
    results["bh_fedd"] = bhmdot / (edd_factor * mass)
    results["bh_mass_max"] = max_mass

    return results