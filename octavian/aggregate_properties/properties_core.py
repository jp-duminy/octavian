"""

Core aggregate properties. These include simple computations (per-ptype number counts, masses, angular momenta, etc.); quantities derived from the basics (baryon/total quantities); then group-specific derived quantities (e.g. virial quantities for halos or morphology indicators for galaxies).

"""

# semantic
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_management import ParticleStore, GroupStore, SimulationAttributes, SimulationData
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # suppresses expected warnings for NaN (empty) groups.

# others
import numpy as np
from octavian.data_management.conventions import CONSTANTS, DTYPES

from octavian.aggregate_properties.aggregate_computations import (
    compute_kinematics,
    compute_rotational_quantities,
    compute_enclosed_mass_radii,
    compute_virial_quantities,
    compute_centre_of_mass,
    compute_vmax_and_rmax,
    compute_radii,
)

from octavian.aggregate_properties.aggregate_helpers import (
    sum_per_group,
    count_per_group,
    min_idx_per_group,
    first_idx_per_group,
    build_group_csr
)

def run_core_properties(simulation_data: SimulationData, config: dict) -> None:
    """
    Top-level executor for the core aggregate properties.
    """
    for group_type in simulation_data.groups: # halos must run first (halo groupstore is built first)

        group_store = simulation_data.groups[group_type]
        particles = simulation_data.particles
        sim = simulation_data.simulation

        available_ptypes = list(particles.keys())
        available_baryonic = [pt for pt, s in particles.items() if s.is_baryonic]

        if group_type == "halos":
        
            global_minimum = _prepare_global_minimum_potential(particles=particles, group_store=group_store, 
                                                               ptypes=available_ptypes, group_key=group_store.group_key)
            group_store.write_batch(results=global_minimum)

        run_core_ptype_pass(particles=particles, store=group_store, group_type=group_type, sim=sim, 
                            config=config)
        run_combine(store=group_store, group_type=group_type, available_ptypes=available_ptypes, 
                    available_baryonic_ptypes=available_baryonic, boxsize=sim.boxsize)
        
        run_combined_radial_quantiles(particles=particles, store=group_store, group_type=group_type,
                                      available_ptypes=available_ptypes, available_baryonic_ptypes=available_baryonic, sim=sim,
                                      config=config)

        if group_type == "halos":
        
            run_halo_stages(particles=particles, store=group_store, sim=sim, config=config)

        elif group_type == "galaxies":
        
            run_galaxy_stages(particles=particles, galaxies=group_store, halos=simulation_data.groups["halos"], available_baryonic_ptypes=available_baryonic, sim=sim)

def run_core_ptype_pass(
    particles: dict[str, ParticleStore],
    store: GroupStore,
    group_type: str,
    sim: SimulationAttributes,
    config: dict,
) -> None:
    """
    Runs the core common quantities (counts & mass, com, kinematics, radials); writes to GroupStore.
    """
    for ptype in particles:

        data = particles[ptype]
        n_groups = store.n_groups
        group_ids = data[store.group_key]  
        group_idx = store.get_indexer(group_id=group_ids)

        valid = group_idx >= 0
        group_idx = group_idx[valid]
        masses = data["mass"][valid]
        positions = data["pos"][valid]
        velocities = data["vel"][valid]

        counts_and_mass = _compute_counts_and_mass(masses, group_idx=group_idx, n_groups=n_groups, ptype=ptype)
        store.write_batch(results=counts_and_mass)

        centre_of_mass = _compute_centre_of_mass(positions=positions, velocities=velocities, masses=masses,
                                                 group_idx=group_idx, group_mass=store[f"mass_{ptype}"], n_groups=n_groups,
                                                 boxsize=sim.boxsize)
        store.write_batch(results=centre_of_mass, suffix=ptype)

        if group_type == "halos":
            ref_pos = store.get_columns(["minpot_x", "minpot_y", "minpot_z"])
            ref_vel = store.get_columns(["minpot_vx", "minpot_vy", "minpot_vz"])
        else:
            ref_pos = store.get_columns([f"x_{ptype}", f"y_{ptype}", f"z_{ptype}"])
            ref_vel = store.get_columns([f"vx_{ptype}", f"vy_{ptype}", f"vz_{ptype}"])

        com_vel = store.get_columns([f"vx_{ptype}", f"vy_{ptype}", f"vz_{ptype}"])

        kinematics, radii = _compute_kinematics(positions=positions, velocities=velocities, masses=masses,
                                         group_idx=group_idx, ref_pos=ref_pos, ref_vel=ref_vel,
                                         com_vel=com_vel, n_groups=n_groups, n_particles=len(group_idx),
                                         boxsize=sim.boxsize)
        store.write_batch(results=kinematics, suffix=ptype)

        derived = _derive_kinematics(
            L=kinematics["_L_vector"],
            dispersion_sum=kinematics["_dispersion_sum"],
            counts=counts_and_mass[f"n{ptype}"],
        )
        store.write_batch(results=derived, suffix=ptype)

        quantile_names = list(config["radial_quantiles"])
        quantiles = np.array(list(config["radial_quantiles"].values()), dtype=np.float64)

        radial = _compute_radial_quantities(
            radii=radii, masses=data["mass"],
            group_idx=group_idx, n_groups=n_groups,
            quantiles=quantiles,
            quantile_names=quantile_names,
        )
        store.write_batch(results=radial, suffix=ptype)

def run_combine(
    store: GroupStore,
    group_type: str,
    available_ptypes: list[str],
    available_baryonic_ptypes: list[str],
    boxsize: float,
) -> None:
    """
    Runs functions to combine per-ptype results into baryon and total aggregates (counts, mass, CoM, L, ke_tot, dispersion_sum).
    """
    combined_baryon = _combine_ptype_sums(
        group_store=store, collective_name="baryon",
        constituent_ptypes=available_baryonic_ptypes, boxsize=boxsize,
    )
    store.write_batch(results=combined_baryon)

    if group_type == "halos":   

        combined_total = _combine_ptype_sums(
            group_store=store, collective_name="total",
            constituent_ptypes=available_ptypes, boxsize=boxsize,
        )
        store.write_batch(results=combined_total)

def run_halo_stages(
    particles: dict[str, ParticleStore],
    store: GroupStore,
    sim: SimulationAttributes,
    config: dict,
) -> None:
    """
    Halo-specific virial/mass profile quantities.
    """
    group_key = store.group_key
    n_groups = store.n_groups
    ref_pos = store.get_columns(["minpot_x", "minpot_y", "minpot_z"])

    all_radii_list, all_masses_list, all_group_idx_list = [], [], [] # ghastly concatenation

    for ptype in particles:

        data = particles[ptype]
        group_idx = store.get_indexer(group_id=data[group_key])
        valid = group_idx >= 0

        radii = compute_radii(
            positions=data["pos"][valid], ref_pos=ref_pos,
            group_idx=group_idx[valid], n_particles=valid.sum(),
            boxsize=sim.boxsize,
        )
        all_radii_list.append(radii)
        all_masses_list.append(data["mass"][valid])
        all_group_idx_list.append(group_idx[valid])

    all_radii = np.concatenate(all_radii_list)
    all_masses = np.concatenate(all_masses_list)
    all_group_idx = np.concatenate(all_group_idx_list)

    factors = np.array(config["virial_factors"])
    mass_profile = _compute_mass_profile_quantities(
        radii=all_radii, masses=all_masses,
        group_idx=all_group_idx, n_groups=n_groups,
        factors=factors,
        rhocrit_comoving=sim.rhocrit_comoving,
    )
    store.write_batch(results=mass_profile)

    derived = _derive_halo_quantities(
        group_mass=store["mass_total"],
        L_mag=store["_L_mag_total"],
        counts=store["_ntotal"],
        r200_factor=sim.r200_factor,
        scale_factor=sim.a,
    )
    store.write_batch(results=derived)

def run_galaxy_stages(
    particles: dict[str, ParticleStore],
    galaxies: GroupStore,
    halos: GroupStore,
    available_baryonic_ptypes: list[str],
    sim: SimulationAttributes,
) -> None:
    """
    Galaxy-specific morphological quantities (requires another pass for alignment of particle L axes).
    """
    group_key = galaxies.group_key
    n_groups = galaxies.n_groups
    combined_L = galaxies.get_columns(["Lx_baryon", "Ly_baryon", "Lz_baryon"])
    ref_pos = galaxies.get_columns(["x_baryon", "y_baryon", "z_baryon"])
    ref_vel = galaxies.get_columns(["vx_baryon", "vy_baryon", "vz_baryon"])

    combined_ke_rot = np.zeros(n_groups)
    combined_counter_rotating_mass = np.zeros(n_groups)

    for ptype in available_baryonic_ptypes:

        data = particles[ptype]
        group_idx = galaxies.get_indexer(group_id=data[group_key])
        valid = group_idx >= 0

        counter_rot, ke_rot = compute_rotational_quantities(
            positions=data["pos"][valid], velocities=data["vel"][valid],
            masses=data["mass"][valid], group_idx=group_idx[valid],
            ref_pos=ref_pos, ref_vel=ref_vel,
            L_group=combined_L, n_groups=n_groups,
            n_particles=valid.sum(), boxsize=sim.boxsize)
        
        combined_ke_rot += ke_rot
        combined_counter_rotating_mass += counter_rot

        per_ptype_derived = _derive_galaxy_quantities(
            ke_tot=galaxies[f"_ke_tot_{ptype}"], ke_rot=ke_rot,
            counter_rotating_mass=counter_rot, group_mass=galaxies[f"mass_{ptype}"],
            counts=galaxies[f"n{ptype}"])

        galaxies.write_batch(results=per_ptype_derived, suffix=ptype)

    derived = _derive_galaxy_quantities(
        ke_tot=galaxies["_ke_tot_baryon"],
        ke_rot=combined_ke_rot,
        counter_rotating_mass=combined_counter_rotating_mass,
        group_mass=galaxies["mass_baryon"],
        counts=galaxies["_nbaryon"],
    )
    galaxies.write_batch(results=derived, suffix="baryon")

    parent = _assign_parent_halo_indices(
        particles=particles, galaxies=galaxies, halos=halos, available_baryonic_ptypes=available_baryonic_ptypes
    )
    galaxies.write_batch(results=parent)

def _combined_quantiles(
    particles: dict[str, ParticleStore],
    store: GroupStore,
    ptypes: list[str],
    ref_pos: np.ndarray,
    sim: SimulationAttributes,
    quantiles: np.ndarray,
    quantile_names: list[str],
) -> None:
    """
    Concatenates, then computes combined quantiles, returning a dict of the same quantities computed by +compute_radial_quantities.
    """
    radii_list, masses_list, group_idx_list = [], [], []

    for ptype in ptypes:
        data = particles[ptype]
        group_idx = store.get_indexer(group_id=data[store.group_key])
        valid = group_idx >= 0

        radii = compute_radii(
            positions=data["pos"][valid], ref_pos=ref_pos,
            group_idx=group_idx[valid], n_particles=valid.sum(),
            boxsize=sim.boxsize,
        )
        radii_list.append(radii)
        masses_list.append(data["mass"][valid])
        group_idx_list.append(group_idx[valid])

    radial = _compute_radial_quantities(
        radii=np.concatenate(radii_list), masses=np.concatenate(masses_list),
        group_idx=np.concatenate(group_idx_list), n_groups=store.n_groups,
        quantiles=quantiles, quantile_names=quantile_names,
    )

    return radial

def run_combined_radial_quantiles(
    particles: dict[str, ParticleStore],
    store: GroupStore,
    group_type: str,
    available_ptypes: list[str],
    available_baryonic_ptypes: list[str],
    sim: SimulationAttributes,
    config: dict,
) -> None:
    """
    Computes radial quantiles for combined ptype sets (baryon, and total for halos).
    """
    quantile_names = list(config["radial_quantiles"])
    quantiles = np.array(list(config["radial_quantiles"].values()), dtype=np.float64)

    baryon_ref = store.get_columns(["x_baryon", "y_baryon", "z_baryon"])
    baryon_quantiles = _combined_quantiles(
        particles=particles, store=store, ptypes=available_baryonic_ptypes,
        ref_pos=baryon_ref, sim=sim, quantiles=quantiles, quantile_names=quantile_names)
    
    store.write_batch(results=baryon_quantiles, suffix="baryon")

    if group_type == "halos":
        total_ref = store.get_columns(["minpot_x", "minpot_y", "minpot_z"])
        total_quantiles = _combined_quantiles(
            particles=particles, store=store, ptypes=available_ptypes,
            ref_pos=total_ref, sim=sim, quantiles=quantiles, 
            quantile_names=quantile_names)
        
        store.write_batch(results=total_quantiles, suffix="total")

def _prepare_global_minimum_potential(
    particles: dict[str, ParticleStore], 
    group_store: GroupStore,
    ptypes: list[str], 
    group_key: str
) -> dict[str, np.ndarray]:
    """
    Finds the global minimum potential across multiple particle types.

    Neccesary for baryon/total aggregate properties. Returns a dict of:

    - minpot_{d} for d in x, y, z
    - minpot_v{d} 
    """
    results: dict[str, np.ndarray] = {}
    n_groups = group_store.n_groups
    best_pot = np.full(shape=n_groups, fill_value=np.inf)
    best_pos = np.full(shape=(n_groups,3), fill_value=np.nan)
    best_vel = np.full(shape=(n_groups,3), fill_value=np.nan)

    for ptype in ptypes:

        data = particles[ptype]
        group_ids = data[group_key]
        group_idx = group_store.get_indexer(group_id=group_ids)

        in_group = group_idx >= 0
        idx, potentials = group_idx[in_group], data["potential"][in_group]
        positions, velocities = data["pos"][in_group], data["vel"][in_group]

        min_idx = min_idx_per_group(values=potentials, group_idx=idx, n_groups=n_groups)
        has_min = min_idx >= 0 
        ptype_min_pot = np.full(n_groups, np.inf)
        ptype_min_pot[has_min] = potentials[min_idx[has_min]]

        better = ptype_min_pot < best_pot
        best_pot[better] = ptype_min_pot[better]
        best_pos[better] = positions[min_idx[better]] # you get the idea
        best_vel[better] = velocities[min_idx[better]]

    for i, d in enumerate(["x", "y", "z"]):
        results[f"minpot_{d}"] = best_pos[:, i]
        results[f"minpot_v{d}"] = best_vel[:, i]

    return results

def _compute_counts_and_mass(
    masses: np.ndarray, 
    group_idx: np.ndarray, 
    n_groups: int, 
    ptype: str
) -> dict[str, np.ndarray]:
    """
    Computes number counts and total masses for the input ptype, returning a dict of:

    - n{ptype}
    - mass_{ptype}
    """
    results: dict[str, np.ndarray] = {}

    results[f"n{ptype}"] = count_per_group(group_idx=group_idx, n_groups=n_groups)
    results[f"mass_{ptype}"] = sum_per_group(values=masses, group_idx=group_idx, n_groups=n_groups)

    return results

def _compute_centre_of_mass(
    positions: np.ndarray, 
    velocities: np.ndarray, 
    masses: np.ndarray, 
    group_idx: np.ndarray,
    group_mass: np.ndarray, 
    n_groups: int, 
    boxsize: float
) -> dict[str, np.ndarray]:
    """
    Computes centre-of-mass positions and velocities (with PBC handling), returning a dict of:

    - {d} for d in x, y, z
    - v{d}
    """
    results: dict[str, np.ndarray] = {}

    anchor_idx = first_idx_per_group(group_idx=group_idx, n_groups=n_groups) # HACK: anchor can be any particle, so use first member of each group
    anchor_positions = np.full((n_groups, 3), np.nan)
    valid_anchors = anchor_idx >= 0
    anchor_positions[valid_anchors] = positions[anchor_idx[valid_anchors]]

    com_positions, com_velocities = compute_centre_of_mass(
        positions=positions, velocities=velocities, masses=masses, group_idx=group_idx, 
        anchor_pos=anchor_positions, group_mass=group_mass, n_groups=n_groups, boxsize=boxsize)

    for i, d in enumerate(["x", "y", "z"]):
        results[f"{d}"] = com_positions[:, i]
    for i, d in enumerate(["x", "y", "z"]):
        results[f"v{d}"] = com_velocities[:, i]

    return results

def _compute_kinematics(
    positions: np.ndarray, 
    velocities: np.ndarray, 
    masses: np.ndarray, 
    group_idx: np.ndarray,
    ref_pos: np.ndarray, 
    ref_vel: np.ndarray, 
    com_vel: np.ndarray, 
    n_groups: int,
    n_particles: int, 
    boxsize: float
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Computes kinematic quantities, where ref_pos/vel is wrt the group centre (com for galaxies, minpot for halos), returning a tuple with a dict of:

    - L{d} for d in x, y, z
    - _dispersion_sum
    - _ke_tot
    - _ke_rot
    - _counter_rotating_mass

    And an (n_particles) array of radii relative to the ref_pos.
    """
    results: dict[str, np.ndarray] = {}

    L, ke_tot, dispersion_sum, radii = compute_kinematics(positions=positions, velocities=velocities,
                                                            masses=masses, group_idx=group_idx,
                                                            ref_pos=ref_pos, ref_vel=ref_vel,
                                                            com_vel=com_vel, n_groups=n_groups,
                                                            n_particles=n_particles, boxsize=boxsize)

    counter_rotating_mass, ke_rot = compute_rotational_quantities(positions=positions, velocities=velocities,
                                                                masses=masses, group_idx=group_idx,
                                                                ref_pos=ref_pos, ref_vel=ref_vel,
                                                                L_group=L, n_groups=n_groups,
                                                                n_particles = n_particles, boxsize=boxsize)
    
    for i, d in enumerate(["x", "y", "z"]):
        results[f"L{d}"] = L[:,i]
    
    results["_L_vector"] = L
    results["_ke_tot"], results["_dispersion_sum"]= ke_tot, dispersion_sum
    results["_counter_rotating_mass"], results["_ke_rot"] = counter_rotating_mass, ke_rot

    return results, radii

def _compute_radial_quantities(
    radii: np.ndarray, 
    masses: np.ndarray, 
    group_idx: np.ndarray, 
    n_groups: int,
    quantiles: np.ndarray, 
    quantile_names: list[str]
) -> dict[str, np.ndarray]:
    """
    Computes radial quantities (r20, half_mass, etc.) defined from config, returning a dict of:

    - r20
    - half mass
    - r80

    Or others specified in config.yaml
    """
    results: dict[str, np.ndarray] = {}

    offsets, sorted_idx = build_group_csr(group_idx=group_idx, n_groups=n_groups)

    radial_results = compute_enclosed_mass_radii(radii=radii, masses=masses, offsets=offsets, idx_sorted=sorted_idx,
                                                 n_groups=n_groups, quantiles=quantiles)

    for q, col_name in enumerate(quantile_names):
        results[f"radius_{col_name}"] = radial_results[:, q]

    return results

def _compute_mass_profile_quantities(
    radii: np.ndarray, 
    masses: np.ndarray, 
    group_idx: np.ndarray, 
    n_groups: int,
    factors: np.ndarray, 
    rhocrit_comoving: float
) -> dict[str, np.ndarray]:
    """
    Computes mass profile quantities (virial and vmax/rmax), (which as of 19/06/26 are only required for halos). Returns a dict of:

    - radius_{f}c for f in {factors}
    - mass_{f}c
    - vmax
    - rmax
    """
    results: dict[str, np.ndarray] = {}

    offsets, sorted_idx = build_group_csr(group_idx=group_idx, n_groups=n_groups)

    virial_radius, virial_mass = compute_virial_quantities(radii=radii, masses=masses, offsets=offsets, idx_sorted=sorted_idx,
                                                             n_groups=n_groups, rhocrit=rhocrit_comoving, factors=factors)

    for f, factor in enumerate(factors.astype(int)): # cast array to ints for f-string name
        results[f"radius_{factor}c"] = virial_radius[:, f]
        results[f"mass_{factor}c"] = virial_mass[:, f]

    vmax, rmax = compute_vmax_and_rmax(radii=radii, masses=masses, offsets=offsets, idx_sorted=sorted_idx,
                                       G=CONSTANTS.G_VCIRC, n_groups=n_groups)
    
    results["vmax"], results["rmax"] = vmax, rmax

    return results

def _derive_kinematics(
    L: np.ndarray, 
    dispersion_sum: np.ndarray, 
    counts: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Compute derived kinematic quantities (simple arithmetic on the outputs of _compute_kinematics), returning a dict of:

    - velocity_dispersion
    - alpha, beta: rotation angles to rotate the galaxy to align with angular momentum
    """
    results: dict[str, np.ndarray] = {}

    L_mag = np.linalg.norm(L, axis=1)
    alpha = np.arctan2(L[:, 1], L[:, 2])
    beta = np.arcsin(L[:, 0] / L_mag)
    velocity_dispersions = np.where(counts > 0, np.sqrt(dispersion_sum / np.maximum(counts, 1)), np.nan)

    small = (counts > 0) & (counts < 3) # groups with fewer than 3 counts have ill-defined rotational quantities (mask away)
    empty = counts == 0

    for quantity in [velocity_dispersions, L_mag, alpha, beta]:
        quantity[empty] = np.nan
        quantity[small] = 0.0
    for i in range(3):
        L[empty, i] = np.nan
        L[small, i] = 0.0

    results["_L_mag"] = L_mag
    results["ALPHA"], results["BETA"] = alpha, beta
    results["velocity_dispersion"] = velocity_dispersions

    return results

def _derive_halo_quantities(
    group_mass: np.ndarray, 
    L_mag: np.ndarray, 
    counts: np.ndarray, 
    r200_factor: float,
    scale_factor: float
) -> dict[str, np.ndarray]:
    """
    Derived halo quantities, returning a dict of:

    - r200: radius at which matter overdensity is 200x the mean matter density
    - circular_velocity
    - virial_temperature
    - spin_param: (Bullock) spin parameter
    """
    results: dict[str, np.ndarray] = {}
    r200 = r200_factor * group_mass**(1./3.) # NOTE: comoving
    v_circ = np.sqrt(CONSTANTS.G_VCIRC * group_mass / (r200 * scale_factor)) # v_circ needs physical r200

    virial_temperature = CONSTANTS.VIRIAL_TEMP_FACTOR * v_circ** 2
    spin_param = L_mag / (np.sqrt(2) * group_mass * v_circ * r200)

    empty = counts == 0
    for arr in [r200, v_circ, virial_temperature, spin_param]:
        arr[empty] = np.nan

    results[f"r200"] = r200
    results[f"circular_velocity"] = v_circ
    results[f"virial_temperature"] = virial_temperature 
    results[f"spin_param"] = spin_param

    return results

def _derive_galaxy_quantities(    
    ke_tot: np.ndarray, 
    ke_rot: np.ndarray,
    counter_rotating_mass: np.ndarray, 
    group_mass: np.ndarray, 
    counts: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Derived galaxy quantities, returning a dict of:

    - BoverT: bulge-to-total kinematic ratio
    - kappa_rot: fraction of kinetic energy in rotation (Sales et al. (2012))
    """
    results: dict[str, np.ndarray] = {}

    BoverT = (2 * counter_rotating_mass) / group_mass
    kappa_rot = ke_rot / ke_tot

    small = (counts > 0) & (counts < 3) # groups with fewer than 3 counts have ill-defined rotational quantities (mask away)
    empty = counts == 0

    for quantity in [BoverT, kappa_rot]:
        quantity[empty] = np.nan
        quantity[small] = 0.0

    results["BoverT"], results["kappa_rot"] = BoverT, kappa_rot

    return results

def _combine_counts_and_mass(group_store: GroupStore, collective_name: str, constituent_ptypes: list[str]) -> dict[str, np.ndarray]:
    """
    Combines counts and centre-of-mass results from constituent_ptypes (additive), returning a dict of:

    - _n{collective_name}
    - mass_{collective_name}
    """
    results: dict[str, np.ndarray] = {}

    counts = sum(group_store[f"n{pt}"] for pt in constituent_ptypes)
    mass = sum(group_store[f"mass_{pt}"] for pt in constituent_ptypes)

    results[f"_n{collective_name}"] = counts
    results[f"mass_{collective_name}"] = mass

    return results

def _combine_centre_of_mass(
    group_store: GroupStore, 
    combined_mass: np.ndarray, 
    collective_name: str, 
    constituent_ptypes: list[str], 
    boxsize: float
) -> dict[str, np.ndarray]:
    """
    Combines centre of masses from constituent_ptypes. PBC-aware, returns a dict of

    - {d}_{collective_name} for d in x, y, z
    - v{d}_{collective_name}
    """
    results: dict[str, np.ndarray] = {}
    n_groups = group_store.n_groups
    anchor_ptype = constituent_ptypes[0] # since the anchor can be anywhere, use first ptype
    anchor_com = group_store.get_columns([f"x_{anchor_ptype}", f"y_{anchor_ptype}", f"z_{anchor_ptype}"])
    
    weighted_shift = np.zeros(shape=(n_groups,3))
    weighted_vel = np.zeros(shape=(n_groups, 3))

    for pt in constituent_ptypes:

        pt_com = group_store.get_columns([f"x_{pt}", f"y_{pt}", f"z_{pt}"])
        pt_vel = group_store.get_columns([f"vx_{pt}", f"vy_{pt}", f"vz_{pt}"])
        pt_mass = group_store[f"mass_{pt}"][:, np.newaxis]  # upgrade dimensionality so this works with (n, 3) arrays
        
        shift = pt_com - anchor_com
        shift -= boxsize * np.round(shift / boxsize)
        weighted_shift += pt_mass * shift
        weighted_vel += pt_mass * pt_vel

    combined_com = anchor_com + weighted_shift / combined_mass[:, np.newaxis]
    combined_com %= boxsize
    combined_vel = weighted_vel / combined_mass[:, np.newaxis]

    for i, d in enumerate(["x", "y", "z"]):
        results[f"{d}_{collective_name}"] = combined_com[:,i]
        results[f"v{d}_{collective_name}"] = combined_vel[:,i]

    return results

def _combine_ptype_sums(
    group_store: GroupStore, 
    collective_name: str, 
    constituent_ptypes: list[str], 
    boxsize: float
) -> dict[str, np.ndarray]:
    """
    Combines results from individual particle type arrays (wraps above functions), returning a dict of:

    - quantities in combine_counts_and_mass
    - quantities in combine_centre_of_mass
    - L{d}_{collective_name} for d in x, y, z
    """
    results: dict[str, np.ndarray] = {}
    n_groups = group_store.n_groups

    counts_and_mass = _combine_counts_and_mass(group_store=group_store, collective_name=collective_name,
                                               constituent_ptypes=constituent_ptypes)
    centre_of_mass = _combine_centre_of_mass(group_store=group_store, combined_mass=counts_and_mass[f"mass_{collective_name}"],
                                             collective_name=collective_name, constituent_ptypes=constituent_ptypes,
                                             boxsize=boxsize)
    
    combined_L = np.zeros(shape=(n_groups, 3))
    combined_ke = np.zeros(shape=n_groups)
    combined_dispersion_sum = np.zeros(shape=n_groups)

    for pt in constituent_ptypes:
        ptype_L = group_store.get_columns([f"Lx_{pt}", f"Ly_{pt}", f"Lz_{pt}"])
        np.nan_to_num(ptype_L, copy=False, nan=0.0)
        combined_L += ptype_L

        combined_ke += np.nan_to_num(group_store[f"_ke_tot_{pt}"], nan=0.0)
        combined_dispersion_sum += np.nan_to_num(group_store[f"_dispersion_sum_{pt}"], nan=0.0)

    for i, d in enumerate(["x", "y", "z"]):
        results[f"L{d}_{collective_name}"] = combined_L[:,i]

    combined_counts = counts_and_mass[f"_n{collective_name}"]

    combined_L_mag = np.linalg.norm(combined_L, axis=1)
    combined_velocity_dispersion = np.where(combined_counts > 0, np.sqrt(combined_dispersion_sum / np.maximum(combined_counts, 1)), np.nan)
    combined_alpha = np.arctan2(combined_L[:, 1], combined_L[:, 2])
    combined_beta = np.arcsin(combined_L[:, 0] / combined_L_mag)

    small = (combined_counts > 0) & (combined_counts < 3)
    empty = combined_counts == 0

    for quantity in [combined_L_mag, combined_alpha, combined_beta]:
        quantity[empty] = np.nan
        quantity[small] = 0.0

    for i in range(3):
        combined_L[empty, i] = np.nan
        combined_L[small, i] = 0.0

    results[f"ALPHA_{collective_name}"] = combined_alpha
    results[f"BETA_{collective_name}"] = combined_beta
    results[f"_L_mag_{collective_name}"] = combined_L_mag
    results[f"_ke_tot_{collective_name}"] = combined_ke
    results[f"_dispersion_sum_{collective_name}"] = combined_dispersion_sum
    results[f"velocity_dispersion_{collective_name}"] = combined_velocity_dispersion

    return results | counts_and_mass | centre_of_mass

def _assign_parent_halo_indices(particles: dict[str, ParticleStore], galaxies: GroupStore, halos: GroupStore, 
                                available_baryonic_ptypes: list[str]) -> dict[str, np.ndarray]:
    """
    Assigns galaxies their parent halo indices (slightly hacky, assigns based on membership of first).

    This may move elsewhere depending on how we do subhalos with FOF6D; returns a dictionary of:

    - parent_halo_index
    """    
    results: dict[ str, np.ndarray] = {}

    gids_list, hids_list = [], []

    for ptype in available_baryonic_ptypes:

        store = particles[ptype]
        gids_list.append(store["GalID"])
        hids_list.append(store["HaloID"])

    all_gids, all_hids = np.concatenate(gids_list, dtype=DTYPES["GalID"]), np.concatenate(hids_list, dtype=DTYPES["HaloID"])
    in_galaxy = all_gids >= 0
    all_gids, all_hids = all_gids[in_galaxy], all_hids[in_galaxy]

    galaxy_idx = galaxies.get_indexer(group_id=all_gids)
    first_particle_idx = first_idx_per_group(group_idx=galaxy_idx, n_groups=galaxies.n_groups)
    valid = first_particle_idx >= 0 # NOTE: add note in logger if this triggers
    galaxy_halo_id = np.full(shape=galaxies.n_groups, fill_value=-1, dtype=DTYPES["HaloID"])
    galaxy_halo_id[valid] = all_hids[first_particle_idx[valid]]

    parent_halo_index = halos.get_indexer(group_id=galaxy_halo_id)
    results["parent_halo_index"] = parent_halo_index

    return results
