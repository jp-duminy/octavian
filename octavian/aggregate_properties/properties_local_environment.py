"""

Properties related to a structure's local environment (number densities, aperture masses, etc.)

"""

# semantic
from __future__ import annotations
from typing import TYPE_CHECKING

# others
import numpy as np
from scipy.spatial import KDTree  # remember, always pass boxsize
from scipy.sparse import csr_array

# other Octavian data structures
if TYPE_CHECKING:
    from octavian.data_management import ParticleStore, GroupStore, SimulationData

from octavian.data_management import get_logger
from octavian.data_management.conventions import DTYPES
from octavian.aggregate_properties.aggregate_helpers import (
    sort_by_group,
)

logger = get_logger()


def run_local_environment(simulation_data: SimulationData, config: dict) -> None:
    """
    Top-level executor for local environment properties.
    """
    galaxies = simulation_data.groups["galaxies"]
    logger.info(f"Running local environment properties: {galaxies.n_groups} members")
    sim = simulation_data.simulation

    density_results = compute_local_densities(
        pos=galaxies["com_pos_baryon"],
        mass=galaxies["mass_baryon"],
        n_groups=galaxies.n_groups,
        boxsize=sim.boxsize,
        radii=config["density_radii"],
    )
    galaxies.write_batch(results=density_results)

    for aperture in config["aperture_size"]:
        aperture_results = compute_galaxy_aperture_masses(
            particles=simulation_data.particles,
            galaxies=galaxies,
            halos=simulation_data.groups["halos"],
            boxsize=sim.boxsize,
            aperture_size=aperture,
        )
        galaxies.write_batch(results=aperture_results)

    logger.info("Local environment properties computed.")


def compute_local_densities(
    pos: np.ndarray, mass: np.ndarray, n_groups: int, boxsize: float, radii: list[float]
) -> dict[str, np.ndarray]:
    """
    Computes local mass and number densities for groups, returning a dict of:

    - local_mass_density_{r} for r in {radii}
    - local_number_density_{r}
    """
    results: dict[str, np.ndarray] = {}
    r_max = np.max(radii)
    tree = KDTree(data=pos, boxsize=boxsize)
    sdm = tree.sparse_distance_matrix(
        other=tree, max_distance=r_max, output_type="coo_array"
    )  # if your scipy is pre-1.18, use "coo_matrix"

    for radius in radii:
        in_range = (
            sdm.data <= radius
        )  # NOTE: the matrix does include self-distance 0 as we passed other=tree ^, I verified this manually with test data
        valid_array = np.ones(in_range.sum())
        adj = csr_array(
            arg1=(valid_array, (sdm.row[in_range], sdm.col[in_range])), shape=(n_groups, n_groups)
        )  # unhelpful argument name

        local_mass = adj @ mass  # adjacency matrix algebra gets the quantities in a vectorised way
        local_number_count = adj @ np.ones(shape=n_groups)
        volume = 4.0 / 3.0 * np.pi * radius**3

        results[f"local_mass_density_{radius:.0f}kpc"] = local_mass / volume
        results[f"local_number_density_{radius:.0f}kpc"] = local_number_count / volume

    return results


def compute_galaxy_aperture_masses(
    particles: dict[str, ParticleStore], galaxies: GroupStore, halos: GroupStore, boxsize: float, aperture_size: float
) -> dict[str, np.ndarray]:
    """
    Computes mass in an aperture of defined size around galaxies, returning a dict of:

    - mass_{p}_{aperture_size}kpc for p in HI, H2, {ptypes}, total

    I know this code looks really messy but I don't know how it can be tidied up.
    """
    results: dict[str, np.ndarray] = {}
    pos_list, mass_list = [], []
    ptype_list, hids_list = [], []
    ptypes = ["star", "gas", "bh", "dm"]
    ptype_to_int = {p: i for i, p in enumerate(ptypes)}  # integer comparison is more straightforward

    # originally this got kind of hacked in by inserting these as ptypes, do it explicitly instead
    particle_counts = [len(particles[ptype]) for ptype in ptypes]
    gas_offset = particle_counts[ptypes.index("gas")]  # cumsum up to gas
    gas_start = sum(particle_counts[: ptypes.index("gas")])
    gas_end = gas_start + gas_offset

    total_particles = sum(particle_counts)
    all_mass_HI = np.zeros(total_particles, dtype=DTYPES["mass"])
    all_mass_H2 = np.zeros(total_particles, dtype=DTYPES["mass"])
    all_mass_HI[gas_start:gas_end] = particles["gas"]["mass_HI"]
    all_mass_H2[gas_start:gas_end] = particles["gas"]["mass_H2"]

    for ptype in ptypes:
        data = particles[ptype]
        pos_list.append(data["pos"])
        mass_list.append(data["mass"])
        ptype_list.append(np.full(shape=len(data), fill_value=ptype_to_int[ptype], dtype=DTYPES["ptype"]))
        hids_list.append(data["HaloID"])

    all_pos, all_mass = np.concatenate(pos_list, dtype=DTYPES["pos"]), np.concatenate(mass_list, dtype=DTYPES["mass"])
    all_ptypes, all_hids = (
        np.concatenate(ptype_list, dtype=DTYPES["ptype"]),
        np.concatenate(hids_list, dtype=DTYPES["HaloID"]),
    )

    order, unique_halos, halo_start, halo_end = sort_by_group(group_ids=all_hids)  # sort particles by halo
    all_pos, all_mass = all_pos[order], all_mass[order]
    all_ptypes = all_ptypes[order]
    all_mass_HI, all_mass_H2 = all_mass_HI[order], all_mass_H2[order]

    # pre-sort galaxies by parent halo
    parent_halo_ids = halos.group_ids[galaxies["parent_halo_index"]]  #
    gal_order, halos_with_galaxies, gal_start, gal_end = sort_by_group(
        group_ids=parent_halo_ids
    )  # sort galaxies by halo
    gal_pos = galaxies["com_pos_baryon"]

    result = np.zeros(shape=(galaxies.n_groups, len(ptypes)))
    result_HI, result_H2 = np.zeros(shape=galaxies.n_groups), np.zeros(shape=galaxies.n_groups)

    for h in range(len(unique_halos)):
        halo_id = unique_halos[h]
        halo_pos, halo_mass = all_pos[halo_start[h] : halo_end[h]], all_mass[halo_start[h] : halo_end[h]]
        halo_mass_HI, halo_mass_H2 = all_mass_HI[halo_start[h] : halo_end[h]], all_mass_H2[halo_start[h] : halo_end[h]]
        halo_ptypes = all_ptypes[halo_start[h] : halo_end[h]]
        galaxy_halo_idx = np.searchsorted(
            a=halos_with_galaxies, v=halo_id
        )  # find where the hid sits in halos_with_galaxies

        if galaxy_halo_idx >= len(halos_with_galaxies) or halos_with_galaxies[galaxy_halo_idx] != halo_id:
            continue  # skip halos with no galaxies

        gal_indices = gal_order[gal_start[galaxy_halo_idx] : gal_end[galaxy_halo_idx]]

        tree = KDTree(data=halo_pos, boxsize=boxsize)
        neighbor_lists = tree.query_ball_point(x=gal_pos[gal_indices], r=aperture_size)

        for galaxies_idx_local, neighbours in enumerate(neighbor_lists):
            if len(neighbours) == 0:
                continue

            neighbours = np.array(neighbours)
            result_HI[gal_indices[galaxies_idx_local]] = halo_mass_HI[neighbours].sum()
            result_H2[gal_indices[galaxies_idx_local]] = halo_mass_H2[neighbours].sum()
            masses_by_type = np.bincount(halo_ptypes[neighbours], weights=halo_mass[neighbours], minlength=len(ptypes))
            result[gal_indices[galaxies_idx_local], :] = masses_by_type

    for i, name in enumerate(ptypes):
        results[f"mass_{name}_{int(aperture_size)}kpc"] = result[:, i]

    results[f"mass_HI_{int(aperture_size)}kpc"] = result_HI
    results[f"mass_H2_{int(aperture_size)}kpc"] = result_H2
    results[f"mass_total_{int(aperture_size)}kpc"] = result[:, : len(ptypes)].sum(axis=1)

    return results
