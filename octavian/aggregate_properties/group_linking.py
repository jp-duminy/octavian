"""

Functionality to assign galaxies their parent halos. This has two paths: if there is no subhalo information, it naïvely assigns galaxies a parent_halo_index corresponding to the HaloID of the first particle in the galaxy; the parent_membership_fraction is then trivially also 1.0. If subhalo information is present, it uses a plurality vote to determine parent_halo_index and registers parent_membership_fraction accordingly.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_management import ParticleStore, GroupStore, SimulationData
    from octavian.external_halo_sources import SubhaloInformation
from octavian.data_management import DTYPES, build_group_csr
from .aggregate_helpers import first_idx_per_group_2

import numpy as np
from numba import njit
from octavian.log import get_logger

logger = get_logger()


def assign_membership(
    simulation_data: SimulationData,
    subhalo_info: SubhaloInformation | None = None,
) -> None:
    """
    Assigns the galaxy groupstore the parent information depending on whether subhalo info exists.
    """
    results: dict[str, np.ndarray] = {}
    halos = simulation_data.groups["halos"]
    galaxies = simulation_data.groups["galaxies"]
    particles = simulation_data.particles
    available_baryonic = [pt for pt, s in particles.items() if s.is_baryonic]
    n_field_halos = int((halos["depth"] == 0).sum()) if "depth" in halos else halos.n_groups

    field_halo_index = assign_galaxy_field_indices(
        particles=particles,
        galaxies=galaxies,
        halos=halos,
        available_baryonic_ptypes=available_baryonic,
        n_field_halos=n_field_halos,
    )

    if subhalo_info is not None:
        raw_winners, parent_membership_frac = assign_galaxy_halo_indices(
            particles=particles,
            galaxies=galaxies,
            available_baryonic_ptypes=available_baryonic,
            n_subhalos=len(subhalo_info.depth),
        )
        parent_halo_index = np.where(raw_winners >= 0, raw_winners + n_field_halos, field_halo_index)
        parent_membership_frac[raw_winners < 0] = 1.0  # interlopers: unambiguous field membership
    else:
        parent_halo_index = field_halo_index.copy()
        parent_membership_frac = np.ones(galaxies.n_groups, dtype=np.float32)

    results["field_halo_index"] = field_halo_index
    results["parent_halo_index"] = parent_halo_index
    results["parent_membership_fraction"] = parent_membership_frac
    galaxies.write_batch(results=results)


def assign_galaxy_field_indices(
    particles: dict[str, ParticleStore],
    galaxies: GroupStore,
    halos: GroupStore,
    available_baryonic_ptypes: list[str],
    n_field_halos: int,
) -> np.ndarray:
    """
    Naïvely assigns galaxies their parent halo indices based on the HaloID of the first particle in the galaxy.
    """
    gids_list, hids_list = [], []

    for ptype in available_baryonic_ptypes:
        store = particles[ptype]
        gids_list.append(store["GalID"])
        hids_list.append(store["HaloID"])

    all_gids, all_hids = (
        np.concatenate(gids_list, dtype=DTYPES["GalID"]),
        np.concatenate(hids_list, dtype=DTYPES["HaloID"]),
    )
    in_galaxy = all_gids >= 0
    all_gids, all_hids = all_gids[in_galaxy], all_hids[in_galaxy]

    galaxy_idx = galaxies.get_indexer(group_id=all_gids)
    offsets, idx_sorted = build_group_csr(group_idx=galaxy_idx, n_groups=galaxies.n_groups)
    first_particle_idx = first_idx_per_group_2(offsets=offsets, idx_sorted=idx_sorted, n_groups=galaxies.n_groups)
    valid = first_particle_idx >= 0

    if not valid.all():
        n_orphan = (~valid).sum()
        logger.warning(f"{n_orphan} galaxies have no baryonic particles!")

    galaxy_halo_id = np.full(shape=galaxies.n_groups, fill_value=-1, dtype=DTYPES["HaloID"])
    galaxy_halo_id[valid] = all_hids[first_particle_idx[valid]]

    field_ids = halos.group_ids[:n_field_halos]
    positions = np.searchsorted(field_ids, galaxy_halo_id)
    positions = np.clip(positions, 0, len(field_ids) - 1)
    found = field_ids[positions] == galaxy_halo_id
    field_halo_index = np.where(found, positions, -1)

    if np.any(field_halo_index == -1):
        n_orphan = (field_halo_index == -1).sum()
        logger.warning(f"{n_orphan} galaxies exist with no valid parent halo.")

    return field_halo_index


def assign_galaxy_halo_indices(
    particles: dict[str, ParticleStore], galaxies: GroupStore, available_baryonic_ptypes: list[str], n_subhalos: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assigns parent halo index by plurality of subhalo membership.
    """
    all_subhids = []
    all_galaxy_idx = []

    for ptype in available_baryonic_ptypes:
        particle_idx, group_idx = galaxies.expand_csr_membership(ptype)
        all_subhids.append(particles[ptype]["SubhaloID"][particle_idx])
        all_galaxy_idx.append(group_idx)

    subhids, galaxy_idx = np.concatenate(all_subhids), np.concatenate(all_galaxy_idx)

    order = np.argsort(galaxy_idx, kind="stable")  # make each galaxy's members contiguous across ptypes
    subhids = subhids[order]  # sorted here (so the numba function can assume so)

    lengths = np.bincount(galaxy_idx, minlength=galaxies.n_groups)
    gal_offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(DTYPES["csr_offsets"])

    raw_winners, parent_membership_frac = _find_galaxy_parent(
        gal_offsets=gal_offsets, subhids=subhids, n_galaxies=galaxies.n_groups, n_subhalos=n_subhalos
    )

    return raw_winners, parent_membership_frac


@njit(cache=True)
def _find_galaxy_parent(
    gal_offsets: np.ndarray,
    subhids: np.ndarray,
    n_galaxies: int,
    n_subhalos: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each galaxy, determine its parent subhalo index and parent membership fraction from the subhalo info. Returns a tuple of (parent_halo_index, parent_membership_fraction).
    """
    counts = np.zeros(n_subhalos, dtype=np.int64)
    parent_halo_index = np.full(shape=n_galaxies, fill_value=-1, dtype=np.int64)
    parent_membership_frac = np.full(shape=n_galaxies, fill_value=0, dtype=np.float32)

    for g in range(n_galaxies):
        start = gal_offsets[g]
        end = gal_offsets[g + 1]

        best_id = -1  # 'best' in this case is the subhalo with the plurality vote
        best_count = 0
        total = 0

        for p in range(start, end):  # keep a running track of which subhalo ID is the winner
            subhid = subhids[p]

            if subhid < 0:  # SubhaloIDs contains sentinels too
                continue

            counts[subhid] += 1
            total += 1
            if counts[subhid] > best_count:
                best_count = counts[subhid]
                best_id = subhid

        if total > 0:  # id/total are already initialised to sentinel values (np.full above)
            parent_membership_frac[g] = best_count / total
            parent_halo_index[g] = best_id

        for p in range(
            start, end
        ):  # re-zero the subhids array before the next iteration of the loop (so you don't repeat np.zeros)
            if subhids[p] >= 0:
                counts[subhids[p]] = 0

    return parent_halo_index, parent_membership_frac
