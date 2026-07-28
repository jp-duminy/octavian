"""

I/O functions for reading in parallel from the same .hdf5 snapshot file.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_management.data_structures import ParticleStore
    from octavian.data_management.conventions import OctavianConfig
    from octavian.external_halo_sources import HaloAssignments, SubhaloInformation
    from mpi4py.MPI import Comm

# others
import numpy as np
from dataclasses import dataclass, replace
from mpi4py.util.dtlib import from_numpy_dtype

# octavian
from octavian.log import get_logger

logger = get_logger()


@dataclass(frozen=True, slots=True)
class RedistributionMap:
    """
    Contains global information for where the particles read by each rank need to go:

    - send_order: rank-sorted order for arrays to mask their particles from (np.argsort(owner))
    - send_counts: the number of particles each rank must send
    - rec_counts: the number of particles each rank must receive
    """

    send_order: np.ndarray
    send_counts: np.ndarray
    rec_counts: np.ndarray


def redistribute_particles(
    local_data: np.ndarray,
    redistribution_map: RedistributionMap,
    comm: Comm,
) -> np.ndarray:
    """
    Redistributes particles from a slab to their ranks.
    """
    ordered_data = local_data[redistribution_map.send_order]

    if comm.size == 1:
        return ordered_data

    # displacements for MPI to know whether to write memory to
    send_displacements = np.zeros(shape=comm.size, dtype=np.int64)
    send_displacements[1:] = np.cumsum(redistribution_map.send_counts[:-1])
    rec_displacements = np.zeros(comm.size, dtype=np.int64)
    rec_displacements[1:] = np.cumsum(redistribution_map.rec_counts[:-1])

    n_cols = ordered_data.size // len(ordered_data)  # for 3D columns (pos, vel)

    send_counts = redistribution_map.send_counts * n_cols
    rec_counts = redistribution_map.rec_counts * n_cols
    send_displacements = send_displacements * n_cols
    rec_displacements = rec_displacements * n_cols

    received_data = np.empty(
        rec_counts.sum(), dtype=ordered_data.dtype
    )  # MPI needs a preallocated memory block to write data to
    ordered_data = ordered_data.ravel()  # .ravel returns a view into memory where possible, .flatten always copies
    mpi_dtype = from_numpy_dtype(dtype=ordered_data.dtype)  # mpi4py requires its own dtype

    comm.Alltoallv(
        [ordered_data, send_counts, send_displacements, mpi_dtype],
        [received_data, rec_counts, rec_displacements, mpi_dtype],
    )

    if local_data.ndim > 1:
        return np.reshape(
            received_data, shape=(-1, local_data.shape[1])
        )  # -1 means np infers the number of rows from received_data
    return received_data


def build_redistribution_map(
    halo_to_rank: np.ndarray,
    slab_halo_ids: np.ndarray,
    comm: Comm,
) -> tuple[RedistributionMap, np.ndarray]:
    """
    Constructs the global RedistributionMap object for a slab of a dataset. Returns:

    - redistribution_map: a RedistributionMap object
    - mask: mask for the slab to filter sentinel/below min_dm_per_halo haloes
    """
    in_halo = (
        slab_halo_ids != -1
    )  # even though rank_assignments masks sentinel particles you still read the dataset (so ignore them again)
    mask = np.zeros(len(slab_halo_ids), dtype=bool)
    mask[in_halo] = (
        halo_to_rank[slab_halo_ids[in_halo]] != -1
    )  # then halos < min_dm_per_halo also have -1 in the halo_to_rank array
    owner_ranks = halo_to_rank[slab_halo_ids[mask]]

    send_order = np.argsort(owner_ranks, stable=True)
    send_counts = np.bincount(
        owner_ranks, minlength=comm.size
    )  # minlength arg is for the case where a rank owns no particles in the slab
    rec_counts = np.empty_like(send_counts)  # MPI needs a memory block to write to (for C memory allocation)

    comm.Alltoall(sendbuf=send_counts, recvbuf=rec_counts)  # Alltoall is high-performance (not alltoall)

    redistribution_map = RedistributionMap(
        send_order=send_order,
        send_counts=send_counts,
        rec_counts=rec_counts,
    )

    return redistribution_map, mask


def generate_rank_assignments_2(
    halo_assignments: HaloAssignments,
    config: OctavianConfig,
    n_ranks: int,
) -> np.ndarray:
    """
    Intermediate function (docstring incomplete).
    """
    logger.info(f"Constructing per-rank indices for {n_ranks} ranks.")
    logger.debug(f"FOF6D weight: {config.fof6d_weight}, Aggregate Properties weight: {config.properties_weight}")
    ptype_counts = {}

    for ptype, halo_ids in halo_assignments.halo_ids.items():
        valid = halo_ids[halo_ids != -1]  # masks valid HaloIDs here
        ptype_counts[ptype] = np.bincount(
            valid, minlength=halo_assignments.n_total_halos
        )  # same logic as sum_per_group in aggregate_helpers.py

    haloes_exist = sum(ptype_counts.values()) > 0
    valid_halo_mask = (
        haloes_exist & (ptype_counts["dm"] >= config.min_dm_per_halo) if "dm" in ptype_counts else haloes_exist
    )  # masks min_dm_per_halo here
    all_valid_hids = np.flatnonzero(valid_halo_mask)

    if all_valid_hids.size == 0:  # prudent guard for a no-halo snapshot
        logger.warning("No valid HaloIDs!")
        return np.full(shape=halo_assignments.n_total_halos, fill_value=-1, dtype=np.int64)  # match type check

    n_valid_halos = halo_assignments.n_total_halos  # at this point the reader has remapped HaloIDs to 0-indexed

    halo_to_rank = np.full(shape=n_valid_halos, fill_value=-1, dtype=np.int64)
    zeros = np.zeros(n_valid_halos, dtype=np.int64)
    star_counts = ptype_counts.get("star", zeros)
    gas_counts = ptype_counts.get("gas", zeros)
    dm_counts = ptype_counts.get("dm", zeros)

    fof6d_cost = star_counts[all_valid_hids] ** 1.2 + gas_counts[all_valid_hids]  # NOTE: this power law is empirical
    aggregates_cost = star_counts[all_valid_hids] + gas_counts[all_valid_hids] + dm_counts[all_valid_hids]
    halo_weights = config.fof6d_weight * fof6d_cost + config.properties_weight * aggregates_cost

    # greedy binning according to halo weight: sort halos by size descending then sequentially assign to rank with lightest load
    weight_order = np.argsort(halo_weights)[::-1]  # TODO: move to argsort(descending=True) in numpy 2.5.0
    rank_loads = np.zeros(n_ranks)

    for idx in weight_order:
        lightest = int(
            np.argmin(rank_loads)
        )  # the actual binning algorithm, which is naturally sequential (not performance-heavy)
        halo_to_rank[all_valid_hids[idx]] = lightest
        rank_loads[lightest] += halo_weights[idx]

    return halo_to_rank


def generate_rank_assignments(
    halo_assignments: HaloAssignments,
    config: OctavianConfig,
    n_ranks: int,
) -> list[dict[str, np.ndarray]]:
    """
    Uses the weighted greedy binning algorithm to produce balanced rank particle assignments, returning:

    - A list of per-rank dictionaries where each dictionary is keyed by Octavian ptypes and contains index arrays for accessing the data belonging to the rank on the snapshot.
    """
    logger.info(f"Constructing per-rank indices for {n_ranks} ranks.")
    logger.debug(f"FOF6D weight: {config.fof6d_weight}, Aggregate Properties weight: {config.properties_weight}")
    ptype_counts = {}

    for ptype, halo_ids in halo_assignments.halo_ids.items():
        valid = halo_ids[halo_ids != -1]  # masks valid HaloIDs here
        ptype_counts[ptype] = np.bincount(
            valid, minlength=halo_assignments.n_total_halos
        )  # same logic as sum_per_group in aggregate_helpers.py

    haloes_exist = sum(ptype_counts.values()) > 0
    valid_halo_mask = (
        haloes_exist & (ptype_counts["dm"] >= config.min_dm_per_halo) if "dm" in ptype_counts else haloes_exist
    )  # masks min_dm_per_halo here
    all_valid_hids = np.flatnonzero(valid_halo_mask)

    if all_valid_hids.size == 0:  # prudent guard for a no-halo snapshot
        logger.warning("No valid HaloIDs!")
        return [
            {pt: np.array([], dtype=np.int64) for pt in halo_assignments.halo_ids} for _ in range(n_ranks)
        ]  # match type check

    n_valid_halos = all_valid_hids.max() + 1  # at this point the reader has remapped HaloIDs to 0-indexed

    halo_to_rank = np.full(shape=n_valid_halos, fill_value=-1, dtype=np.int64)
    zeros = np.zeros(n_valid_halos, dtype=np.int64)
    star_counts = ptype_counts.get("star", zeros)
    gas_counts = ptype_counts.get("gas", zeros)
    dm_counts = ptype_counts.get("dm", zeros)

    fof6d_cost = star_counts[all_valid_hids] ** 1.2 + gas_counts[all_valid_hids]  # NOTE: this power law is empirical
    aggregates_cost = star_counts[all_valid_hids] + gas_counts[all_valid_hids] + dm_counts[all_valid_hids]
    halo_weights = config.fof6d_weight * fof6d_cost + config.properties_weight * aggregates_cost

    # greedy binning according to halo weight: sort halos by size descending then sequentially assign to rank with lightest load
    weight_order = np.argsort(halo_weights)[::-1]  # TODO: move to argsort(descending=True) in numpy 2.5.0
    rank_loads = np.zeros(n_ranks)

    for idx in weight_order:
        lightest = int(
            np.argmin(rank_loads)
        )  # the actual binning algorithm, which is naturally sequential (not performance-heavy)
        halo_to_rank[all_valid_hids[idx]] = lightest
        rank_loads[lightest] += halo_weights[idx]

    result: list[dict[str, np.ndarray]] = [{} for _ in range(n_ranks)]

    for ptype, halo_ids in halo_assignments.halo_ids.items():
        mask = np.zeros(shape=len(halo_ids), dtype=bool)
        in_halo = halo_ids != -1
        mask[in_halo] = valid_halo_mask[
            halo_ids[in_halo]
        ]  # have to match with halos > min_dm_per_halo mask and sentinel value
        filtered_indices = np.flatnonzero(mask)  # ignore particles not in halo/not matched to criteria

        particle_rank_assignments = halo_to_rank[halo_ids[filtered_indices]]
        particle_counts = np.bincount(
            particle_rank_assignments, minlength=n_ranks
        )  # NOTE: reimplements build_group_csr to avoid circular import
        offsets = np.empty(n_ranks + 1, dtype=np.int64)
        offsets[0] = 0
        offsets[1:] = np.cumsum(particle_counts)

        sorted_indices = np.argsort(particle_rank_assignments, kind="stable")

        for r in range(n_ranks):
            result[r][ptype] = filtered_indices[sorted_indices[offsets[r] : offsets[r + 1]]]

    return result


def generate_slabs(
    rank: int,
    n_ranks: int,
    particle_counts: dict[str, int],
) -> dict[str, slice]:
    """
    Returns a slice of size [this_rank:next_rank] particles to be read from the raw snapshot.
    """
    slabs: dict[str, slice] = {}

    for ptype, n_particles in particle_counts.items():
        remainder = n_particles % n_ranks  # need to account for the remainder in division
        base_allocation = n_particles // n_ranks

        particles_per_rank = base_allocation + 1 if rank < remainder else base_allocation

        start = rank * base_allocation + min(rank, remainder)
        end = start + particles_per_rank
        slabs[ptype] = slice(start, end)

    return slabs


def assign_local_subhalos(
    particles: dict[str, ParticleStore],
    subhalo_info: SubhaloInformation | None,
) -> SubhaloInformation:
    """
    Returns SubhaloInformation masked to a rank's allocation.
    """
    if subhalo_info is None:
        return None

    all_hids = [particles[ptype]["HaloID"] for ptype in particles]
    non_empty = [hids for hids in all_hids if len(hids) > 0]

    if not non_empty:
        return None

    max_hid = max(int(hids.max()) for hids in non_empty)
    present = np.zeros(
        shape=(max_hid + 1), dtype=bool
    )  # the reason we don't use np datatypes here is np.bool was deprecated (unsure of the deeper reasons why)
    for ptype in particles:
        hids = particles[ptype]["HaloID"]
        present[hids[hids != -1]] = True  # this works because hids are contiguous and 0-indexed

    keep = present[subhalo_info.host_halo_ids]

    global_to_local_map = np.full(len(subhalo_info.depth), -1, dtype=np.int64)
    global_to_local_map[np.flatnonzero(keep)] = np.arange(keep.sum())

    for ptype in particles:
        subhid = particles[ptype]["SubhaloID"]
        particles[ptype]["SubhaloID"] = np.where(
            (subhid == -1) | ~keep[subhid], -1, global_to_local_map[subhid]
        )  # set subhid to -1 for non-rank particles

    new_subhalo_info = replace(
        subhalo_info,  # NOTE: n_total_halos should not be replaced
        host_halo_ids=subhalo_info.host_halo_ids[keep],
        global_index=subhalo_info.global_index[keep],
        parent_index=np.where(
            subhalo_info.parent_index[keep] < 0, -1, global_to_local_map[subhalo_info.parent_index[keep]]
        ),
        depth=subhalo_info.depth[keep],
        n_bound=subhalo_info.n_bound[keep],
    )

    return new_subhalo_info


def assign_rank_halo_assignments(
    halo_assignments: HaloAssignments, all_indices: list[dict[str, np.ndarray]]
) -> list[HaloAssignments]:
    """
    Returns a list of per-rank HaloAssignments dataclasses, sliced to the rank's allocation from the binning algorithm which produces all_indices.
    """
    per_rank_assignments: list[HaloAssignments] = []

    for rank in all_indices:
        sliced_hids = {ptype: halo_assignments.halo_ids[ptype][rank[ptype]] for ptype in halo_assignments.halo_ids}

        sliced_subhids = (
            {ptype: halo_assignments.subhalo_ids[ptype][rank[ptype]] for ptype in halo_assignments.subhalo_ids}
            if halo_assignments.subhalo_ids
            else None
        )  # guard for snapshot reads

        per_rank_assignments.append(replace(halo_assignments, halo_ids=sliced_hids, subhalo_ids=sliced_subhids))

    return per_rank_assignments
