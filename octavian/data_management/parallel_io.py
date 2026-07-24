"""

I/O functions for reading in parallel from the same .hdf5 snapshot file.

"""

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from octavian.data_management.data_structures import ParticleStore
    from octavian.data_management.conventions import OctavianConfig
    from octavian.external_halo_sources import HaloAssignments, SubhaloInformation
    from .pipeline_management import Internals
    from mpi4py.MPI import Comm

# others
import numpy as np
from dataclasses import replace

# octavian
from octavian.log import get_logger
from .conventions import DTYPES
from .write_data import RankPackedData

logger = get_logger()
GatheredData: TypeAlias = tuple[
    list[RankPackedData], dict[tuple[str, str], np.ndarray]
]  # this is to mask the hideous nature of this type annotation


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


def assign_local_subhalos(
    particles: dict[str, ParticleStore],
    subhalo_info: SubhaloInformation | None,
) -> SubhaloInformation:
    """
    Returns SubhaloInformation masked to a rank's allocation.
    """
    if subhalo_info is None:
        return None

    max_hid = max(int(particles[ptype]["HaloID"].max()) for ptype in particles)
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


def gather_datasets(
    local_data: RankPackedData,
    internals: Internals,
    comm: Comm | None,
) -> GatheredData | None:
    """
    Gathers data from other ranks (docstring unfinished).
    """
    if comm is None:
        gathered_csr: dict[tuple[str, str], np.ndarray] = {}
        for group_params in internals.group_types.values():
            hdf5_name = group_params["hdf5_group"]
            local_group = local_data.groups.get(hdf5_name)
            if not local_group:
                continue
            for ptype, membership in local_group.particle_lists.items():
                gathered_csr[(hdf5_name, ptype)] = membership.indices
        return [local_data.without_indices()], gathered_csr  # match type check

    rank = comm.Get_rank()
    lightweight = local_data.without_indices()
    all_lightweight: list[RankPackedData] = comm.gather(
        lightweight, root=0
    )  # lightweight means everything except the indices array (which is n_particles big, everything else is n_groups big)

    for group_params in internals.group_types.values():
        hdf5_name = group_params["hdf5_group"]

        for ptype in group_params["ptypes"]:
            local_group = local_data.groups.get(hdf5_name)

            if local_group and ptype in local_group.particle_lists:
                send_indices = local_group.particle_lists[ptype].indices
            else:
                send_indices = np.empty(
                    0, dtype=DTYPES["csr_indices"]
                )  # if for whatever reason a list is missing, not doing this will wrong the offsets in parallel

            if rank == 0:
                gathered_membership: dict[tuple[str, str], np.ndarray] = {}  # strings are hdf5 name/ptype
                per_rank_counts = []

                for rank_lightweight in all_lightweight:
                    rank_group = rank_lightweight.groups.get(hdf5_name)
                    if rank_group and ptype in rank_group.particle_lists:
                        count = rank_group.particle_lists[ptype].offsets[-1]  # final offset = total number of particles
                    else:
                        count = 0
                    per_rank_counts.append(count)

                displacements = np.concatenate(([0], np.cumsum(per_rank_counts[:-1])))
                memory_block = np.empty(
                    sum(per_rank_counts), dtype=DTYPES["csr_indices"]
                )  # Gatherv needs a pre-allocated block of memory to write to
                gathered_membership[(hdf5_name, ptype)] = memory_block
                comm.Gatherv(
                    send_indices, [memory_block, per_rank_counts, displacements], root=0
                )  # fills gathered_membership by writing to memory_block

            else:
                comm.Gatherv(send_indices, None, root=0)

    if rank == 0:
        return all_lightweight, gathered_membership
    return None
