"""

I/O functions for reading in parallel from the same .hdf5 snapshot file.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_management.data_structures import SnapshotReader
    from octavian.data_management.conventions import OctavianConfig
    from octavian.external_halo_sources import HaloSource

# others
import numpy as np

# octavian
from octavian.log import get_logger

logger = get_logger()


def compute_rank_assignments(
    reader: SnapshotReader,
    halo_source: HaloSource,
    config: OctavianConfig,
    n_ranks: int,
) -> list[dict[str, np.ndarray]]:
    """
    Uses the weighted greedy binning algorithm to produce balanced rank particle assignments, returning:

    - A list of per-rank dictionaries where each dictionary is keyed by Octavian ptypes and contains index arrays for accessing the data belonging to the rank on the snapshot.
    """
    logger.info(f"Constructing per-rank indices for {n_ranks} ranks.")
    logger.debug(f"FOF6D weight: {config.fof6d_weight}, Aggregate Properties weight: {config.properties_weight}")

    available_ptypes = (
        reader.available_ptypes()
    )  # reader functions open the snapshot (so no need to wrap this code block)

    assignments = halo_source.read_halo_ids(
        ptypes=available_ptypes
    )  # named this assignments because it will eventually contain subhalos too

    ptype_counts = {}

    for ptype, halo_ids in assignments.halo_ids.items():
        valid = halo_ids[halo_ids != -1]  # masks valid HaloIDs here
        ptype_counts[ptype] = np.bincount(
            valid, minlength=assignments.n_total_halos
        )  # same logic as sum_per_group in aggregate_helpers.py

    haloes_exist = sum(ptype_counts.values()) > 0
    valid_halo_mask = (
        haloes_exist & (ptype_counts["dm"] >= config.min_dm_per_halo) if "dm" in ptype_counts else haloes_exist
    )  # masks min_dm_per_halo here
    all_valid_hids = np.flatnonzero(valid_halo_mask)

    if all_valid_hids.size == 0:  # prudent guard for a no-halo snapshot
        logger.warning("No valid HaloIDs!")
        return [
            {pt: np.array([], dtype=np.int64) for pt in available_ptypes} for _ in range(n_ranks)
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

    for ptype, halo_ids in assignments.halo_ids.items():
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
