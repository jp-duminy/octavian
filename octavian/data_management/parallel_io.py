"""

I/O functions for reading in parallel from the same .hdf5 snapshot file.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_management.data_structures import SnapshotReader
    from octavian.data_management.conventions import OctavianConfig

# defaults

# others
import numpy as np

# octavian
from octavian.log import get_logger

logger = get_logger()

HDF5_GROUP_NAMES = {
    "halos": "halo_data",
    "galaxies": "galaxy_data",
}


def compute_rank_assignments(
    reader: SnapshotReader, config: OctavianConfig, n_ranks: int
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

    ptype_counts = {}
    raw_halo_id_cache: dict[str, np.ndarray] = {}  # worth caching the IDs to avoid re-reading them in the second loop

    for pt in available_ptypes:
        halo_ids = reader.read_halo_ids(ptype=pt)  # autoconverts to raw snapshot convention
        raw_halo_id_cache[pt] = halo_ids
        halo_ids = halo_ids[halo_ids != -1]  # masks sentinel value here
        unique, counts = np.unique(halo_ids, return_counts=True)
        ptype_counts[pt] = (unique, counts)

    if "dm" in ptype_counts and len(ptype_counts["dm"][0]) > 0:
        dm_unique, dm_per_halo = ptype_counts["dm"]
        valid_mask = dm_per_halo >= config.min_dm_per_halo
        valid_halo_set = set(dm_unique[valid_mask])  # masks min_dm_per_halo here
    else:
        valid_halo_set = None

    all_hids_raw = np.unique(np.concatenate([unique for unique, _ in ptype_counts.values()]))
    all_hids = (
        all_hids_raw if valid_halo_set is None else all_hids_raw[np.isin(all_hids_raw, np.array(list(valid_halo_set)))]
    )

    if len(all_hids) == 0:  # prudent guard for a no-halo snapshot
        logger.warning("No valid HaloIDs!")
        return [
            {pt: np.array([], dtype=np.int64) for pt in available_ptypes} for _ in range(n_ranks)
        ]  # match type check

    n_halos = all_hids.max() + 1  # at this point the reader has remapped HaloIDs to 0-indexed

    star_counts = np.zeros(shape=n_halos)
    gas_counts = np.zeros(shape=n_halos)
    dm_counts = np.zeros(shape=n_halos)

    weight_ptypes = {"star": star_counts, "gas": gas_counts, "dm": dm_counts}

    for ptype, count_array in weight_ptypes.items():
        if ptype in ptype_counts:
            halo_ids, counts = ptype_counts[ptype]
            count_array[halo_ids] = counts

    fof6d_cost = star_counts[all_hids] ** 1.2 + gas_counts[all_hids]  # this power law is empirical
    aggregates_cost = star_counts[all_hids] + gas_counts[all_hids] + dm_counts[all_hids]
    halo_weights = config.fof6d_weight * fof6d_cost + config.properties_weight * aggregates_cost

    # greedy binning according to halo weight: sort halos by size descending then sequentially assign to rank with lightest load
    weight_order = np.argsort(halo_weights)[::-1]  # TODO: move to argsort(descending=True) in numpy 2.5.0
    rank_assignments = [set() for _ in range(n_ranks)]
    rank_loads = np.zeros(n_ranks)

    for idx in weight_order:  # the actual binning algorithm, which is naturally sequential (not performance-heavy)
        lightest = np.argmin(rank_loads)
        rank_assignments[lightest].add(all_hids[idx])
        rank_loads[lightest] += halo_weights[idx]

    result: list[dict[str, np.ndarray]] = [{} for _ in range(n_ranks)]

    for ptype in available_ptypes:
        halo_ids = raw_halo_id_cache[ptype]  # cached from the first loop (MVP burnt this practice into my brain)
        all_indices = np.arange(len(halo_ids), dtype=np.int64)

        in_halo = (halo_ids != -1) if valid_halo_set is None else np.isin(halo_ids, np.array(list(valid_halo_set)))
        filtered_ids = halo_ids[in_halo]  # ignore particles not assigned to a halo
        filtered_indices = all_indices[in_halo]

        for rank_idx in range(n_ranks):
            rank_mask = np.isin(filtered_ids, np.array(list(rank_assignments[rank_idx])))
            result[rank_idx][ptype] = filtered_indices[rank_mask]

    return result
