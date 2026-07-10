"""

Functions to filter & split a snapshot into subfiles for per-rank MPI access; and remerge analysis hdf5 files into an output catalogue.

NOTE: this will eventually be legacy code, as we intend to move away from intermediate files.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_management import GizmoReader, Internals, OctavianConfig

# defaults
from pathlib import Path

# others
import h5py
import numpy as np

# octavian
from octavian.data_management.conventions import DTYPES
from octavian.data_management.log import get_logger

logger = get_logger()

HDF5_GROUP_NAMES = {
    "halos": "halo_data",
    "galaxies": "galaxy_data",
}

# NOTE: will likely be unnecessary after move to HDF5 MPI.


def filter_snapshot(
    snapshot_file: Path,
    intermediate_directory: Path,
    reader: GizmoReader,
    config: OctavianConfig,
    n_split: int = 4,
) -> None:
    """
    Divides the snapshot into n_split (where n_split should be the number of MPI ranks) intermediate HDF5 files with all particles not in a halo filtered out. Employs a weighted binning algorithm to evenly distribute computational load amongst ranks.

    alpha: FOF6D weighting constant (default 0.6)
    beta: aggregate properties weighting constant (default 0.4)

    Constant defaults are empirically chosen but work well, only change with good reason.
    """
    logger.info(f"Splitting snapshot into {n_split} intermediate files.")
    logger.debug(f"FOF6d weight: {config.fof6d_weight}, Aggregate Properties weight: {config.properties_weight}")

    with h5py.File(snapshot_file, "r") as f:
        for i in range(n_split):
            with h5py.File(intermediate_directory / f"rank_{i}.hdf5", "a") as intermediate:
                f.copy(f["Header"], intermediate, "Header")

        available_ptypes = reader.available_ptypes()
        ptype_counts = {}

        for pt in available_ptypes:
            halo_ids = reader.read_halo_ids(ptype=pt)  # autoconverts to raw snapshot convention
            halo_ids = halo_ids[halo_ids != -1]
            unique, counts = np.unique(halo_ids, return_counts=True)
            ptype_counts[pt] = (unique, counts)

        all_hids = np.unique(np.concatenate([unique for unique, _ in ptype_counts.values()]))

        # guard (necessary for high-redshift snapshots with no HaloIDs)
        if len(all_hids) == 0:
            return

        n_halos = all_hids.max() + 1  # use hid as direct index

        star_counts = np.zeros(shape=n_halos)
        gas_counts = np.zeros(shape=n_halos)
        dm_counts = np.zeros(shape=n_halos)

        weight_ptypes = {"star": star_counts, "gas": gas_counts, "dm": dm_counts}

        for raw_ptype_name, count_array in weight_ptypes.items():
            if raw_ptype_name in ptype_counts:
                halo_ids, counts = ptype_counts[raw_ptype_name]
                count_array[halo_ids] = counts

        fof6d_cost = star_counts[all_hids] ** 1.2 + gas_counts[all_hids]
        aggregates_cost = star_counts[all_hids] + gas_counts[all_hids] + dm_counts[all_hids]
        halo_weights = config.fof6d_weight * fof6d_cost + config.properties_weight * aggregates_cost

        # greedy binning according to halo weight
        weight_order = np.argsort(halo_weights)[::-1]
        rank_assignments = [set() for _ in range(n_split)]
        rank_loads = np.zeros(n_split)

        for idx in weight_order:
            lightest = np.argmin(rank_loads)
            rank_assignments[lightest].add(all_hids[idx])
            rank_loads[lightest] += halo_weights[idx]

        # filter, tosses particles not in halos
        rank_particle_counts: dict[int, dict[str, int]] = {i: {} for i in range(n_split)}

        for pt in available_ptypes:
            raw_ptype = reader.inverse_ptype_map[pt]
            datasets = list(f[raw_ptype].keys())
            halo_ids = reader.read_halo_ids(ptype=pt)
            particle_index = np.arange(len(halo_ids), dtype=np.int64)
            in_halo = halo_ids != -1
            ids_filtered = halo_ids[in_halo]
            order = np.argsort(ids_filtered)
            ids_sorted = ids_filtered[order]

            datasets = datasets + ["particle_index"]

            for i in range(n_split):
                with h5py.File(intermediate_directory / f"rank_{i}.hdf5", "a") as f_out:
                    f_out.require_group(raw_ptype)

                    halo_set = np.array(list(rank_assignments[i]))
                    rank_mask = np.isin(ids_sorted, halo_set)
                    rank_particle_counts[i][pt] = int(rank_mask.sum())

                    for dataset in datasets:
                        if dataset == "particle_index":
                            data = particle_index[in_halo][order]

                        else:
                            data = f[raw_ptype][dataset][:][in_halo][order]

                        f_out[raw_ptype][dataset] = data[rank_mask]

        for i in range(n_split):
            with h5py.File(intermediate_directory / f"rank_{i}.hdf5", "a") as f_out:
                diag = f_out.require_group("Diagnostics")

                for ptype, count in rank_particle_counts[i].items():
                    diag.attrs[f"n_{ptype}"] = count

                diag.attrs["n_halos"] = len(rank_assignments[i])
                diag.attrs["total_weight"] = rank_loads[i]

    logger.info("Intermediate files created.")


def merge_intermediate_catalogues(files: list[Path], output_path: Path, internals: Internals) -> None:
    """
    Merges per-rank HDF5 catalogues into a single output file, groups sorted by mass descending.
    """
    logger.info(f"Merging {len(files)} intermediate files into output catalogue.")

    sort_column = {"halos": "properties/core/mass_total", "galaxies": "properties/core/mass_baryon"}

    # first pass: collect lengths and sort keys per group type
    group_lengths: dict[str, list[int]] = {}
    sort_arrays: dict[str, list[np.ndarray]] = {}
    parent_halo_chunks: list[np.ndarray] = []

    cumulative_halos = 0

    for file in files:
        with h5py.File(file, "r") as f:
            for group_type, hdf5_name in HDF5_GROUP_NAMES.items():
                if hdf5_name not in f:
                    group_lengths.setdefault(group_type, []).append(0)
                    continue

                grp = f[hdf5_name]
                n_groups = len(grp[internals.group_types[group_type]["key"]])
                group_lengths.setdefault(group_type, []).append(n_groups)
                sort_arrays.setdefault(group_type, []).append(grp[sort_column[group_type]][:])

                if group_type == "galaxies" and n_groups > 0:
                    parent_halo_chunks.append(grp["properties/core/parent_halo_index"][:] + cumulative_halos)

            cumulative_halos += group_lengths["halos"][-1]

    # compute sort orders and inverse map for parent reindexing
    sort_orders: dict[str, np.ndarray] = {}

    for group_type in sort_arrays:
        merged = np.concatenate(sort_arrays[group_type])
        sort_orders[group_type] = np.argsort(-merged, kind="stable")  # largest -> smallest ascending

    if "halos" in sort_orders:
        inverse_halo_order = np.argsort(sort_orders["halos"], kind="stable")

    # second pass: merge datasets
    with h5py.File(output_path, "w") as f_out:
        for group_type, hdf5_name in HDF5_GROUP_NAMES.items():
            if group_type not in sort_orders:
                continue

            group_config = internals.group_types[group_type]

            order = sort_orders[group_type]
            out_grp = f_out.create_group(hdf5_name)
            n_total = len(order)

            # discover datasets from first non-empty rank file
            dataset_names: set[str] = set()
            csr_suffixes = ("_indices", "_offsets", "_lengths")

            for file, length in zip(files, group_lengths[group_type]):
                if length == 0:
                    continue
                with h5py.File(file, "r") as f:
                    dataset_names = _discover_datasets(f[hdf5_name], csr_suffixes)

                break

            # skip group IDs — we reassign sequentially
            group_id_name = internals.group_types[group_type]["key"]
            dataset_names.discard(group_id_name)
            dataset_names.discard("properties/core/parent_halo_index")

            # concatenate and reorder each dataset
            for dataset_name in sorted(dataset_names):
                chunks = []
                source_attrs = {}  # unit/description metadata

                for file, length in zip(files, group_lengths[group_type]):
                    if length == 0:
                        continue

                    with h5py.File(file, "r") as f:
                        grp = f[hdf5_name]

                        if dataset_name in grp:
                            chunks.append(grp[dataset_name][:])

                            if not source_attrs:
                                source_attrs = dict(grp[dataset_name].attrs)

                        else:
                            chunks.append(np.full(length, np.nan))

                merged = np.concatenate(chunks)

                parent = (
                    dataset_name.rsplit("/", 1)[0] if "/" in dataset_name else None
                )  # slightly weird but this will never break under standard conventions

                if parent:
                    out_grp.require_group(parent)

                ds = out_grp.create_dataset(dataset_name, data=merged[order], compression=1)

                for attr_name, attr_value in source_attrs.items():
                    ds.attrs[attr_name] = attr_value

            # sequential IDs
            out_grp.create_dataset(group_id_name, data=np.arange(n_total), compression=1)

            # parent halo reindexing
            if group_type == "galaxies" and parent_halo_chunks:
                merged_parents = np.concatenate(parent_halo_chunks)
                reindexed = inverse_halo_order[merged_parents]
                out_grp.require_group("properties/core")
                out_grp.create_dataset("properties/core/parent_halo_index", data=reindexed[order], compression=1)

            # CSR lists

            for ptype in group_config["ptypes"]:
                all_indices, all_lengths = [], []

                for file, length in zip(files, group_lengths[group_type]):
                    if length == 0:
                        continue

                    with h5py.File(file, "r") as f:
                        grp = f[hdf5_name]

                        if f"membership/{ptype}_indices" in grp:
                            all_indices.append(grp[f"membership/{ptype}_indices"][:])
                            all_lengths.append(grp[f"membership/{ptype}_lengths"][:])

                if not all_indices:
                    continue

                membership_grp = out_grp.require_group("membership")
                indices, offsets, lengths = _reorder_csr_lists(all_indices, all_lengths, order)
                membership_grp.create_dataset(f"{ptype}_indices", data=indices, compression=1)
                membership_grp.create_dataset(f"{ptype}_offsets", data=offsets, compression=1)
                membership_grp.create_dataset(f"{ptype}_lengths", data=lengths, compression=1)

    logger.info("Created merged analysis catalogue.")


def _discover_datasets(group: h5py.Group, exclude_suffixes: tuple[str, ...]) -> set[str]:
    """
    Uses h5py's visititems to go into sub-groups (e.g. properties/core) etc. and populate column fields.
    """
    datasets: set[str] = set()
    group.visititems(
        lambda name, obj: (
            datasets.add(name) if isinstance(obj, h5py.Dataset) and not name.endswith(exclude_suffixes) else None
        )
    )
    return datasets


def _reorder_csr_lists(
    all_indices: list[np.ndarray], all_lengths: list[np.ndarray], order: np.ndarray
) -> tuple[np.ndarray, ...]:
    """
    Helper to reorders CSR-format particle lists according to the group sort order (which should be largest mass descending).
    """
    flat_lengths = np.concatenate(all_lengths)
    flat_offsets = np.concatenate([[0], np.cumsum(flat_lengths[:-1])]).astype(DTYPES["csr_offsets"])
    flat_indices = np.concatenate(all_indices)

    reordered_lengths = flat_lengths[order]
    reordered_indices = np.concatenate(
        [flat_indices[flat_offsets[i] : flat_offsets[i] + flat_lengths[i]] for i in order]
    )
    reordered_offsets = np.concatenate([[0], np.cumsum(reordered_lengths[:-1])]).astype(DTYPES["csr_offsets"])

    return reordered_indices, reordered_offsets, reordered_lengths
