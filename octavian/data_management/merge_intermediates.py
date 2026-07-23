"""

Functions to filter & split a snapshot into subfiles for per-rank MPI access; and remerge analysis hdf5 files into an output catalogue.

NOTE: this will eventually be legacy code, as we intend to move away from intermediate files.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_management.pipeline_management import Internals
    from octavian.data_management.conventions import OctavianConfig

# defaults
from pathlib import Path

# others
import h5py
import numpy as np

# octavian
from octavian.data_management.conventions import DTYPES, intermediate_catalogue_path
from octavian.log import get_logger, intermediate_log_path

logger = get_logger()

SORT_COLUMN_BY_KIND: dict[str, str] = {
    "halo": "properties/core/mass_total",
    "galaxy": "properties/core/mass_baryon",
}
HALO_POINTER_COLUMNS: frozenset[str] = frozenset({"parent", "field_halo_index", "parent_halo_index"})
FILL_VALUES: dict[str, int] = {
    "parent": -1,
    "depth": 0,
}  # this is so data doesn't get sent through the pipeline if it doesn't exist (but we still need corresponding sentinel values for the tree pointers)


def merge_intermediate_catalogues(files: list[Path], output_path: Path, internals: Internals) -> None:
    """
    Merges per-rank HDF5 catalogues into a single output file, groups sorted by mass descending.

    # TODO: sort out type-checking on dicts
    # TODO: move this to MPI scatter/gather
    """
    logger.info(f"Merging {len(files)} intermediate files into output catalogue.")

    # NOTE: with hierarchical membership this becomes quite messy; it'd be nice to break up this function
    group_lengths: dict[str, list[int]] = {}
    sort_arrays: dict[str, list[np.ndarray]] = {}
    membership_chunks: dict[str, dict[str, list[np.ndarray]]] = {
        group_type: {
            column_name: []
            for column_name in internals.membership_columns.get(group_type, {})
            if column_name
            != "central_galaxy_index"  # this has to be done by merge_intermediates so isn't in data at this point
        }
        for group_type in internals.group_types
    }

    cumulative_halos = 0

    for file in files:
        with h5py.File(file, "r") as f:
            for group_type in internals.group_types:
                group_params = internals.group_types[group_type]
                hdf5_name = group_params["hdf5_group"]

                if hdf5_name not in f:
                    group_lengths.setdefault(group_type, []).append(0)
                    continue

                kind = group_params["kind"]
                sort_path = SORT_COLUMN_BY_KIND[kind]

                grp = f[hdf5_name]
                n_groups = len(grp[group_params["key"]])
                group_lengths.setdefault(group_type, []).append(n_groups)
                sort_arrays.setdefault(group_type, []).append(grp[sort_path][:])

                if n_groups == 0:
                    logger.warning(f"No groups found for {group_type} in {file}.")
                    continue

                for column_name in membership_chunks[group_type]:
                    dataset_path = f"membership/{column_name}"

                    if dataset_path in grp:
                        chunk = grp[dataset_path][:]
                    elif column_name in FILL_VALUES:
                        column_meta = internals.membership_columns[group_type][column_name]
                        chunk = np.full(n_groups, FILL_VALUES[column_name], dtype=np.dtype(column_meta.dtype))
                    else:
                        raise KeyError(
                            f"{file} is missing membership columns {dataset_path!r} (which is declared in internals.yaml)"
                        )

                    if column_name in HALO_POINTER_COLUMNS:
                        chunk = np.where(chunk == -1, -1, chunk + cumulative_halos)

                    membership_chunks[group_type][column_name].append(chunk)

            cumulative_halos += group_lengths["halos"][-1]

    # compute sort orders and inverse map for hierarchical parent reindexing
    sort_orders: dict[str, np.ndarray] = {}

    for group_type in sort_arrays:
        merged = np.concatenate(sort_arrays[group_type])
        sort_orders[group_type] = np.argsort(-merged, kind="stable")

    if "halos" in sort_orders:
        inverse_halo_order = np.argsort(sort_orders["halos"], kind="stable")

    # now resolve the per-rank orders into the final descending order
    resolved: dict[tuple[str, str], np.ndarray] = {}

    for group_type, columns in membership_chunks.items():
        if group_type not in sort_orders:
            continue

        order = sort_orders[group_type]

        for column_name, chunks in columns.items():
            if column_name in HALO_POINTER_COLUMNS:
                resolved[(group_type, column_name)] = _resolve_pointer_column(chunks, inverse_halo_order, order)
            else:
                resolved[(group_type, column_name)] = np.concatenate(chunks)[order]

    # second pass: merge datasets
    with h5py.File(output_path, "w") as f_out:
        for group_type in internals.group_types:
            group_params = internals.group_types[group_type]
            hdf5_name = group_params["hdf5_group"]

            if group_type not in sort_orders:
                continue

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

            # skip group IDs (done with np.arange below, more user-friendly for these to be sequential)
            dataset_names.discard(internals.group_types[group_type]["key"])

            # skip membership columns (these are handled explicitly with their own path)
            for column_name in internals.membership_columns.get(group_type, {}):
                dataset_names.discard(f"membership/{column_name}")

            # concatenate then reorder the physics data accordingly
            for dataset_name in sorted(dataset_names):
                chunks = []
                source_attrs = {}

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
                parent_group_path = (
                    dataset_name.rsplit("/", 1)[0] if "/" in dataset_name else None
                )  # slightly weird but this will never break under standard conventions

                if parent_group_path:
                    out_grp.require_group(parent_group_path)

                ds = out_grp.create_dataset(dataset_name, data=merged[order], compression=1)

                for attr_name, attr_value in source_attrs.items():
                    ds.attrs[attr_name] = attr_value

            # ^ see above, final IDs need to be sequential and therefore it is done with np.arange
            out_grp.create_dataset(internals.group_types[group_type]["key"], data=np.arange(n_total), compression=1)

            # likewise, membership columns are in internals.yaml
            membership_grp = out_grp.require_group("membership")

            for column_name, column_meta in internals.membership_columns.get(group_type, {}).items():
                if (group_type, column_name) not in resolved:
                    continue  # central_galaxy_index handled below

                dataset = membership_grp.create_dataset(
                    column_name, data=resolved[(group_type, column_name)], compression=1
                )
                dataset.attrs["unit"] = column_meta.unit
                dataset.attrs["description"] = column_meta.description

            # particle membership lists are handled explicitly
            for ptype in group_params["ptypes"]:
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

                indices, offsets, lengths = _reorder_csr_lists(
                    all_indices, all_lengths, order
                )  # moved to its own function for readability
                membership_grp.create_dataset(f"{ptype}_indices", data=indices, compression=1)
                membership_grp.create_dataset(f"{ptype}_offsets", data=offsets, compression=1)
                membership_grp.create_dataset(f"{ptype}_lengths", data=lengths, compression=1)

        # now another loop galaxy memberships (required halos to be done already)
        if "halos" in sort_orders and ("galaxies", "parent_halo_index") in resolved:
            n_halos = len(sort_orders["halos"])
            galaxy_indices, galaxy_offsets, galaxy_lengths, central_galaxy_index = _build_galaxy_membership_csr(
                resolved[("galaxies", "parent_halo_index")], resolved[("halos", "parent")], n_halos
            )

            halo_membership_grp = f_out[internals.group_types["halos"]["hdf5_group"]]["membership"]
            halo_membership_grp.create_dataset("galaxy_indices", data=galaxy_indices, compression=1)
            halo_membership_grp.create_dataset("galaxy_offsets", data=galaxy_offsets, compression=1)
            halo_membership_grp.create_dataset("galaxy_lengths", data=galaxy_lengths, compression=1)

            central_meta = internals.membership_columns["halos"]["central_galaxy_index"]
            central_dataset = halo_membership_grp.create_dataset(
                "central_galaxy_index", data=central_galaxy_index, compression=1
            )
            central_dataset.attrs["unit"] = central_meta.unit
            central_dataset.attrs["description"] = central_meta.description

    logger.info("Created merged analysis catalogue.")


def _resolve_pointer_column(chunks: list[np.ndarray], inverse_halo_order: np.ndarray, order: np.ndarray) -> np.ndarray:
    """
    Concatenates per-rank pointers and reindexes them into final merged catalogue order.
    """
    merged_pointers = np.concatenate(chunks)
    reindexed = np.where(merged_pointers == -1, -1, inverse_halo_order[merged_pointers])
    return reindexed[order]


def _build_galaxy_membership_csr(
    galaxy_parent_halo_index: np.ndarray, halo_parent: np.ndarray, n_halos: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds the galaxy membershipn CSR and derives the central galaxies too.
    """
    lengths = np.zeros(n_halos, dtype=DTYPES["csr_lengths"])
    level_halo_rows: list[np.ndarray] = []
    level_galaxy_indices: list[np.ndarray] = []

    current_rows = galaxy_parent_halo_index.astype(np.int64, copy=True)
    galaxy_indices = np.arange(len(galaxy_parent_halo_index), dtype=DTYPES["csr_indices"])

    while True:
        valid_mask = current_rows >= 0
        if not valid_mask.any():
            break

        lengths += np.bincount(current_rows[valid_mask], minlength=n_halos).astype(DTYPES["csr_lengths"])
        level_halo_rows.append(current_rows[valid_mask])
        level_galaxy_indices.append(galaxy_indices[valid_mask])
        current_rows[valid_mask] = halo_parent[current_rows[valid_mask]]  # walk one level up; roots become -1

    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(DTYPES["csr_offsets"])

    if level_halo_rows:
        flat_halo_rows = np.concatenate(level_halo_rows)
        flat_galaxy_indices = np.concatenate(level_galaxy_indices)
        pair_order = np.lexsort((flat_galaxy_indices, flat_halo_rows))  # by halo row, then galaxy ascending
        indices = flat_galaxy_indices[pair_order]
    else:
        indices = np.empty(0, dtype=DTYPES["csr_indices"])

    central_galaxy_index = np.full(n_halos, -1, dtype=np.int64)
    non_empty = lengths > 0
    central_galaxy_index[non_empty] = indices[offsets[:-1][non_empty]]

    return indices, offsets, lengths, central_galaxy_index


def clean_intermediates(
    intermediate_dir: Path,
    output_dir: Path,
    n_ranks: int,
    config: OctavianConfig,
) -> None:
    """
    Cleans the working directory by removing intermediate analysis catalogues and compressing the log into one file/removing the log, depending on what was specified in config.yaml.
    """
    for i in range(n_ranks):  # remove intermediate analysis files
        (intermediate_catalogue_path(directory=intermediate_dir, rank=i)).unlink(missing_ok=True)
    logger.info(f"Removed {n_ranks} intermediate analysis catalogues.")

    if config.keep_logs:  # concatenates the per-rank logs rather than time-based zipper merging
        merged_log = output_dir / "octavian.log"
        with open(merged_log, "w") as out:
            for i in range(n_ranks):
                rank_log = intermediate_dir / f"octavian_rank{i}.log"
                if rank_log.exists():
                    out.write(rank_log.read_text())
                    rank_log.unlink()
            logger.info(f"Merged then cleaned up {n_ranks} log files.")
    else:
        for i in range(n_ranks):
            (intermediate_log_path(directory=intermediate_dir, rank=i)).unlink(
                missing_ok=True
            )  # remove intermediate logs
        logger.info(f"Removed {n_ranks} log files.")


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
    flat_offsets = np.concatenate([[0], np.cumsum(flat_lengths)]).astype(
        DTYPES["csr_offsets"]
    )  # this is technically the same as line above
    flat_indices = np.concatenate(all_indices)

    reordered_lengths = flat_lengths[order]
    reordered_indices = np.concatenate(
        [flat_indices[flat_offsets[i] : flat_offsets[i] + flat_lengths[i]] for i in order]
    )
    reordered_offsets = np.concatenate([[0], np.cumsum(reordered_lengths)]).astype(DTYPES["csr_offsets"])

    return reordered_indices, reordered_offsets, reordered_lengths
