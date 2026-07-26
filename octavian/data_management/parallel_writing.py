"""

Functions to filter & split a snapshot into subfiles for per-rank MPI access; and remerge analysis hdf5 files into an output catalogue.

NOTE: this will eventually be legacy code, as we intend to move away from intermediate files.

"""

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from octavian.data_management.pipeline_management import Internals
    from octavian.data_management.conventions import OctavianConfig
    from .pack_catalogue_data import RankPackedData, MembershipArrays
    from mpi4py.MPI import Comm

# defaults
from pathlib import Path
from dataclasses import dataclass

# others
import h5py
import numpy as np

# octavian
from octavian.data_management.conventions import DTYPES
from octavian.log import get_logger, intermediate_log_path

GroupLengths: TypeAlias = dict[str, list[int]]  # temporary to avoid horrific type annotations
SortArrays: TypeAlias = dict[str, list[np.ndarray]]
MembershipChunks: TypeAlias = dict[str, dict[str, list[np.ndarray]]]

logger = get_logger()

SORT_COLUMN_BY_KIND: dict[str, str] = {  # TODO: figure out a way to get around this
    "halo": "properties/core/mass_total",
    "galaxy": "properties/core/mass_baryon",
}
HALO_POINTER_COLUMNS: frozenset[str] = frozenset({"parent", "field_halo_index", "parent_halo_index"})
FILL_VALUES: dict[str, int] = {
    "parent": -1,
    "depth": 0,
}  # this is so data doesn't get sent through the pipeline if it doesn't exist (but we still need corresponding sentinel values for the tree pointers)


@dataclass(slots=True)
class UnpackedRankData:
    """
    Group-level membership and physics data arrays unpacked from the PackedRankData class.

    - sort_arrays: the array (total mass, but with specific halo/data key for total/baryon) to sort groups by
    - membership_chunks: per-rank membership data
    - group_ids: all group IDs
    - physics_chunks: per-rank physics data

    Key by HDF5 output group name (likewise for columns contained therein).
    """

    sort_arrays: dict[str, list[np.ndarray]]
    membership_chunks: dict[str, dict[str, list[np.ndarray]]]
    group_ids: dict[str, list[np.ndarray]]
    physics_chunks: dict[str, dict[str, list[np.ndarray]]]


def write_catalogue_parallel(
    packed_data: RankPackedData,
    catalogue_path: Path,
    internals: Internals,
    comm: Comm | None,
) -> None:
    """
    Drop in for merge_intermediate_catalogues 2 (write catalogue in parallel). Docstring unfinished.
    """
    rank = (
        comm.Get_rank() if comm is not None else 0
    )  # duplicates what's in run_octavian.py for conftest.py to work too
    n_ranks = comm.Get_size() if comm is not None else 1

    lightweight_data = packed_data.without_indices()  # indices array is sizen n_particles (others are n_groups)

    all_lightweight: list[RankPackedData] = (
        comm.gather(lightweight_data, root=0) if comm is not None else [lightweight_data]
    )  # comm.gather is safe on the smaller datasets

    if rank == 0:
        logger.info(f"Synthesising analysis data from {len(all_lightweight)} ranks into output catalogue.")
        unpacked = unpack_data(all_rank_data=all_lightweight, internals=internals)
        sort_orders, resolved = sort_and_resolve(unpacked=unpacked)

        write_catalogue_physics(
            output_path=catalogue_path,
            sort_orders=sort_orders,
            resolved=resolved,
            physics_chunks=unpacked.physics_chunks,
            internals=internals,
        )

        per_rank_write_positions: dict[tuple[str, str], list[np.ndarray]] = {}

        for group_type in internals.group_types:
            group_params = internals.group_types[group_type]
            hdf5_name = group_params["hdf5_group"]

            for ptype in group_params["ptypes"]:
                key = (hdf5_name, ptype)
                rank_lengths = []

                for rank_data in (
                    all_lightweight
                ):  # there is a fair bit of nestage in the dataclass here but hopefully this reads okay
                    if hdf5_name in rank_data.groups and ptype in rank_data.groups[hdf5_name].particle_lists:
                        rank_lengths.append(
                            np.diff(rank_data.groups[hdf5_name].particle_lists[ptype].offsets)
                        )  # diff(offsets) gives (n-1) lengths array
                    else:
                        rank_lengths.append(
                            np.empty(0, dtype=DTYPES["csr_lengths"])
                        )  # if data isn't on a rank (think high-z snaps) the ordering must be preserved

                flat_lengths = np.concatenate(rank_lengths)
                order = sort_orders[group_type]
                sorted_lengths = flat_lengths[order]
                sorted_offsets = np.empty(len(sorted_lengths) + 1, dtype=DTYPES["csr_offsets"])
                sorted_offsets[0] = 0
                np.cumsum(
                    sorted_lengths, out=sorted_offsets[1:]
                )  # doing np.concat with dtype arg doesn't work for the prepended 0 list

                inverse_order = np.argsort(order, stable=True)
                write_displacements = sorted_offsets[:-1][
                    inverse_order
                ]  # inverse_order[global_index] = sorted_position
                nonzero_mask = sorted_lengths > 0
                nonzero_displacements = sorted_offsets[:-1][nonzero_mask]
                unique_count = len(np.unique(nonzero_displacements))
                assert unique_count == len(nonzero_displacements), (
                    f"{key}: {len(nonzero_displacements) - unique_count} duplicate displacements among nonzero-length groups"
                )

                rank_group_counts = [len(rl) for rl in rank_lengths]
                per_rank_write_positions[key] = np.split(
                    write_displacements, np.cumsum(rank_group_counts[:-1])
                )  # gives per-rank boundaries

                with h5py.File(catalogue_path, "a") as f:  # use "a" not "w"  (group data written above ^)
                    membership_grp = f[hdf5_name]["membership"]
                    membership_grp.create_dataset(f"{ptype}_offsets", data=sorted_offsets, compression=1)
                    membership_grp.create_dataset(f"{ptype}_lengths", data=sorted_lengths, compression=1)
                    membership_grp.create_dataset(
                        f"{ptype}_indices", shape=(sorted_offsets[-1],), dtype=DTYPES["csr_indices"]
                    )  # indices initialised to empty

    rank_write_positions: dict[tuple[str, str], np.ndarray] = {}
    rank_membership_data: dict[tuple[str, str], MembershipArrays] = {}

    # each rank is given its HDF5 file displacements so it knows where to write the indices array to
    for group_type in internals.group_types:
        group_params = internals.group_types[group_type]
        hdf5_name = group_params["hdf5_group"]

        for ptype in group_params["ptypes"]:
            key = (hdf5_name, ptype)
            send_data = per_rank_write_positions[key] if rank == 0 else None
            rank_write_positions[key] = comm.scatter(send_data, root=0) if comm is not None else send_data[0]

            if hdf5_name in packed_data.groups and ptype in packed_data.groups[hdf5_name].particle_lists:
                rank_membership_data[key] = packed_data.groups[hdf5_name].particle_lists[ptype]

    # ranks take turns writing their indices array (globally-ordered) to the output catalogue
    for writer_rank in range(n_ranks):
        if rank == writer_rank and rank_membership_data:  # == writer_rank so ranks don't overwrite each others' data
            with h5py.File(catalogue_path, "a") as f:  # also use "a" not "w"!
                for (hdf5_name, ptype), csr in rank_membership_data.items():
                    displacements = rank_write_positions[(hdf5_name, ptype)]
                    dataset = f[hdf5_name][f"membership/{ptype}_indices"]
                    for group_idx in range(len(displacements)):
                        start, end = csr.offsets[group_idx], csr.offsets[group_idx + 1]
                        if end > start:
                            dataset[displacements[group_idx] : displacements[group_idx] + (end - start)] = csr.indices[
                                start:end
                            ]
        if comm is not None:
            comm.Barrier()  # remember to Barrier() because each rank needs to finish its write first


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


def unpack_data(
    all_rank_data: list[RankPackedData],
    internals: Internals,
) -> UnpackedRankData:
    """
    Will finish docstring once returns are done.
    """
    group_lengths: dict[str, list[int]] = {}
    sort_arrays: dict[str, list[np.ndarray]] = {}
    all_group_ids: dict[str, list[np.ndarray]] = {}
    membership_chunks: dict[str, dict[str, list[np.ndarray]]] = {
        group_type: {
            column_name: []
            for column_name in internals.membership_columns.get(group_type, {})
            if column_name
            != "central_galaxy_index"  # this has to be done by merge_intermediates so isn't in data at this point, TODO: deprecate
        }
        for group_type in internals.group_types
    }
    physics_chunks: dict[str, dict[str, list[np.ndarray]]] = {}
    cumulative_halos = 0

    for rank_data in all_rank_data:  # the rank_data dataclass is keyed by hdf5 names
        for group_type in internals.group_types:
            group_params = internals.group_types[group_type]
            hdf5_name = group_params["hdf5_group"]

            if hdf5_name not in rank_data.groups:
                group_lengths.setdefault(group_type, []).append(0)
                continue

            kind = group_params["kind"]
            sort_column = SORT_COLUMN_BY_KIND[kind].rsplit("/", 1)[
                -1
            ]  # the rank data doesn't include hdf5 group prefixes

            group = rank_data.groups[hdf5_name]
            n_groups = len(group.group_ids)
            group_lengths.setdefault(group_type, []).append(n_groups)
            sort_arrays.setdefault(group_type, []).append(group.physics_columns[sort_column])

            if n_groups == 0:
                logger.warning(f"No groups found for {group_type}.")
                continue

            # membership
            for column_name in membership_chunks[group_type]:
                if column_name in group.membership_columns:
                    chunk = group.membership_columns[column_name]
                elif column_name in FILL_VALUES:
                    column_meta = internals.membership_columns[group_type][column_name]
                    chunk = np.full(n_groups, FILL_VALUES[column_name], dtype=np.dtype(column_meta.dtype))

                if column_name in HALO_POINTER_COLUMNS:
                    chunk = np.where(chunk == -1, -1, chunk + cumulative_halos)

                membership_chunks[group_type][column_name].append(chunk)

            # physics
            for column_name, column_data in group.physics_columns.items():
                physics_chunks.setdefault(group_type, {}).setdefault(column_name, []).append(column_data)

            all_group_ids.setdefault(group_type, []).append(
                group.group_ids
            )  # match type annotation for now, TODO: just a dict

        cumulative_halos += group_lengths["halos"][
            -1
        ]  # use the previous rank's contribution so the next knows where to start

    unpacked_data = UnpackedRankData(
        sort_arrays=sort_arrays,
        membership_chunks=membership_chunks,
        group_ids=all_group_ids,
        physics_chunks=physics_chunks,
    )

    return unpacked_data


def sort_and_resolve(
    unpacked: UnpackedRankData,
) -> tuple[dict[str, np.ndarray], dict[tuple[str, str], np.ndarray]]:
    """
    Sorts and resolves the concatenated per-rank data (for catalogue invariance).
    """
    sort_orders: dict[str, np.ndarray] = {}
    for group_type in unpacked.sort_arrays:
        merged = np.concatenate(unpacked.sort_arrays[group_type])
        global_ids = np.concatenate(unpacked.group_ids[group_type])
        depth_chunks = unpacked.membership_chunks.get(group_type, {}).get("depth")

        if depth_chunks:
            depth = np.concatenate(depth_chunks)
            sort_orders[group_type] = np.lexsort(
                (global_ids, depth, -merged)
            )  # lexsort sorts by last key first, so in this case, halos
        else:
            sort_orders[group_type] = np.lexsort((global_ids, -merged))

    if "halos" in sort_orders:
        inverse_halo_order = np.argsort(sort_orders["halos"], kind="stable")

    # now resolve the per-rank orders into the final descending order
    resolved: dict[tuple[str, str], np.ndarray] = {}

    for group_type, columns in unpacked.membership_chunks.items():
        if group_type not in sort_orders:
            continue

        order = sort_orders[group_type]

        for column_name, chunks in columns.items():
            if column_name in HALO_POINTER_COLUMNS:
                resolved[(group_type, column_name)] = _resolve_pointer_column(chunks, inverse_halo_order, order)
            else:
                resolved[(group_type, column_name)] = np.concatenate(chunks)[order]

    return sort_orders, resolved


def write_catalogue_physics(
    output_path: Path,
    sort_orders: dict[str, np.ndarray],
    resolved: dict[tuple[str, str], np.ndarray],
    physics_chunks: dict[str, dict[str, list[np.ndarray]]],
    internals: Internals,
) -> None:
    """
    Writes the final catalogue.
    """
    with h5py.File(output_path, "w") as f_out:
        for group_type in internals.group_types:
            group_params = internals.group_types[group_type]
            hdf5_name = group_params["hdf5_group"]

            if group_type not in sort_orders:
                continue

            order = sort_orders[group_type]
            out_grp = f_out.create_group(hdf5_name)
            n_total = len(order)

            for column_name, chunks in physics_chunks.get(group_type, {}).items():
                merged = np.concatenate(chunks)[order]
                column_meta = internals.output_columns[column_name]
                label = column_meta.label
                label_group = out_grp.require_group(f"properties/{label}")
                ds = label_group.create_dataset(column_name, data=merged, compression=1)
                ds.attrs["unit"] = column_meta.unit
                ds.attrs["description"] = column_meta.description

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


def clean_intermediates(
    intermediate_dir: Path,
    output_dir: Path,
    n_ranks: int,
    config: OctavianConfig,
) -> None:
    """
    Cleans the working directory by removing intermediate analysis catalogues and compressing the log into one file/removing the log, depending on what was specified in config.yaml.
    """
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
