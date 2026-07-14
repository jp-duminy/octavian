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

HDF5_GROUP_NAMES = {
    "halos": "halo_data",
    "galaxies": "galaxy_data",
}


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
    flat_offsets = np.concatenate([[0], np.cumsum(flat_lengths[:-1])]).astype(DTYPES["csr_offsets"])
    flat_indices = np.concatenate(all_indices)

    reordered_lengths = flat_lengths[order]
    reordered_indices = np.concatenate(
        [flat_indices[flat_offsets[i] : flat_offsets[i] + flat_lengths[i]] for i in order]
    )
    reordered_offsets = np.concatenate([[0], np.cumsum(reordered_lengths[:-1])]).astype(DTYPES["csr_offsets"])

    return reordered_indices, reordered_offsets, reordered_lengths
