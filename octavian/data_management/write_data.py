"""

Functions which write data from analysis stages into CSR format lists for HDF5 compatibility (and fast, straightforward access) and create output HDF5 files (per-rank currently).

"""

# type checking (semantic)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_management import SimulationData, Internals

# octavian modules
from octavian.data_management.conventions import (
    DTYPES,
    OctavianConfig,
)  # NOTE: import from within-file, not module level (to avoid circular import)
from octavian.log import get_logger
from octavian.version import __version__

# others
import h5py
from pathlib import Path  # NOTE: migrated fully to pathlib in v0.3
import numpy as np
from datetime import datetime, timezone
import subprocess
from dataclasses import dataclass, asdict

logger = get_logger()


@dataclass(frozen=True, slots=True)
class MembershipArrays:
    """
    Contains membership arrays in CSR format:

    - indices
    - offsets
    - lengths
    """

    indices: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray


@dataclass(frozen=True, slots=True)
class GroupPackedData:
    """
    Contains packed data from a GroupStore, keyed by output catalogue HDF5 names.

    - group_ids: array of IDs
    - membership_columns: group-group membership dict
    - physics_columns: properties output columns
    - particle_lists: particle-level (key by ptype) MembershipArray dataclasses
    """

    group_ids: np.ndarray
    membership_columns: dict[str, np.ndarray]
    physics_columns: dict[str, np.ndarray]
    particle_lists: dict[str, MembershipArrays]  # keyed by ptype


@dataclass(frozen=True, slots=True)
class RankPackedData:
    """
    Contains the GroupPackedData dataclasses of the groups present on a rank, keyed by output catalogue HDF5 names.
    """

    groups: dict[str, GroupPackedData]


def construct_particle_csr_lists(
    data: SimulationData, internals: Internals, indices: dict[str, np.ndarray]
) -> dict[str, dict[str, dict]]:
    """
    Extracts particle lists from SimulationData (matching GroupStore & ParticleStore) and converts them to the CSR format for hdf5.
    """
    logger.info("Constructing particle membership lists.")

    result = {group: {} for group in data.groups}

    for (
        group_name
    ) in data.groups:  # NOTE: sorts both halos & galaxies as opposed to previous function which took group_name
        group_store = data.groups[group_name]
        particles = data.particles

        for ptype in internals.group_types[group_name]["ptypes"]:
            if ptype not in particles:
                continue

            offsets, sorted_local = group_store.csr_membership[ptype]
            lengths = np.diff(offsets).astype(DTYPES["csr_lengths"])
            snapshot_indices = indices[ptype][sorted_local].astype(DTYPES["csr_indices"])

            result[group_name][ptype] = {
                "indices": snapshot_indices,
                "offsets": offsets.astype(DTYPES["csr_offsets"]),
                "lengths": lengths,
            }

    logger.info("Constructed membership lists.")

    return result


def write_analysis_to_output_file(
    data: SimulationData, particle_lists: dict, internals: Internals, output_file: Path
) -> None:
    """
    Takes in the SimulationData object and writes it to a .hdf5 file.
    """
    logger.info("Writing analysis to .hdf5 file.")

    if output_file.is_file():  # pathlib version of previous os logic
        logger.debug("Removed old analysis file.")
        output_file.unlink()

    with h5py.File(output_file, "w") as out:
        for group_name in internals.group_types:
            group_params = internals.group_types[group_name]
            hdf5_name = group_params["hdf5_group"]

            if group_name not in data.groups:
                continue

            group_store = data.groups[group_name]  # quickhand
            hdf5_group = out.create_group(hdf5_name)
            membership_group = hdf5_group.create_group("membership")

            hdf5_group.create_dataset(name=group_params["key"], data=group_store.group_ids, compression=1)

            for ptype in group_params["ptypes"]:
                if ptype not in particle_lists[group_name]:
                    continue

                pl = particle_lists[group_name][ptype]
                membership_group.create_dataset(f"{ptype}_indices", data=pl["indices"], compression=1)
                membership_group.create_dataset(f"{ptype}_offsets", data=pl["offsets"], compression=1)
                membership_group.create_dataset(f"{ptype}_lengths", data=pl["lengths"], compression=1)

            for column_name, column_meta in internals.membership_columns.get(group_name, {}).items():
                if column_name not in group_store.columns:
                    continue

                dataset = membership_group.create_dataset(
                    column_name,
                    data=group_store[column_name],
                    compression=1,
                )
                dataset.attrs["unit"] = column_meta.unit
                dataset.attrs["description"] = column_meta.description

            # group columns by stage label (what was previously dicts)
            columns_by_label: dict[str, list[str]] = {}

            for column_name in group_store.columns:
                if column_name.startswith("_"):
                    continue

                if column_name not in internals.output_columns:
                    continue

                label = internals.output_columns[column_name].label
                columns_by_label.setdefault(label, []).append(column_name)

            for label, column_names in columns_by_label.items():
                label_group = hdf5_group.require_group(f"properties/{label}")

                for column_name in column_names:
                    column_meta = internals.output_columns[column_name]
                    dataset = label_group.create_dataset(
                        column_name,
                        data=group_store[column_name],
                        compression=1,
                    )
                    dataset.attrs["unit"] = column_meta.unit
                    dataset.attrs["description"] = column_meta.description

    logger.info("Created intermediate analysis file.")


def pack_rank_data(
    data: SimulationData,
    particle_lists: dict[str, dict[str, dict]],
    internals: Internals,
) -> RankPackedData:
    """
    Packs per-rank analysis data into a RankPackedData dataclass which follows the HDF5 output catalogue naming conventions; stripping of GroupStore columns prefixed with a _ is handled here.
    """
    logger.info("Packing data for MPI gather.")
    groups_packed: dict[str, GroupPackedData] = {}

    for group_name in internals.group_types:
        group_params = internals.group_types[group_name]
        hdf5_name = group_params["hdf5_group"]  # dict should be keyed by hdf5 name

        if group_name not in data.groups:
            continue

        group_store = data.groups[group_name]

        # particle-group membership
        particle_membership: dict[str, np.ndarray] = {}
        for ptype in group_params["ptypes"]:
            if ptype not in particle_lists[group_name]:
                logger.debug(f"{ptype} is not in the particle lists.")
                continue

            pl = particle_lists[group_name][ptype]
            membership_arrays = MembershipArrays(
                indices=pl["indices"],
                offsets=pl["offsets"],
                lengths=pl["lengths"],
            )
            particle_membership[ptype] = membership_arrays

        # group-group membership
        membership_columns: dict[str, np.ndarray] = {}
        for column_name in internals.membership_columns[group_name]:
            if column_name not in group_store.columns:
                logger.debug(f"{column_name} is in internals.yaml but not found in the {group_name} GroupStore.")
                continue

            membership_columns[column_name] = group_store[column_name]

        # physics data
        physics_columns: dict[
            str, np.ndarray
        ] = {}  # write all properties as flat columns; the file-write does hdf5 hierarchies
        for column_name in group_store.columns:
            if column_name.startswith(
                "_"
            ):  # the convention I went with was to use a _ prefix for columns which go on GroupStore but not catalogue
                continue

            if column_name not in internals.output_columns:
                continue

            physics_columns[column_name] = group_store[column_name]

        groups_packed[hdf5_name] = GroupPackedData(
            group_ids=group_store.group_ids,
            membership_columns=membership_columns,
            physics_columns=physics_columns,
            particle_lists=particle_membership,
        )

    return RankPackedData(groups=groups_packed)


def write_catalogue_metadata(
    catalogue_path: Path,
    snapshot_path: Path,
    config: OctavianConfig,
    n_ranks: int,
) -> None:
    commit_hash = _get_git_commit()

    with h5py.File(catalogue_path, "a") as f:
        metadata = f.create_group("metadata")
        metadata.attrs["octavian_version"] = __version__
        metadata.attrs["timestamp"] = datetime.now(timezone.utc).isoformat()
        metadata.attrs["original_snapshot_path"] = str(snapshot_path.resolve())
        metadata.attrs["simulation_type"] = config.simulation_type
        metadata.attrs["number_of_mpi_ranks"] = n_ranks

        config_group = metadata.create_group("config_parameters")

        for key, value in asdict(config).items():
            if isinstance(value, dict):
                subgroup = config_group.create_group(key)
                for sub_key, sub_value in value.items():
                    subgroup.attrs[sub_key] = sub_value
            elif isinstance(value, Path) or value is None:
                config_group.attrs[key] = str(value)
            else:
                config_group.attrs[key] = value

        if commit_hash is not None:
            metadata.attrs["git_commit"] = commit_hash


def _get_git_commit() -> str | None:
    """
    Tries to retrieve the git commit; wraps in try/except to avoid stalling on clusters.
    """
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):  # https://github.com/astral-sh/ruff/issues/25901 (PEP 758 py3.14 syntax)
        return None
