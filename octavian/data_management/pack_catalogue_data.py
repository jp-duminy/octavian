"""

Functions which write data from analysis stages into CSR format lists for HDF5 compatibility (and fast, straightforward access) and create output HDF5 files (per-rank currently).

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_management import SimulationData, Internals, SimulationAttributes

# default libraries
from pathlib import Path
from datetime import datetime, timezone
import subprocess
from dataclasses import dataclass, asdict, replace

# other packages
import h5py
import numpy as np

# internal imports
from octavian.data_management.conventions import (
    DTYPES,
    OctavianConfig,
)  # NOTE: import from within-file, not module level (to avoid circular import)
from octavian.log import get_logger
from octavian.version import __version__

logger = get_logger()


@dataclass(frozen=True, slots=True)
class MembershipArrays:
    """
    Contains membership arrays in CSR format:

    - indices
    - offsets
    - lengths
    """

    indices: np.ndarray | None
    offsets: np.ndarray

    def without_indices(self) -> MembershipArrays:
        """
        Deletes indices (memory problem).
        """
        return replace(self, indices=None)


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

    def without_indices(self) -> RankPackedData:
        """
        Deletes the particle lists (enormous) from the rankdata (useful for gather)
        """
        return replace(
            self,
            groups={
                hdf5_name: replace(
                    group,
                    particle_lists={
                        ptype: membership.without_indices() for ptype, membership in group.particle_lists.items()
                    },
                )
                for hdf5_name, group in self.groups.items()
            },
        )


def construct_membership_arrays(
    data: SimulationData, internals: Internals, indices: dict[str, np.ndarray]
) -> dict[str, dict[str, MembershipArrays]]:
    """
    Extracts particle lists from SimulationData (matching GroupStore & ParticleStore) and converts them to the CSR format for hdf5.
    """
    logger.info("Constructing particle membership arrays.")

    result: dict[str, dict[str, MembershipArrays]] = {group_name: {} for group_name in data.groups}

    for (
        group_name  # I really have no idea why ruff insists on the line looking like this.
    ) in data.groups:
        group_store = data.groups[group_name]
        particles = data.particles

        for ptype in internals.group_types[group_name]["ptypes"]:
            if ptype not in particles:
                continue

            offsets, sorted_local = group_store.csr_membership[ptype]

            result[group_name][ptype] = MembershipArrays(
                indices=indices[ptype][sorted_local].astype(DTYPES["csr_indices"]),
                offsets=offsets.astype(DTYPES["csr_offsets"]),
            )

    logger.info("Constructed particle-level membership arrays.")

    return result


def pack_rank_data(
    data: SimulationData,
    particle_membership_arrays: dict[str, dict[str, MembershipArrays]],
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
        particle_membership: dict[str, MembershipArrays] = {}
        for ptype in group_params["ptypes"]:
            if ptype not in particle_membership_arrays[group_name]:
                logger.debug(f"{ptype} is not in the particle lists.")
                continue
            particle_membership[ptype] = particle_membership_arrays[group_name][ptype]

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
        ] = {}  # write all properties as flat "physics" columns; the file-write does hdf5 hierarchies
        for column_name in group_store.columns:
            if column_name.startswith(
                "_"
            ):  # the convention I went with was to use a _ prefix for columns which go on GroupStore but not catalogue
                continue

            if (
                column_name not in internals.output_columns
            ):  # make sure whatever property you want in the catalogue is in internals.yaml
                continue

            physics_columns[column_name] = group_store[column_name]

        groups_packed[hdf5_name] = GroupPackedData(
            group_ids=group_store.group_ids,
            membership_columns=membership_columns,
            physics_columns=physics_columns,
            particle_lists=particle_membership,
        )

    return RankPackedData(groups=groups_packed)


def write_catalogue_headers(
    catalogue_path: Path,
    snapshot_path: Path,
    config: OctavianConfig,
    internals: Internals,
    sim_attrs: SimulationAttributes,
    n_ranks: int,
) -> None:
    commit_hash = _get_git_commit()

    with h5py.File(catalogue_path, "a") as f:
        # basic snapshot metadata
        metadata = f.create_group("metadata")
        metadata.attrs["octavian_version"] = __version__
        metadata.attrs["timestamp"] = datetime.now(timezone.utc).isoformat()
        metadata.attrs["original_snapshot_path"] = str(snapshot_path.resolve())
        metadata.attrs["simulation_type"] = config.simulation_type
        metadata.attrs["number_of_mpi_ranks"] = n_ranks
        if commit_hash is not None:
            metadata.attrs["git_commit"] = commit_hash

        # config parameters
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

        # cosmology (+ mis)
        header = f.create_group("header")

        for field_name, meta in internals.header_fields.items():
            value = getattr(sim_attrs, field_name)
            dataset = header.create_dataset(field_name, data=value)
            dataset.attrs["unit"] = meta.unit
            dataset.attrs["description"] = meta.description
            dataset.attrs["a_exp"] = meta.a_exp


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
