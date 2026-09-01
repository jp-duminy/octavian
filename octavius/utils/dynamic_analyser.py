"""

Functionality for dynamic, on-the-fly analysis of raw snapshots by calling the pipeline stages
on a subset of groups in a provided catalogue. Allows users to recompute properties with different
config params, alleviating some of the expensive cost of running the full pipeline.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loader import OctaviusCatalogue, GroupCollection
    from ..data_management import SnapshotReader
    import h5py

# default libraries
from dataclasses import dataclass
from pathlib import Path

# other packages
import numpy as np

# internal imports
from ..data_management import (
    ParticleStore,
    GroupStore,
    OctaviusConfig,
    OctaviusConstants,
    build_reader,
)
from ..data_management.pipeline_management import load_internals, Internals, PipelineStage
from ..log import get_logger, configure_logger

INTERNALS_FILEPATH = Path(__file__).parent.parent / "internals.yaml"

logger = get_logger()


def build_analyser(
    snapshot_path: Path,
    catalogue: OctaviusCatalogue,
    config: OctaviusConfig,
) -> OctaviusAnalyser:
    """
    Constructs an ``Analyser`` object for dynamic, on-the-fly analysis of groups in an Octavius catalogue
    via the existing runtime stages.

    Parameters
    ----------
    snapshot_path: pathlib.Path
        Path object pointing to the original, raw snapshot.
    catalogue: OctaviusCatalogue
        OctaviusCatalogue object corresponding to the snapshot.
    config: OctaviusConfig
        OctaviusConfig object.

    Returns
    -------
    analyser: OctaviusAnalyser
        An OctaviusAnalyser object.

    Notes
    -----
    The config can be modified from the one used to create the snapshot. Pipeline-specific parameters will be disregarded.
    """
    configure_logger(snapshot_path=snapshot_path, rank=0, output_level=config.terminal_output_level)

    # initialise data structures
    constants = OctaviusConstants(MU=config.MU, FRAD=config.FRAD)
    reader = build_reader(snapshot_path=snapshot_path, constants=constants, config=config)
    internals = load_internals(internals_filepath=INTERNALS_FILEPATH, config=config)

    return OctaviusAnalyser(
        reader=reader,
        catalogue=catalogue,
        config=config,
        constants=constants,
        internals=internals,
    )


@dataclass(frozen=True, slots=True)
class StageResult:
    """
    Contains the outputs of a pipeline stage when run in dynamic, on-the-fly mode.

    Attributes
    ----------
    group_type: str
        The type of group the result belongs to.
    group_indices: np.ndarray
        Indices into the catalogue for the groups.
    columns: dict[str, np.ndarray]
        The output columns.
    """

    group_type: str
    group_indices: np.ndarray
    columns: dict[str, np.ndarray]

    def __getitem__(self, key: str) -> np.ndarray:
        """
        Controls behaviour of StageResult["column_name"].
        """
        return self.columns[key]

    def __len__(self) -> int:
        """
        Returns the number of groups.
        """
        return len(self.group_indices)

    def keys(self) -> list[str]:
        """
        Returns a list of available output columns.
        """
        return list(self.columns.keys())


class OctaviusAnalyser:
    """
    Object-oriented interface for dynamic, on-the-fly analysis of groups in a snapshot from the corresponding
    Octavius catalogue.
    """

    def __init__(
        self,
        reader: SnapshotReader,
        catalogue: OctaviusCatalogue,
        config: OctaviusConfig,
        constants: OctaviusConstants,
        internals: Internals,
    ) -> None:

        self._reader = reader
        self._catalogue = catalogue
        self._config = config
        self._constants = constants
        self._internals = internals

        self._collections: dict[str, GroupCollection] = {}
        if catalogue.galaxies is not None:
            self._collections["galaxies"] = catalogue.galaxies
        if catalogue.haloes is not None:
            self._collections["haloes"] = catalogue.haloes

        if not self._collections:  # high-z snapshots
            raise ValueError("Catalogue contains no group data")

    def __repr__(self) -> str:
        """
        Controls behaviour of print(OctaviusAnalyser)
        """
        return f"<OctaviusAnalyser> | {self._reader.snapshot_path.stem} |"

    def _extract_groups(
        self,
        group_type: str,
        group_indices: np.ndarray,
        ptype_columns: dict[str, list[str]],
    ) -> dict[str, SubsetParticleData]:
        """
        Extracts groups from the catalogue and then their particle-level requisite data. Returns:

        - result: a dictionary of SubsetParticleData classes, keyed by ptype.
        """
        membership_group = self._collections[group_type]._data["membership"]  # NOTE: hardcoded, update if changing

        result: dict[str, SubsetParticleData] = {}

        for ptype, columns in ptype_columns.items():
            result[ptype] = self._extract_ptype(
                ptype=ptype, membership_group=membership_group, group_indices=group_indices, columns=columns
            )

        return result

    def _extract_ptype(
        self,
        ptype: str,
        membership_group: h5py.Group,
        group_indices: np.ndarray,
        columns: list[str],
    ) -> SubsetParticleData:
        """
        Extracts data for requested columns for the specified ptype. Returns:

        - result: a SubsetParticleData for the requested ptype.
        """
        indices_dataset = membership_group[f"{ptype}_indices"]  # slab-read indices (large)
        offsets = membership_group[f"{ptype}_offsets"][:]  # full read offsets

        starts = offsets[group_indices]
        ends = offsets[group_indices + 1]
        lengths = ends - starts  # to make code cleaner
        total_particles = np.sum(lengths)

        # allocate outputs
        particle_indices = np.empty(total_particles, dtype=np.int64)
        new_offsets = np.empty(len(group_indices) + 1, dtype=np.int64)  # not snapshot aligned

        # construct new offsets from old offsets masked by group indices
        new_offsets[0] = 0
        np.cumsum(lengths, out=new_offsets[1:])

        # loop over groups, collect their particles
        position = 0  # positional index
        for i in range(len(group_indices)):
            if lengths[i] > 0:  # need this because a group can miss a ptype
                particle_indices[position : position + lengths[i]] = indices_dataset[starts[i] : ends[i]]

            position += lengths[i]

        # sort indices first (must be increasing)
        sort_order = np.argsort(particle_indices, stable=True)
        sorted_indices = particle_indices[sort_order]

        column_data = self._reader.read_requested_columns(
            ptype=ptype,
            datasets=columns,
            sorted_snapshot_indices=sorted_indices,
        )

        # then unsort back to group order (necessary for sorted_idx in pipeline stages)
        unsort_order = np.argsort(sort_order)
        for col in column_data:
            column_data[col] = column_data[col][unsort_order]

        data = SubsetParticleData(columns=column_data, offsets=new_offsets)

        return data

    def _resolve_stage_columns(
        self,
        stage: PipelineStage,
        group_type: str,
    ) -> dict[str, list[str]]:
        """
        Stages require different columns for each ptype and not all columns exist for each ptype; this
        helper resolves this. Returns:

        - result: dict keyed by ptype containing the list of columns needed for that ptype.
        """
        group_config = self._internals.group_types[group_type]

        # discern which ptypes to access from what exists, what the stage needs, and what the user wants
        all_group_ptypes = group_config["ptypes"]  # from internals
        requested = {pt for pt, on in self._config.process_ptypes.items() if on}  # from config
        available = set(self._reader.available_ptypes())  # from raw snap
        valid_ptypes = [pt for pt in all_group_ptypes if pt in requested and pt in available]

        result: dict[str, list[str]] = {}
        # columns required by all ptypes (e.g. mass)
        shared_required = stage.needs_particle_columns.get("all", frozenset())
        shared_optional = stage.optional_particle_columns.get("all", frozenset())

        for ptype in valid_ptypes:
            # union the shared reqs with the ptype-specific needs
            columns = shared_required | stage.needs_particle_columns.get(ptype, frozenset())

            # union optional columns
            optional = shared_optional | stage.optional_particle_columns.get(ptype, frozenset())
            columns |= {col for col in optional if self._reader.has_dataset(ptype, col)}

            # ensure columns are loaded in the same order
            if columns:
                result[ptype] = sorted(columns)

        return result

    def _verify_dependencies(
        self,
        stage: PipelineStage,
        group_type: str,
    ) -> None:
        """
        Verifies the analysis can be done from what exists in the catalogue (stage dependencies).
        """
        data = self._collections[group_type]._data

        for dependent in stage.requires:
            dependent_label = self._internals.stages[dependent].label
            if f"properties/{dependent_label}" in data:
                continue
            else:
                raise KeyError(f"Dependent stage {dependent_label} is not present in the catalogue.")


@dataclass(frozen=True, slots=True)
class SubsetParticleData:
    """
    Container for a subset of snapshot particle data, to be passed to ParticleStores.
    """

    columns: dict[str, np.ndarray]
    offsets: np.ndarray


def _extract_results(
    group_store: GroupStore,
    group_type: str,
    group_indices: np.ndarray,
    pre_columns: set[str],
) -> StageResult:
    """
    Diffs the GroupStore from before & after the stage (and strips _-prefixed columns) to return the
    StageResult of outputs you would expect from the pipeline.
    """
    new_columns = set(group_store.columns.keys()) - pre_columns
    result_columns = {
        col: group_store[col]
        for col in new_columns
        if not col.startswith("_")  # convention in pipeline is to prefix non-HDF5 columns with _
    }

    return StageResult(
        group_type=group_type,
        group_indices=group_indices,
        columns=result_columns,
    )


def _build_particle_stores(
    subset_data: dict[str, SubsetParticleData], internals: Internals
) -> dict[str, ParticleStore]:
    """
    Constructs subset ParticleStores from the snapshot data stored in subset_data. Returns the
    particles dict, same as the pipeline.
    """
    particles: dict[str, ParticleStore] = {}

    for ptype, subset in subset_data.items():
        n_particles = subset.offsets[-1]
        is_baryonic = ptype in internals.baryonic_ptypes
        store = ParticleStore(ptype=ptype, n_particles=n_particles, is_baryonic=is_baryonic)

        for col_name, col_data in subset.columns.items():
            store[col_name] = col_data

        particles[ptype] = store

    return particles


def _build_group_store(
    group_type: str,
    group_indices: np.ndarray,
    subset_data: dict[str, SubsetParticleData],
    internals: Internals,
) -> GroupStore:

    group_config = internals.group_types[group_type]

    store = GroupStore(
        group_ids=group_indices,  # this works because IDs aren't really IDs, rather, indices
        group_key=group_config["key"],
        kind=group_config["kind"],
    )

    for ptype, subset in subset_data.items():
        sorted_idx = np.arange(subset.offsets[-1], dtype=np.int64)
        store.csr_membership[ptype] = (subset.offsets, sorted_idx)

    return store


def _resolve_linked_groups(
    collection: GroupCollection,
    group_idx: np.ndarray,
    linking_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolves groups for stages which require another group to be present (e.g. photometry needs halo gas).Returns:

    - unique_map_idx: unique indices of mapped groups which need to be loaded (e.g. field haloes)
    - multi_map_link: array where multi_map_link[g] links to mapped group (for representing many-to-one)
    """
    link_values = collection._data["membership"][linking_column][group_idx]

    unique_map_idx = np.unique(link_values)  # only load parent/child groups once
    unique_map_idx = unique_map_idx[unique_map_idx != -1]  # filter sentinel values

    # map from catalogue level to the subset we're running on
    subset_map = np.full(shape=(np.max(unique_map_idx) + 1), fill_value=-1, dtype=np.int64)
    subset_map[unique_map_idx] = np.arange(len(unique_map_idx), dtype=np.int64)
    multi_map_link = subset_map[link_values]  # multi-membership mapping (since we did np.unique)

    return unique_map_idx, multi_map_link


def _collect_all_dependencies(internals: Internals, stage: PipelineStage) -> list[str]:
    """
    The simple first half of pipeline_management/resolve_dependencies, to get all dependencies beyond
    the immediate.
    """
    visited: set[str] = set()
    stack = list(stage.requires)

    while stack:
        dep_name = stack.pop()
        if dep_name in visited:
            continue
        visited.add(dep_name)
        stack.extend(internals.stages[dep_name].requires)

    return list(visited)


def _preload_stage_dependencies(
    collection: GroupCollection,
    group_store: GroupStore,
    group_indices: np.ndarray,
    stage: PipelineStage,
    internals: Internals,
) -> None:
    """
    Load columns output by a stage's dependencies from the catalogue (file checks will run before this
    to raise an error if they don't exist)
    """
    all_deps = _collect_all_dependencies(internals=internals, stage=stage)

    for dep_name in all_deps:
        dep_label = internals.stages[dep_name].label
        dep_group = collection._data[f"properties/{dep_label}"]

        for col_name in dep_group:
            group_store[col_name] = dep_group[col_name][group_indices]
