"""

Internal agnostic halo source infrastructure, for passing to the likewise-agnostic pipeline.

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_management import SnapshotReader, OctavianConfig
    from mpi4py.MPI import Comm

# default libraries
from dataclasses import dataclass

# other packages
import numpy as np
from numba import njit

# internal imports
from octavian.log import get_logger

logger = get_logger()


@dataclass(slots=True, frozen=True)
class HaloAssignments:
    """
    Dictionaries of per-ptype HaloID/SubhaloID assignments aligned with the original snapshot. These use the internal -1 sentinel.
    """

    halo_ids: dict[str, np.ndarray]
    n_total_halos: int
    subhalo_ids: dict[str, np.ndarray] | None = None


@dataclass(slots=True, frozen=True)
class SubhaloInformation:
    """
    Basic subhalo information.

    - host_halo_ids: top-level HaloID
    - parent_index: immediate parent subhalo index
    - depth: the level of nestage, always >=1
    - n_bound: the number of bound particles (inclusive)
    """

    host_halo_ids: np.ndarray  # top-level HaloID
    parent_index: np.ndarray  # immediate parent subhalo index
    depth: np.ndarray  # >= 1
    global_index: np.ndarray
    n_bound: np.ndarray


def build_halo_source(config: OctavianConfig, reader: SnapshotReader) -> HaloSource:
    """
    Construct the (Sub)HaloID parser based on what was requested in the config. Returns:

    - HaloSource corresponding to the config source.
    """
    if config.halo_id_source == "SNAPSHOT":
        logger.info("Using snapshot-assigned HaloIDs.")
        return SnapshotHaloSource(reader=reader)
    elif config.halo_id_source == "AHF":
        from .ahf import AHFHaloSource  # I had to stick this in here to avoid a circular import

        prefix = config.halo_id_filepath  # renamed for explicitness
        logger.info("Using AHF-assigned HaloIDs.")
        logger.info(f"Finding AHF catalogues at {prefix} .")
        return AHFHaloSource(
            halos_path=prefix.with_suffix(".AHF_halos"),
            particles_path=prefix.with_suffix(".AHF_particles"),
            reader=reader,
        )
    else:
        raise ValueError("Unknown halo ID source, please check config?")


def build_contiguous_id_lookup(ids: np.ndarray) -> np.ndarray:
    """
    Thin wrapper around np.searchsorted which returns an array of the indices which would remap a (sub)halo_id array to be contiguous
    """
    unique_ids = np.unique(ids[ids != -1])
    lookup = np.full(shape=unique_ids.max() + 1, fill_value=-1, dtype=np.int64)
    lookup[unique_ids] = np.arange(len(unique_ids), dtype=np.int64)

    return lookup


class HaloSource:
    def read_halo_ids(self, ptypes: list[str]) -> HaloAssignments:
        """
        Reads particles in raw snapshot their HaloIDs based on the conventions and quirks of the source implementation; returns a HaloAssignments dataclass.
        """
        raise NotImplementedError

    def read_subhalo_info(self) -> SubhaloInformation | None:
        """
        Reads subhalo information from the source, if this exists.
        """
        return None

    def distribute_raw_halo_ids(
        self,
        slabs: dict[str, slice],
        comm: Comm | None,
        global_ids: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Distributes the global raw IDs (MPI).
        """
        raise NotImplementedError

    def distribute_raw_subhalo_ids(
        self,
        slabs: dict[str, slice],
        comm: Comm | None,
        global_subhalo_ids: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray] | None:
        """
        Distributes the global raw subhalo IDs (MPI).
        """
        raise NotImplementedError


class SnapshotHaloSource(HaloSource):
    """
    Raw snapshot halo sources (no external finder.) This means HaloIDs for GIZMO and FOFGroupIDs from SWIFT.
    """

    def __init__(self, reader: SnapshotReader):

        self.reader = reader

    def read_halo_ids(self, ptypes: list[str], comm=None, global_ids=None) -> HaloAssignments:
        """
        Iterates over ptypes with the reader to produce HaloAssignments.
        """
        halo_ids: dict[str, np.ndarray] = {}

        for pt in ptypes:
            halo_ids[pt] = self.reader.read_halo_ids(ptype=pt)  # these are already contiguous

        max_id = max(int(ids.max()) for ids in halo_ids.values() if len(ids) > 0)
        n_total_halos = max_id + 1 if max_id >= 0 else 0  # derived, not read (could read from the actual field)

        return HaloAssignments(
            halo_ids=halo_ids, n_total_halos=n_total_halos, subhalo_ids=None
        )  # on-the-fly currently does not do subhalos

    def distribute_raw_halo_ids(self, slabs: dict[str, slice], comm=None, global_ids=None) -> dict[str, np.ndarray]:
        """
        Iterates over ptypes to produce slabs of HaloIDs per-ptype.
        """
        raw_ids: dict[str, np.ndarray] = {}

        for pt in self.reader.available_ptypes():
            raw_ids[pt] = self.reader.read_halo_ids(ptype=pt, slab=slabs[pt])  # these are already contiguous

        return raw_ids

    def distribute_raw_subhalo_ids(self, slabs: dict[str, slice], comm=None, global_subhalo_ids=None) -> None:
        """
        Exists for compatibility with AHF.
        """
        return None


@njit
def compute_depths(parent_ids: np.ndarray, max_allowed_depth: int = 15) -> np.ndarray:
    """
    Uses the parent_id array to return a corresponding depths array where depth=1 means its immediate parent is the field halo.
    """
    n_subhalos = len(parent_ids)
    depths = np.empty(n_subhalos, dtype=np.int64)

    for idx in range(n_subhalos):
        current = idx
        depth = 0

        while parent_ids[current] >= 0:
            current = parent_ids[current]
            depth += 1

            if depth > max_allowed_depth:
                raise ValueError("Subhalo depth loop has entered a recursion (cyclic relationship somewhere in data?)")

        depths[idx] = depth

    return depths


def apply_lookup(ids: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    """
    Small helper to use the global id lookup with the ptype id arrays.
    """
    result = np.full_like(ids, fill_value=-1)
    matched = ids != -1  # there will be unmatched particles and [-1] on an ndarray picks last element, so must mask
    result[matched] = lookup[ids[matched]]
    return result
