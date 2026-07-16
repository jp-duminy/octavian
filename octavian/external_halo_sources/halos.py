"""

Internal agnostic halo source infrastructure, for passing to the likewise-agnostic pipeline.

"""

from octavian.data_management import SnapshotReader
from dataclasses import dataclass
import numpy as np


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
    Basic subhalo information: the parent HaloID and the number of bound particles. For HBT HERONS this will include the TrackID (not provided by AHF).
    """

    parent_ids: np.ndarray
    n_bound: np.ndarray
    track_ids: np.ndarray | None = None  # only provided by HBT HERONS


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


class SnapshotHaloSource(HaloSource):
    """
    Raw snapshot halo sources (no external finder.) This means HaloIDs for GIZMO and FOFGroupIDs from SWIFT.
    """

    def __init__(self, reader: SnapshotReader):

        self.reader = reader

    def read_halo_ids(self, ptypes: list[str]) -> HaloAssignments:
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
