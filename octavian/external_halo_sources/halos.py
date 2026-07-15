"""

Internal agnostic halo source infrastructure, for passing to the likewise-agnostic pipeline.

"""

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True, frozen=True)
class HaloAssignments:
    """
    Dictionaries of per-ptype HaloID/SubhaloID assignments aligned with the original snapshot. These use the internal -1 sentinel.
    """

    halo_ids: dict[str, np.ndarray]
    subhalo_ids: dict[str, np.ndarray] | None = None


@dataclass(slots=True, frozen=True)
class SubhaloInformation:
    """
    Basic subhalo information: the parent HaloID and the number of bound particles. For HBT HERONS this will include the TrackID (not provided by AHF).
    """

    parent_ids: np.ndarray
    n_bound: np.ndarray
    track_ids: np.ndarray | None = None  # only provided by HBT HERONS


def make_halo_ids_contiguous(halo_ids: np.ndarray) -> np.ndarray:
    """
    Returns halo_ids remapped to a contiguous array; assumes the halo_ids array has had its sentinel handled appropriately.
    """
    valid = halo_ids != -1  # NOTE: assumes Octavian -1 sentinel has already been handled
    unique_ids = np.unique(halo_ids[valid])
    remapped_ids = np.full(shape=halo_ids, fill_value=-1)
    remapped_ids[valid] = np.searchsorted(unique_ids, halo_ids[valid])

    return remapped_ids
