"""

Parser for AHF: Amiga's Halo Finder.

AHF paper: https://iopscience.iop.org/article/10.1088/0067-0049/182/2/608

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_management import SnapshotReader

import numpy as np
from pathlib import Path
from numba import njit
from functools import (
    cached_property,
)  # for avoiding rereading files across methods but also not holding too much in __init__

from .halos import (
    HaloAssignments,
    SubhaloInformation,
    HaloSource,
    build_contiguous_id_lookup,
    compute_depths,
    apply_lookup,
)


class AHFHaloSource(HaloSource):
    def __init__(self, halos_path: Path, particles_path: Path, reader: SnapshotReader):

        self.halos_path = halos_path
        self.particles_path = particles_path
        self.reader = reader

    @cached_property
    def _halos_catalogue(self) -> tuple[np.ndarray, ...]:
        """
        Parses and stores AHF_halos file information, deriving raw ahf ids, parent indices, subhalo depths, and lookup arrays.
        """
        ahf_ids, raw_host_ids, n_particles = parse_ahf_halos(self.halos_path)

        id_lookup = build_contiguous_id_lookup(ids=ahf_ids)
        parent_indices = np.where(raw_host_ids == 0, -1, id_lookup[raw_host_ids])
        depths = compute_depths(parent_ids=parent_indices)

        is_field = parent_indices == -1

        field_lookup = np.full(len(ahf_ids), fill_value=-1, dtype=np.int64)
        field_lookup[is_field] = np.arange(is_field.sum(), dtype=np.int64)

        sub_lookup = np.full(len(ahf_ids), fill_value=-1, dtype=np.int64)
        sub_lookup[~is_field] = np.arange((~is_field).sum(), dtype=np.int64)

        return ahf_ids, parent_indices, depths, n_particles, field_lookup, sub_lookup

    @cached_property
    def _particles(self) -> tuple[np.ndarray, ...]:
        """
        Parses AHF_particles and deduplicates inclusive membership.
        Returns (unique_pids, field_halo_indices, deepest_halo_indices), sorted by unique_pids.
        """
        particles_array = np.loadtxt(self.particles_path, skiprows=1, dtype=np.int64)
        _, _, depths, _, _, _ = self._halos_catalogue

        pids, halo_indices = parse_ahf_particles(ahf_particle_array=particles_array, n_halos=len(depths))

        return deduplicate_ahf_particles(pids=pids, halo_indices=halo_indices, depths=depths)

    def read_halo_ids(self, ptypes: list[str]) -> HaloAssignments:
        """
        Returns a HaloAssignments dataclass containing the assignments made by AHF.
        """
        _, _, depths, _, field_lookup, sub_lookup = self._halos_catalogue
        unique_pids, field_halo_indices, deepest_halo_indices = self._particles

        halo_assignments: dict[str, np.ndarray] = {}
        subhalo_assignments: dict[str, np.ndarray] = {}

        for ptype in ptypes:
            snapshot_pids = self.reader.read_particle_ids(ptype=ptype)

            positional_hids, positional_subhids = match_ahf_particle_ids(
                snapshot_pids=snapshot_pids,
                unique_ahf_pids=unique_pids,
                field_halo_indices=field_halo_indices,
                deepest_halo_indices=deepest_halo_indices,
                depths=depths,
            )

            halo_assignments[ptype] = apply_lookup(ids=positional_hids, lookup=field_lookup)
            subhalo_assignments[ptype] = apply_lookup(ids=positional_subhids, lookup=sub_lookup)

        n_total_halos = int((depths == 0).sum())

        return HaloAssignments(halo_ids=halo_assignments, n_total_halos=n_total_halos, subhalo_ids=subhalo_assignments)

    def read_subhalo_info(self) -> SubhaloInformation:
        """
        Returns the AHF subhalo information, sliced to subhalo-only rows in contiguous SubhaloID order.
        """
        _, parent_indices, depths, n_particles, field_lookup, sub_lookup = self._halos_catalogue

        sub_mask = depths > 0

        # parents may be field halos (depth-1 subs) or other subhalos (deeper): remap each namespace
        sub_parents = parent_indices[sub_mask]
        parent_is_field = depths[sub_parents] == 0

        parent_index = np.where(parent_is_field, -1, sub_lookup[sub_parents])
        host_halo_ids = field_lookup[compute_field_index(parent_ids=parent_indices)[sub_mask]]

        return SubhaloInformation(
            host_halo_ids=host_halo_ids,
            parent_index=parent_index,
            depth=depths[sub_mask],
            n_bound=n_particles[sub_mask],
        )


def parse_ahf_halos(ahf_halos_path: Path) -> tuple[np.ndarray, ...]:
    """
    Parses a .AHF_halos file, returning a tuple of (ahf_ids, raw_host_ids, n_particles) (all raw AHF data).
    """
    #  col0=ID, col1=hostHalo, col4=n_particles; includes a header; tab-delimited
    ahf_ids, raw_host_ids, n_particles = np.loadtxt(
        fname=ahf_halos_path, dtype=np.int64, usecols=[0, 1, 4], skiprows=1, delimiter="\t", unpack=True
    )

    return ahf_ids, raw_host_ids, n_particles


@njit
def parse_ahf_particles(ahf_particle_array: np.ndarray, n_halos: int) -> tuple[np.ndarray, ...]:
    """
    Iterates on an .AHF_particles file which has been converted into an (n, 2) array by np.loadtxt, returning a tuple of (pids, halo_ids).
    """
    n_particles = len(ahf_particle_array) - n_halos

    pids = np.empty(n_particles, dtype=np.int64)
    halo_indices = np.empty(n_particles, dtype=np.int64)

    current_halo = -1
    write_idx = 0
    # these appear in a columnar structure where for each halo you have an initial n_particles | HaloID row followed by rows of PID | hostID till next halo

    for row_idx in range(len(ahf_particle_array)):
        if (
            ahf_particle_array[row_idx, 1] > 5
        ):  # maximum ptype is 5 (GIZMO convention), so assuming this doesn't change we know we hit a halo
            current_halo += 1  # NOTE: AHF halos also have comically large IDs (6 quintillion or so)

        else:
            pids[write_idx] = ahf_particle_array[row_idx, 0]
            halo_indices[write_idx] = current_halo
            write_idx += 1

    return pids, halo_indices


def deduplicate_ahf_particles(
    pids: np.ndarray,
    halo_indices: np.ndarray,
    depths: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Resolves the duplicate appearances of particles in AHF halos by assigning them to their deepest membership, which follows the convention of then propagating child subhalo membership into parents; returns a tuple of unique particle ids, field halo indices, and (deepest) subhalo indices.
    """
    particle_depths = depths[halo_indices]

    sort_order = np.lexsort(
        (particle_depths, pids)
    )  # NOTE: lexsort sorts by second key first (so pids then depths, giving you depth-first pid appearance)

    sorted_pids = pids[sort_order]
    sorted_halo_indices = halo_indices[sort_order]

    first_appearance = np.empty(len(sorted_pids), dtype=np.bool_)
    first_appearance[0] = True
    first_appearance[1:] = sorted_pids[1:] != sorted_pids[:-1]

    last_appearance = np.empty(len(sorted_pids), dtype=np.bool_)
    last_appearance[-1] = True
    last_appearance[:-1] = sorted_pids[:-1] != sorted_pids[1:]

    return sorted_pids[last_appearance], sorted_halo_indices[first_appearance], sorted_halo_indices[last_appearance]


def match_ahf_particle_ids(
    snapshot_pids: np.ndarray,
    unique_ahf_pids: np.ndarray,
    field_halo_indices: np.ndarray,
    deepest_halo_indices: np.ndarray,
    depths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Matches ahf particle ids to their counterparts in the raw snapshot, returning an array of HaloIDs and SubhaloIDs.
    """
    insertion_idx = np.searchsorted(unique_ahf_pids, snapshot_pids)
    insertion_idx = np.minimum(
        insertion_idx, len(unique_ahf_pids) - 1
    )  # prevent OOB when max(snapshot_pids) > max(ahf_pids)
    matched = unique_ahf_pids[insertion_idx] == snapshot_pids

    aligned_hids = np.full(shape=len(snapshot_pids), fill_value=-1, dtype=np.int64)
    aligned_subhids = np.full(shape=len(snapshot_pids), fill_value=-1, dtype=np.int64)

    matched_insertion = insertion_idx[matched]

    aligned_hids[matched] = field_halo_indices[matched_insertion]

    deepest = deepest_halo_indices[matched_insertion]
    aligned_subhids[matched] = np.where(depths[deepest] > 0, deepest, -1)

    return aligned_hids, aligned_subhids


@njit
def compute_field_index(parent_ids: np.ndarray) -> np.ndarray:
    """
    Follows the same logic as compute_depths but instead returns the index of the field halo.
    """
    n_halos = len(parent_ids)
    field_index = np.empty(n_halos, dtype=np.int64)

    for halo_idx in range(n_halos):
        current = halo_idx
        while parent_ids[current] != -1:
            current = parent_ids[current]
        field_index[halo_idx] = current

    return field_index
