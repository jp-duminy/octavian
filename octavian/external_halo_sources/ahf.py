"""

Parser for AHF: Amiga's Halo Finder.

AHF paper: https://iopscience.iop.org/article/10.1088/0067-0049/182/2/608

NOTE: the AHF parser uses np.loadtxt with a numba parser on the resulting array. The call to loadtxt is hard-coded to the structure of an AHF catalogue, and the structure of these catalogues is somewhat finicky/not too user-friendly. Therefore the parser is quite exposed to any changes AHF makes to how they store their information. If catalogues take a long time to parse, a better parsing method would perhaps be welcome.

"""

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from octavian.data_management import SnapshotReader
    from mpi4py.MPI import Comm

import numpy as np
from pathlib import Path
from numba import njit
from functools import (
    cached_property,
)  # for avoiding rereading files across methods but also not holding too much in __init__

from .halo_data_structures import (
    HaloAssignments,
    SubhaloInformation,
    HaloSource,
    compute_depths,
    apply_lookup,
)

from octavian.data_management.parallel_reading import generate_slabs


class AHFCatalogue(NamedTuple):  # for code readability
    """
    Container for AHF catalogue info.
    """

    parent_indices: np.ndarray
    depths: np.ndarray
    n_particles: np.ndarray
    field_lookup: np.ndarray
    sub_lookup: np.ndarray
    field_of: np.ndarray


class AHFHaloSource(HaloSource):
    """
    AHF Amiga Halo Finder parser with object-oriented interface. Methods:

    - read_halo_ids: returns HaloAssignments object
    - read_subhalo_ids: returns SubhaloInformation object
    - distribute_raw_halo_ids: distributes slab-based HaloID info from rank 0 to other ranks
    - distribute_raw_subhalo_ids: distributes slab-based SubhaloID info from rank 0 to other ranks
    """

    def __init__(self, halos_path: Path, particles_path: Path, reader: SnapshotReader) -> None:

        self.halos_path = halos_path
        self.particles_path = particles_path
        self.reader = reader

    @cached_property  # cached_property allows AHFHaloSource to parse catalogues once while meeting the inheritance requirements of a HaloSource class
    def _halos_catalogue(self) -> AHFCatalogue:
        """
        Parses and stores AHF_halos file information, deriving raw ahf ids, parent indices, subhalo depths, and lookup arrays.
        """
        raw_ahf_ids, raw_host_ids, n_particles = parse_ahf_halos(self.halos_path)

        # you now need to remap the comically-large AHF ids (use indices instead)
        parent_indices = remap_ahf_ids(
            ahf_ids=raw_ahf_ids, raw_host_ids=raw_host_ids
        )  # this uses searchsorted so no need for the contiguous ID helper

        field_of = compute_field_index(parent_ids=parent_indices)
        depths = compute_depths(parent_ids=parent_indices)

        is_field = parent_indices == -1

        field_lookup = np.full(len(raw_ahf_ids), fill_value=-1, dtype=np.int64)
        field_lookup[is_field] = np.arange(is_field.sum(), dtype=np.int64)

        sub_lookup = np.full(len(raw_ahf_ids), fill_value=-1, dtype=np.int64)
        sub_lookup[~is_field] = np.arange((~is_field).sum(), dtype=np.int64)

        catalogue = AHFCatalogue(
            parent_indices=parent_indices,
            depths=depths,
            n_particles=n_particles,
            field_lookup=field_lookup,
            sub_lookup=sub_lookup,
            field_of=field_of,
        )

        return catalogue

    @cached_property
    def _particles(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Parses AHF_particles and deduplicates inclusive membership.
        Returns (unique_pids, field_halo_indices, deepest_halo_indices), sorted by unique_pids.
        """
        particles_array = np.loadtxt(self.particles_path, skiprows=1, dtype=np.int64)
        depths = self._halos_catalogue.depths

        pids, halo_indices = parse_ahf_particles(ahf_particle_array=particles_array, n_halos=len(depths))

        return deduplicate_ahf_particles(pids=pids, halo_indices=halo_indices, depths=depths)

    def read_halo_ids(self, ptypes: list[str]) -> HaloAssignments:
        """
        Interfaces with the provided SnapshotReader and parses the .AHF_halos/AHF_particles files to match snapshot particle IDs with their AHF equivalents and produce the final assignments made by AHF. Returns:

        - HaloAssignments dataclass.
        """
        catalogue = self._halos_catalogue
        unique_pids, deepest_halo_indices = self._particles

        halo_assignments: dict[str, np.ndarray] = {}
        subhalo_assignments: dict[str, np.ndarray] = {}

        for ptype in ptypes:
            snapshot_pids = self.reader.read_particle_ids(ptype=ptype)

            positional_hids, positional_subhids = match_ahf_particle_ids(
                snapshot_pids=snapshot_pids,
                unique_ahf_pids=unique_pids,
                field_of=catalogue.field_of,
                deepest_halo_indices=deepest_halo_indices,
                depths=catalogue.depths,
            )

            halo_assignments[ptype] = apply_lookup(ids=positional_hids, lookup=catalogue.field_lookup)
            subhalo_assignments[ptype] = apply_lookup(ids=positional_subhids, lookup=catalogue.sub_lookup)

        n_total_halos = int((catalogue.depths == 0).sum())

        sub_info = self.read_subhalo_info()
        for ptype, sub_ids in subhalo_assignments.items():
            in_sub = sub_ids != -1
            assert np.array_equal(sub_info.host_halo_ids[sub_ids[in_sub]], halo_assignments[ptype][in_sub]), (
                f"{ptype}: particle HaloID disagrees with its subhalo's host tree."
            )

        return HaloAssignments(halo_ids=halo_assignments, n_total_halos=n_total_halos, subhalo_ids=subhalo_assignments)

    def read_subhalo_info(self) -> SubhaloInformation:
        """
        Uses the supplied AHF catalogues to map out hierarchy and parent pointers for subhaloes. Returns:

        - SubhaloInformation dataclass
        """
        catalogue = self._halos_catalogue

        sub_mask = catalogue.depths > 0

        # parents may be field halos (depth-1 subs) or other subhalos (deeper): remap each namespace
        sub_parents = catalogue.parent_indices[sub_mask]
        parent_is_field = catalogue.depths[sub_parents] == 0

        parent_index = np.where(parent_is_field, -1, catalogue.sub_lookup[sub_parents])
        host_halo_ids = catalogue.field_lookup[catalogue.field_of[sub_mask]]

        return SubhaloInformation(
            host_halo_ids=host_halo_ids,
            parent_index=parent_index,
            global_index=np.arange(sub_mask.sum(), dtype=np.int64),
            depth=catalogue.depths[sub_mask],
            n_bound=catalogue.n_particles[sub_mask],
        )

    def distribute_raw_halo_ids(
        self,
        slabs: dict[str, slice],
        comm: Comm | None,
        global_ids: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Distributes rank 0's raw halo ID information to the other ranks so they know the HaloIDs of the particles on their slab. Returns:

        - local_halo_ids: dict keyed per-ptype containing the rank's HaloID arrays

        # TODO: can move to comm.Scatter on MPI-4 architectures rather than sequential send/receive.
        """
        if comm is None or comm.size == 1:
            return {ptype: ids[slabs[ptype]].copy() for ptype, ids in global_ids.items()}

        rank = comm.Get_rank()
        size = comm.Get_size()
        local_halo_ids: dict[str, np.ndarray] = {}

        if rank == 0:
            all_slabs = {
                dest: generate_slabs(rank=dest, n_ranks=size, particle_counts=self.reader.particle_counts)
                for dest in range(size)
            }

        for ptype_index, ptype in enumerate(
            global_ids if rank == 0 else slabs
        ):  # NOTE: done with send/receive as comm.scatter is capped to < int32 elements (https://github.com/PyLops/pylops-mpi/issues/115) kept for backwards-compatibility with MPI-3 as I am unsure what versions clusters use
            slab = slabs[ptype]
            slab_length = slab.stop - slab.start

            if rank == 0:
                local_halo_ids[ptype] = global_ids[ptype][slab].copy()  # .copy() instead of sending to itself

                for dest in range(1, size):
                    comm.Send(global_ids[ptype][all_slabs[dest][ptype]], dest=dest, tag=ptype_index)
            else:
                memory_block = np.empty(slab_length, dtype=np.int64)
                comm.Recv(memory_block, source=0, tag=ptype_index)
                local_halo_ids[ptype] = memory_block

        return local_halo_ids

    def distribute_raw_subhalo_ids(
        self,
        slabs: dict[str, slice],
        comm: Comm | None,
        global_subhalo_ids: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Wrapper around distribute_raw_halo_ids, but for subhalos (made so SnapshotHaloSource and AHFHaloSource can match). Returns:

        - local_subhalo_ids: dict keyed per-ptype containing the rank's SubhaloID arrays
        """
        return self.distribute_raw_halo_ids(slabs=slabs, comm=comm, global_ids=global_subhalo_ids)


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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolves the duplicate appearances of particles in AHF halos by assigning them to their deepest membership, which follows the convention of then propagating child subhalo membership into parents; returns a tuple of unique particle ids, field halo indices, and (deepest) subhalo indices.
    """
    particle_depths = depths[halo_indices]

    sort_order = np.lexsort(
        (particle_depths, pids)
    )  # NOTE: lexsort sorts by second key first (so pids then depths, giving you depth-first pid appearance)

    sorted_pids = pids[sort_order]
    sorted_halo_indices = halo_indices[sort_order]

    last_appearance = np.empty(len(sorted_pids), dtype=np.bool_)
    last_appearance[-1] = True
    last_appearance[:-1] = sorted_pids[:-1] != sorted_pids[1:]

    return sorted_pids[last_appearance], sorted_halo_indices[last_appearance]


def match_ahf_particle_ids(
    snapshot_pids: np.ndarray,
    unique_ahf_pids: np.ndarray,
    field_of: np.ndarray,
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
    deepest = deepest_halo_indices[matched_insertion]

    aligned_hids[matched] = field_of[deepest]
    aligned_subhids[matched] = np.where(depths[deepest] > 0, deepest, -1)

    return aligned_hids, aligned_subhids


def remap_ahf_ids(ahf_ids: np.ndarray, raw_host_ids: np.ndarray) -> np.ndarray:
    """
    Map the comically-large AHF IDs to positional indices (necessary otherwise you will allocate an unfathomably large array of several exobytes).
    """
    sort_order = np.argsort(ahf_ids, stable=True)
    sorted_ids = ahf_ids[sort_order]
    is_field = raw_host_ids == 0
    parent_indices = np.full(len(ahf_ids), fill_value=-1, dtype=np.int64)

    insertion_idx = np.searchsorted(
        sorted_ids, raw_host_ids[~is_field]
    )  # position within the sorted array gives ascending unique sensible IDs
    parent_indices[~is_field] = sort_order[insertion_idx]

    return parent_indices


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
