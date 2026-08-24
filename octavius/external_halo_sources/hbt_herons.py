"""

Machinery for reading and parsing the SWIFT-native subhalo finder HBT-HERONS. This draws on similar methods to what you will find in their toolbox folder.

HERONS outputs a number split of SubSnap files owing to MPI. In their source code, toolbox/catalogue_cleanup/SortCatalogues.py can combine these into the format Octavius supports. The reason we do not support both formats is because variable-length HDF5 reads caused painful nightmares in early Octavius development and HERONS is post-processing this for us.

HBT-HERONS website: https://hbt-herons.strw.leidenuniv.nl/
HBT-HERONS source code: https://github.com/SWIFTSIM/HBT-HERONS
HBT-HERONS source paper: https://academic.oup.com/mnras/article/543/2/1339/8250004
HBT algorithm source paper: https://academic.oup.com/mnras/article/474/1/604/4566529

# NOTE: this file is currently outdated and does not match codebase conventions yet, as I am missing a HERONS catalogue to verify information against. The filenames and general structure are built with reference to HERONS' toolbox folder and SOAP's catalogue_readers/read_hbtplus.py file. Once these become available, it should be fairly easy to get this code working (the infrastructure is there already).

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import SnapshotReader

# default libraries
from pathlib import Path
from functools import (
    cached_property,
)  # for avoiding rereading files across methods but also not holding too much in __init__

# other packages
import h5py
import numpy as np

# internal imports
from .halo_data_structures import (
    HaloAssignments,
    SubhaloInformation,
    HaloSource,
    build_contiguous_id_lookup,
    apply_lookup,
)


class HeronsHaloSource(HaloSource):
    """
    HBT-HERONS halo ID source; only compatible with SWIFT, and requires SWIFT halo FOF groups to be identified.
    """

    def __init__(self, catalogue_dir: Path, snap_nr: int, reader: SnapshotReader) -> None:

        self.catalogue_path = resolve_subsnap_paths(catalogue_dir=catalogue_dir, snap_nr=snap_nr)
        self.reader = reader

    @cached_property
    def _properties(self) -> tuple[np.ndarray, ...]:
        """
        All subsnap file properties.
        """
        return read_subsnap_properties(catalogue_path=self.catalogue_path)

    @cached_property
    def hid_lookup(self) -> np.ndarray:
        """
        Returns global HostHaloID lookup array.
        """
        _, host_halo_ids, _ = self._properties
        return build_contiguous_id_lookup(
            ids=host_halo_ids
        )  # need a global lookup; not all hids are necessarily found in each ptype

    @cached_property
    def subhid_lookup(self) -> np.ndarray:
        """
        Returns global TrackID lookup array.
        """
        track_ids, _, _ = self._properties
        return build_contiguous_id_lookup(ids=track_ids)

    def read_halo_ids(self, ptypes: list[str]) -> HaloAssignments:
        """
        Returns a HaloAssignments dataclass containing the assignments made by HERONS.
        """
        herons_pids, offsets = read_subsnap_particles(catalogue_path=self.catalogue_path)
        order = np.argsort(
            herons_pids, stable=True
        )  # this is best precomputed otherwise you argsort the pid array 4 times in the merge
        sorted_herons_pids = herons_pids[order]

        track_ids, host_halo_ids, _ = self._properties

        halo_assignments, subhalo_assignments = {}, {}

        for (
            ptype
        ) in ptypes:  # to sort the ids we first need all ptypes present to cover all HaloIDs to avoid desync issues
            snapshot_pids = self.reader.read_particle_ids(ptype=ptype)

            halo_ids, subhalo_ids = match_particle_ids(
                snapshot_pids=snapshot_pids,
                sorted_herons_pids=sorted_herons_pids,
                order=order,
                particle_offsets=offsets,
                track_ids=track_ids,
                host_halo_ids=host_halo_ids,
            )

            halo_assignments[ptype] = apply_lookup(
                ids=halo_ids, lookup=self.hid_lookup
            )  # handles unmatched particles' sentinels
            subhalo_assignments[ptype] = apply_lookup(ids=subhalo_ids, lookup=self.subhid_lookup)

        return HaloAssignments(
            halo_ids=halo_assignments, subhalo_ids=subhalo_assignments
        )  # FIXME: now 4 fields on HaloAssignments

    def read_subhalo_info(self) -> SubhaloInformation:
        """
        Reads the HERONS subhalo information.
        """
        track_ids, host_halo_ids, n_bound = self._properties
        return SubhaloInformation(
            host_halo_ids=apply_lookup(ids=host_halo_ids, lookup=self.hid_lookup),
            track_ids=track_ids,
            n_bound=n_bound,
        )  # FIXME: now 6 fields on SubhaloInformation


def match_particle_ids(
    snapshot_pids: np.ndarray,
    sorted_herons_pids: np.ndarray,
    order: np.ndarray,
    particle_offsets: np.ndarray,
    track_ids: np.ndarray,
    host_halo_ids: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Matches the raw snapshot PIDs with the HERONS-assigned IDs, returning a tuple of particle (HaloIDs, SubhaloIDs) aligned with the raw snapshot PIDs.
    """
    insertion_idx = np.searchsorted(sorted_herons_pids, snapshot_pids)
    insertion_idx = np.minimum(
        insertion_idx, len(sorted_herons_pids) - 1
    )  # prevent OOB index when max(snapshot_pids) > max(herons_pids)
    matched = sorted_herons_pids[insertion_idx] == snapshot_pids

    aligned_hids = np.full(shape=len(snapshot_pids), fill_value=-1, dtype=np.int64)
    aligned_subhids = np.full(shape=len(snapshot_pids), fill_value=-1, dtype=np.int64)

    herons_idx = order[
        insertion_idx[matched]
    ]  # RHS fancy indexing maps matched, which is sorted, to unsorted id arrays
    subhalo_idx = (
        np.searchsorted(particle_offsets, herons_idx, side="right") - 1
    )  # use side=right to match [offsets[i]:offsets[i+1]]; -1 for the subhalo itself

    aligned_hids[matched] = host_halo_ids[subhalo_idx]
    aligned_subhids[matched] = track_ids[subhalo_idx]

    return aligned_hids, aligned_subhids


def read_subsnap_particles(catalogue_path: Path) -> tuple[np.ndarray, ...]:
    """
    Returns a tuple of ParticleIDs and ParticleOffsets from the merged HERONS catalogue.
    """
    with h5py.File(catalogue_path, "r") as catalogue:
        if "Particles" not in catalogue:
            raise FileNotFoundError(
                f"{catalogue_path} does not contain particle info, please run HERONS' SortedCatalogues.py with --with-particles."
            )

        particle_ids = catalogue["Particles/ParticleIDs"][:]
        particle_offsets = catalogue["Subhaloes/ParticleOffset"][:]

    return particle_ids, particle_offsets


def read_subsnap_properties(catalogue_path: Path) -> tuple[np.ndarray, ...]:
    """
    Returns a tuple of TrackID, HostHaloID and nbound arrays from the merged HERONS catalogue.

    NOTE: n_bound == 0 for orphan particles and is not filtered here.
    """
    with h5py.File(catalogue_path, "r") as catalogue:
        track_ids = catalogue["Subhaloes/TrackId"][:]
        host_halo_ids = catalogue["Subhaloes/HostHaloId"][:]
        n_bound = catalogue["Subhaloes/Nbound"][:]

    return track_ids, host_halo_ids, n_bound


def resolve_subsnap_paths(catalogue_dir: Path, snap_nr: int) -> Path:
    """
    Returns a Path object pointing to the sorted HERONS catalogue. To produce this catalogue you must please run HBT-HERONS/toolbox/catalogue_cleanup/SortCatalogues.py, and run it with the --with-particles flag for Octavius.
    """
    pattern = f"**/OrderedSubSnap_{snap_nr}.hdf5"
    matches = sorted(catalogue_dir.glob(pattern))

    if not matches:
        raise FileNotFoundError(f"Could not locate the HERONS catalogue for snapshot {snap_nr} in {catalogue_dir}.")

    if len(matches) > 1:
        raise FileNotFoundError(f"{matches} output catalogues found in {catalogue_dir}, please check the directory.")

    return matches[0]
