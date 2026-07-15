"""

Machinery for reading and parsing the SWIFT-native subhalo finder HBT-HERONS. This draws on similar methods to what you will find in their toolbox folder.

HERONS outputs a number split of SubSnap files owing to MPI. In their source code, toolbox/catalogue_cleanup/SortCatalogues.py can combine these into the format Octavian supports. The reason we do not support both formats is because variable-length HDF5 reads caused painful nightmares in early Octavian development and HERONS is post-processing this for us.

HBT-HERONS website: https://hbt-herons.strw.leidenuniv.nl/
HBT-HERONS source code: https://github.com/SWIFTSIM/HBT-HERONS
HBT-HERONS source paper: https://academic.oup.com/mnras/article/543/2/1339/8250004
HBT algorithm source paper: https://academic.oup.com/mnras/article/474/1/604/4566529

"""

import h5py
import numpy as np
from pathlib import Path


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
        particle_offsets = catalogue["Subhalos/ParticleOffset"][:]

    return particle_ids, particle_offsets


def read_subsnap_properties(catalogue_path: Path) -> tuple[np.ndarray, ...]:
    """
    Returns a tuple of TrackID, HostHaloID and nbound arrays from the merged HERONS catalogue.

    NOTE: n_bound == 0 for orphan particles and is not filtered here.
    """
    with h5py.File(catalogue_path, "r") as catalogue:
        track_ids = catalogue["Subhalos/TrackId"][:]
        host_halo_ids = catalogue["Subhalos/HostHaloId"][:]
        n_bound = catalogue["Subhalos/Nbound"][:]

    return track_ids, host_halo_ids, n_bound


def resolve_subsnap_paths(catalogue_dir: Path, snap_nr: int) -> Path:
    """
    Returns a Path object pointing to the sorted HERONS catalogue. To produce this catalogue you must please run HBT-HERONS/toolbox/catalogue_cleanup/SortCatalogues.py, and run it with the --with-particles flag for Octavian.
    """
    pattern = f"**/OrderedSubSnap_{snap_nr}.hdf5"
    matches = sorted(catalogue_dir.glob(pattern))

    if not matches:
        raise FileNotFoundError(f"Could not locate the HERONS catalogue for snapshot {snap_nr} in {catalogue_dir}.")

    if len(matches) > 1:
        raise FileNotFoundError(f"{matches} output catalogues found in {catalogue_dir}, please check the directory.")

    return matches[0]
