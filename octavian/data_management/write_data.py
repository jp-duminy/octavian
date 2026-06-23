"""

Functions which write data from analysis stages into CSR format lists for HDF5 compatibility (and fast, straightforward access) and create output HDF5 files (per-rank currently).

"""

# type checking (semantic)
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_management import SimulationData, convert_data_manager

# octavian modules
from octavian.data_management.conventions import DTYPES # NOTE: import from within-file, not module level (to avoid circular import)

# others
import h5py
from pathlib import Path # NOTE: migrated fully to pathlib in v0.3
import numpy as np

GROUP_PTYPE_LISTS = {
    "halos":    ["glist", "slist", "dmlist", "bhlist"],
    "galaxies": ["glist", "slist", "bhlist"],
}

PLIST_TO_PTYPE = {
    "slist": "star",
    "glist": "gas",
    "bhlist": "bh",
    "dmlist": "dm",
}


HDF5_GROUP_NAMES = {
    "halos": "halo_data",
    "galaxies": "galaxy_data",
}

map_group_to_gid_name = {
    "halos": "haloID",
    "galaxies": "galaxyID",
}

def _resolve_columns(group_store, column: list[str]):
    """
    Small helper function for resolving vectors in columns (which are stored as separate entries).
    """
    if isinstance(column, list):
        return np.column_stack([group_store[c] for c in column])
    
    return group_store[column]

def construct_particle_csr_lists(data: SimulationData, config: dict) -> dict[str, dict[str, dict]]:
    """
    Extracts particle lists from SimulationData (matching GroupStore & ParticleStore) and converts them to the CSR format for hdf5.
    """
    result = {group: {} for group in data.groups}

    for group_name in GROUP_PTYPE_LISTS: # NOTE: sorts both halos & galaxies as opposed to previous function which took group_name

        sentinel = -1 if group_name == "galaxies" else 0 # REVIEW: fix this ideally?
        group_ID = config['groupIDs'][group_name]

        for ptype_list in GROUP_PTYPE_LISTS[group_name]: # let it be known this was a pain to write

            ptype = PLIST_TO_PTYPE[ptype_list]
            particle_group_ids = data.particles[ptype][group_ID]
            particle_indices = data.particles[ptype]["particle_index"] # positional index of particles in original snapshot

            # mask out non-group particles (technically redundant for halos) // sort
            mask = particle_group_ids != sentinel
            particle_group_ids = particle_group_ids[mask]
            particle_indices = particle_indices[mask]
            order = np.argsort(particle_group_ids)
            sorted_particle_group_ids = particle_group_ids[order] 
            sorted_indices = particle_indices[order]

            group_store = data.groups[group_name]

            if len(sorted_particle_group_ids) == 0:
                result[group_name][ptype_list] = {
                    "indices": np.array([], dtype=DTYPES["csr_indices"]),
                    "offsets": np.zeros(shape=group_store.n_groups, dtype=DTYPES["csr_offsets"]),
                    "lengths": np.zeros(shape=group_store.n_groups, dtype=DTYPES["csr_lengths"]),
                }
                continue

            breaks = np.flatnonzero(np.diff(sorted_particle_group_ids)) + 1 # +1 shifts index array right
            split_lengths = np.diff(np.concatenate([[0], breaks, [len(sorted_particle_group_ids)]])) # prepend/append position of first/last group ID
            split_ids = sorted_particle_group_ids[np.concatenate([[0], breaks])] # equivalent to np.unique

            lengths = np.zeros(shape=group_store.n_groups, dtype=DTYPES["csr_lengths"]) # a group can be empty for a certain ptype
            group_indices = group_store.get_indexer(group_id=split_ids)
            lengths[group_indices] = split_lengths

            offsets = np.concatenate([[0], np.cumsum(lengths[:-1])], dtype=DTYPES["csr_offsets"]) # shift array left (first offset is 0)

            group_slots = group_store.get_indexer(group_id=sorted_particle_group_ids)
            reorder = np.argsort(group_slots) # in case GroupStore order is not sorted
            indices = sorted_indices[reorder].astype(DTYPES["csr_indices"])

            result[group_name][ptype_list] = {
                "indices": indices,
                "offsets": offsets,
                "lengths": lengths,
            }
            
    return result


def write_analysis_to_output_file(data: SimulationData, particle_lists: dict, config: dict, output_file: Path) -> None:
    """
    Takes in the SimulationData object and writes it to a .hdf5 file.
    """
    if output_file.is_file(): # pathlib version of previous os logic
        output_file.unlink()

    with h5py.File(output_file, "w") as out:

        for group_name, hdf5_name in HDF5_GROUP_NAMES.items():
           
            if group_name not in data.groups:
                continue

            group_store = data.groups[group_name] # quickhand
            hdf5_group = out.create_group(hdf5_name)

            hdf5_group.create_dataset(name=f"{map_group_to_gid_name[group_name]}", data=group_store.group_ids, compression=1)

            for ptype in GROUP_PTYPE_LISTS[group_name]: # in theory these could be split up into different functions

                if ptype not in particle_lists[group_name]:
                    continue

                pl = particle_lists[group_name][ptype]
                hdf5_group.create_dataset(f'{ptype}_indices', data=pl['indices'], compression=1)
                hdf5_group.create_dataset(f'{ptype}_offsets', data=pl['offsets'], compression=1)
                hdf5_group.create_dataset(f'{ptype}_lengths', data=pl['lengths'], compression=1)

            for dataset_name, column_key in config['dataset_columns'].items():

                if dataset_name in GROUP_PTYPE_LISTS[group_name]:
                    continue

                if column_key not in GROUP_PTYPE_LISTS[group_name]:

                    if isinstance(column_key, list): # for 3D attributes (pos, vel)
                        if all(c in group_store.columns for c in column_key):
                            hdf5_group.create_dataset(dataset_name, data=_resolve_columns(group_store, column_key), compression=1)
                    else:
                        if column_key in group_store.columns:
                            hdf5_group.create_dataset(dataset_name, data=group_store[column_key], compression=1)