"""

Functions which write data from analysis stages into CSR format lists for HDF5 compatibility (and fast, straightforward access) and create output HDF5 files (per-rank currently).

"""

# type checking (semantic)
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_management import SimulationData, Internals

# octavian modules
from octavian.data_management.conventions import DTYPES # NOTE: import from within-file, not module level (to avoid circular import)

# others
import h5py
from pathlib import Path # NOTE: migrated fully to pathlib in v0.3
import numpy as np

HDF5_GROUP_NAMES = {
    "halos": "halo_data",
    "galaxies": "galaxy_data",
}

def construct_particle_csr_lists(data: SimulationData, internals: Internals) -> dict[str, dict[str, dict]]:
    """
    Extracts particle lists from SimulationData (matching GroupStore & ParticleStore) and converts them to the CSR format for hdf5.
    """
    result = {group: {} for group in data.groups}

    for group_name in data.groups: # NOTE: sorts both halos & galaxies as opposed to previous function which took group_name

        sentinel = -1 if group_name == "galaxies" else 0 # REVIEW: fix this ideally?
        group_store = data.groups[group_name]
        group_key = group_store.group_key

        for ptype_list in internals.group_ptype_lists[group_name]: # let it be known this was a pain to write

            ptype = internals.plist_to_ptype[ptype_list]
            particle_group_ids = data.particles[ptype][group_key]
            particle_indices = data.particles[ptype]["particle_index"] # positional index of particles in original snapshot

            # mask out non-group particles (technically redundant for halos) // sort
            mask = particle_group_ids != sentinel
            particle_group_ids = particle_group_ids[mask]
            particle_indices = particle_indices[mask]
            order = np.argsort(particle_group_ids)
            sorted_particle_group_ids = particle_group_ids[order] 
            sorted_indices = particle_indices[order]

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

def write_analysis_to_output_file(data: SimulationData, particle_lists: dict, internals: Internals, output_file: Path) -> None:
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

            hdf5_group.create_dataset(name=f"{group_store.group_key}", data=group_store.group_ids, compression=1)

            for ptype in internals.group_ptype_lists[group_name]: # in theory these could be split up into different functions

                if ptype not in particle_lists[group_name]:
                    continue

                pl = particle_lists[group_name][ptype]
                hdf5_group.create_dataset(f'{ptype}_indices', data=pl['indices'], compression=1)
                hdf5_group.create_dataset(f'{ptype}_offsets', data=pl['offsets'], compression=1)
                hdf5_group.create_dataset(f'{ptype}_lengths', data=pl['lengths'], compression=1)

            # group columns by stage label (what was previously dicts)
            columns_by_label: dict[str, list[str]] = {}

            for column_name in group_store.columns:

                if column_name.startswith("_"):
                    continue

                if column_name not in internals.output_columns:
                    continue

                label = internals.output_columns[column_name].label
                columns_by_label.setdefault(label, []).append(column_name)

            for label, column_names in columns_by_label.items():

                label_group = hdf5_group.require_group(f"properties/{label}")
                
                for column_name in column_names:

                    column_meta = internals.output_columns[column_name]
                    dataset = label_group.create_dataset(
                        column_name, data=group_store[column_name], compression=1,
                    )
                    dataset.attrs["unit"] = column_meta.unit
                    dataset.attrs["description"] = column_meta.description