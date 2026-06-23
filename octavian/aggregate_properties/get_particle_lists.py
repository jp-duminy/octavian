from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_management import DataManager, SimulationData

from octavian.data_management import DTYPES

import numpy as np
import pandas as pd

# FIXME: same problems as save_group_properties and remerge_catalogues.

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


def get_group_particle_indexes(data_manager: DataManager, group_name: str) -> None:

    config = data_manager.config
    group_data = data_manager.group_data[group_name]
    groupID = config['groupIDs'][group_name]

    for ptype in config['ptypes']:

        data = data_manager.data[ptype][['HaloID', 'GalID', 'particle_index']]
        ptype_list = config['ptype_lists'][ptype]

        if group_name == 'galaxies':
            data = data.loc[data['GalID'] != -1]

        if len(data) == 0:
            data_manager.particle_lists[group_name][ptype_list] = {
                'indices': np.array([], dtype='int32'),
                'offsets': np.zeros(len(group_data), dtype='int64'),
                'lengths': np.zeros(len(group_data), dtype='int32'),
            }
            continue

        sorted_data = data.sort_values(groupID)
        ids = sorted_data[groupID].values
        indices = sorted_data['particle_index'].values.astype('int32')

        breaks = np.flatnonzero(np.diff(ids)) + 1
        split_lengths = np.diff(np.concatenate([[0], breaks, [len(ids)]]))
        split_ids = ids[np.concatenate([[0], breaks])]

        # map to group_data index (some groups may have no particles of this ptype)
        length_series = pd.Series(split_lengths, index=split_ids).reindex(group_data.index, fill_value=0)
        lengths = length_series.values.astype('int32')
        offsets = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype('int64')

        # reorder indices to match group_data index order
        # split_ids order may differ from group_data.index order
        reordered = []
        old_offsets = np.concatenate([[0], np.cumsum(split_lengths)])
        id_to_pos = {gid: i for i, gid in enumerate(split_ids)}
        for gid in group_data.index:
            if gid in id_to_pos:
                pos = id_to_pos[gid]
                reordered.append(indices[old_offsets[pos]:old_offsets[pos+1]])
        indices = np.concatenate(reordered) if reordered else np.array([], dtype='int32')

        data_manager.particle_lists[group_name][ptype_list] = {
            'indices': indices,
            'offsets': offsets,
            'lengths': lengths,
        }

def get_particle_lists(data_manager: DataManager) -> None:
    config = data_manager.config

    data_manager.particle_lists = {group: {} for group in config['groups']}

    for ptype in config['ptypes']:
        data_manager.load_property('particle_index', ptype)

    for group in config['groups']:
        get_group_particle_indexes(data_manager, group)