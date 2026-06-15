from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_management import DataManager

import numpy as np
import pandas as pd

# FIXME: same problems as save_group_properties and remerge_catalogues.

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