from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_management import DataManager

import h5py
import os
import numpy as np
import warnings

# REVIEW: needs refactoring to avoid all the conditionals

warnings.filterwarnings("ignore", category=RuntimeWarning)

def save_group_properties(data_manager: DataManager, filename: str) -> None:
  config = data_manager.config

  if os.path.exists(filename):
    os.remove(filename)

  with h5py.File(filename, 'w') as f:
    halo_data = f.create_group('halo_data')
    halo_columns = data_manager.group_data['halos'].columns

    if 'galaxies' in config['groups']:
      galaxy_data = f.create_group('galaxy_data')
      galaxy_columns = data_manager.group_data['galaxies'].columns
    else:
      galaxy_columns = []

    # write particle lists in flat CSR format
    ptype_lists = ['glist', 'slist', 'dmlist', 'bhlist']
    for group_name, hdf5_group in [('halos', halo_data), ('galaxies', galaxy_data if 'galaxies' in config['groups'] else None)]:
      if hdf5_group is None:
        continue
      for ptype_list in ptype_lists:
        if ptype_list not in data_manager.particle_lists[group_name]:
          continue
        pl = data_manager.particle_lists[group_name][ptype_list]
        hdf5_group.create_dataset(f'{ptype_list}_indices', data=pl['indices'], compression=1)
        hdf5_group.create_dataset(f'{ptype_list}_offsets', data=pl['offsets'], compression=1)
        hdf5_group.create_dataset(f'{ptype_list}_lengths', data=pl['lengths'], compression=1)

    # TODO: temporary fix for missing IDs 
    # needs addressing in the full refactor
    halo_data.create_dataset('haloID', data=data_manager.group_data['halos'].index.to_numpy(), compression=1)
    if 'galaxies' in config['groups']:
        galaxy_data.create_dataset('galaxyID', data=data_manager.group_data['galaxies'].index.to_numpy(), compression=1)
        
    # write all other datasets
    for dataset_name, column in config['dataset_columns'].items():
      if dataset_name in ptype_lists:
        continue

      if np.all(np.isin(column, halo_columns)):
        halo_data.create_dataset(dataset_name, data=data_manager.group_data['halos'][column].to_numpy(), compression=1)
      if 'galaxies' in config['groups'] and np.all(np.isin(column, galaxy_columns)):
        galaxy_data.create_dataset(dataset_name, data=data_manager.group_data['galaxies'][column].to_numpy(), compression=1)