from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_management import DataManager, SimulationData, convert_data_manager

import h5py
import os
from pathlib import Path
import numpy as np
import warnings

# REVIEW: needs refactoring to avoid all the conditionals

warnings.filterwarnings("ignore", category=RuntimeWarning) # FIXME:  aim to get rid of this (I think this is pre-csr?)

GROUP_PTYPE_LISTS = {
    "halos":    ["glist", "slist", "dmlist", "bhlist"],
    "galaxies": ["glist", "slist", "bhlist"],
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