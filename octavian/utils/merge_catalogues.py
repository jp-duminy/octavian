import h5py
import numpy as np
import warnings
from yaml import safe_load

warnings.filterwarnings("ignore", category=RuntimeWarning)

def merge_catalogues(files: list[str], outfile: str, configfile: str) -> None:
  with open(configfile, 'r') as f:
    config = safe_load(f)

  galaxy_parent_halo = []
  halo_masses = []
  galaxy_masses = []
  file_lengths = {'halos': {}, 'galaxies': {}}

  cumulative_halos = 0
  for file in files:
    with h5py.File(file, 'r') as f:
      file_lengths['halos'][file] = len(f['halo_data']['dicts/masses.total'])
      try:
        file_lengths['galaxies'][file] = len(f['galaxy_data']['dicts/masses.total'])
      except:
        file_lengths['galaxies'][file] = 0

      halo_masses.append(f['halo_data']['dicts/masses.total'][:])
      try:
        file_galaxy_masses = f['galaxy_data']['dicts/masses.stellar'][:]
        if len(file_galaxy_masses) != 0:
          galaxy_parent_halo.append(f['galaxy_data']['parent_halo_index'][:] + cumulative_halos)
          galaxy_masses.append(file_galaxy_masses)
      except: pass
      
      cumulative_halos += file_lengths['halos'][file]
  halo_masses = np.concatenate(halo_masses)
  galaxy_masses = np.concatenate(galaxy_masses)
  galaxy_parent_halo = np.concatenate(galaxy_parent_halo)

  halo_order = np.argsort(halo_masses)
  inverse_order = np.argsort(halo_order)  # maps old index -> new index
  galaxy_parent_halo = inverse_order[galaxy_parent_halo]
  galaxy_order = np.argsort(galaxy_masses)

  with h5py.File(outfile, 'w') as f_out:
    halo_group = f_out.create_group('halo_data')
    galaxy_group = f_out.create_group('galaxy_data')

    for dataset in config['dataset_columns'].keys():
      if dataset in ['haloID', 'galaxyID', 'parent_halo_index']: continue
      if dataset in ['glist', 'slist', 'dmlist', 'bhlist']: continue # old code guard
      halo_data = []
      galaxy_data = []
      for file in files:
        with h5py.File(file, 'r') as f:

          try: halo_data.append(f['halo_data'][dataset][:])
          except:
            if '_L' in dataset: halo_data.append(np.full((file_lengths['halos'][file], 3), np.nan))
            else: halo_data.append(np.full(file_lengths['halos'][file], np.nan))

          if file_lengths['galaxies'][file] == 0: continue
          try: galaxy_data.append(f['galaxy_data'][dataset][:])
          except:
            if '_L' in dataset: galaxy_data.append(np.full((file_lengths['galaxies'][file], 3), np.nan))
            else: galaxy_data.append(np.full(file_lengths['galaxies'][file], np.nan))

      halo_group[dataset] = np.concatenate(halo_data)[halo_order]
      galaxy_group[dataset] = np.concatenate(galaxy_data)[galaxy_order]

    halo_group['HaloID'] = np.arange(np.sum(list(file_lengths['halos'].values())))
    galaxy_group['GalID'] = np.arange(np.sum(list(file_lengths['galaxies'].values())))
    galaxy_group['parent_halo_index'] = galaxy_parent_halo[galaxy_order]
    
    ptype_lists = ['glist', 'slist', 'dmlist', 'bhlist']
    for ptype_list in ptype_lists:
      for group_key, out_group in [('halo_data', halo_group), ('galaxy_data', galaxy_group)]:
        all_indices = []
        all_lengths = []
        for file in files:
          with h5py.File(file, 'r') as f_in:
            try:
              all_indices.append(f_in[group_key][f'{ptype_list}_indices'][:])
              all_lengths.append(f_in[group_key][f'{ptype_list}_lengths'][:])
            except KeyError:
              continue

        if not all_indices:
          continue

        # reorder by the sort order computed earlier
        order = halo_order if group_key == 'halo_data' else galaxy_order
        merged_lengths = np.concatenate(all_lengths)[order]
        merged_offsets = np.concatenate([[0], np.cumsum(merged_lengths[:-1])]).astype('int64')

        # reorder indices to match
        old_lengths = np.concatenate(all_lengths)
        old_offsets = np.concatenate([[0], np.cumsum(old_lengths[:-1])]).astype('int64')
        all_flat = np.concatenate(all_indices)
        reordered = np.concatenate([all_flat[old_offsets[i]:old_offsets[i]+old_lengths[i]] for i in order])

        out_group.create_dataset(f'{ptype_list}_indices', data=reordered, compression=1)
        out_group.create_dataset(f'{ptype_list}_offsets', data=merged_offsets, compression=1)
        out_group.create_dataset(f'{ptype_list}_lengths', data=merged_lengths, compression=1)

