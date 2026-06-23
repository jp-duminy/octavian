"""

Functions to filter & split a snapshot into subfiles for per-rank MPI access; and remerge analysis hdf5 files into an output catalogue.

NOTE: this will eventually be legacy code, as we intend to move away from intermediate files.

"""

import h5py
import numpy as np
from pathlib import Path
from yaml import safe_load

# NOTE: will likely be unnecessary after move to HDF5 MPI.

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_id_filter(f: h5py.File, ptypes: list[str], nsplit: int) -> list[list[int]]:
  ids = []
  for ptype in ptypes:
    ids_ptype = f[ptype]['HaloID'][:]
    ids.append(ids_ptype[ids_ptype != 0])

  ids = np.sort(np.concatenate(ids))
  unique_ids, counts = np.unique(ids, return_counts=True)
  cumulative_counts = np.cumsum(counts)
  total = len(ids)

  split_ids = [0]
  split_fractions = np.linspace(0., 1., nsplit + 1)
  split_fractions = split_fractions[1:]

  for fraction in split_fractions:
    fraction_count = total * fraction
    split_ids.append(unique_ids[(np.abs(cumulative_counts - fraction_count)).argmin()])

  id_filter = list(zip(split_ids[:-1], split_ids[1:]))

  return id_filter

def filter_snapshot(snapfile: str, outfile: Path, nsplit: int=4):
  """
  Weighted snapshot filter.

  This snapshot filter is designed to be weighted towards balancing FOF6D. It does so by applying a 
  power law to star/gas counts when deciding how to divide the snapshot. FOF6D can take extremely long
  and ranks can have wildly different runtimes if the snapshot is not weighted when filtered.
  """

  # these are weighting constants. cgp scales better than fof6d so ideally lean towards fof6d
  ALPHA = 0.6 # arbitrary fof6d constant
  BETA = 0.4 # arbitrary cgp constant

  with h5py.File(snapfile, 'r') as f:
    for i in range(nsplit):
      with h5py.File(outfile / f"rank_{i}.hdf5", 'a') as f_out:
        f.copy(f['Header'], f_out, 'Header')

    #
    # algorithm to weight split snapshot
    #

    ptypes = [group for group in list(f.keys()) if 'HaloID' in list(f[group].keys())] # from Jakub's code
    # initialise weight dictionaries
    ptype_counts = {}
    for ptype_name in ['PartType0', 'PartType1', 'PartType4', 'PartType5']: # no datamanager mapping so use default ptype names
        if ptype_name not in f:
            continue
        ids = f[ptype_name]['HaloID'][:]
        ids = ids[ids != 0]
        unique, counts = np.unique(ids, return_counts=True)
        ptype_counts[ptype_name] = (unique, counts)

    # build a unified halo ID array
    all_hids_list = []
    for ptype_name in ptypes:  # ptypes from the existing detection logic
        ids = f[ptype_name]['HaloID'][:]
        all_hids_list.append(np.unique(ids[ids != 0]))
    all_hids = np.unique(np.concatenate(all_hids_list))

    # guard (necessary for high-redshift snapshots with no HaloIDs)
    if len(all_hids) == 0:
      return
    
    n_halos = all_hids.max() + 1  # use hid as direct index

    star_counts = np.zeros(n_halos)
    gas_counts = np.zeros(n_halos)
    dm_counts = np.zeros(n_halos)

    for ptype_name, arr in [('PartType4', star_counts), ('PartType0', gas_counts), ('PartType1', dm_counts)]:
        if ptype_name in ptype_counts:
            hids, cnts = ptype_counts[ptype_name]
            arr[hids] = cnts

    fof6d_cost = star_counts[all_hids] ** 1.2 + gas_counts[all_hids]
    cgp_cost = star_counts[all_hids] + gas_counts[all_hids] + dm_counts[all_hids]
    halo_weights = ALPHA * fof6d_cost + BETA * cgp_cost

    # greedy binning — sort heaviest first
    weight_order = np.argsort(halo_weights)[::-1]
    rank_assignments = [set() for _ in range(nsplit)]
    rank_loads = np.zeros(nsplit)
    for idx in weight_order:
        lightest = np.argmin(rank_loads)
        rank_assignments[lightest].add(all_hids[idx])
        rank_loads[lightest] += halo_weights[idx]

    # and now the actual filter
    # toss particles not in a halo
    for ptype in ptypes:
      datasets = list(f[ptype].keys())
      ids = f[ptype]['HaloID'][:]
      particle_index = np.arange(len(ids), dtype='int')
      in_halo = ids != 0 # find ids not in a halo
      ids_filtered = ids[in_halo]
      order = np.argsort(ids_filtered)
      ids_sorted = ids_filtered[order]
      datasets = datasets + ['particle_index']

      # Jakub's code masks once per dataset but we could mask once per ptype
      rank_masks = []
      for i in range(nsplit):
          halo_set = np.array(list(rank_assignments[i]))
          rank_masks.append(np.isin(ids_sorted, halo_set))

      for dataset in datasets:
        if dataset == 'particle_index':        
            data = particle_index[in_halo][order]
        else:
          data = f[ptype][dataset][:][in_halo][order]
        for i in range(nsplit):
            with h5py.File(outfile / f"rank_{i}.hdf5", 'a') as f_out:
                f_out.require_group(ptype)
                f_out[ptype][dataset] = data[rank_masks[i]]

    for i in range(nsplit):
      with h5py.File(outfile / f"rank_{i}.hdf5", 'a') as f_out:
          diag = f_out.require_group('Diagnostics')
          for pt in ptypes:  
              if pt in f_out:
                  diag.attrs[f'n_{pt}'] = len(f_out[pt]['HaloID'])
          diag.attrs['n_halos'] = len(rank_assignments[i])
          diag.attrs['total_weight'] = rank_loads[i]

# FIXME: needs a refactoring pass, very messy and try/excepts can cause unnoticeable bugs
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
