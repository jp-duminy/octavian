import h5py
import numpy as np

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

def filter_snapshot(snapfile: str, outfile: str, nsplit: int=4):
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
      with h5py.File(f'{outfile}_{i}.hdf5', 'a') as f_out:
        f.copy(f['Header'], f_out, 'Header')

    #
    # algorithm to weight split snapshot
    #

    ptypes = [group for group in list(f.keys()) if 'HaloID' in list(f[group].keys())] # from Jakub's code
    # initialise weight dictionaries
    star_weights = {}
    gas_weights = {}
    dm_weights = {} 
    
    for ptype_name, weight_dict in [
    ('PartType4', star_weights),
    ('PartType0', gas_weights),
    ('PartType1', dm_weights),  
    ]: # config is not passed so refer to them by PartType
      ptype_ids = f[ptype_name]['HaloID'][:] # access star/gas particles and their halo IDs
      ptype_ids = ptype_ids[ptype_ids != 0] # access only the stars/gas in a halo
      unique, counts = np.unique(ptype_ids, return_counts=True) # find the counts of that particle for a unique halo

      # find a raw weight
      for hid, count in zip(unique, counts):
          weight_dict[hid] = count

    weights = {}
    for hid in set(star_weights) | set(gas_weights) | set(dm_weights): # union operator: find halos in both sets
      n_total = star_weights.get(hid, 0) + gas_weights.get(hid, 0) + dm_weights.get(hid, 0)
      fof6d_cost = (star_weights.get(hid, 0))**1.2 + gas_weights.get(hid, 0)
      cgp_cost = n_total  # roughly linear in total particles per halo
      weights[hid] = ALPHA * fof6d_cost + BETA * cgp_cost

    # account for theoretical pure dark matter halo (these still need to be assigned)
    # this could maybe be removed
    all_ids = set()
    for ptype in ptypes:
        ptype_ids = f[ptype]['HaloID'][:]
        all_ids.update(ptype_ids[ptype_ids != 0])
    for hid in all_ids:
        weights.setdefault(hid, 0)

    # simple sequential binning algorithm
    rank_assignments = [set() for _ in range(nsplit)] # initialise a set
    rank_loads = [0] * nsplit 
    for hid in sorted(weights, key=weights.get, reverse=True): # sort by heaviest first
        # we go from heaviest -> lightest, adding the next halo to the bin with the smallest load
        lightest = np.argmin(rank_loads) # find which rank has the lowest load 
        rank_assignments[lightest].add(hid)
        rank_loads[lightest] += weights[hid]

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
        if dataset == 'particle_index':        # <-- add
            data = particle_index[in_halo][order]
        else:
          data = f[ptype][dataset][:][in_halo][order]
        for i in range(nsplit):
            with h5py.File(f'{outfile}_{i}.hdf5', 'a') as f_out:
                f_out.require_group(ptype)
                f_out[ptype][dataset] = data[rank_masks[i]]