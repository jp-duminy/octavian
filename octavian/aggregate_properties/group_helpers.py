"""

This file contains sorting algorithms with numpy/numba.

This is to improve readability and reduce clutter in calculate_group_properties.py

"""

import numpy as np
from numba import njit

# NOTE: group_idx is a list of particles where the value of each particle is the index of its group.

#
# per-group properties
#

def sum_per_group(values, group_idx, n_groups):
    """
    Sum values per group.
    
    Takes advantage of np.bincount. For any group g:
    - Bin all of its particles (which have group_idx[i] == g)
    - weights=values means the value of the physical quantity is added instead of the index
    - minlength handles a group with zero particles

    The output is a an array of the total values of the quantity of interest for each group.
    """
    return np.bincount(group_idx, weights=values, minlength=n_groups)

def count_per_group(group_idx, n_groups):
    """
    Count occurences per group.

    Same logic as sum_per_group() but without the weighting. This means we simply count the number of 
    occurences, so for example the number of particles.
    """
    return np.bincount(group_idx, minlength=n_groups)

def weighted_mean_per_group(values, weights, group_idx, n_groups):
    """
    Weighted mean per group.

    Same logic as above. Here 'weighted' means the actual physical weight of another quantity
    e.g. mass-weighted metallicity
    """
    weighted_sum = np.bincount(group_idx, weights=values * weights, minlength=n_groups)
    total_weight = np.bincount(group_idx, weights=weights, minlength=n_groups)
    return weighted_sum / total_weight

#
# per-group min/max and for indices too
#

@njit
def max_value_per_group(values, group_idx, n_groups):
    """
    Find the maximum value in each group of an array.
    np.max() only works on a total array.
    """
    result = np.full(n_groups, -np.inf) # array of -infinities
    for i in range(len(values)):
        g = group_idx[i] # individual group
        if values[i] > result[g]: # loop over and find highest value
            result[g] = values[i]
    return result

@njit
def min_value_per_group(values, group_idx, n_groups):
    """
    Find the minimum value in each group of an array.
    Same as above.
    """
    result = np.full(n_groups, np.inf)
    for i in range(len(values)):
        g = group_idx[i]
        if values[i] < result[g]:
            result[g] = values[i]
    return result

@njit
def max_idx_per_group(values, group_idx, n_groups):
    """
    Find the index of the maximum value in each group.
    
    Same as above but with one extra layer.
    """
    result_val = np.full(n_groups, -np.inf)
    result_idx = np.full(n_groups, -1, dtype=np.int64)
    for i in range(len(values)):
        g = group_idx[i]
        if values[i] > result_val[g]:
            result_val[g] = values[i]
            result_idx[g] = i
    return result_idx

@njit
def min_idx_per_group(values, group_idx, n_groups):
    """
    Find the index of the minimum value in each group.
    """
    result_val = np.full(n_groups, np.inf)
    result_idx = np.full(n_groups, -1, dtype=np.int64)
    for i in range(len(values)):
        g = group_idx[i]
        if values[i] < result_val[g]:
            result_val[g] = values[i]
            result_idx[g] = i
    return result_idx

#
# I/O: broadcasting and sorting
#

def broadcast_to_particles(group_values, group_idx):
    """
    Transform an array into one where val[i] = group_value

    e.g. setting particles to their groupIDs
    """
    return group_values[group_idx]

def sort_by_group(group_ids):
    """
    Constructs slices of the bulk for efficient data processing.

    - sorts array by group_ids
    - finds where each group starts and ends

    Meaning we now have a flat array for quick vectorised operations.
    Similar to the CSR format that forms the basis of Octavian's I/O.
    """
    # guard which in practice should never happen (see common_group_properties)
    if len(group_ids) == 0:
        return np.array([], dtype=np.int64), np.array([]), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    
    # https://en.wikipedia.org/wiki/Merge_sort
    # it's safer to use a 'stable' sorting algorithm for the best science in exchange for a slight runtime cost
    order = np.argsort(group_ids, kind='mergesort') 
    sorted_ids = group_ids[order]

    changes = np.flatnonzero(np.diff(sorted_ids)) # find where the difference is nonzero (where a new group starts)

    # find where each group starts
    start = np.empty(len(changes) + 1, dtype=np.int64) # np.diff shifts the array left; compensate
    start[0] = 0
    start[1:] = changes + 1

    # find where each group ends
    end = np.empty(len(start), dtype=np.int64)
    end[:-1] = start[1:] # group ends at the next group's starting index
    end[-1] = len(group_ids) # no such boundary exists from start[] but can find it from number of particles (len(group_ids))

    unique_ids = sorted_ids[start]
    
    return order, unique_ids, start, end

def extract_particle_arrays(data_manager, config, include_hydrogen=False):
    """
    Extract flat particle arrays from all ptypes.
    Useful for anytime you want a list of particles, positions, masses and ptypes in a given area (like a halo).
    """

    ptype_names = list(config['ptypes'])
    ptype_to_int = {p: i for i, p in enumerate(ptype_names)}

    pos_list, mass_list, code_list, halo_list = [], [], [], []

    for ptype in config['ptypes']:

        df = data_manager.data[ptype]
        n = len(df)
        pos_list.append(df[['x', 'y', 'z']].to_numpy())
        mass_list.append(df['mass'].to_numpy())
        code_list.append(np.full(n, ptype_to_int[ptype], dtype=np.int32)) # always easier to store ptypes as int rather than str
        halo_list.append(df['HaloID'].to_numpy())

    # this is for aperture_masses currently, not sure whether this is useful
    if include_hydrogen:
        gas = data_manager.data['gas']
        gas_pos = gas[['x', 'y', 'z']].to_numpy()
        gas_halos = gas['HaloID'].to_numpy()
        for col, name in [('mass_HI', 'HI'), ('mass_H2', 'H2')]:
            ptype_names.append(name)
            ptype_to_int[name] = len(ptype_names) - 1
            pos_list.append(gas_pos)
            mass_list.append(gas[col].to_numpy())
            code_list.append(np.full(len(gas), ptype_to_int[name], dtype=np.int32))
            halo_list.append(gas_halos)

    return (
        np.concatenate(pos_list),
        np.concatenate(mass_list),
        np.concatenate(code_list),
        np.concatenate(halo_list),
        ptype_names,
    )
