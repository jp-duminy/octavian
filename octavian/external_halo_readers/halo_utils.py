"""

Here you will find the halo data management classes.

HaloReader ties the whole thing together. It is an agnostic class which handles all the data management
from a specified halo finder. The goal of the halo-finder-specific files is to then translate their outputs
into the format this class expects.

HaloTree provides the framework for storing substructure (subhalos, sub-subhalos, etc.) that come from
halo finders such as AHF and HBT+. It is useful to do this thoroughly for things like progenitors, and
because it allows us to capture more science data rather than tossing everything.

HaloMembership characterises the existing halos.

I think OOP is good here because bespoke halo readers can use inheritance.

"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from octavian.data_management import DataManager

import numpy as np
import pandas as pd
from numba import njit

# the readers extract Octavian-compatible particles
# {'gas': 0, 'dm': 1, 'star': 2, 'bh': 3} is our convention for what readers must return
# readers must also align halo IDs so they read 0, 1, 2 etc.

# easier to work with integers than strings
PTYPE_ENCODE = {'gas': 0, 'dm': 1, 'star': 2, 'bh': 3}
PTYPE_DECODE = {i: j for j, i in PTYPE_ENCODE.items()} # the inverse operation

class HaloReader:
    """
    Base class for external halo finder integration.
    
    Subclasses use read() to parse their specific file formats.

    This class handles remapping, tree construction, snapshot matching, 
    and DataManager assignment for Octavian-friendly analysis.
    """

    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
        self.tree = None
        self.membership = None

        # this cache will speed up ptype assignment
        self._snap_cache = {}

    def remap_ids(self, halo_ids, parent_ids, member_hids):
        """
        Map halo-finder-specific IDs to an agnostic 0, 1, 2 (speeds up agnostic classes, easier to work with).
        You need to use searchsorted here as halo IDs can be enormous numbers.
        """
        # sorted, unique hids
        unique_raw = np.unique(halo_ids)
        # find where all halo IDs can be inserted such that the order of unique_raw is preserved
        new_halo_ids = np.searchsorted(unique_raw, halo_ids)
        
        # parents: map known IDs, anything else is -1 as is Octavian tradition
        new_parent_ids = np.full_like(parent_ids, -1) # set all equal to -1 initially then replace
        valid_parents = parent_ids != -1 # -1 handles orphan/parent case
        if np.any(valid_parents): # if any valid parents exist
            parent_positions = np.searchsorted(unique_raw, parent_ids[valid_parents]) # find where parents can be inserted
            parent_matched = unique_raw[np.clip(parent_positions, 0, len(unique_raw) - 1)] == parent_ids[valid_parents] # match positions to IDs
            new_parent_ids[valid_parents] = np.where(parent_matched, parent_positions, -1) # insert IDs
        
        # membership: same logic and structure
        member_positions = np.searchsorted(unique_raw, member_hids)
        member_matched = unique_raw[np.clip(member_positions, 0, len(unique_raw) - 1)] == member_hids
        new_member_hids = np.where(member_matched, member_positions, -1)
        
        return new_halo_ids, new_parent_ids, new_member_hids

    def assign(self, membership, mode):
        """
        Resolve memberships, match to snapshot, assign HaloIDs.
        Does one enormous sweep over the snapshot.
        """
        # admin
        pids, ptypes, hids = membership.branch_membership(mode=mode)
        config = self.dm.config

        # build a giant snapshot array 
        all_snap_pids = []
        all_snap_offsets = {} # track where each ptype starts
        offset = 0 # this will be used to move to the next ptype

        for ptype in config['ptypes']:
            ptype_code = PTYPE_ENCODE.get(ptype)
            if ptype_code is None:
                continue
            self.dm.load_property('pid', ptype)
            snap_pids = self.dm.data[ptype]['pid'].to_numpy(dtype=np.int64)
            all_snap_pids.append(snap_pids)
            all_snap_offsets[ptype] = (offset, offset + len(snap_pids))
            offset += len(snap_pids)
        all_snap_pids = np.concatenate(all_snap_pids)

        # this is the massive bottleneck
        is_sorted = np.all(all_snap_pids[:-1] <= all_snap_pids[1:])
        # conditional: if satisfied, can save an enormous amount of time
        if is_sorted:
            snap_order = np.arange(len(all_snap_pids), dtype=np.int64)
            snap_pids_sorted = all_snap_pids
        else:
            snap_order = np.argsort(all_snap_pids)
            snap_pids_sorted = all_snap_pids[snap_order]

        # are these externally sorted (HBT+) or not (AHF)
        ext_sorted = not membership.exclusive

        if ext_sorted:
            # skip the sort
            ext_pids_sorted = pids
            ext_hids_sorted = hids
        else:
            ext_order = np.argsort(pids)
            ext_pids_sorted = pids[ext_order]
            ext_hids_sorted = hids[ext_order]

        # linear merge matches halo finder and original snapshot IDs
        halo_ids_sorted = merge_match(snap_pids_sorted, ext_pids_sorted, ext_hids_sorted) 
    
        # then insert hids
        halo_ids_all = np.empty_like(halo_ids_sorted)
        halo_ids_all[snap_order] = halo_ids_sorted

        # now distribute the ids to their ptypes
        for ptype in config['ptypes']:
            if ptype not in all_snap_offsets:
                continue
            start, end = all_snap_offsets[ptype]
            self.dm.data[ptype]['HaloID'] = pd.Series(halo_ids_all[start:end], dtype='category') # send to data manager

    def match_ptype(self, ptype, ext_pids, ext_hids, ext_sorted=False):
        """
        Sorted merge matching of external particle IDs against snapshot.

        Defunct since assign() moved to sweep the entire snapshot. But perhaps a useful skeleton?
        """
        self.dm.load_property('pid', ptype)
        snap_pids = self.dm.data[ptype]['pid'].to_numpy(dtype=np.int64)

        # cache these to avoid repeated argsort on up to 600m particles (I ran on simba m100n1024)
        cache = self._snap_cache.setdefault(ptype, {})
        if 'order' not in cache:
            cache['order'] = np.argsort(snap_pids)
            cache['pids_sorted'] = snap_pids[cache['order']]
        snap_order = cache['order']
        snap_pids_sorted = cache['pids_sorted']

        if ext_sorted:
            ext_pids_sorted = ext_pids
            ext_hids_sorted = ext_hids
        else:
            ext_order = np.argsort(ext_pids)
            ext_pids_sorted = ext_pids[ext_order]
            ext_hids_sorted = ext_hids[ext_order]
        
        # linear merge using numba function
        halo_ids_sorted = merge_match(snap_pids_sorted, ext_pids_sorted, ext_hids_sorted)
        
        # unsort to return to original snapshot order
        halo_ids = np.empty_like(halo_ids_sorted)
        halo_ids[snap_order] = halo_ids_sorted
        
        self.dm.data[ptype]['HaloID'] = halo_ids

class HaloTree:
    """
    Handles halo hierarchy for a single snapshot.
    """

    def __init__(self, halo_ids, parent_ids, properties=None):

        # base structure
        self.halo_ids = np.asarray(halo_ids, dtype=np.int64)
        self.parent_ids = np.asarray(parent_ids, dtype=np.int64)
        self.properties = properties

        # guard against no halos being present
        if len(self.halo_ids) == 0:
            self._id_to_idx = np.empty(0, dtype=np.int32)
            self.depths = np.empty(0, dtype=np.int32)
            self.field_map = np.empty(0, dtype=np.int64)
            self._depth_lookup = np.empty(0, dtype=np.int32)
            return

        # hid to index: map halo ID to its place in the hierarchy
        max_id = self.halo_ids.max()
        self._id_to_idx = np.full(max_id + 1, -1, dtype=np.int32) # array of -1s with len(nhalos)
        self._id_to_idx[self.halo_ids] = np.arange(len(self.halo_ids)) # val[i] = unique halo

        # halo structuring
        # simply sorts halos and their subhalos by hid
        order = np.argsort(self.parent_ids)
        self._parents_sorted = self.parent_ids[order]
        self._children_sorted = self.halo_ids[order]

        # csr format
        changes = np.flatnonzero(np.diff(self._parents_sorted)) + 1 # +1 helps because np.diff technically shifts the array
        self._child_offsets = np.concatenate(([0], changes))
        self._child_lengths = np.diff(np.concatenate((self._child_offsets, [len(self._parents_sorted)])))
        self._child_parent_keys = self._parents_sorted[self._child_offsets]

        self._parent_to_child_idx = np.full(max_id + 1, -1, dtype=np.int32)
        self._parent_to_child_idx[self._child_parent_keys] = np.arange(len(self._child_parent_keys))
        
        # compute on instantiation
        # both passed to numba functions
        self.depths = compute_depths(self.halo_ids, self.parent_ids, self._id_to_idx)
        self.field_map = build_field_map(self.halo_ids, self.parent_ids, self._id_to_idx)

        self._depth_lookup = np.zeros(max_id + 1, dtype=np.int32)
        self._depth_lookup[self.halo_ids] = self.depths

        # field halos
        field_mask = self.parent_ids == -1
        self._field_halos = self.halo_ids[field_mask]

    def get_depth(self, halo_id):
        """
        Returns the depth of a halo (0 = field halo)
        """
        return self._depth_lookup[halo_id]

    def get_children(self, halo_id):
        """
        Returns children of a halo.
        """
        idx = self._parent_to_child_idx[halo_id]
        if idx == -1:
            return np.empty(0, dtype=np.int64)
        start = self._child_offsets[idx]
        end = start + self._child_lengths[idx]
        return self._children_sorted[start:end]

class HaloMembership:
    """
    Handles halo membership for a single snapshot.
    """

    def __init__(self, tree: HaloTree, halo_ids, particle_ids, ptype_codes, exclusive):

        self.tree = tree
        self.exclusive = exclusive

        # guard for no halos
        if len(halo_ids) == 0:
            self._member_hids = np.empty(0, dtype=np.int64)
            self._member_pids = np.empty(0, dtype=np.int64)
            self._member_ptypes = np.empty(0, dtype=np.int8)
            self._offsets = np.empty(0, dtype=np.int64)
            self._lengths = np.empty(0, dtype=np.int32)
            self._unique_hids = np.empty(0, dtype=np.int64)
            self._hid_to_idx = np.empty(0, dtype=np.int32)
            return

        # sort by hID for csr structure
        order = np.argsort(halo_ids)
        self._member_hids = halo_ids[order]
        self._member_pids = particle_ids[order]
        self._member_ptypes = ptype_codes[order]

        changes = np.flatnonzero(np.diff(self._member_hids)) + 1 # find where halo IDs change: +1 accounts for the zeroth case
        self._offsets = np.concatenate(([0], changes)) # where each halo begins
        self._lengths = np.diff(np.concatenate((self._offsets, [len(self._member_hids)]))) # size of halo membership
        self._unique_hids = self._member_hids[self._offsets] 

        max_hid = self._unique_hids.max()
        # halo ID to index: map hid to csr position
        self._hid_to_idx = np.full(max_hid + 1, -1, dtype=np.int32) # array of -1s with len(nhalos)
        self._hid_to_idx[self._unique_hids] = np.arange(len(self._unique_hids)) # set val[i] = unique halo

    def get_halo_particles(self, halo_id, ptype=None):
        """
        Gets the pIDs for a single halo. 
        Can also do specific ptypes if desired.
        """
        # dict lookup to find the index
        idx = self._hid_to_idx[halo_id]
        if idx == -1: # if it lands on a particle
            return np.empty(0, dtype=np.int64) # return nothing
        
        # extract csr format
        start = self._offsets[idx]
        end = start + self._lengths[idx]
        
        # default case: grab all particles (halo readers filter in Octavian particles)
        if ptype is None:
            return self._member_pids[start:end]
        
        # in case you fancy a specific ptype
        mask = self._member_ptypes[start:end] == ptype
        return self._member_pids[start:end][mask]
    
    def get_all_memberships(self, ptype=None):
        """
        Creates flat, aligned (hids, pids) arrays.
        For Caesar-esque progenitor matching.
        """
        if ptype is None:
            return self._member_hids, self._member_pids
        mask = self._member_ptypes == ptype
        return self._member_hids[mask], self._member_pids[mask]
    
    def branch_membership(self, mode='field'):
        """
        Decides on the 'final' membership of a particle (particles can appear in multiple halos)
        
        Field mode: particle belongs to top-level halo (field halo, in AHF paper)
        Subhalo mode: particle belongs to bottom-level halo 
        """
        if mode == 'field':
            resolved_hids = self.tree.field_map[self._member_hids]
        elif mode == 'subhalo':
            resolved_hids = self._member_hids
        else:
            raise ValueError(f"Mode {mode} not supported (yet...)")
        
        # some halo finders (HBT+) have no duplicates, so we can skip those steps
        if self.exclusive:
            return self._member_pids, self._member_ptypes, resolved_hids
        
        return self._deduplicate(
            self._member_pids, self._member_ptypes, 
            resolved_hids, prefer_deepest=(mode == 'subhalo')
        )

    def _deduplicate(self, pids, ptypes, hids, prefer_deepest=False):
        """
        Masks particles to one halo.

        Field mode: order is agnostic
        Subhalo mode: deepest assignment (sub-est halo)
        """
        if prefer_deepest:
            
            sort_key = np.lexsort((-self.tree._depth_lookup[hids], pids)) # https://numpy.org/devdocs/reference/generated/numpy.lexsort.html
        else:
            sort_key = np.argsort(pids)
        
        sorted_pids = pids[sort_key]
        sorted_ptypes = ptypes[sort_key]
        sorted_hids = hids[sort_key]
        
        # locate a pID's first occurrence with boolean masking (array is already sorted)
        mask = np.empty(len(sorted_pids), dtype=bool)
        mask[0] = True # first and last are always unique
        mask[1:] = sorted_pids[1:] != sorted_pids[:-1] # shifting the array one to the left catches the boundaries

        return sorted_pids[mask], sorted_ptypes[mask], sorted_hids[mask]

# optimised numba functions; gets us to C++ speed on these large loops.

@njit
def compute_depths(halo_ids, parent_ids, id_to_idx):
    depths = np.zeros(len(halo_ids), dtype=np.int32)
    for i in range(len(halo_ids)):
        d = 0
        current_parent = parent_ids[i]
        while current_parent != -1:
            idx = id_to_idx[current_parent]
            if idx == -1:
                break
            d += 1
            current_parent = parent_ids[idx]
        depths[i] = d
    return depths

@njit
def build_field_map(halo_ids, parent_ids, id_to_idx):
    n = id_to_idx.shape[0]
    field_map = np.arange(n, dtype=np.int64)
    for i in range(len(halo_ids)):
        hid = halo_ids[i]
        current = hid
        while True:
            idx = id_to_idx[current]
            if idx == -1:
                break
            parent = parent_ids[idx]
            if parent == -1:
                break
            current = parent
        field_map[hid] = current
    return field_map

@njit
def merge_match(snap_pids_sorted, ext_pids_sorted, ext_hids_sorted):
    n_snap = len(snap_pids_sorted)
    n_ext = len(ext_pids_sorted)
    out = np.full(n_snap, -1, dtype=np.int64)
    
    i = 0
    j = 0
    
    while i < n_snap and j < n_ext:
        if snap_pids_sorted[i] == ext_pids_sorted[j]:
            out[i] = ext_hids_sorted[j]
            i += 1
            j += 1
        elif snap_pids_sorted[i] < ext_pids_sorted[j]:
            i += 1
        else:
            j += 1
    
    return out