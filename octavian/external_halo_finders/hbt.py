"""

HBT+ halo finder integration with Octavian; again, assumes we are analysing a HBT+ output.

Source paper: https://ui.adsabs.harvard.edu/abs/2018MNRAS.474..604H/abstract
Github: https://github.com/Kambrian/HBTplus/wiki/Outputs
Code is based on the architecture outlined therein.

"""

# FIXME: likely dead code as we want to use HBT HERONS

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from octavian.data_management import DataManager

from pathlib import Path

import numpy as np
import pandas as pd
import h5py

from octavian.external_halo_finders.halo_utils import HaloMembership, HaloReader, HaloTree, PTYPE_ENCODE

def gather_subsnap_file(subhalo_path, snap_index) -> Path:
    """
    HBT+ outputs a SubSnap_*.hdf5 where * ranges from 0 to nsnapshots-1.
    """
    subhalo_path = Path(subhalo_path)
    filepath = subhalo_path / f'SubSnap_{snap_index}.hdf5'
    
    if not filepath.exists():
        raise FileNotFoundError(f'SubSnap file not found: {filepath}')
    
    return filepath

def read_subhalos(filepath) -> pd.DataFrame:
    """
    Read subhalos from the hdf5 file.
    Pass to polars dataframe.
    """
    with h5py.File(filepath, 'r') as f:
        subhalos = f['Subhalos'][:]
    
    columns = {name: subhalos[name] for name in subhalos.dtype.names}
    return pd.DataFrame(columns)

def read_particles(filepath) -> tuple[np.ndarray, np.ndarray]:
    """
    Read particle memberships from a SubSnap file.
    HBT+ stores variable-length particle ID arrays per subhalo.
    Returns flat aligned arrays (halo_indices, particle_ids).
    Key thing: HBT+ does not store particle types.
    """
    with h5py.File(filepath, 'r') as f:
        particles = f['SubhaloParticles']
        
        # bulk read all variablen-length arrays at once
        all_particles = particles[:]  # h5py returns object array of numpy arrays
    
    # vectorised length computation
    lengths = np.array([len(p) for p in all_particles], dtype=np.int64)
    total = lengths.sum()
    
    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    
    # build halo index array via np.repeat
    out_hids = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    out_pids = np.concatenate(all_particles)
    
    return out_hids, out_pids

def build_parent_ids(properties) -> np.ndarray:
    """
    Construct parent-child relationships from HBT+ FoF grouping.
    
    HBT+ does not have a ranked hierarchy like AHF. It instead makes an FoF group
    where there is a central halo with rank = 0 and the others halos are satellites.
    Then the satellites point to the central halo's TrackID.

    Orphan halos and central halos are assigned a parent -1.
    """
    track_ids = properties['TrackId'].to_numpy().astype(np.int64)
    host_ids = properties['HostHaloId'].to_numpy().astype(np.int64)
    ranks = properties['Rank'].to_numpy().astype(np.int32)
    
    # central halos
    central_mask = (ranks == 0) & (host_ids != -1)
    central_fof_ids = host_ids[central_mask] # the group they belong to
    central_track_ids = track_ids[central_mask] # their TrackIDs
    
    # guard if there are no central halo ids
    if len(central_fof_ids) == 0:
        return np.full(len(track_ids), -1, dtype=np.int64)
    
    # map fof group to central IDs
    max_fof = central_fof_ids.max()
    fof_to_central = np.full(max_fof + 1, -1, dtype=np.int64) # full np array of -1s for each central ID
    fof_to_central[central_fof_ids] = central_track_ids # insert TrackIDs of central halo
    
    # map satellites to their central; orphans and parents to -1
    parent_ids = np.full(len(track_ids), -1, dtype=np.int64) # set everything to -1
    satellite_mask = (ranks > 0) & (host_ids != -1) & (host_ids <= max_fof) # criteria for a satellite
    parent_ids[satellite_mask] = fof_to_central[host_ids[satellite_mask]] # insert satellite mapping
    
    return parent_ids

def label_ptypes(data_manager: DataManager, member_hids: np.ndarray,
                member_pids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HBT+ stores particle IDs but not their types.
    We therefore check the original snapshot to label the ptypes.
    Particles not found in any Octavian ptype are dropped.
    """
    config = data_manager.config
    
    # sort membership pids for searchsorted
    order = np.argsort(member_pids)
    sorted_pids = member_pids[order]
    sorted_hids = member_hids[order]
    
    out_ptypes = np.full(len(member_pids), -1, dtype=np.int8)
    
    for ptype in config['ptypes']:
        ptype_code = PTYPE_ENCODE.get(ptype)
        if ptype_code is None:
            continue
        
        data_manager.load_property('pid', ptype)
        snap_pids = data_manager.data[ptype]['pid'].to_numpy(dtype=np.int64)
        
        # find which membership particles match this ptype
        positions = np.searchsorted(sorted_pids, snap_pids)
        positions = np.clip(positions, 0, len(sorted_pids) - 1)
        matched = sorted_pids[positions] == snap_pids
        
        out_ptypes[order[positions[matched]]] = ptype_code
    
    # drop particles not used in Octavian
    valid = out_ptypes != -1
    return member_hids[valid], member_pids[valid], out_ptypes[valid]

def load_hbt(data_manager: DataManager, subhalo_path: str, snap_index: int, mode='field'):
    """
    Load HBT+ output into Octavian.
    data_manager needs to be loaded on the original snapshot for cross-referencing ptypes.
    This also has the option to just do field halos or capture substructure.
    """
    filepath = gather_subsnap_file(subhalo_path, snap_index)
    properties = read_subhalos(filepath)
    
    track_ids = properties['TrackId'].to_numpy().astype(np.int64)
    parent_ids = build_parent_ids(properties)
    
    member_hids, member_pids = read_particles(filepath)

    # member_hids are subhalo indices (0, 1, 2...) — map to TrackIds
    member_hids = track_ids[member_hids]
    member_hids, member_pids, member_ptypes = label_ptypes(
        data_manager, member_hids, member_pids
    )
    
    # remap TrackIds to the HaloReader-friendly 0, 1, 2 etc. format
    # HBT+ already uses -1 for field halos
    reader = HaloReader(data_manager)
    track_ids, parent_ids, member_hids = reader.remap_ids(track_ids, parent_ids, member_hids)
    tree = HaloTree(track_ids, parent_ids, properties)
    membership = HaloMembership(tree, member_hids, member_pids, member_ptypes, exclusive=True)
    
    data_manager.halo_tree = tree
    data_manager.halo_membership = membership
    
    reader.assign(membership, mode)
    
    # we adore diagnostics
    n_orphans = (properties['Nbound'].to_numpy() <= 1).sum()
    n_centrals = (properties['Rank'].to_numpy() == 0).sum()