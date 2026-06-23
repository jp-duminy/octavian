"""

AHF Halo Finder integration to Octavian; assumes we are analysing a snapshot which AHF has already run on.

Source paper: https://arxiv.org/pdf/0904.3662
Github: https://github.com/weiguangcui/AHF
Code is based on the architecture outlined therein.

"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_management import DataManager

import ctypes # for accelerated reading
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from octavian.external_halo_finders.halo_utils import HaloMembership, HaloReader, HaloTree

# AHF assigns ptype codes, we change these to ptype names for Octavian compatibility
_PTYPE_MAP = {0: 0, # gas
              1: 1, # dm
              4: 2, # star
              5: 3} # bh

# AHF compresses its output (which is what gzip is for)
def _open_catalog(path: Path):
    """
    Open an AHF catalogue file.
    """
    if path.suffix == '.gz':
        return gzip.open(path, 'rt')
    return open(path, 'r')

def read_ahf_halos(path: Path) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """
    Parse an AHF file.
    """
    # headers are #-commented
    with _open_catalog(path) as f:
        header_line = f.readline().strip().lstrip('#').split() # strip the #

    clean_headers = [h.split('(')[0] for h in header_line] # AHF adds numbering so strip this too

    # whitespace-delimited rows
    # NOTE: only pandas can handle this (at present March 2026)
    df = pd.read_csv(path, comment='#', delim_whitespace=True, names=clean_headers)

    return df # convert to polars

# previously defined as two functions: now convolved
def read_ahf_particles(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse AHF _particles file into flat numpy arrays (only Octavian particles).
    The format of these is slightly fiddly because the format alternates.
    Data structure of AHF catalog:

    Number of Particles | Halo ID 
    Particle ID | Particle Type
    Particle ID | Particle Type
    etc. with alternating lengths under the halo header, making reading the output finicky.
    """
    # AHF uses gzip
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f:
        raw_lines = f.readlines()

    n = len(raw_lines) # start with all lines
    out_hids = np.empty(n, dtype=np.int64)
    out_pids = np.empty(n, dtype=np.int64)
    out_ptypes = np.empty(n, dtype=np.int8) # small numbers, use smallest footprint data structure

    # this is a loop-based state machine
    idx = 0
    remaining = 0
    current_hid = 0

    for line in raw_lines:
        parts = line.split()

        # this avoids MPI separators in the file (potentially redundant)
        if len(parts) != 2:
            continue

        # if we are at a header (nparticles | hid)
        if remaining == 0:
            try:
                remaining = int(parts[0])
                current_hid = int(parts[1])
            except ValueError:
                remaining = 0
            continue

        # ptype
        ptype_code = int(parts[1])
        remaining -= 1 # then register this as completed, move on

        # the alternating structure makes this finicky because if it were columnar we could mask these
        if ptype_code not in _PTYPE_MAP:
            continue # dict lookup is famously fast: but unavoidable overhead

        # write to numpy
        out_hids[idx] = current_hid
        out_pids[idx] = int(parts[0])
        out_ptypes[idx] = _PTYPE_MAP[ptype_code]
        idx += 1

    return out_hids[:idx].copy(), out_pids[:idx].copy(), out_ptypes[:idx].copy()

def read_ahf_particles_c(filepath, n_estimate=None):
    """
    C-accelerated AHF particle parser.
    Requires ahf_parser.so to be compiled in the same directory.
    """
    import ctypes
    
    so_path = Path(__file__).parent / 'ahf_parser.so'
    if not so_path.exists():
        raise FileNotFoundError(f'Compiled parser not found at {so_path}. Compile with: gcc -O2 -shared -fPIC -o ahf_parser.so ahf_parser.c')
    
    lib = ctypes.CDLL(str(so_path))
    lib.parse_ahf_particles.restype = ctypes.c_long
    
    filepath = Path(filepath)
    if n_estimate is None:
        n_estimate = filepath.stat().st_size // 8
    
    out_hids = np.empty(n_estimate, dtype=np.int64)
    out_pids = np.empty(n_estimate, dtype=np.int64)
    out_ptypes = np.empty(n_estimate, dtype=np.int8)
    valid_ptypes = np.array([0, 1, 4, 5], dtype=np.int32)
    
    n = lib.parse_ahf_particles(
        str(filepath).encode(),
        out_hids.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        out_pids.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        out_ptypes.ctypes.data_as(ctypes.POINTER(ctypes.c_char)),
        valid_ptypes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        len(valid_ptypes)
    )
    
    if n < 0:
        raise IOError(f'Failed to open {filepath}')
    
    # C parser returns raw AHF ptype codes (0, 1, 4, 5)
    # map to Octavian encoding (0, 1, 2, 3)
    ptype_remap = np.full(6, -1, dtype=np.int8)  # max AHF code is 5
    for ahf_code, octavian_code in _PTYPE_MAP.items():
        ptype_remap[ahf_code] = octavian_code
    
    result_ptypes = ptype_remap[out_ptypes[:n]]
    
    return out_hids[:n].copy(), out_pids[:n].copy(), result_ptypes.copy()

def load_ahf(data_manager, particles_path, halos_path=None, mode='field'):

    particles_path = Path(particles_path)
    
    if halos_path is None:
        halos_path = particles_path.with_name(
            particles_path.name.replace('particles', 'halos')
        )
    else:
        halos_path = Path(halos_path)

    properties = read_ahf_halos(halos_path)
    member_hids, member_pids, member_ptypes = read_ahf_particles_c(particles_path)
    
    halo_ids = properties['ID'].to_numpy().astype(np.int64)
    parent_ids = properties['hostHalo'].to_numpy().astype(np.int64)
    parent_ids[parent_ids == 0] = -1 # AHF sets field halos equal to 0 but we want them as -1

    reader = HaloReader(data_manager)
    halo_ids, parent_ids, member_hids = reader.remap_ids(halo_ids, parent_ids, member_hids)
    tree = HaloTree(halo_ids, parent_ids, properties)
    membership = HaloMembership(tree, member_hids, member_pids, member_ptypes, exclusive=False)

    data_manager.halo_tree = tree
    data_manager.halo_membership = membership

    reader.assign(membership, mode)
