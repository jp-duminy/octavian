"""

Octavian galaxy finding, calling the internal FOF6D algorithm in fof6d_algorith.py

If this is very slow, numba may not be jitting on your cluster. Please check whether this is the case.

"""

# type checking (semantic)
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_management import ParticleStore, SimulationAttributes, OctavianConstants

# octavians
from octavian.data_management import DTYPES
from octavian.galaxy_finding.fof6d_algorithm import run_fof6d_algorithm, unwrap_positions
# default library
from dataclasses import dataclass

# other libraries
import numpy as np
from joblib import Parallel, delayed

# slots=True turns off unneeded dict behaviour which avoids accidental mutation & improves memory usage 
# frozen=True adds tiny overhead but is safe
# see https://github.com/orgs/community/discussions/168147#discussioncomment-15464120 if curious

@dataclass(slots=True, frozen=True) 
class FOF6DItem: 
    """
    The attributes of a halo (but, in future, perhaps gas clouds?) being passed into FOF6D.
    """
    pos: np.ndarray
    vel: np.ndarray
    ptype: np.ndarray # which particles are what ptype (move to integer codings eventually)
    write_key: np.ndarray # for writing back to data layers
  
@dataclass(slots=True, frozen=True) 
class FOF6DParameters:
    """
    Fixed simulation/runtime parameters which the algorithm needs.
    """
    linking_length: float
    velocity_factor: float
    boxsize: float
    minstars: int
    cores_per_rank: int

@dataclass(slots=True, frozen=True)
class FOF6DResult:
    """
    Assignments made by FOF6D for writing back to datamanager.
    """
    write_keys: np.ndarray # indexes back into the ParticleStores
    galaxy_ids: np.ndarray 
    ptypes: np.ndarray # move to integer codings eventually
    n_galaxies: int

def prepare_fof6d_data(
    particles: dict[str, ParticleStore], 
    simulation: SimulationAttributes, 
    config: dict, 
    constants: OctavianConstants
) -> tuple[list[FOF6DItem], FOF6DParameters]:
    """
    Extracts relevant arrays/parameters & initialises dataclasses for the FOF6D algorithm.
    """
    star_halo_ids = particles["star"]["HaloID"]
    star_counts = np.bincount(star_halo_ids[star_halo_ids >= 0]) # NOTE: work sentinel value into here

    eligible_halos = np.where(star_counts >= config['MINIMUM_STARS_PER_GALAXY'])[0] # disregard halos which would have no galaxies
    eligible_set = np.zeros(star_counts.shape[0], dtype=bool)
    eligible_set[eligible_halos] = True

    gas = particles["gas"]
    rho, sfr = gas["rho"], gas["sfr"]
    temperature = gas["temperature"] 
    nH = rho * config["XH"] / constants.PROTON_MASS_G # TODO: move to reader

    dense_mask = (nH > config['nHlim']) & ((temperature < config['Tlim']) | (sfr > 0)) # NOTE: sfr > 0 overrides of the density criterion

    pos_list, vel_list, ptype_list, index_list, hid_list = [], [], [], [], []

    for ptype in ["star", "gas", "bh"]:

        if ptype not in particles:
            continue

        halo_ids = particles[ptype]["HaloID"]
        in_range = (halo_ids >= 0) & (halo_ids < len(eligible_set))
        masked_hids = np.where(in_range, halo_ids, 0) # have to mask before & operator 
        
        if ptype == "gas":
            mask = dense_mask & in_range & eligible_set[masked_hids] # dense criterion for gas specifically
        else:
            mask = in_range & eligible_set[masked_hids]
        
        pos_list.append(particles[ptype]["pos"][mask])
        vel_list.append(particles[ptype]["vel"][mask])
        ptype_list.append(particles[ptype]["ptype"][mask])
        index_list.append(np.arange(particles[ptype].n_particles)[mask])
        hid_list.append(halo_ids[mask])
    
    linking_length = simulation.mis * config["b"]

    params = FOF6DParameters(
        linking_length=linking_length,
        velocity_factor=config["velocity_factor"],
        boxsize=simulation.boxsize,
        minstars=config['MINIMUM_STARS_PER_GALAXY'],
        cores_per_rank=config['nproc'],
    )

    # NOTE: in-halo concatenation, could be expensive
    all_pos, all_vel = np.concatenate(pos_list, dtype=DTYPES["pos"]), np.concatenate(vel_list, dtype=DTYPES["vel"]) 
    all_ptype, all_halo_ids = np.concatenate(ptype_list), np.concatenate(hid_list)
    all_write_key = np.concatenate(index_list)

    order = np.argsort(all_halo_ids, stable=True) # keep stable=True on for deterministic results
    all_pos, all_vel = all_pos[order], all_vel[order]
    all_ptype, all_halo_ids = all_ptype[order], all_halo_ids[order]
    all_write_key = all_write_key[order]

    sorted_halo_ids = all_halo_ids # just for readability
    changes = np.flatnonzero(np.diff(sorted_halo_ids)) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(sorted_halo_ids)]))

    sizes = ends - starts
    size_order = np.argsort(sizes)[::-1] # largest halos first

    items = []

    for i in size_order:

        s, e = starts[i], ends[i]

        halo_pos = all_pos[s:e].copy()
        unwrap_positions(positions=halo_pos, boxsize=simulation.boxsize)

        items.append(FOF6DItem(
            pos=halo_pos,
            vel=all_vel[s:e],
            ptype=all_ptype[s:e],
            write_key=all_write_key[s:e],
        ))

    return items, params

def collect_fof6d_results(results: tuple[np.ndarray, np.ndarray, np.ndarray, int]) -> FOF6DResult:
    """
    Collates and unpacks all the results from the FOF6DItems and concatenates them into a result for broadcasting.
    """
    not_empty = [(k, g, p, n) for k, g, p, n in results if n > 0]

    if not not_empty: # NOTE: flag this in the logger when added

        return FOF6DResult(write_keys=np.empty(0, dtype=np.int64), 
                           galaxy_ids=np.empty(0, dtype=np.int64),
                           ptypes=np.empty(0, dtype=object),
                           n_galaxies=0)
    
    keys, gids, ptypes, counts = zip(*not_empty)

    all_keys = np.concatenate(keys)
    all_gids = np.concatenate(gids)
    all_ptypes = np.concatenate(ptypes)
    counts_array = np.array(counts, dtype=np.int64)
    offsets = np.cumsum(counts_array) - counts_array

    particles_per_halo = np.array([len(k) for k in keys], dtype=np.int64)
    all_gids += np.repeat(offsets, particles_per_halo)

    return FOF6DResult(
        write_keys=all_keys,
        galaxy_ids=all_gids,
        ptypes=all_ptypes,
        n_galaxies=counts_array.sum())

def dispatch_fof6d(items: list[FOF6DItem], params: FOF6DParameters) -> FOF6DResult:
    """
    Runs the FOF6D algorithm by dispatching work items to cores using joblib.
    """
    per_halo_results = Parallel(n_jobs=params.cores_per_rank, batch_size=1)(
        delayed(run_fof6d_algorithm)(work_item=w, params=params) for w in items)

    return collect_fof6d_results(per_halo_results)

def store_fof6d_results(particles: dict[str, ParticleStore], result: FOF6DResult) -> None:
    """
    Appends the galaxy-finding results to ParticleStore (and is vectorised compared to the original).
    """
    for ptype in particles:

        particles[ptype]["GalID"] = np.full(particles[ptype].n_particles, -1, dtype=np.int64) # NOTE: sentinel value

        mask = result.ptypes == ptype

        if not mask.any():
            continue

        particles[ptype]["GalID"][result.write_keys[mask]] = result.galaxy_ids[mask]

def find_galaxies(
    particles: dict[str, ParticleStore], 
    simulation: SimulationAttributes, 
    config: dict,
    constants: OctavianConstants,
) -> FOF6DResult:
    """
    Handles the end-to-end galaxy-finding with FOF6D pipeline; writes back to ParticleStore.
    """
    work_items, params = prepare_fof6d_data(particles=particles, simulation=simulation, config=config, constants=constants)
    result = dispatch_fof6d(items=work_items, params=params)
    store_fof6d_results(particles=particles, result=result)

    return result