"""

Octavian galaxy finding, calling the internal FOF6D algorithm in fof6d_algorith.py

If this is very slow, numba may not be jitting on your cluster. Please check whether this is the case.

"""

# type checking (semantic)
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from octavian.data_management import ParticleStore, SimulationAttributes, OctavianConstants

# defaults
from dataclasses import dataclass

# octavians
from octavian.data_management import DTYPES
from octavian.galaxy_finding.fof6d_algorithm import dispatch_fof6d, unwrap_positions

# other libraries
import numpy as np
import numba

PTYPE_CODES = {"gas": np.int8(0), "star": np.int8(4), "bh": np.int8(5), "dm": np.int8(1)} # GIZMO number conventions

# slots=True turns off unneeded dict behaviour which avoids accidental mutation & improves memory usage 
# frozen=True adds tiny overhead but is safe
# see https://github.com/orgs/community/discussions/168147#discussioncomment-15464120 if curious

@dataclass(slots=True, frozen=True) 
class FOF6DData: 
    """
    Dataclass to concisely store the arrays being passed into FOF6D. Ensure pos is PBC-unwrapped!
    """
    pos: np.ndarray # these should be unwrapped
    vel: np.ndarray
    ptype_codes: np.ndarray
    write_keys: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
  
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
    Assignments made by FOF6D for writing back into the ParticleStores; n_galaxies is used to flag whether galaxies were found, in which case the executor should build the GroupStore for galaxies.
    """
    write_keys: np.ndarray # indexes back into the ParticleStores
    galaxy_ids: np.ndarray 
    ptype_codes: np.ndarray 
    n_galaxies: int

def find_galaxies(
    particles: dict[str, ParticleStore], 
    simulation: SimulationAttributes, 
    config: dict,
    constants: OctavianConstants,
) -> FOF6DResult:
    """
    Handles the end-to-end galaxy-finding with FOF6D pipeline; writes back to ParticleStore.
    """
    work_data, params = prepare_fof6d_data(particles=particles, simulation=simulation, config=config, constants=constants)

    numba.set_num_threads(params.cores_per_rank) # same as joblib nproc

    parents = np.full(len(work_data.pos), -1, dtype=np.int32)
    dispatch_fof6d(
        positions=work_data.pos, velocities=work_data.vel, parents=parents, 
        ptype_codes=work_data.ptype_codes, starts=work_data.starts, ends=work_data.ends,
        linking_length=params.linking_length, velocity_factor=params.velocity_factor, minstars=params.minstars,
        star_ptype_code=PTYPE_CODES["star"])

    result = extract_galaxies_from_parents(work_data=work_data, parents=parents, minstars=params.minstars)

    store_fof6d_results(particles=particles, result=result)

    return result

def prepare_fof6d_data(
    particles: dict[str, ParticleStore], 
    simulation: SimulationAttributes, 
    config: dict, 
    constants: OctavianConstants
) -> tuple[FOF6DData, FOF6DParameters]:
    """
    Extracts relevant arrays from ParticleStores and FOF6D parameters from the SimulationAttributes & user config. Returns a tuple of:

    - data: FOF6DData dataclass
    - params: FOF6DParameters dataclass
    """
    star_halo_ids = particles["star"]["HaloID"]
    star_counts = np.bincount(star_halo_ids[star_halo_ids >= 0]) # NOTE: work sentinel value into here

    eligible_halos = np.where(star_counts >= config["MINIMUM_STARS_PER_GALAXY"])[0] # disregard halos which would have no galaxies
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
        ptype_list.append(np.full(mask.sum(), PTYPE_CODES[ptype], dtype=np.int8))
        index_list.append(np.arange(particles[ptype].n_particles)[mask])
        hid_list.append(halo_ids[mask])
    
    linking_length = simulation.mis * config["b"]

    params = FOF6DParameters(
        linking_length=linking_length,
        velocity_factor=config["velocity_factor"],
        boxsize=simulation.boxsize,
        minstars=config["MINIMUM_STARS_PER_GALAXY"],
        cores_per_rank=config["nproc"],
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

    ordered_starts = starts[size_order]
    ordered_ends = ends[size_order]

    for halo_idx in range(len(ordered_starts)):
        s, e = ordered_starts[halo_idx], ordered_ends[halo_idx]
        unwrap_positions(positions=all_pos[s:e], boxsize=simulation.boxsize)

    data = FOF6DData(
        pos=all_pos,
        vel=all_vel,
        ptype_codes=all_ptype,
        write_keys=all_write_key,
        starts=ordered_starts,
        ends=ordered_ends,
    )

    return data, params

def extract_galaxies_from_parents(
    work_data: FOF6DData,
    parents: np.ndarray,
    minstars: int,
) -> FOF6DResult:
    """
    Extracts GalIDs from the parents array returned by the FOF6D algorithm.
    """
    n_halos = len(work_data.starts)
    all_keys, all_gids, all_ptype_codes = [], [], []
    galaxy_id_offset = 0

    for halo_idx in range(n_halos):

        s, e = work_data.starts[halo_idx], work_data.ends[halo_idx]

        if parents[s] == -1:
            continue

        halo_parents = parents[s:e]
        component_sizes = np.bincount(halo_parents)
        star_mask = work_data.ptype_codes[s:e] == PTYPE_CODES["star"]
        star_counts = np.bincount(halo_parents[star_mask], minlength=len(component_sizes))

        valid_parents = np.where((component_sizes >= minstars) & (star_counts >= minstars))[0]

        if len(valid_parents) == 0:
            continue

        parent_to_gal_id = np.full(len(component_sizes), -1, dtype=np.int64)
        parent_to_gal_id[valid_parents] = np.arange(len(valid_parents), dtype=np.int64) + galaxy_id_offset

        particle_gal_ids = parent_to_gal_id[halo_parents]
        assigned = particle_gal_ids >= 0

        all_keys.append(work_data.write_keys[s:e][assigned])
        all_gids.append(particle_gal_ids[assigned])
        all_ptype_codes.append(work_data.ptype_codes[s:e][assigned])
        galaxy_id_offset += len(valid_parents)

    if galaxy_id_offset == 0: # guard for an empty group, needs to match type check in signature
        result = FOF6DResult(write_keys=np.empty(0, dtype=np.int64),
                             galaxy_ids=np.empty(0, dtype=np.int64),
                             ptype_codes=np.empty(0, dtype=np.int8),
                             n_galaxies=0)
    else:
        result = FOF6DResult(write_keys=np.concatenate(all_keys),
                             galaxy_ids=np.concatenate(all_gids),
                             ptype_codes=np.concatenate(all_ptype_codes),
                             n_galaxies=galaxy_id_offset)
        
    return result

def store_fof6d_results(particles: dict[str, ParticleStore], result: FOF6DResult) -> None:
    """
    Appends the galaxy-finding results to ParticleStore (and is vectorised compared to the original).
    """
    for ptype in particles:

        particles[ptype]["GalID"] = np.full(particles[ptype].n_particles, -1, dtype=np.int64) # NOTE: sentinel value

        mask = result.ptype_codes == PTYPE_CODES[ptype]

        if not mask.any():
            continue

        particles[ptype]["GalID"][result.write_keys[mask]] = result.galaxy_ids[mask]