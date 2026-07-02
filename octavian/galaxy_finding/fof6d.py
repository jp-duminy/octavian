"""

Octavian 6D friends-of-friends galaxy-finding algorithm.

For the deeper workings of the algorithm please refer to the scipy.spatial framework
Original FoF: Davis et al. 1985, doi: 10.1086/163168

"""

# type checking (semantic)
from __future__ import annotations
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_management import ParticleStore, GroupStore, SimulationAttributes

from octavian.data_management import CONSTANTS, DTYPES
# default library
from dataclasses import dataclass

# other libraries
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

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
    kernel_table: np.ndarray
    position_LL: float
    velocity_LL: float
    boxsize: float
    minstars: int
    cores_per_rank: int

@dataclass(slots=True, frozen=True)
class FOF6DResult:
    """
    Assignments made by FOF6D for writing back to datamanager.
    """
    write_keys: np.ndarray # for now, indexes back into datamanager
    galaxy_ids: np.ndarray 
    ptypes: np.ndarray # move to integer codings eventually
    n_galaxies: int

# NOTE: just pass the simulationattributes dataclass when this gets moved to read-in 
def compute_mean_interparticle_separation(dm_mass_total: float, n_dm: int, baryonic_mass_total: float, omega_matter: float,
                                          h: float, a: float, time_s: float) -> float:
    """
    Computes the mean separation between particles (lambda) across the box.
    """
    G_CODE = CONSTANTS.G_CGS / CONSTANTS.KPC_CM**3 * CONSTANTS.M_SUN_G * (time_s / a)**2 # G [kpc^3/(M_sun * t/a)^2]
    H0_CODE = 100 * CONSTANTS.HUBBLE_UNIT * (time_s / a) # H0 = 100h^-1 in hubble units (work in h^-1)

    omega_baryon = baryonic_mass_total / (baryonic_mass_total + dm_mass_total) * omega_matter
    rho_dm = (omega_matter - omega_baryon) * 3.0 * H0_CODE**2 / (8.0 * np.pi * G_CODE) / h
    mean_interparticle_separation = ((dm_mass_total / n_dm / rho_dm)**(1./3.)) / h

    return mean_interparticle_separation # NOTE: old code returned efres and omega_baryon but they were unused

def prepare_fof6d_data(particles: dict[str, ParticleStore], simulation: SimulationAttributes, 
                           config: dict,) -> tuple[list[FOF6DItem], FOF6DParameters]:
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
    nH = rho * config["XH"] / CONSTANTS.PROTON_MASS_G # TODO: move to reader

    dense_mask = (nH > config['nHlim']) & ((temperature < config['Tlim']) | (sfr > 0)) # NOTE: sfr > 0 overrides of the density criterion

    pos_list, vel_list, ptype_list, index_list, hid_list = [], [], [], [], []

    n_dm, dm_mass_total = particles["dm"].n_particles, np.sum(particles["dm"]["mass"])
    baryonic_mass_total = sum(
        particles[pt]["mass"].sum() for pt in ["gas", "star", "bh"] if pt in particles
    )

    for ptype in ["star", "gas", "bh"]:

        if ptype not in particles:
            continue

        halo_ids = particles[ptype]["HaloID"]
        in_range = (halo_ids >= 0) & (halo_ids < len(eligible_set))
        masked_hids = np.where(in_range, halo_ids, 0) # have to mask before & operator 
        
        if ptype == 'gas':
            mask = dense_mask & in_range & eligible_set[masked_hids] # dense criterion for gas specifically
        else:
            mask = in_range & eligible_set[masked_hids]
        
        pos_list.append(particles[ptype]["pos"][mask])
        vel_list.append(particles[ptype]["vel"][mask])
        ptype_list.append(particles[ptype]["ptype"][mask])
        index_list.append(np.arange(particles[ptype].n_particles)[mask])
        hid_list.append(halo_ids[mask])

    mis = compute_mean_interparticle_separation(dm_mass_total=dm_mass_total, n_dm=n_dm, baryonic_mass_total=baryonic_mass_total,
                                                omega_matter=simulation.omega_matter, h=simulation.h, a=simulation.a,
                                                time_s=simulation.time)
    
    b = 0.02 # NOTE: move to config
    fof_LL = mis * b
    vel_LL = 1. # NOTE: represents deviation from velocity dispersion rather than a distance in velocity space
    boxsize = simulation.boxsize
    kernel_table = create_kernel_table(fof_LL)

    params = FOF6DParameters(
        kernel_table=kernel_table,
        position_LL=fof_LL,
        velocity_LL=vel_LL,
        boxsize=boxsize,
        minstars=config['MINIMUM_STARS_PER_GALAXY'],
        cores_per_rank=config['nproc'],
    )

    # NOTE: this copies the same pattern as CGP does on read-in
    all_pos, all_vel = np.concatenate(pos_list, dtype=DTYPES["pos"]), np.concatenate(vel_list, dtype=DTYPES["vel"]) 
    all_ptype, all_halo_ids = np.concatenate(ptype_list), np.concatenate(hid_list)
    all_write_key = np.concatenate(index_list)

    order = np.argsort(all_halo_ids, kind='mergesort')
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
        items.append(FOF6DItem(
            pos=all_pos[s:e],
            vel=all_vel[s:e],
            ptype=all_ptype[s:e],
            write_key=all_write_key[s:e],
        ))

    return items, params

# REVIEW: necessity of these two functions, hangover from Caesar; we are probably efficient enough to refactor this.
# kernel table for fof6d velocity criterion distance weights
def create_kernel_table(fof_LL,ntab=1000):
    kerneltab = np.zeros(ntab+1)
    hinv = 1./fof_LL
    for i in range(ntab):
        r = 1. * i / ntab
        q = 2 * r * hinv # FIXME: double normalisation
        if q > 2: kerneltab[i] = 0.0
        elif q > 1: kerneltab[i] = 0.25 * (2 - q)**3
        else: kerneltab[i] = 1 - 1.5 * q * q * (1 - 0.5 * q)
    return kerneltab

# kernel table lookup
def kernel(r_over_h,kerneltab):
    ntab = len(kerneltab) - 1
    rtab = ntab * r_over_h + 0.5
    itab = rtab.astype(int)
    return kerneltab[itab]

def run_fof6d_algorithm(work_item: FOF6DItem, params: FOF6DParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:

    n = len(work_item.pos)
    if n < params.minstars:
        return []

    tree = KDTree(work_item.pos, boxsize=params.boxsize) # REVIEW: move this to PBCs
    sdm = tree.sparse_distance_matrix(tree, params.position_LL, output_type='coo_matrix')

    rows = sdm.row
    cols = sdm.col
    dists = sdm.data

    # vectorised kernel weights (adapted from Jakub)
    q = dists / params.position_LL
    w = kernel(q, params.kernel_table)  # already works on arrays

    # vectorised velocity differences
    vel_diff = np.linalg.norm(work_item.vel[cols] - work_item.vel[rows], axis=1)

    # vectorised sigma per particle
    weighted_dv_sq = w * vel_diff**2 # same as Jakub (I renamed variables for readability)
    sigmas = np.sqrt(np.bincount(rows, weights=weighted_dv_sq, minlength=n)) #/ np.bincount(rows, weights=w, minlength=n))  FIXME: unnormalised

    # vectorised velocity criterion
    valid = vel_diff <= (params.velocity_LL * sigmas[rows])

    adj = csr_matrix((np.ones(valid.sum()), (rows[valid], cols[valid])), shape=(n, n)) # np.ones matrix; boolean mask with rows, cols
    n_components, labels = connected_components(adj, directed=False) # directed=False means we only care about connections (preserves original logic)

    # split by label — numpy instead of python loop
    label_order = np.argsort(labels)
    sorted_labels = labels[label_order]
    label_splits = np.flatnonzero(np.diff(sorted_labels)) + 1
    component_groups = np.split(label_order, label_splits)

    out_keys = []
    out_gids = []
    out_ptypes = []
    local_gal_id = 0

    for component in component_groups:

        if len(component) < params.minstars:
            continue

        component_ptype = work_item.ptype[component]

        if np.sum(component_ptype == 'star') < params.minstars:
            continue

        out_keys.append(work_item.write_key[component])
        out_gids.append(np.full(len(component), local_gal_id, dtype=np.int64))
        out_ptypes.append(component_ptype)
        local_gal_id += 1

    if local_gal_id == 0: # no galaxies

        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=object), 0

    return np.concatenate(out_keys), np.concatenate(out_gids), np.concatenate(out_ptypes), local_gal_id # local_gal_id is proxy for n_galaxies

def collect_fof6d_results(results: tuple[np.ndarray, np.ndarray, np.ndarray, int]) -> FOF6DResult:
    """
    Collates and unpacks all the results from the FOF6DItems and concatenates them into a result for broadcasting.
    """
    not_empty = [(k, g, p, n) for k, g, p, n in results if n > 0]

    if not not_empty: # NOTE: flag this in the logger when added

        return FOF6DResult(write_keys=np.empty(shape=not_empty), 
                           galaxy_ids=np.empty(shape=not_empty),
                           ptypes=np.empty(shape=not_empty),
                           n_galaxies=0)
    
    keys, gids, ptypes, counts = zip(*results)

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

def find_galaxies(particles: dict[str, ParticleStore], simulation: SimulationAttributes, 
              config: dict) -> FOF6DResult:
    """
    Handles the end-to-end galaxy-finding with FOF6D pipeline; writes back to ParticleStore.
    """
    work_items, params = prepare_fof6d_data(particles=particles, simulation=simulation, config=config)
    result = dispatch_fof6d(items=work_items, params=params)
    store_fof6d_results(particles=particles, result=result)

    return result
