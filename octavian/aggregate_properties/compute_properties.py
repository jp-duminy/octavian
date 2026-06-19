"""

Aggregate properties for halos & galaxies. 
The engine room for this file is in group_computations.py & group_helpers.py

"""

# defaults
from dataclasses import dataclass

# others
import numpy as np
from scipy.spatial import KDTree # remember, always pass boxsize

# other Octavian data structures
from octavian.data_management.data_structures import ParticleStore, GroupStore, SimulationAttributes
from octavian.data_management.conventions import CONSTANTS, DTYPES

from octavian.aggregate_properties.group_computations import (
    compute_angular_momentum,
    compute_rotation_quantities,
    compute_radial_quantiles,
    compute_virial_quantities,
)

from octavian.aggregate_properties.group_helpers import (
    sum_per_group,
    count_per_group,
    max_value_per_group,
    max_idx_per_group,
    min_idx_per_group,
    first_idx_per_group,
    broadcast_to_particles,
    sort_by_group,
    weighted_mean_per_group,
)

@dataclass(slots=True) 
class GroupContext: # I'm sure this could have a better name but I feel context -> ctx improves readability
    """
    Dataclass containing information about a group particle type.
    """
    # make these on instantiation
    group_name: str
    particle_type: str
    group_idx: np.ndarray       
    n_groups: int

    positions: np.ndarray       
    velocities: np.ndarray      
    masses: np.ndarray          

    # append these as properties run
    counts: np.ndarray | None = None
    group_mass: np.ndarray | None = None
    ref_positions: np.ndarray | None = None     
    ref_velocities: np.ndarray | None = None
    com_positions: np.ndarray | None = None
    com_velocities: np.ndarray | None = None    
    positions_rel: np.ndarray | None = None     
    velocities_rel_com: np.ndarray | None = None
    velocities_rel_ref: np.ndarray | None = None
    radii: np.ndarray | None = None             
    L_mag: np.ndarray | None = None 

def extract_particles(particles: dict[str, ParticleStore], ptypes: list[str], group_key: str) -> tuple[np.ndarray, ...]: 
    """
    Unpacks the ParticleStore for the group, then concatenates the particle arrays across ptypes for vectorisation.

    group_key: the corresponding ID, so HaloID or GalID
    """
    halo_ids_list, group_ids_list = [], [] # ids: group_ids will be halo_ids for halos (redundant)
    masses_list, potentials_list = [], [] # physical quantities
    positions_list, velocities_list = [], [] # kinematics

    for ptype in ptypes: 

        data = particles[ptype]

        halo_ids_list.append(data[group_key])
        halo_ids_list.append(data["HaloID"])
        masses_list.append(data["mass"])
        potentials_list.append(data["potential"])
        positions_list.append(data.get_columns(["x", "y", "z"]))
        velocities_list.append(data.get_columns(["vx", "vy", "vz"]))

    halo_ids, group_ids = np.concatenate(halo_ids_list, dtype=DTYPES["HaloID"]), np.concatenate(group_ids_list, dtype=DTYPES["GalID"])
    masses, potentials = np.concatenate(masses_list, dtype=DTYPES["mass"]), np.concatenate(potentials_list, dtype=DTYPES["potential"])
    positions, velocities = np.vstack(positions_list, dtype=DTYPES["pos"]), np.vstack(velocities_list, dtype=DTYPES["vel"])

    if group_key == "GalID":
        keep = group_ids != -1
        group_ids = group_ids[keep]
        halo_ids = halo_ids[keep]
        masses = masses[keep]
        potentials = potentials[keep]
        positions = positions[keep]
        velocities = velocities[keep]

    return halo_ids, group_ids, masses, potentials, positions, velocities

def compute_counts_and_mass(ctx: GroupContext, group_store: GroupStore) -> None:
    """
    Computes number counts and masses of groups.
    """
    ctx.counts = count_per_group(group_idx=ctx.group_idx, n_groups=ctx.n_groups)

    if ctx.particle_type == "bh":
        ctx.group_mass = max_value_per_group(values=ctx.masses, group_idx=ctx.group_idx, n_groups=ctx.n_groups) # REVIEW: why?
        ctx.group_mass = np.where(np.isfinite(ctx.group_mass), ctx.group_mass, 0.0) # mask out -inf for no-bh groups
    else:
        ctx.group_mass = sum_per_group(values=ctx.masses, group_idx=ctx.group_idx, n_groups=ctx.n_groups)

    group_store[f"n{ctx.particle_type}"] = ctx.counts
    group_store[f"mass_{ctx.particle_type}"] = ctx.group_mass
    
def compute_minimal_potential(ctx: GroupContext, group_store: GroupStore, potentials: np.ndarray | None) -> None:
    """
    Finds potential well of a halo. Do not pass galaxy GroupStore.
    """
    if ctx.particle_type == "total":

        minimum_potential_idx = min_idx_per_group(values=potentials, group_idx=ctx.group_idx, n_groups=ctx.n_groups)
        valid = minimum_potential_idx >= 0 # min_idx_per_group puts -1 for groups where min_idx is undefined

        minimum_potential_pos = np.full((ctx.n_groups, 3), np.nan)
        minimum_potential_vel = np.full((ctx.n_groups, 3), np.nan)
        minimum_potential_pos[valid] = ctx.positions[minimum_potential_idx[valid]]
        minimum_potential_vel[valid] = ctx.velocities[minimum_potential_idx[valid]]

        ctx.ref_positions = minimum_potential_pos
        ctx.ref_velocities = minimum_potential_vel

        for i, d in enumerate(['x', 'y', 'z']):
            group_store[f"minpot_{d}"] = minimum_potential_pos[:, i]
            group_store[f"minpot_v{d}"] = minimum_potential_vel[:, i]

    else:

        ctx.ref_positions = group_store.get_columns(['minpot_x', 'minpot_y', 'minpot_z'])
        ctx.ref_velocities = group_store.get_columns(['minpot_vx', 'minpot_vy', 'minpot_vz'])

def compute_centre_of_mass(ctx: GroupContext, group_store: GroupStore, boxsize: float) -> None:
    """
    Computes centre-of-mass, accounting for PBCs.
    """
    com_positions = np.zeros((ctx.n_groups, 3))
    com_velocities = np.zeros((ctx.n_groups, 3))

    anchor_idx = first_idx_per_group(group_idx=ctx.group_idx, n_groups=ctx.n_groups) # HACK: use first member of each group
    anchor_positions = np.full((ctx.n_groups, 3), np.nan)
    valid_anchors = anchor_idx >= 0
    anchor_positions[valid_anchors] = ctx.positions[anchor_idx[valid_anchors]]

    pos_shifted = ctx.positions - anchor_positions[ctx.group_idx]
    pos_shifted -= boxsize * np.round(pos_shifted / boxsize)

    for d in range(3):
        com_positions[:, d] = sum_per_group(values=(pos_shifted[:,d]*ctx.masses), group_idx=ctx.group_idx, n_groups=ctx.n_groups) / ctx.group_mass
        com_velocities[:, d] = sum_per_group(values=(ctx.velocities[:,d]*ctx.masses), group_idx=ctx.group_idx, n_groups=ctx.n_groups) / ctx.group_mass

    com_positions += anchor_positions # return to box frame
    com_positions %= boxsize # modulo in python handles negatives automatically so this is safe
    ctx.com_velocities = com_velocities

    if ctx.group_name == "galaxies":
        ctx.ref_positions = com_positions
        ctx.ref_velocities = com_velocities

    for i, d in enumerate(['x', 'y', 'z']):
        group_store[f"{d}_{ctx.particle_type}"] = com_positions[:, i]
    for i, d in enumerate(['x', 'y', 'z']):
        group_store[f"v{d}_{ctx.particle_type}"] = com_velocities[:, i]

def compute_relative_quantities(ctx: GroupContext, boxsize: float) -> None:    
    """
    Relative positions and velocities, with PBCs: middle man, does not write to output.
    """
    ctx.positions_rel = ctx.positions - ctx.ref_positions[ctx.group_idx]
    ctx.positions_rel -= boxsize * np.round(ctx.positions_rel / boxsize)
    ctx.velocities_rel_com = ctx.velocities - ctx.com_velocities[ctx.group_idx]
    ctx.velocities_rel_ref = ctx.velocities - ctx.ref_velocities[ctx.group_idx]

    ctx.radii = np.linalg.norm(ctx.positions_rel, axis=1)

def compute_kinematics(ctx: GroupContext, group_store: GroupStore) -> None:
    """
    Kinematics: velocity dispersions & angular momentum.
    """
    dispersion_sums = sum_per_group(values=np.sum(ctx.velocities_rel_com**2, axis=1), group_idx=ctx.group_idx, n_groups=ctx.n_groups)
    velocity_dispersions = np.where(ctx.counts > 0, np.sqrt(dispersion_sums / np.maximum(ctx.counts, 1)), np.nan)

    L, ktot = compute_angular_momentum(pos_rel=ctx.positions_rel, vel_rel=ctx.velocities_rel_ref, 
                                       mass=ctx.masses, group_idx=ctx.group_idx, n_groups=ctx.n_groups)
    ctx.L_mag = np.linalg.norm(L, axis=1)
    alpha = np.arctan2(L[:, 1], L[:, 2])
    beta = np.arcsin(L[:, 0] / ctx.L_mag)

    counter_mass, krot, ktot = compute_rotation_quantities(pos_rel=ctx.positions_rel, vel_rel=ctx.velocities_rel_ref, 
                                                           mass=ctx.masses, group_idx=ctx.group_idx, L_group=L, n_groups=ctx.n_groups)
    BoverT = 2 * counter_mass / ctx.group_mass
    kappa_rot = krot / ktot

    small = (ctx.counts > 0) & (ctx.counts < 3) # groups with fewer than 3 counts have ill-defined rotational quantities (mask away)
    empty = ctx.counts == 0

    for quantity in [velocity_dispersions, ctx.L_mag, alpha, beta, BoverT, kappa_rot]:
        quantity[empty] = np.nan
        quantity[small] = 0.0
    for i in range(3):
        L[empty, i] = np.nan
        L[small, i] = 0.0

    # write back
    group_store[f"velocity_dispersion_{ctx.particle_type}"] = velocity_dispersions

    for i, d in enumerate(['x', 'y', 'z']):
        group_store[f"L{d}_{ctx.particle_type}"] = L[:, i]

    group_store[f"L_{ctx.particle_type}"] = ctx.L_mag
    group_store[f"ALPHA_{ctx.particle_type}"] = alpha
    group_store[f"BETA_{ctx.particle_type}"] = beta
    group_store[f'BoverT_{ctx.particle_type}'] = BoverT
    group_store[f'kappa_rot_{ctx.particle_type}'] = kappa_rot

def compute_radial_quantities(ctx: GroupContext, group_store: GroupStore) -> None:
    """
    Radial quantities: r20, half-mass, r80.
    """
    quantiles = np.array([0.2, 0.5, 0.8])
    quantile_names = ["r20", "half_mass", "r80"]

    radial_results = compute_radial_quantiles(radius=ctx.radii, mass=ctx.masses, group_idx=ctx.group_idx, n_groups=ctx.n_groups, 
                                              quantiles=quantiles)

    for q, col_name in enumerate(quantile_names):
        group_store[f"radius_{ctx.particle_type}_{col_name}"] = radial_results[:, q]

def compute_halo_quantities(ctx: GroupContext, group_store: GroupStore, r200_factor: float, rhocrit: float) -> None:
    """
    Quantities (which as of 19/06 should only be computed for halos).
    """
    r200 = r200_factor * ctx.group_mass**(1./3.) # REVIEW: units, comoving?
    v_circ = np.sqrt(CONSTANTS.G_VCIRC * ctx.group_mass / r200)
    temperature = CONSTANTS.VIRIAL_TEMP_FACTOR * (v_circ / 100.0) ** 2
    spin_param = ctx.L_mag / (np.sqrt(2) * ctx.group_mass * v_circ * r200)

    factors = np.array([200., 500., 2500.])
    virial_radius, virial_mass = compute_virial_quantities(radius=ctx.radii, mass=ctx.masses, group_idx=ctx.group_idx, n_groups=ctx.n_groups,
                                                   rhocrit=rhocrit, factors=factors)

    for f, factor in enumerate(int(factor)): # cast array to ints for f-string name
        group_store[f"radius_{factor}_c"] = virial_radius[:, f]
        group_store[f"mass_{factor}_c"] = virial_mass[:, f]

    empty = ctx.counts == 0
    for arr in [r200, v_circ, temperature, spin_param]:
        arr[empty] = np.nan

    group_store[f"r200"] = r200
    group_store[f"circular_velocity"] = v_circ
    group_store[f"temperature"] = temperature # NOTE: this is virial temperature, maybe worth making explicit
    group_store[f"spin_param"] = spin_param

