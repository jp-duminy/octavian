"""

Aggregate properties for halos & galaxies. 
The engine room for this file is in group_computations.py & group_helpers.py

"""

# defaults
from dataclasses import dataclass

# others
import numpy as np
from scipy.spatial import KDTree # remember, always pass boxsize
from scipy.sparse import csr_array

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

BARYONIC_PTYPES = ["star", "gas", "bh"]

@dataclass(slots=True) 
class GroupContext: # I'm sure this could have a better name but I feel context -> ctx improves readability
    """
    Dataclass containing information about a group particle type.

    This information is needed for the pipeline (to avoid functions having ten arguments)
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
    com_velocities: np.ndarray | None = None    
    positions_rel: np.ndarray | None = None     
    velocities_rel_com: np.ndarray | None = None
    velocities_rel_ref: np.ndarray | None = None
    radii: np.ndarray | None = None             
    L_mag: np.ndarray | None = None 

def extract_particles(particles: dict[str, ParticleStore], ptypes: list[str], group_key: str) -> tuple[np.ndarray, ...]: 
    """
    Unpacks the ParticleStore(s) for the group, then concatenates the particle arrays across ptypes for vectorisation.

    group_key: the corresponding ID, so HaloID or GalID
    """
    halo_ids_list, group_ids_list = [], [] # ids: group_ids will be halo_ids for halos (redundant)
    masses_list, potentials_list = [], [] # physical quantities
    positions_list, velocities_list = [], [] # kinematics

    for ptype in ptypes: 

        data = particles[ptype]

        group_ids_list.append(data[group_key])
        halo_ids_list.append(data["HaloID"])
        masses_list.append(data["mass"])
        potentials_list.append(data["potential"])
        positions_list.append(data.get_columns(["x", "y", "z"]))
        velocities_list.append(data.get_columns(["vx", "vy", "vz"]))

    halo_ids, group_ids = np.concatenate(halo_ids_list, dtype=DTYPES["HaloID"]), np.concatenate(group_ids_list, dtype=DTYPES["GalID"])
    masses, potentials = np.concatenate(masses_list, dtype=DTYPES["mass"]), np.concatenate(potentials_list, dtype=DTYPES["potential"])
    positions, velocities = np.vstack(positions_list).astype(DTYPES["pos"], copy=False), np.vstack(velocities_list).astype(DTYPES["vel"], copy=False)

    if group_key == "GalID":
        keep = group_ids != -1
        group_ids = group_ids[keep]
        halo_ids = halo_ids[keep]
        masses = masses[keep]
        potentials = potentials[keep]
        positions = positions[keep]
        velocities = velocities[keep]

    return halo_ids, group_ids, masses, potentials, positions, velocities

def _compute_counts_and_mass(ctx: GroupContext, group_store: GroupStore) -> None:
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
    
def _compute_minimal_potential(ctx: GroupContext, group_store: GroupStore, potentials: np.ndarray | None) -> None:
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

def _compute_centre_of_mass(ctx: GroupContext, group_store: GroupStore, boxsize: float) -> None:
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

def _compute_relative_quantities(ctx: GroupContext, boxsize: float) -> None:    
    """
    Relative positions and velocities, with PBCs: middle man, does not write to output.
    """
    ctx.positions_rel = ctx.positions - ctx.ref_positions[ctx.group_idx]
    ctx.positions_rel -= boxsize * np.round(ctx.positions_rel / boxsize)
    ctx.velocities_rel_com = ctx.velocities - ctx.com_velocities[ctx.group_idx]
    ctx.velocities_rel_ref = ctx.velocities - ctx.ref_velocities[ctx.group_idx]

    ctx.radii = np.linalg.norm(ctx.positions_rel, axis=1)

def _compute_kinematics(ctx: GroupContext, group_store: GroupStore) -> None:
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

def _compute_radial_quantities(ctx: GroupContext, group_store: GroupStore) -> None:
    """
    Radial quantities: r20, half-mass, r80.
    """
    quantiles = np.array([0.2, 0.5, 0.8])
    quantile_names = ["r20", "half_mass", "r80"]

    radial_results = compute_radial_quantiles(radius=ctx.radii, mass=ctx.masses, group_idx=ctx.group_idx, n_groups=ctx.n_groups, 
                                              quantiles=quantiles)

    for q, col_name in enumerate(quantile_names):
        group_store[f"radius_{ctx.particle_type}_{col_name}"] = radial_results[:, q]

def _compute_halo_quantities(ctx: GroupContext, group_store: GroupStore, r200_factor: float, rhocrit: float) -> None:
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

    for f, factor in enumerate(factors.astype(int)): # cast array to ints for f-string name
        group_store[f"radius_{factor}_c"] = virial_radius[:, f]
        group_store[f"mass_{factor}_c"] = virial_mass[:, f]

    empty = ctx.counts == 0
    for arr in [r200, v_circ, temperature, spin_param]:
        arr[empty] = np.nan

    group_store[f"r200"] = r200
    group_store[f"circular_velocity"] = v_circ
    group_store[f"temperature"] = temperature # NOTE: this is virial temperature, maybe worth making explicit
    group_store[f"spin_param"] = spin_param

def _prepare_hydrogen_fractions(gas: ParticleStore, XH: float) -> None:
    """
    Derive HI/H2 masses from snapshot information, mutates the gas ParticleStore.
    """
    fHI = gas["nh"].copy()
    fH2 = gas["fH2"]

    # enforce mass conservation: fHI + fH2 <= 1
    not_conserving = (fHI + fH2) > 1.0
    fHI[not_conserving] = 1.0 - fH2[not_conserving]

    mass = gas["mass"]
    gas["fHI"] = fHI
    gas["mass_HI"] = XH * fHI * mass
    gas["mass_H2"] = XH * fH2 * mass

def _assign_parent_halo_indices(particles: dict[str, ParticleStore], galaxies: GroupStore, halos: GroupStore) -> None:
    """
    Assigns galaxies their parent halo indices (slightly hacky, assigns based on membership of first).

    This may move elsewhere depending on how we do subhalos with FOF6D.
    """    
    gids_list, hids_list = [], []

    for ptype in BARYONIC_PTYPES:

        store = particles[ptype]
        gids_list.append(store["GalID"])
        hids_list.append(store["HaloID"])

    all_gids, all_hids = np.concatenate(gids_list, dtype=DTYPES["GalID"]), np.concatenate(hids_list, dtype=DTYPES["HaloID"])
    in_galaxy = all_gids >= 0
    all_gids, all_hids = all_gids[in_galaxy], all_hids[in_galaxy]

    galaxy_idx = galaxies.get_indexer(group_id=all_gids)
    first_particle_idx = first_idx_per_group(group_idx=galaxy_idx, n_groups=galaxies.n_groups)
    valid = first_particle_idx >= 0 # NOTE: add note in logger if this triggers
    galaxy_halo_id = np.full(shape=galaxies.n_groups, fill_value=-1, dtype=DTYPES["HaloID"])
    galaxy_halo_id[valid] = all_hids[first_particle_idx[valid]]

    parent_halo_index = halos.get_indexer(group_id=galaxy_halo_id)
    galaxies["parent_halo_index"] = parent_halo_index

def compute_gas_properties(gas: ParticleStore, group_store: GroupStore, group_idx: np.ndarray, nHlim: float) -> None:
    """
    Computes gas-specific properties; second block is for CGM.
    """
    valid = group_idx >= 0 # indexer assigns -1 to particles not in groups
    group_idx = group_idx[valid]
    n_groups = group_store.n_groups

    temperatures = gas["temperature"][valid]
    metallicities = gas["metallicity"][valid]
    sfrs = gas["sfr"][valid]
    masses = gas["mass"][valid] # particle-level
    group_mass = group_store["mass_gas"] # group-level (computed earlier in cgp)

    mass_HI = sum_per_group(values=gas["mass_HI"][valid], group_idx=group_idx, n_groups=n_groups)
    mass_H2 = sum_per_group(values=gas["mass_H2"][valid], group_idx=group_idx, n_groups=n_groups)
    sfr = sum_per_group(values=sfrs, group_idx=group_idx, n_groups=n_groups)
    metal_mass = sum_per_group(values=(metallicities * masses), group_idx=group_idx, n_groups=n_groups)
    metal_sfr = sum_per_group(values=(metallicities * sfrs), group_idx=group_idx, n_groups=n_groups)
    temp_mass = sum_per_group(values=(temperatures * masses), group_idx=group_idx, n_groups=n_groups)

    group_store["mass_HI"] = mass_HI
    group_store["mass_H2"] = mass_H2
    group_store["sfr"] = sfr
    group_store["metallicity_mass_weighted"] =  metal_mass / group_mass
    group_store["metallicity_sfr_weighted"] =  metal_sfr / sfr
    group_store["temp_mass_weighted"] = temp_mass / group_mass

    # cgm 
    rhos = gas["rho"][valid]
    cgm_criterion = rhos < nHlim
    cgm_idx = group_idx[cgm_criterion]
    cgm_masses = masses[cgm_criterion]
    cgm_temperatures = temperatures[cgm_criterion]
    cgm_metallicities = metallicities[cgm_criterion]

    cgm_mass = sum_per_group(values=cgm_masses, group_idx=cgm_idx, n_groups=n_groups)
    cgm_temp_mass = sum_per_group(values=(cgm_temperatures * cgm_masses), group_idx=cgm_idx, n_groups=n_groups)
    cgm_temp_metal = sum_per_group(values=(cgm_temperatures * cgm_masses * cgm_metallicities), group_idx=cgm_idx, n_groups=n_groups)
    cgm_metal_mass = sum_per_group(values=(cgm_masses * cgm_metallicities), group_idx=cgm_idx, n_groups=n_groups)

    group_store["mass_cgm"] = cgm_mass
    group_store["temp_mass_weighted_cgm"] = cgm_temp_mass / cgm_mass
    group_store["temp_metal_weighted_cgm"] = cgm_temp_metal / cgm_temp_mass
    group_store["metallicity_mass_weighted_cgm"] = cgm_metal_mass / cgm_mass
    group_store["metallicity_temp_weighted_cgm"] = cgm_temp_metal / cgm_metal_mass

def compute_star_properties(star: ParticleStore, group_store: GroupStore, group_idx: np.ndarray) -> None:
    """
    Computes star-specific properties.
    """
    valid = group_idx >= 0 # indexer assigns -1 to particles not in groups
    group_idx = group_idx[valid]
    n_groups = group_store.n_groups

    metallicities = star["metallicity"][valid]
    ages = star["age"][valid]
    masses = star["mass"][valid] # particle-level
    group_mass = group_store["mass_star"] # group-level

    metal_mass = sum_per_group(values=(masses * metallicities), group_idx=group_idx, n_groups=n_groups)
    age_mass = sum_per_group(values=(ages * masses), group_idx=group_idx, n_groups=n_groups)
    age_metal = sum_per_group(values=(ages * masses * metallicities), group_idx=group_idx, n_groups=n_groups)

    group_store["metallicity_stellar"] = metal_mass / group_mass
    group_store["age_mass_weighted"] = age_mass / group_mass
    group_store["age_metal_weighted"] = age_metal / metal_mass

def compute_bh_properties(bh: ParticleStore, group_store: GroupStore, group_idx: np.ndarray, edd_factor: float) -> None:
    """
    Computes black-hole specific properties.

    Note bh mass for galaxies is the most massive black hole, not the total.
    """
    valid = group_idx >= 0 # indexer assigns -1 to particles not in groups
    group_idx = group_idx[valid]
    n_groups = group_store.n_groups

    masses = bh["mass"][valid]
    bhmdots = bh["bhmdot"][valid]

    max_idx = max_idx_per_group(values=masses, group_idx=group_idx, n_groups=n_groups) # also assigns -1 as sentinel
    with_bh = max_idx >= 0

    mass = np.full(shape=n_groups, fill_value=np.nan) # split across line for when more properties are added
    bhmdot = np.full(shape=n_groups, fill_value=np.nan)

    mass[with_bh] = masses[max_idx[with_bh]]
    bhmdot[with_bh] = bhmdots[max_idx[with_bh]]

    group_store["bhmdot"] = bhmdot
    group_store["bh_fedd"] = bhmdot / (edd_factor * mass)

def compute_local_densities(group_store: GroupStore, boxsize: float, radii: list[float]) -> None:
    """
    Computes local mass and number densities for groups.
    """
    pos = group_store.get_columns(keys=["x_total", "y_total", "z_total"])
    mass = group_store["mass_total"]
    n_groups = len(mass)

    r_max = np.max(radii)
    tree = KDTree(data=pos, boxsize=boxsize)
    sdm = tree.sparse_distance_matrix(other=tree, max_distance=r_max, output_type="coo_array") # if your scipy is pre-1.18, use "coo_matrix"

    for radius in radii:

        in_range = sdm.data <= radius
        valid_array = np.ones(in_range.sum())
        adj = csr_array(arg1=(valid_array, (sdm.row[in_range],sdm.col[in_range])), shape=(n_groups,n_groups)) # unhelpful argument name

        local_mass = adj @ mass
        local_number_count = adj @ np.ones(shape=n_groups)
        volume = 4./3. * np.pi * radius**3

        group_store[f"local_mass_density_{int(radius)}"] = local_mass / volume
        group_store[f"local_number_density_{int(radius)}"] = local_number_count / volume  

def compute_galaxy_aperture_masses(particles: dict[str, ParticleStore], galaxies: GroupStore, boxsize: float, aperture_size: float) -> None:
    """
    Computes mass in an aperture of defined size around galaxies.

    I know this code looks really messy but I don't know how it can be tidied up.
    """
    pos_list, mass_list = [], []
    ptype_list, hids_list = [], []

    ptypes = ["star", "gas", "bh", "dm"]
    ptype_to_int = {p: i for i, p in enumerate(ptypes)} # integer comparison is more straightforward

    for ptype in ptypes:

        data = particles[ptype]
        pos_list.append(data.get_columns(["x", "y", "z"]))
        mass_list.append(data["mass"])
        ptype_list.append(np.full(shape=len(data), fill_value=ptype_to_int[ptype], dtype=DTYPES["ptype"]))
        hids_list.append(data["HaloID"])

    all_pos, all_mass = np.concatenate(pos_list, dtype=DTYPES["pos"]), np.concatenate(mass_list, dtype=DTYPES["mass"])
    all_ptypes, all_hids = np.concatenate(ptype_list, dtype=DTYPES["ptype"]), np.concatenate(hids_list, dtype=DTYPES["HaloID"])
    
    # original version kind of hacked the hydrogens into ptypes, do them explicitly instead
    all_mass_HI, all_mass_H2 = np.zeros_like(a=all_mass), np.zeros_like(a=all_mass)
    gas_start = len(particles["star"]) # HACK: gas offset is length of preceding start array (will break if changed)
    gas_end = gas_start + len(particles["gas"])
    all_mass_HI[gas_start:gas_end] = particles["gas"]["mass_HI"]
    all_mass_H2[gas_start:gas_end] = particles["gas"]["mass_H2"]

    order, unique_halos, halo_start, halo_end = sort_by_group(group_ids=all_hids) # sort particles by halo
    all_pos, all_mass = all_pos[order], all_mass[order]
    all_ptypes = all_ptypes[order]
    all_mass_HI, all_mass_H2 = all_mass_HI[order], all_mass_H2[order]

    # pre-sort galaxies by parent halo
    gal_order, halos_with_galaxies, gal_start, gal_end = sort_by_group(group_ids=galaxies["parent_halo_index"]) # sort galaxies by halo
    gal_pos = galaxies.get_columns(["x_total", "y_total", "z_total"])

    result = np.zeros(shape=(galaxies.n_groups,len(ptypes)))
    result_HI, result_H2 = np.zeros(shape=galaxies.n_groups), np.zeros(shape=galaxies.n_groups)

    for h in range(len(unique_halos)):

        halo_id = unique_halos[h]
        halo_pos, halo_mass = all_pos[halo_start[h]:halo_end[h]], all_mass[halo_start[h]:halo_end[h]]
        halo_mass_HI, halo_mass_H2 = all_mass_HI[halo_start[h]:halo_end[h]], all_mass_H2[halo_start[h]:halo_end[h]]
        halo_ptypes = all_ptypes[halo_start[h]:halo_end[h]]
        galaxy_halo_idx = np.searchsorted(a=halos_with_galaxies, v=halo_id) # find where the hid sits in halos_with_galaxies

        if galaxy_halo_idx >= len(halos_with_galaxies) or halos_with_galaxies[galaxy_halo_idx] != halo_id:
            continue # skip halos with no galaxies

        gal_indices = gal_order[gal_start[galaxy_halo_idx]:gal_end[galaxy_halo_idx]]

        tree = KDTree(data=halo_pos, boxsize=boxsize)
        neighbor_lists = tree.query_ball_point(x=gal_pos[gal_indices], r=aperture_size)

        for galaxies_idx_local, neighbours in enumerate(neighbor_lists):

            if len(neighbours) == 0:
                continue

            neighbours = np.array(neighbours)
            result_HI[gal_indices[galaxies_idx_local]] = halo_mass_HI[neighbours].sum()
            result_H2[gal_indices[galaxies_idx_local]] = halo_mass_H2[neighbours].sum()
            masses_by_type = np.bincount(x=halo_ptypes[neighbours], weights=halo_mass[neighbours], minlength=len(ptypes))
            result[gal_indices[galaxies_idx_local], :] = masses_by_type

    for i, name in enumerate(ptypes):
        galaxies[f"mass_{name}_30kpc"] = result[:, i]

    galaxies["mass_total_30kpc"] = result[:, :len(ptypes)].sum(axis=1)

def compute_common_properties(particles: dict[str, ParticleStore], group_store: GroupStore, sim: SimulationAttributes, group_name: str, 
                              ptype: str, ptypes: list[str], group_key: str) -> None:
    """
    Computes properties common to both halos & galaxies for specified particle type(s).
    """
    halo_ids, group_ids, masses, potentials, positions, velocities = \
        extract_particles(particles=particles, ptypes=ptypes, group_key=group_key) # NOTE: potentials only used for halos
    
    if len(masses) == 0:
        return
    
    group_idx = group_store.get_indexer(group_ids)

    ctx = GroupContext(
        group_name=group_name,
        particle_type=ptype,
        group_idx=group_idx,
        n_groups=group_store.n_groups,
        positions=positions,
        velocities=velocities,
        masses=masses,
    )

    _compute_counts_and_mass(ctx=ctx, group_store=group_store)
    _compute_centre_of_mass(ctx=ctx, group_store=group_store, boxsize=sim.boxsize)

    if group_name == "halos": # halo and galaxy centres are defined differently
        _compute_minimal_potential(ctx=ctx, group_store=group_store, potentials=potentials)

    _compute_relative_quantities(ctx=ctx, boxsize=sim.boxsize)
    _compute_kinematics(ctx=ctx, group_store=group_store)
    _compute_radial_quantities(ctx=ctx, group_store=group_store)

    if group_name == "halos" and ptype == "total":
        # NOTE: this is not a common property but requires the context dataclass so for sensibility purposes it goes here
        _compute_halo_quantities(ctx=ctx, group_store=group_store, r200_factor=sim.r200_factor, rhocrit=sim.rhocrit)

def compute_aggregate_properties(particles: dict[str, ParticleStore], groups: dict[str, GroupStore], sim: SimulationAttributes,
                                 config: dict) -> None:
    """
    Aggregate properties orchestrator, loads all requisite data and drops unneeded columns after use. 

    Calls all compute functions for both halos and galaxies.
    """    
    PTYPE_PASSES = [
        ("total",   config["ptypes"]),
        ("dm",      ["dm"]),
        ("baryon",  BARYONIC_PTYPES), # NOTE: this means we reallocate large arrays for baryons, but I'm unsure how to optimise this
        ("gas",     ["gas"]),
        ("star",    ["star"]),
        ("bh",      ["bh"]),
    ]

    group_keys = {"halos": "HaloID", "galaxies": "GalID"}
    _prepare_hydrogen_fractions(gas=particles["gas"], XH=config["XH"])

    for group_name in config["groups"]:

        store = groups[group_name]
        group_key = group_keys[group_name]

        for particle_type, ptypes in PTYPE_PASSES:

            compute_common_properties(
                particles=particles, group_store=store, group_name=group_name, sim=sim,
                ptype=particle_type, ptypes=ptypes, group_key=group_key,
            )

        gas_idx = store.get_indexer(group_id=particles["gas"][group_key])
        compute_gas_properties(gas=particles["gas"], group_store=store, group_idx=gas_idx, nHlim=config["nHlim"])

        star_idx = store.get_indexer(group_id=particles["star"][group_key])
        compute_star_properties(star=particles["star"], group_store=store, group_idx=star_idx)

        bh_idx = store.get_indexer(group_id=particles["bh"][group_key])
        compute_bh_properties(bh=particles["bh"], group_store=store, group_idx=bh_idx, edd_factor=CONSTANTS.EDD_FACTOR)

        if group_name == "galaxies":
            _assign_parent_halo_indices(particles=particles, galaxies=store, halos=groups["halos"])
            compute_galaxy_aperture_masses(particles=particles, galaxies=store, boxsize=sim.boxsize, aperture_size=30.)

        compute_local_densities(group_store=store, boxsize=sim.boxsize, radii=[300., 1000., 3000.])
        