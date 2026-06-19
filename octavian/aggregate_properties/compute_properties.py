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



    

