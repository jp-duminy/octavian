"""

The Octavian CAP engine room.

Vectorised CAP means intermediate array allocations are made by numpy. JIT-compiled numba functions avoid creating these intermediates; a good example is the centre-of-mass computations, which need intermediate position and velocity Nx3 arrays. Sometimes numba also makes inherently more sense.

Sometimes we get one physical quantity along the way of finding another, in which case these functions return those for free. This means at first glance you may wonder why, for example, L and KE are returned together, but otherwise we're doing wasted computations.

"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from octavian.aggregate_properties.compute_properties import GroupParticles, GroupRefs

import numpy as np
from numba import njit # NOTE: prange and parallel=True can lead to non-deterministic results https://stackoverflow.com/questions/68236463/python-numba-non-deterministic-results

@njit(cache=True)
def compute_kinematics(particles: GroupParticles, refs: GroupRefs, n_groups: int, n_particles: int,
                             boxsize: float) -> tuple[np.ndarray, ...]:
    """
    Returns a tuple of four arrays. In order:

    - (n_groups, 3) array of angular momenta.
    - (n_groups) array of total KEs.
    - (n_groups) array of velocity dispersions.
    - (n_particles) array of radii relative to reference positions.
    """
    L = np.zeros(shape=(n_groups, 3)) # loop avoids 3 np.bincount calls
    k_tot = np.zeros(shape=n_groups)
    dispersion_sum = np.zeros(shape=n_groups)
    radii = np.empty(shape=n_particles)

    for i in range(n_particles):

        g = particles.group_idx[i] # corresponding group for each value
        mass = particles.masses[i]

        pos_rel = np.empty(3) # originally I had dx, dy, dz and dvx, etc. but the code becomes long
        vel_rel = np.empty(3)
        r_sq = 0.0
        vel_sq_com = 0.0
        ke = 0.0

        for d in range(3):

            pos_shifted = particles.positions[i,d] - refs.ref_pos[g,d]
            pos_shifted -= boxsize * np.round(pos_shifted / boxsize) # PBCs

            pos_rel[d] = pos_shifted
            r_sq += pos_shifted**2

            vel_shifted = particles.velocities[i,d] - refs.ref_vel[g,d]
            vel_rel[d] = vel_shifted
            ke += vel_shifted**2 # NOTE: at this point ke is unphysical

            vel_rel_com = particles.velocities[i,d] - refs.com_vel[g,d]
            vel_sq_com += vel_rel_com**2
        
        radii[i] = np.sqrt(r_sq)
        dispersion_sum[g] += vel_sq_com
        k_tot[g] += 0.5 * mass * ke

        rx, ry, rz = pos_rel[0], pos_rel[1], pos_rel[2]
        px, py, pz = mass * vel_rel[0], mass * vel_rel[1], mass * vel_rel[2]

        L[g,0] += (ry * pz) - (rz * py)
        L[g,1] += (rz * px) - (rx * pz)
        L[g,2] += (rx * py) - (ry * px)

    return L, k_tot, dispersion_sum, radii

@njit(cache=True)
def compute_rotational_quantities(particles: GroupParticles, ref_pos: np.ndarray, ref_vel: np.ndarray, L_group: np.ndarray,
                                    n_groups: int, n_particles: int, boxsize: float) -> tuple[np.ndarray, ...]:
    """
    Returns a tuple of two arrays. In order:

    - (n_groups) array of counter rotating masses.
    - (n_groups) array of total angular KEs.
    """
    counter_rotating_mass = np.zeros(shape=n_groups) 
    k_rot = np.zeros(shape=n_groups) 

    for i in range(n_particles):

        g = particles.group_idx[i]
        mass = particles.masses[i]

        pos_rel = np.empty(3) # originally I had dx, dy, dz and dvx, etc. but the code becomes long
        vel_rel = np.empty(3)

        # the idea here is we don't want to make enormous intermediate arrays, meaning we need to recompute these for memory purposes
        for d in range(3): # get coords into the correct reference frame

            pos_shifted = particles.positions[i,d] - ref_pos[g,d]
            pos_shifted -= boxsize * np.round(pos_shifted / boxsize) # PBCs
            pos_rel[d] = pos_shifted

            vel_shifted = particles.velocities[i,d] - ref_vel[g,d]
            vel_rel[d] = vel_shifted

        rx, ry, rz = pos_rel[0], pos_rel[1], pos_rel[2]
        px, py, pz = mass * vel_rel[0], mass * vel_rel[1], mass * vel_rel[2]

        Lx = ry * pz - rz * py
        Ly = rz * px - rx * pz
        Lz = rx * py - ry * px

        group_Lx, group_Ly, group_Lz = L_group[g,0], L_group[g,1], L_group[g,2]
        L_dot = Lx * group_Lx + Ly * group_Ly + group_Lz * Lz

        if L_dot < 0: # if particle moves opposite to ordered rotation
            counter_rotating_mass[g] += mass

        # transform to cylindrical coords
        cx = ry * group_Lz - rz * group_Ly
        cy = rz * group_Lx - rx * group_Lz
        cz = rx * group_Ly - ry * group_Lx
        rotation_axis_distance = np.sqrt(cx**2 + cy**2 + cz**2)

        if rotation_axis_distance > 0.0:
            circular_velocity = L_dot / (rotation_axis_distance * mass)
            k_rot[g] += 0.5 * mass * circular_velocity**2 

    return counter_rotating_mass, k_rot

@njit(cache=True)
def compute_enclosed_mass_radii(radii: np.ndarray, masses: np.ndarray, offsets: np.ndarray, idx_sorted: np.ndarray, 
                                n_groups: int, quantiles: list[float]) -> np.ndarray:
    """
    Returns an (n_groups, n_quantiles) array of the radii where for array[:,i] the values correspond to the radii at which the quantile[i] percentage of mass is enclosed.
    """
    result = np.full(shape=(n_groups, len(quantiles)), fill_value=np.nan)

    for g in range(n_groups):

        indices = idx_sorted[offsets[g]:offsets[g+1]]
        
        radius = radii[indices]
        mass = masses[indices]

        order = np.argsort(radius) # NOTE: numba freaks out if you try to do stable sort here
        radius = radius[order]
        mass = mass[order]
        cumulative_mass = np.cumsum(mass)
        fraction = cumulative_mass / cumulative_mass[-1]

        for q in range(len(quantiles)): # slightly odd syntax, but this is the best way to do it for array shape

            idx = np.searchsorted(fraction, quantiles[q]) # returns left

            if idx < len(radius): # idx from searchsorted flips to right when exceeding the enclosed quantile
                result[g, q] = radius[idx]

    return result

@njit(cache=True)
def compute_virial_quantities(radii: np.ndarray, masses: np.ndarray, offsets: np.ndarray, idx_sorted: np.ndarray, 
                              n_groups: int, rhocrit: float, factors: list[float]) -> tuple[np.ndarray, ...]:
    """
    Returns a tuple of (n_groups, n_factors) arrays for virial radius and virial mass at those factors respectively.
    """
    volume_prefactor = (4. * np.pi) / 3. # I'm splitting hairs with this optimisation

    result_r, result_m = np.full((n_groups, len(factors)), np.nan), np.full((n_groups, len(factors)), np.nan)

    for g in range(n_groups):

       indices = idx_sorted[offsets[g]:offsets[g+1]]

       radius = radii[indices]
       mass = masses[indices]

       order = np.argsort(radius) # NOTE: numba freaks out if you try to do stable sort here
       radius = radius[order]
       mass = mass[order]
       cumulative_mass = np.cumsum(mass)

       for i in range(len(radius)): # revert to go outwards (multiple factors, so inward breaks)
           
           if radius[i] > 0: # guard
               
               overdensity = cumulative_mass[i] / (volume_prefactor * radius[i]**3) / rhocrit # rhocrit should be comoving

               for f in range(len(factors)):
                   
                   if overdensity >= factors[f]:
                       
                       result_r[g,f] = radius[i]
                       result_m[g,f] = cumulative_mass[i]

    return result_r, result_m

@njit(cache=True)
def compute_centre_of_mass(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, group_idx: np.ndarray,
                           anchor_pos: np.ndarray, group_mass: np.ndarray,  n_groups: int, boxsize: float) -> tuple[np.ndarray, ...]:
    """
    Returns a tuple of (n-groups, 3) arrays. In order:

    - com positions
    - com velocities
    """
    com_pos, com_vel = np.zeros((n_groups, 3)), np.zeros((n_groups, 3))

    # numerator
    for i in range(len(masses)):

        g = group_idx[i]
        mass = masses[i]

        for d in range(3):

            pos_shifted = positions[i,d] - anchor_pos[g,d]
            pos_shifted -= boxsize * np.round(pos_shifted / boxsize) # handles PBCs
            com_pos[g,d] += pos_shifted * mass # NOTE: com_pos and com_vel are unphysical when this loop finishes
            com_vel[g,d] += velocities[i,d] * mass

    # denominator + return to box frame
    for group in range(n_groups): # g is used above

        if group_mass[group] > 0.0:

            for d in range(3):

                com_pos[group, d] = (com_pos[group, d] / group_mass[group] + anchor_pos[group, d]) % boxsize
                com_vel[group, d] /= group_mass[group]

    return com_pos, com_vel

@njit(cache=True)
def compute_radii(positions: np.ndarray, ref_pos: np.ndarray, group_idx: np.ndarray, n_particles: int,
                  boxsize: float) -> np.ndarray:
    """
    Returns an (n_particles) array of radii relative to whatever reference position is passed in (different for groups, hence why a different function is necessary)
    """
    radii = np.empty(shape=n_particles)

    for i in range(n_particles):

        g = group_idx[i]
        r_squared = 0.0

        for d in range(3):

            dx = positions[i,d] - ref_pos[g,d]
            dx -= boxsize * np.round(dx / boxsize)
            r_squared += dx**2

        radii[i] = np.sqrt(r_squared)

    return radii

