"""

The Octavian CAP engine room.

Vectorised CAP means intermediate array allocations are made by numpy. JIT-compiled numba functions avoid creating these intermediates; a good example is the centre-of-mass computations, which need intermediate position and velocity Nx3 arrays. Sometimes numba also makes inherently more sense.

Sometimes we get one physical quantity along the way of finding another, in which case these functions return those for free. This means at first glance you may wonder why, for example, L and KE are returned together, but otherwise we're doing wasted computations.

"""

import numpy as np
from numba import njit # NOTE: prange and parallel=True can lead to non-deterministic results https://stackoverflow.com/questions/68236463/python-numba-non-deterministic-results

@njit(cache=True)
def compute_L_and_KE(pos_rel: np.ndarray, vel_rel: np.ndarray, masses: np.ndarray, group_idx: np.ndarray, 
                               n_groups: int) -> np.ndarray:
    """
    Returns a tuple of two arrays. In order:

    - (n_groups, 3) array of angular momenta.
    - (n_groups) array of total KEs.
    """
    L = np.zeros(shape=(n_groups, 3)) # the nature of the cross product means we'd have to do 3 bincount calls, so do cross product explicitly
    k_tot = np.zeros(shape=n_groups) # there's no reason for these two to go together, but you get KE for free in the loop

    for i in range(len(masses)):

        g = group_idx[i] # corresponding group for each value

        mass = masses[i]
        rx, ry, rz = pos_rel[i,0], pos_rel[i,1], pos_rel[i,2]
        vx, vy, vz = vel_rel[i, 0], vel_rel[i, 1], vel_rel[i, 2]

        k_tot[g] += 0.5 * mass * (vx**2 + vy**2 + vz**2)

        px, py, pz = mass * vx, mass * vy, mass * vz

        L[g,0] += (ry * pz) - (rz * py)
        L[g,1] += (rz * px) - (rx * pz)
        L[g,2] += (rx * py) - (ry * px)
    
    return L, k_tot

@njit(cache=True)
def compute_rotational_quantities(pos_rel: np.ndarray, vel_rel: np.ndarray, masses: np.ndarray, group_idx: np.ndarray,
                                  L_group: np.ndarray, n_groups: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    Returns a tuple of two arrays. In order:

    - (n_groups) array of counter rotating masses.
    - (n_groups) array of total angular KEs.
    """
    counter_rotating_mass = np.zeros(shape=n_groups) 
    k_rot = np.zeros(shape=n_groups) # like L and KE, you get rotational KE for free in this one

    for i in range(len(masses)):

        g = group_idx[i] # corresponding group for each value

        mass = masses[i]
        rx, ry, rz = pos_rel[i,0], pos_rel[i,1], pos_rel[i,2]
        px, py, pz = mass * vel_rel[i, 0], mass * vel_rel[i, 1], mass * vel_rel[i, 2]

        Lx = ry * pz - rz * py
        Ly = rz * px - rx * pz
        Lz = rx * py - ry * px

        group_Lx, group_Ly, group_Lz = L_group[g, 0], L_group[g, 1], L_group[g, 2]
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
def compute_relative_quantities(positions: np.ndarray, velocities: np.ndarray, ref_pos: np.ndarray, ref_vel: np.ndarray,
                                com_vel: np.ndarray, group_idx: np.ndarray, n_groups: int, n_particles: int,
                                boxsize: float) -> tuple[np.ndarray, ...]:
    """
    Returns a tuple of 3x (n_particles, 3) arrays, 1x (n_particles), and 1x (n_groups). In order:

    - relative positions
    - velocities relative to com
    - velocities relative to reference point
    - radii
    - dispersion_sum
    """
    pos_rel = np.empty(shape=(n_particles, 3))
    vel_rel_com = np.empty(shape=(n_particles, 3))
    vel_rel_ref = np.empty(shape=(n_particles, 3))
    radii = np.empty(shape=n_particles)
    v_squared_sum = np.zeros(shape=n_groups)

    for i in range(n_particles):

        g = group_idx[i]
        r_squared = 0.0
        v_squared = 0.0

        for d in range(3):

            dx = positions[i,d] - ref_pos[g,d]
            dx -= boxsize * np.round(dx / boxsize)
            pos_rel[i,d] = dx

            dv_com = velocities[i,d] - com_vel[g,d]
            vel_rel_com[i,d] = dv_com
            vel_rel_ref[i,d] = velocities[i,d] - ref_vel[g,d]

            r_squared += dx**2
            v_squared += dv_com**2

        radii[i] = np.sqrt(r_squared)
        v_squared_sum[g] += v_squared

    return pos_rel, vel_rel_com, vel_rel_ref, radii, v_squared_sum

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

