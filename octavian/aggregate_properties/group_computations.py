"""

This file contains numba functions used to compute physical quantities in bulk.

Numba requires quite basic syntax meaning these functions are inherently quite readable.

"""

import numpy as np
from numba import njit # NOTE: prange and parallel=True can lead to non-deterministic results https://stackoverflow.com/questions/68236463/python-numba-non-deterministic-results

@njit
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

@njit
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

@njit
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

@njit
def compute_virial_quantities(radius, mass, group_idx, n_groups, rhocrit, factors):

    volume_factor = 4.0 / 3.0 * np.pi

    counts = np.zeros(n_groups, dtype=np.int64)
    for i in range(len(mass)):
        counts[group_idx[i]] += 1

    start = np.zeros(n_groups, dtype=np.int64)
    for g in range(1, n_groups):
        start[g] = start[g-1] + counts[g-1]

    idx_sorted = np.empty(len(mass), dtype=np.int64)
    pos = np.zeros(n_groups, dtype=np.int64)
    for i in range(len(mass)):
        g = group_idx[i]
        idx_sorted[start[g] + pos[g]] = i
        pos[g] += 1

    result_r = np.full((n_groups, len(factors)), np.nan)
    result_m = np.full((n_groups, len(factors)), np.nan)

    for g in range(n_groups):
        s = start[g]
        e = s + counts[g]
        if s == e:
            continue
        indices = idx_sorted[s:e]
        r = radius[indices]
        m = mass[indices]
        order = np.argsort(r)
        r = r[order]
        m = m[order]
        cumulative = np.cumsum(m)

        for i in range(len(r)):
            if r[i] > 0:
                overdensity = cumulative[i] / (volume_factor * r[i]**3) / rhocrit
                for f in range(len(factors)):
                    if overdensity >= factors[f]:
                        result_r[g, f] = r[i]
                        result_m[g, f] = cumulative[i]

    return result_r, result_m