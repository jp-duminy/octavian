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
