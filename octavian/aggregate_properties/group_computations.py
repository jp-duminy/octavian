"""

This file contains numba functions used to compute physical quantities in bulk.

Numba requires quite basic syntax meaning these functions are inherently quite readable.

"""

import numpy as np
from numba import njit, prange, boolean


"""

Physical Quantities

"""

@njit(parallel=True)
def compute_centre_of_mass(pos, vel, mass, group_idx, n_groups):
    com_pos = np.zeros((n_groups, 3))
    com_vel = np.zeros((n_groups, 3))
    total_mass = np.zeros(n_groups)
    
    for i in prange(len(mass)):
        g = group_idx[i]
        m = mass[i]
        total_mass[g] += m
        for d in range(3):
            com_pos[g, d] += pos[i, d] * m
            com_vel[g, d] += vel[i, d] * m
    
    for g in range(n_groups):
        if total_mass[g] > 0:
            for d in range(3):
                com_pos[g, d] /= total_mass[g]
                com_vel[g, d] /= total_mass[g]
    
    return com_pos, com_vel, total_mass

@njit(parallel=True)
def compute_angular_momentum(pos_rel, vel_rel, mass, group_idx, n_groups):
    L = np.zeros((n_groups, 3))
    ktot_sum = np.zeros(n_groups)

    for i in prange(len(mass)):
        g = group_idx[i]
        rx, ry, rz = pos_rel[i, 0], pos_rel[i, 1], pos_rel[i, 2]
        vx, vy, vz = vel_rel[i, 0], vel_rel[i, 1], vel_rel[i, 2]
        m = mass[i]

        px, py, pz = m * vx, m * vy, m * vz
        L[g, 0] += ry * pz - rz * py
        L[g, 1] += rz * px - rx * pz
        L[g, 2] += rx * py - ry * px

        ktot_sum[g] += 0.5 * m * (vx**2 + vy**2 + vz**2)

    return L, ktot_sum

@njit(parallel=True)
def compute_rotation_quantities(pos_rel, vel_rel, mass, group_idx, L_group, n_groups):
    counter_rotating_mass = np.zeros(n_groups)
    krot_sum = np.zeros(n_groups)
    ktot_sum = np.zeros(n_groups)
    
    for i in prange(len(mass)):
        g = group_idx[i]
        rx, ry, rz = pos_rel[i, 0], pos_rel[i, 1], pos_rel[i, 2]
        vx, vy, vz = vel_rel[i, 0], vel_rel[i, 1], vel_rel[i, 2]
        m = mass[i]
        
        px, py, pz = m * vx, m * vy, m * vz
        Lx = ry * pz - rz * py
        Ly = rz * px - rx * pz
        Lz = rx * py - ry * px
        
        Lgx, Lgy, Lgz = L_group[g, 0], L_group[g, 1], L_group[g, 2]
        L_dot = Lx * Lgx + Ly * Lgy + Lz * Lgz
        
        if L_dot < 0:
            counter_rotating_mass[g] += m
        
        cx = ry * Lgz - rz * Lgy
        cy = rz * Lgx - rx * Lgz
        cz = rx * Lgy - ry * Lgx
        rz_cyl = np.sqrt(cx**2 + cy**2 + cz**2)
        
        ktot = 0.5 * m * (vx**2 + vy**2 + vz**2)
        ktot_sum[g] += ktot
        
        if rz_cyl > 0.0:
            krot_sum[g] += 0.5 * (L_dot / rz_cyl)**2 / m
    
    return counter_rotating_mass, krot_sum, ktot_sum

@njit
def compute_radial_quantiles(radius, mass, group_idx, n_groups, quantiles):

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

    result = np.full((n_groups, len(quantiles)), np.nan)
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
        cum = np.cumsum(m)
        frac = cum / cum[-1]
        for q in range(len(quantiles)):
            idx = np.searchsorted(frac, quantiles[q])
            if idx < len(r):
                result[g, q] = r[idx]

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