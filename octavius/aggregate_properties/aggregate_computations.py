"""

The Octavius aggregate properties engine room.

All physics computations are written in numba for JIT compilation. Numba provides an inherent raw speed
advantage (the compilation is cached on the machine, per cache=True), but more importantly, avoids materialising
intermediate arrays where numpy otherwise would.

The engine room functions are designed to collect a bunch of properties in one go, compute_kinematics() at the top
being a good example of such.

Conventions:

g: group-level index
p: particle-level index
d: axis (x, y, z)

Particles belonging to a group are found in offsets[g]:offsets[g+1]

"""

# workhorses
import numpy as np
from numba import (
    njit,
    prange,
)  # NOTE: be careful with prange: it should parallelise over groups but not over particles, otherwise results become non-deterministic
# https://stackoverflow.com/questions/68236463/python-numba-non-deterministic-results


@njit(cache=True, parallel=True)
def compute_kinematics(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    ref_pos: np.ndarray,
    ref_vel: np.ndarray,
    com_vel: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    boxsize: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns a tuple of four arrays. In order:

    - L: (n_groups, 3) array of the angular momentum vector
    - ke_tot: (n_groups) array of total kinetic energy
    - dispersion_sum: (n_groups) array of the mass-weighted velocity dispersion sum
    - I_tensor: (n_groups, 3, 3) array of the inertia tensor
    """
    L = np.zeros(shape=(n_groups, 3))
    ke_tot = np.zeros(shape=n_groups)
    dispersion_sum = np.zeros(shape=n_groups)
    I_tensor = np.zeros(shape=(n_groups, 3, 3))

    for g in prange(n_groups):  # group-level parallel loop
        for idx in range(offsets[g], offsets[g + 1]):  # offsets[g]:offsets[g+1] = n_particles in group
            p = idx_sorted[idx]
            mass = masses[p]

            pos_rel = np.empty(shape=3)
            vel_rel = np.empty(shape=3)
            vel_sq_com = 0.0
            ke = 0.0

            for d in range(3):
                # PBC unwrap
                pos_shifted = positions[p, d] - ref_pos[g, d]
                pos_shifted -= boxsize * np.round(pos_shifted / boxsize)

                # pos/vel relative to centre
                pos_rel[d] = pos_shifted
                vel_shifted = velocities[p, d] - ref_vel[g, d]
                vel_rel[d] = vel_shifted
                vel_rel_com = velocities[p, d] - com_vel[g, d]
                vel_sq_com += vel_rel_com**2
                ke += vel_shifted**2  # NOTE: at this point KE is unphysical

            dispersion_sum[g] += mass * vel_sq_com
            ke_tot[g] += 0.5 * mass * ke

            rx, ry, rz = pos_rel[0], pos_rel[1], pos_rel[2]

            # angular momentum cross product
            px, py, pz = mass * vel_rel[0], mass * vel_rel[1], mass * vel_rel[2]
            L[g, 0] += (ry * pz) - (rz * py)
            L[g, 1] += (rz * px) - (rx * pz)
            L[g, 2] += (rx * py) - (ry * px)

            # inertia tensor diagonals
            I_tensor[g, 0, 0] += mass * (ry**2 + rz**2)
            I_tensor[g, 1, 1] += mass * (rx**2 + rz**2)
            I_tensor[g, 2, 2] += mass * (rx**2 + ry**2)

            # inertia tensor off-diagonals
            I_tensor[g, 0, 1] -= mass * rx * ry
            I_tensor[g, 1, 0] -= mass * rx * ry
            I_tensor[g, 0, 2] -= mass * rx * rz
            I_tensor[g, 2, 0] -= mass * rx * rz
            I_tensor[g, 1, 2] -= mass * ry * rz
            I_tensor[g, 2, 1] -= mass * ry * rz

    return L, ke_tot, dispersion_sum, I_tensor


@njit(cache=True, parallel=True)
def compute_rotational_quantities(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    ref_pos: np.ndarray,
    ref_vel: np.ndarray,
    L_group: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    boxsize: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a tuple of two arrays. In order:

    - counter_rotating_mass: (n_groups) array of the mass in each group which rotates against the bulk motion
    - k_rot: (n_groups) array of total rotational kinetic energies
    """
    counter_rotating_mass = np.zeros(shape=n_groups)
    k_rot = np.zeros(shape=n_groups)

    for g in prange(n_groups):
        for idx in range(offsets[g], offsets[g + 1]):
            p = idx_sorted[idx]
            mass = masses[p]

            pos_rel = np.empty(shape=3)
            vel_rel = np.empty(shape=3)

            # the idea here is we don't want to allocate large intermediate arrays, meaning we need to recompute these for memory purposes
            for d in range(3):  # get coords into the correct reference frame
                pos_shifted = positions[p, d] - ref_pos[g, d]
                pos_shifted -= boxsize * np.round(pos_shifted / boxsize)  # PBCs
                pos_rel[d] = pos_shifted

                vel_shifted = velocities[p, d] - ref_vel[g, d]
                vel_rel[d] = vel_shifted

            rx, ry, rz = pos_rel[0], pos_rel[1], pos_rel[2]
            px, py, pz = mass * vel_rel[0], mass * vel_rel[1], mass * vel_rel[2]

            Lx = ry * pz - rz * py
            Ly = rz * px - rx * pz
            Lz = rx * py - ry * px

            group_Lx, group_Ly, group_Lz = L_group[g, 0], L_group[g, 1], L_group[g, 2]
            L_dot = Lx * group_Lx + Ly * group_Ly + group_Lz * Lz

            if L_dot < 0:  # if particle moves opposite to ordered rotation
                counter_rotating_mass[g] += mass

            # transform to cylindrical coords (much easier to work in this frame; ke_rot formula simplifies)
            cx = (ry * group_Lz) - (rz * group_Ly)
            cy = (rz * group_Lx) - (rx * group_Lz)
            cz = (rx * group_Ly) - (ry * group_Lx)
            rotation_axis_distance = np.sqrt(cx**2 + cy**2 + cz**2)

            if rotation_axis_distance > 0.0:
                circular_velocity = L_dot / (rotation_axis_distance * mass)
                k_rot[g] += 0.5 * mass * circular_velocity**2

    return counter_rotating_mass, k_rot


@njit(cache=True, parallel=True)
def compute_enclosed_mass_radii(
    radii: np.ndarray,
    masses: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    quantiles: list[float],
) -> np.ndarray:
    """
    Returns an (n_groups, n_quantiles) array of the radii where for array[:,i] the values correspond to the radii at which the quantile[i] percentage of mass is enclosed.
    """
    result = np.full(shape=(n_groups, len(quantiles)), fill_value=np.nan)

    for g in prange(n_groups):
        indices = idx_sorted[offsets[g] : offsets[g + 1]]

        radius = radii[indices]
        mass = masses[indices]

        order = np.argsort(radius)  # NOTE: numba freaks out if you try to do stable sort here
        radius = radius[order]
        mass = mass[order]
        cumulative_mass = np.cumsum(mass)
        fraction = cumulative_mass / cumulative_mass[-1]

        for q in range(len(quantiles)):  # slightly odd syntax, but this is the best way to do it for array shape
            idx = np.searchsorted(fraction, quantiles[q])  # returns left

            if idx < len(radius):  # idx from searchsorted flips to right when exceeding the enclosed quantile
                result[g, q] = radius[idx]

    return result


@njit(cache=True, parallel=True)
def compute_virial_quantities(
    radii: np.ndarray,
    masses: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    rhocrit: float,
    factors: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a tuple of (n_groups, n_factors) arrays for virial radius/mass at those factors respectively.
    """
    volume_prefactor = (4.0 * np.pi) / 3.0  # I'm splitting hairs with this optimisation

    result_r, result_m = np.full((n_groups, len(factors)), np.nan), np.full((n_groups, len(factors)), np.nan)

    for g in prange(n_groups):
        indices = idx_sorted[offsets[g] : offsets[g + 1]]
        radius = radii[indices]
        mass = masses[indices]

        order = np.argsort(radius)  # NOTE: numba freaks out if you try to do stable sort here
        radius = radius[order]
        mass = mass[order]
        cumulative_mass = np.cumsum(mass)

        for i in range(len(radius)):
            if radius[i] > 0:  # guard
                overdensity = (
                    cumulative_mass[i] / (volume_prefactor * radius[i] ** 3) / rhocrit
                )  # rhocrit should be comoving

                for f in range(len(factors)):
                    if overdensity >= factors[f]:
                        result_r[g, f] = radius[i]
                        result_m[g, f] = cumulative_mass[i]

    return result_r, result_m


@njit(cache=True, parallel=True)
def compute_vmax_and_rmax(
    radii: np.ndarray,
    masses: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    G: float,
    scale_factor: float,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a tuple of vmax, rmax (n_groups) arrays from the enclosed mass profile; this does do redundant recomputation from virial quantities but separation of concerns is more important.
    """
    result_v, result_r = np.full(shape=n_groups, fill_value=np.nan), np.full(shape=n_groups, fill_value=np.nan)

    for g in prange(n_groups):
        indices = idx_sorted[offsets[g] : offsets[g + 1]]
        radius = radii[indices]
        mass = masses[indices]

        order = np.argsort(radius)  # NOTE: numba freaks out if you try to do stable sort here
        radius = radius[order]
        mass = mass[order]
        cumulative_mass = np.cumsum(mass)
        vmax = -np.inf
        rmax = np.nan

        for i in range(len(radius)):
            if radius[i] > 0:  # guard
                v_circ = np.sqrt(
                    G * cumulative_mass[i] / (radius[i] * scale_factor)
                )  # for unit consistency so all velocities are physical

                if v_circ > vmax:
                    vmax = v_circ
                    rmax = radius[i]

        if vmax > 0:
            result_v[g] = vmax
            result_r[g] = rmax

    return result_v, result_r


@njit(cache=True, parallel=True)
def compute_centre_of_mass(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    anchor_pos: np.ndarray,
    group_mass: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    boxsize: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a tuple of  arrays. In order:

    - com_pos: (n_groups, 3) array of centre-of-mass positions
    - com_vel: (n_groups, 3) array of centre-of-mass velocities
    """
    com_pos = np.zeros(shape=(n_groups, 3))
    com_vel = np.zeros(shape=(n_groups, 3))

    for g in prange(n_groups):
        # numerator
        for idx in range(offsets[g], offsets[g + 1]):
            p = idx_sorted[idx]
            mass = masses[p]

            for d in range(3):
                # PBCs
                pos_shifted = positions[p, d] - anchor_pos[g, d]
                pos_shifted -= boxsize * np.round(pos_shifted / boxsize)

                com_pos[g, d] += pos_shifted * mass  # NOTE: com_pos and com_vel are unphysical at this point
                com_vel[g, d] += velocities[p, d] * mass

        # denominator
        if group_mass[g] > 0.0:
            for d in range(3):
                com_pos[g, d] = (com_pos[g, d] / group_mass[g] + anchor_pos[g, d]) % boxsize
                com_vel[g, d] /= group_mass[g]

    return com_pos, com_vel


@njit(cache=True, parallel=True)
def compute_radii(
    positions: np.ndarray,
    ref_pos: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    boxsize: float,
) -> np.ndarray:
    """
    Computes the radial distance of particles from their group's centre. Returns:

    - radii: (n_particles) array of CSR-index-aligned particle radii
    """
    n_particles = len(idx_sorted)
    radii = np.zeros(shape=n_particles)

    for g in prange(n_groups):
        for idx in range(offsets[g], offsets[g + 1]):
            p = idx_sorted[idx]
            r_sq = 0.0

            for d in range(3):
                dx = positions[p, d] - ref_pos[g, d]
                dx -= boxsize * np.round(dx / boxsize)
                r_sq += dx**2

            radii[idx] = np.sqrt(r_sq)  # slice by idx instead of p, as future functions use CSR-aligned orders

    return radii
