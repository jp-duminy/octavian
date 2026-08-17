"""

The photometry engine room. Photometry has a less well-defined boundary than aggregate_properties between simple helpers and heavyweight physics computations, but for the most part, these functions are the ones which dominate runtime and are called by the run_photometry function.

"""

# other packages
from numba import njit
import numpy as np


@njit(cache=True)
def compute_metal_column_densities(
    star_pos: np.ndarray,
    gas_pos: np.ndarray,
    gas_mass: np.ndarray,
    gas_metallicity: np.ndarray,
    smoothing_lengths: np.ndarray,
    neighbour_offsets: np.ndarray,
    kernel_table: np.ndarray,
    los_axis: int,
    boxsize: float,
) -> np.ndarray:
    """
    Computes dust extinction magnitude A_v for the stars in a galaxy along the LOS. star_pos should be the stars in the galaxy; gas quantities should be halo-level. Returns:

    - Z_col: the total metal column density from gas along the LOS
    """
    # orthogonal axes
    ax0 = (los_axis + 1) % 3
    ax1 = (los_axis + 2) % 3

    sort_order, cell_offsets, n_cells_x, n_cells_y, origin_x, origin_y, cell_width = build_dust_cell_list(
        gas_pos=gas_pos,
        smoothing_lengths=smoothing_lengths,
        ax0=ax0,
        ax1=ax1,
    )

    n_bins = len(kernel_table) - 1
    n_stars = len(star_pos)
    Z_col = np.zeros(shape=n_stars, dtype=np.float64)
    dx = np.empty(3, dtype=np.float64)  # allocate this here and overwrite in the loop

    for i in range(n_stars):  # outer loop over stars
        # the cell in which the star lives
        cx = int((star_pos[i, ax0] - origin_x) / cell_width)
        cy = int((star_pos[i, ax1] - origin_y) / cell_width)
        cx = np.clip(
            cx, a_min=0, a_max=(n_cells_x - 1)
        )  # clip because cells were built on gas so stars can be outside the covered region
        cy = np.clip(cy, a_min=0, a_max=(n_cells_y - 1))

        for j in range(len(neighbour_offsets)):  # loop over cells (cells are 2D so we only check 9 adjacent cells)
            nx = cx + neighbour_offsets[j, 0]
            ny = cy + neighbour_offsets[j, 1]

            if nx < 0 or nx >= n_cells_x or ny < 0 or ny >= n_cells_y:
                continue  # don't need to check neighbouring cells if we are at the edge

            cell_id = nx * n_cells_y + ny  # row major ordering again
            start = cell_offsets[cell_id]  # slice into the sort_order array to get the gas particles in the cell
            end = cell_offsets[cell_id + 1]

            for idx in range(start, end):  # inner loop over gas in each cell
                g = sort_order[idx]

                for d in range(3):  # inherited convention: observer lives at -infinity
                    dx[d] = gas_pos[g, d] - star_pos[i, d]
                    if dx[d] > (0.5 * boxsize):
                        dx[d] -= boxsize
                    if dx[d] < (-0.5 * boxsize):
                        dx[d] += boxsize

                if dx[los_axis] > 0:
                    continue  # if gas is behind star it contributes 0 to attenuation

                b_sq = dx[ax0] ** 2 + dx[ax1] ** 2
                h = smoothing_lengths[g]
                h_sq = h**2

                if b_sq >= h_sq:
                    continue  # gas particles beyond 1 smoothing length away have 0 weight

                b_over_h = np.sqrt(b_sq) / h  # kernel table is keyed by b/h
                table_idx = int(n_bins * b_over_h)
                kernel_weight = kernel_table[table_idx]

                # metal mass with kernel weight normalised to surface element
                Z_col[i] += gas_mass[g] * gas_metallicity[g] * kernel_weight / h_sq

    return Z_col


@njit(cache=True)
def build_dust_cell_list(
    gas_pos: np.ndarray,
    smoothing_lengths: np.ndarray,
    ax0: int,
    ax1: int,
) -> tuple[np.ndarray, ...]:
    """
    Creates a cell linked list for dust attenutation. Returns:

    - sort_order: the indices to sort particles by cell (np.argsort)
    - cell_offsets: where each cell begins in the flat sorted array (classic csr offset)
    - n_cells_{x/y}: the number of cells in the orthogonal directions
    - origin_{x/y}: the origin of the cell list in the orthogonal directions
    - cell_width: the width of each cell (max smoothing length)
    """
    # set the maximum cell width to h_max so a star will always access all gas which contributes to its attenutation
    cell_width = np.max(smoothing_lengths)

    # np.min/max with axis arg is not supported in numba (yet)
    origin_x = np.min(gas_pos[:, ax0])
    origin_y = np.min(gas_pos[:, ax1])
    max_x = np.max(gas_pos[:, ax0])
    max_y = np.max(gas_pos[:, ax1])
    n_cells_x = int((max_x - origin_x) / cell_width) + 1  # padding so particles at the edge are always included
    n_cells_y = int((max_y - origin_y) / cell_width) + 1

    n_particles = len(gas_pos)
    cell_ids = np.empty(shape=n_particles, dtype=np.int32)

    for i in range(n_particles):
        ix = int((gas_pos[i, ax0] - origin_x) / cell_width)
        iy = int((gas_pos[i, ax1] - origin_y) / cell_width)
        cell_idx = ix * n_cells_y + iy  # row-major ordering
        cell_ids[i] = cell_idx

    sort_order = np.argsort(cell_ids, kind="quicksort")
    n_total_cells = n_cells_x * n_cells_y
    cell_offsets = np.zeros(n_total_cells + 1, dtype=np.int64)

    # add number of particles in each cell (offsets becomes the number of particles in cell i-1)
    for i in range(n_particles):
        cell_offsets[cell_ids[i] + 1] += 1

    # then add per-cell counts to the next offset so offsets becomes actually particle-indexable
    for i in range(n_total_cells):
        cell_offsets[i + 1] += cell_offsets[i]

    return sort_order, cell_offsets, n_cells_x, n_cells_y, origin_x, origin_y, cell_width
