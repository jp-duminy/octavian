"""

This file contains helpers which help sorting in CAP. Some of these are wrappers for readability (e.g. the bincounts at the top of the file), and some are numba loops to find things like minimum values etc.

The idea being to improve readability and reduce clutter when working with group-level operations.

"""

import numpy as np
from numba import (
    njit,
    prange,
)  # NOTE: prange and parallel=True can lead to non-deterministic results https://stackoverflow.com/questions/68236463/python-numba-non-deterministic-results

# NOTE: as of 25/06/26, do not njit the bincount wrappers, numba does not support minlength parameter


def sum_per_group(values: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Returns an array where each element is a sum of the quantity of interest (values) per group (bincount wrapper).

    The output is a an array of the total values of the quantity of interest for each group.
    """
    return np.bincount(group_idx, weights=values, minlength=n_groups)  # minlength handles empty groups


@njit(cache=True, parallel=True)
def sum_per_group_2(
    values: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """
    Returns:

    - sum_per_group: (n_groups) array of the sum of quantity 'values' in each group.
    """
    sum_per_group = np.zeros(shape=n_groups)

    for g in prange(n_groups):
        for idx in range(offsets[g], offsets[g + 1]):
            p = idx_sorted[idx]
            sum_per_group[g] += values[p]

    return sum_per_group


def count_per_group(group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Returns an array where each element is the number of occurences per group (bincount wrapper).
    """
    return np.bincount(group_idx, minlength=n_groups)  # minlength handles empty groups


@njit(cache=True, parallel=True)
def count_per_group_2(
    offsets: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """
    Returns:

    count_per_group: an (n_groups) array of the number of members of each group
    """
    count_per_group = np.zeros(shape=n_groups)

    for g in prange(n_groups):
        count_per_group[g] += offsets[g + 1] - offsets[g]

    return count_per_group


@njit(cache=True)
def max_value_per_group(values: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Returns an array of the maximum value (of quantity of interest 'values') for each group.

    np.max() only works on a total array.
    """
    result = np.full(shape=n_groups, fill_value=-np.inf)

    for i in range(len(values)):
        g = group_idx[i]  # corresponding group for each value

        if values[i] > result[g]:  # any real value > -np.inf
            result[g] = values[i]

    return result


@njit(cache=True, parallel=True)
def max_value_per_group_2(values: np.ndarray, offsets: np.ndarray, idx_sorted: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Returns:

    - max_value_per_group: (n_groups) array of the maximum value of values in each group.
    """
    max_value_per_group = np.full(shape=n_groups, fill_value=-np.inf)

    for g in prange(n_groups):
        for idx in range(offsets[g], offsets[g + 1]):
            p = idx_sorted[idx]

            if values[p] > max_value_per_group[g]:
                max_value_per_group[g] = values[p]

    return max_value_per_group


@njit(cache=True)
def min_value_per_group(values: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Returns an array of the minimum value (of quantity of interest 'values') for each group.

    np.min() only works on a total array.
    """
    result = np.full(shape=n_groups, fill_value=np.inf)

    for i in range(len(values)):
        g = group_idx[i]  # corresponding group for each value

        if values[i] < result[g]:  # any real value < np.inf
            result[g] = values[i]

    return result


@njit(cache=True, parallel=True)
def min_value_per_group_2(values: np.ndarray, offsets: np.ndarray, idx_sorted: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Returns:

    - min_value_per_group: (n_groups) array of the minimum value of values in each group.
    """
    min_value_per_group = np.full(shape=n_groups, fill_value=np.inf)

    for g in prange(n_groups):
        for idx in range(offsets[g], offsets[g + 1]):
            p = idx_sorted[idx]

            if values[p] < min_value_per_group[g]:
                min_value_per_group[g] = values[p]

    return min_value_per_group


@njit(cache=True)
def max_idx_per_group(values: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Returns an array of the indices of the member with the maximum value (of quantity of interest 'values') in each group.
    """
    result_val = np.full(shape=n_groups, fill_value=-np.inf)
    result_idx = np.full(shape=n_groups, fill_value=-1, dtype=np.int64)  # -1 sentinel value

    for i in range(len(values)):
        g = group_idx[i]  # corresponding group for each value

        if values[i] > result_val[g]:  # any real value > -np.inf
            result_val[g] = values[i]
            result_idx[g] = i

    return result_idx


@njit(cache=True, parallel=True)
def max_idx_per_group_2(
    values: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """
    Returns:

    - max_idx_per_group: (n_groups) array of the particle-ordered index corresponding to the maximum value of 'values' in each group.
    """
    max_val = np.full(shape=n_groups, fill_value=-np.inf)
    max_idx_per_group = np.full(shape=n_groups, fill_value=-1, dtype=np.int64)  # -1 sentinel value

    for g in prange(n_groups):
        for idx in range(offsets[g], offsets[g + 1]):
            p = idx_sorted[idx]

            if values[p] > max_val[g]:  # any real value > -np.inf
                max_val[g] = values[p]
                max_idx_per_group[g] = p

    return max_idx_per_group


@njit(cache=True)
def min_idx_per_group(values: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Returns an array of the indices of the member with the minimum value (of quantity of interest 'values') in each group.
    """
    result_val = np.full(shape=n_groups, fill_value=np.inf)
    result_idx = np.full(shape=n_groups, fill_value=-1, dtype=np.int64)  # -1 sentinel value

    for i in range(len(values)):
        g = group_idx[i]  # corresponding group for each value

        if values[i] < result_val[g]:  # any real value < np.inf
            result_val[g] = values[i]
            result_idx[g] = i

    return result_idx


@njit(cache=True, parallel=True)
def min_idx_per_group_2(
    values: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """
    Returns:

    - min_idx_per_group: (n_groups) array of the particle-ordered index corresponding to the minimum value of 'values' in each group.
    """
    min_val = np.full(shape=n_groups, fill_value=np.inf)
    min_idx_per_group = np.full(shape=n_groups, fill_value=-1, dtype=np.int64)  # -1 sentinel value

    for g in prange(n_groups):
        for idx in range(offsets[g], offsets[g + 1]):
            p = idx_sorted[idx]

            if values[p] < min_val[g]:  # any real value < np.inf
                min_val[g] = values[p]
                min_idx_per_group[g] = p

    return min_idx_per_group


@njit(cache=True)
def first_idx_per_group(group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Returns an array of the indices of the first member of each group.
    """
    result = np.full(shape=n_groups, fill_value=-1, dtype=np.int64)

    for i in range(len(group_idx)):
        g = group_idx[i]

        if result[g] == -1:
            result[g] = i

    return result


@njit(cache=True, parallel=True)
def first_idx_per_group_2(offsets: np.ndarray, idx_sorted: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Returns:

    - first_idx_per_group: an (n_groups) array of the particle-ordered index corresponding to the first particle in each group.
    """
    first_idx_per_group = np.full(shape=n_groups, fill_value=-1, dtype=np.int64)

    for g in prange(n_groups):
        if offsets[g] < offsets[g + 1]:  # empty group guard
            first_idx_per_group[g] = idx_sorted[offsets[g]]  # offsets[g] is the first particle

    return first_idx_per_group


def guarded_divide(numerator: np.ndarray, denominator: np.ndarray, fill_value: float = np.nan):
    """
    Performs standard division, but returns the fill_value where the denominator is zero.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, fill_value=fill_value, dtype=np.float64),
            where=denominator != 0,
        )


def guarded_arcsin(values: np.ndarray) -> np.ndarray:
    """
    Wrapper around np.arcsin which replicates the behaviour of np.arctan2 to return 0.0 for NaN/inf propagation.
    """
    with np.errstate(invalid="ignore"):
        return np.arcsin(
            np.clip(values, -1.0, 1.0), out=np.zeros_like(values, dtype=np.float64), where=np.isfinite(values)
        )


def sort_by_group(group_ids: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    Constructs slices of the bulk for efficient data processing.

    - sorts array by group_ids
    - finds where each group starts and ends

    Meaning we now have a flat array for quick vectorised operations.
    Similar to the CSR format that forms the basis of Octavian's I/O.
    """
    # guard (in practice, should not happen)
    if len(group_ids) == 0:
        return np.array([], dtype=np.int64), np.array([]), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    order = np.argsort(group_ids, kind="stable")  # stable sort avoids non-deterministic sorting
    sorted_ids = group_ids[order]
    changes = np.flatnonzero(np.diff(sorted_ids))  # find where the difference is nonzero (where a new group starts)

    # find where each group starts
    start = np.empty(len(changes) + 1, dtype=np.int64)  # np.diff shifts the array left; compensate
    start[0] = 0
    start[1:] = changes + 1

    # find where each group ends
    end = np.empty(len(start), dtype=np.int64)
    end[:-1] = start[1:]  # group ends at the next group's starting index
    end[-1] = len(
        group_ids
    )  # no such boundary exists from start[] but can find it from number of particles (len(group_ids))

    unique_ids = sorted_ids[start]

    return order, unique_ids, start, end
