"""

This file contains helpers which help sorting in CAP. Some of these are wrappers for readability (e.g. the bincounts at the top of the file), and some are numba loops to find things like minimum values etc.

The idea being to improve readability and reduce clutter when working with group-level operations.

"""

# workhorses
import numpy as np
from numba import (
    njit,
    prange,
)  # NOTE: be careful with prange: it should parallelise over groups but not over particles, otherwise results become non-deterministic
# https://stackoverflow.com/questions/68236463/python-numba-non-deterministic-results


@njit(cache=True, parallel=True)
def sum_per_group(
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


@njit(cache=True, parallel=True)
def count_per_group(
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


@njit(cache=True, parallel=True)
def max_value_per_group(values: np.ndarray, offsets: np.ndarray, idx_sorted: np.ndarray, n_groups: int) -> np.ndarray:
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


@njit(cache=True, parallel=True)
def min_value_per_group(values: np.ndarray, offsets: np.ndarray, idx_sorted: np.ndarray, n_groups: int) -> np.ndarray:
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


@njit(cache=True, parallel=True)
def max_idx_per_group(
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


@njit(cache=True, parallel=True)
def min_idx_per_group(
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


@njit(cache=True, parallel=True)
def first_idx_per_group(offsets: np.ndarray, idx_sorted: np.ndarray, n_groups: int) -> np.ndarray:
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
