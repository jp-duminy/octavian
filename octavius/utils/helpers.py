"""

Helper functions for numerical stability and no invalid divisions.

"""

# other packages
import numpy as np
from numba import njit


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


@njit(cache=True)
def unwrap_positions(positions: np.ndarray, centre: np.ndarray, boxsize: float) -> None:
    """
    Unwraps PBCs by anchoring a group of positions to the passed centre; mutates the positions array in-place.
    """
    half_box = boxsize * 0.5
    n_particles = len(positions)

    for axis in range(3):
        for i in range(n_particles):
            delta = positions[i, axis] - centre[axis]

            if delta > half_box:
                positions[i, axis] -= boxsize

            elif delta < -half_box:
                positions[i, axis] += boxsize
