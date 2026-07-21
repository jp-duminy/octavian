"""

Utility functions for working with compressed sparse-row format (CSR) representations of data.

"""

import numpy as np


def build_group_csr(group_idx: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a tuple of (offsets, sorted_indices) into the group_idx array.

    group_idx must be raw and full-length (i.e. no sentinel value mask applied).
    """
    in_group = group_idx >= 0
    masked_idx = group_idx[in_group]
    counts = np.bincount(masked_idx, minlength=n_groups)
    offsets = np.empty(n_groups + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts)

    global_indices = in_group.nonzero()[
        0
    ]  # reindexing needs to happen on the original group_idx order / .nonzero() returns a tuple
    sorted_indices = global_indices[
        np.argsort(masked_idx, kind="stable")
    ]  # .nonzero() returns a tuple (array is first)

    return offsets, sorted_indices
