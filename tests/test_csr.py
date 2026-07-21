"""

Tests whether the functions which represent membership arrays in CSR format accurately recover membership and properly propagate the multi-membership model.

"""

from octavian.data_management import (
    build_group_csr,
)

import numpy as np

N_GROUPS = 6
parent = np.array([-1, -1, -1, 0, 0, 3], dtype=np.int64)  # store row indices
depth = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
GROUP_IDX = np.array([0, 0, 0, 3, 3, 5, 4, 4, 1, 1, 1, 2], dtype=np.int64)  # 12 particles


def test_group_csr() -> None:
    """
    Tests aggregate helper build_group_csr.
    """
    offsets, idx_sorted = build_group_csr(group_idx=GROUP_IDX, n_groups=N_GROUPS)

    for g in range(N_GROUPS):
        expected = np.flatnonzero(GROUP_IDX == g)
        result = idx_sorted[offsets[g] : offsets[g + 1]]

        np.testing.assert_array_equal(np.sort(result), expected, err_msg="build_group_csr failed.")
