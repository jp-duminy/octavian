"""

Tests whether the functions which represent membership arrays in CSR format accurately recover membership and properly propagate the multi-membership model. This uses hand-derived solutions as opposed to the procedural generation in the computation tests, since this is fundamentally about indexing and figuring out where data is stored rather than physics.

"""

from octavius.data_management import (
    build_group_csr,
    propagate_membership_csr,
)

import numpy as np

# NOTE: don't change these without good reason as the following functions rely on hand-derived solutions
N_GROUPS = 6
parent = np.array([-1, -1, -1, 0, 0, 3], dtype=np.int64)  # store row indices
depth = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
GROUP_IDX = np.array([0, 0, 0, 3, 3, 5, 4, 4, 1, 1, 1, 2], dtype=np.int64)  # bottom-level depth
# subhalo 3+4 belong to subhalo 0 which is a field halo; subhalo 5 belongs to subhalo 3
# therefore subhalo 0 contains particle sets [0, 1, 2] + [3, 4] + [5] + [6, 7]
# subhalo 1 contains particle set [8, 9, 10]
# subhalo 2 contains particle set [11]


def test_group_csr() -> None:
    """
    Tests csr function build_group_csr.
    """
    offsets, idx_sorted = build_group_csr(group_idx=GROUP_IDX, n_groups=N_GROUPS)

    for g in range(N_GROUPS):
        expected = np.flatnonzero(GROUP_IDX == g)
        result = idx_sorted[offsets[g] : offsets[g + 1]]

        np.testing.assert_array_equal(np.sort(result), expected, err_msg="build_group_csr failed.")


def test_inclusive_csr() -> None:
    """
    Tests the multi-membership list propagation model.
    """
    exclusive_offsets, exclusive_sorted = build_group_csr(group_idx=GROUP_IDX, n_groups=N_GROUPS)

    inclusive_offsets, inclusive_sorted = propagate_membership_csr(
        offsets=exclusive_offsets,
        sorted_indices=exclusive_sorted,
        parent_ids=parent,
        n_groups=N_GROUPS,
    )

    # derived by hand
    expected = {
        0: [0, 1, 2, 3, 4, 5, 6, 7],
        1: [8, 9, 10],
        2: [11],
        3: [3, 4, 5],
        4: [6, 7],
        5: [5],
    }

    for group in range(N_GROUPS):
        result = inclusive_sorted[inclusive_offsets[group] : inclusive_offsets[group + 1]]
        np.testing.assert_array_equal(
            np.sort(result), np.sort(expected[group]), err_msg="propagate_membership_csr failed."
        )
