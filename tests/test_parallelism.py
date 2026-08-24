"""

Tests whether the slabs/local subhalos functions produce sensible masks.

"""

import numpy as np
from octavian.data_management.parallel_reading import (
    generate_slabs,
    assign_local_subhalos,
)
from octavian.external_halo_sources import SubhaloInformation
from octavian.data_management import ParticleStore

store = ParticleStore(ptype="star", n_particles=6, is_baryonic=False)
store["HaloID"] = np.array([0, 2, 3, 0, -1, 1], dtype=np.int64)
store["SubhaloID"] = np.array([0, 1, 2, 0, -1, -1], dtype=np.int64)
subhalo_test_particles = {"star": store}

test_subhalo_info = SubhaloInformation(
    host_halo_ids=np.array([0, 2, 3, 5, 0], dtype=np.int64),
    parent_index=np.array([-1, -1, -1, -1, 0], dtype=np.int64),
    depth=np.array([1, 1, 1, 1, 2], dtype=np.int64),
    global_index=np.array([0, 1, 2, 3, 4], dtype=np.int64),
    n_bound=np.array([10, 5, 8, 3, 4], dtype=np.int64),
)

store = ParticleStore(ptype="star", n_particles=6, is_baryonic=False)
store["HaloID"] = np.array([0, 2, 3, 0, -1, 1], dtype=np.int64)
store["SubhaloID"] = np.array([0, 1, 2, 0, -1, -1], dtype=np.int64)
subhalo_test_particles = {"star": store}

test_subhalo_info = SubhaloInformation(
    host_halo_ids=np.array([0, 2, 3, 5, 0], dtype=np.int64),
    parent_index=np.array([-1, -1, -1, -1, 0], dtype=np.int64),
    depth=np.array([1, 1, 1, 1, 2], dtype=np.int64),
    global_index=np.array([0, 1, 2, 3, 4], dtype=np.int64),
    n_bound=np.array([10, 5, 8, 3, 4], dtype=np.int64),
    original_subhids=np.array([0, 1, 2, 3, 4], dtype=np.int64),
)

# this rank owns HaloIDs 0, 1, 2, 3 and has one sentinel particle
# this rank also owns particles in subhalos 0, 1, 2 (plus two sentinels) (subhaloID is currently global)
# so, this rank needs the global subhalos which have a host_halo_id which appears in this rank's particle HaloIDs
# global subhalo -> field: 0 -> 0, 1 -> 2, 2 -> 3, 4 -> 0
# by association, since this rank owns field 0 which owns subhalo 4, it needs subhalo 4's particles (though there are no stars on this rank in subhalo 4)
# present on rank = [true, true, true, true, false, false]
# keep = present_on_rank[host_halo_ids] = [true, true, true, false, true]
# global_to_local[0, 1, 2, 4] = np.arange(4) -> global_to_local = [0, 1, 2, -1, 3]
# new subhids = [0, 1, 2, 0, -1, -1]


def test_local_subhalos() -> None:
    """
    Tests parallel_reads.py function assign_local_subhalos.
    """
    result = assign_local_subhalos(particles=subhalo_test_particles, subhalo_info=test_subhalo_info)

    # expected derived from above (keep mask on initial arrays)
    assert np.array_equal(result.host_halo_ids, np.array([0, 2, 3, 0], dtype=np.int64)), (
        "assign_local_subhalos failed: host IDs do not match."
    )
    assert np.array_equal(result.depth, np.array([1, 1, 1, 2], dtype=np.int64)), (
        "assign_local_subhalos failed: depths do not match."
    )
    assert np.array_equal(result.global_index, np.array([0, 1, 2, 4], dtype=np.int64)), (
        "assign_local_subhalos failed: global idx do not match."
    )
    assert np.array_equal(result.n_bound, np.array([10, 5, 8, 4], dtype=np.int64)), (
        "assign_local_subhalos failed: n_bound do not match."
    )
    assert np.array_equal(result.parent_index, np.array([-1, -1, -1, 0], dtype=np.int64)), (
        "assign_local_subhalos failed: parent idx do not match."
    )
    assert np.array_equal(
        subhalo_test_particles["star"]["SubhaloID"], np.array([0, 1, 2, 0, -1, -1], dtype=np.int64)
    ), "assign_local_subhalos failed: remapped SubhaloIDs do not match."


test_particle_counts: dict[str, int] = {
    "dm": 7,
    "gas": 5,
    "star": 2,
}
n_ranks = 3

# expected allocations (rank slice) with 3 ranks:
# dm: (0, 3), (3, 5), (5, 7)
# gas: (0, 2), (2, 4), (4, 5)
# star: (0, 1), (1, 2), (2, 2)


def test_slabs() -> None:
    """
    Tests parallel_reads.py function generate_slabs.
    """
    expected_rank_1 = {
        "dm": slice(3, 5),
        "gas": slice(2, 4),
        "star": slice(1, 2),
    }
    result_rank_1 = generate_slabs(rank=1, n_ranks=3, particle_counts=test_particle_counts)
    for ptype in expected_rank_1:
        assert expected_rank_1[ptype] == result_rank_1[ptype], (
            "test_slabs failed: rank 1 result does not match expected"
        )

    expected_rank_2 = {"dm": slice(5, 7), "gas": slice(4, 5), "star": slice(2, 2)}
    result_rank_2 = generate_slabs(
        rank=2,
        n_ranks=3,
        particle_counts=test_particle_counts,
    )
    for ptype in expected_rank_2:
        assert expected_rank_2[ptype] == result_rank_2[ptype], (
            "test_slabs failed: rank 2 result does not match expected"
        )
