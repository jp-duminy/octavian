"""

Configuration file for pytest. Please see https://docs.pytest.org/en/stable/reference/ for more information.

This is for automated CI/CD (did your commit break anything in the analysis?). Please see the validation folder for a more rigorous testing suite (validation_suite.py), which has methods for verifying catalogue correctedness.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octavian.data_management.parallel_reading import RedistributionMap

from pathlib import Path
import h5py
import pytest
from collections.abc import Generator
import numpy as np
from dataclasses import replace

from octavian.run_octavian import execute_pipeline, get_mpi_communicator
from octavian.log import configure_logger
from octavian.data_management import (
    OctavianConfig,
    OctavianConstants,
    load_internals,
    build_reader,
    generate_rank_halo_assignments,
    generate_slabs,
    build_redistribution_map,
    redistribute_data,
    write_catalogue,
)
from octavian.external_halo_sources import (
    build_halo_source,
    HaloAssignments,
)

GIZMO_TEST_PATH = Path(__file__).parent / "data" / "gizmo_test_snapshot.hdf5"
SWIFT_TEST_PATH = Path(__file__).parent / "data" / "swift_test_snapshot.hdf5"
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
INTERNALS_PATH = Path(__file__).parent.parent / "octavian" / "internals.yaml"


@pytest.fixture(scope="session", params=["GIZMO", "SWIFT"])
def mock_catalogue(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Generator[h5py.File, None, None]:
    """
    Generates the mock catalogue for testing (uses a tiny test snapshot of size ~10MB).
    """
    tmp_dir = tmp_path_factory.mktemp("pipeline")
    output_path = tmp_dir / "test_catalogue.hdf5"
    comm = get_mpi_communicator()
    rank = comm.Get_rank() if comm else 0

    configure_logger(rank=rank, output_level="INFO", log_dir=tmp_dir)

    sim_type = request.param
    snapshot_path = GIZMO_TEST_PATH if sim_type == "GIZMO" else SWIFT_TEST_PATH
    config = OctavianConfig.from_yaml(config_path=CONFIG_PATH)
    config = replace(
        config, simulation_type=sim_type, halo_id_source="SNAPSHOT"
    )  # use the built-in dataclass method otherwise you need to modify production code in a weird way
    internals = load_internals(internals_filepath=INTERNALS_PATH, config=config)
    oc = OctavianConstants(mu=config.MU, frad=config.FRAD)

    reader = build_reader(snapshot_path=snapshot_path, constants=oc, config=config)
    halo_source = build_halo_source(config=config, reader=reader)
    all_halo_assignments = halo_source.read_halo_ids(ptypes=reader.available_ptypes())
    subhalo_info = halo_source.read_subhalo_info()

    halo_to_rank = generate_rank_halo_assignments(
        halo_assignments=all_halo_assignments,
        config=config,
        n_ranks=1,
    )

    slabs = generate_slabs(rank=0, n_ranks=1, particle_counts=reader.particle_counts)

    raw_halo_ids = halo_source.distribute_raw_halo_ids(slabs=slabs)
    raw_subhalo_ids = halo_source.distribute_raw_subhalo_ids(slabs=slabs)

    masks: dict[str, np.ndarray] = {}
    maps: dict[str, RedistributionMap] = {}
    local_halo_ids: dict[str, np.ndarray] = {}
    local_subhalo_ids: dict[str, np.ndarray] | None = {} if raw_subhalo_ids is not None else None

    for ptype in raw_halo_ids:
        print(
            f"{ptype}: raw_halo_ids_len={len(raw_halo_ids[ptype])}, slab_len={slabs[ptype].stop - slabs[ptype].start}"
        )
        maps[ptype], masks[ptype] = build_redistribution_map(halo_to_rank, raw_halo_ids[ptype], comm)
        local_halo_ids[ptype] = redistribute_data(raw_halo_ids[ptype][masks[ptype]], maps[ptype], comm)
        if local_subhalo_ids is not None:
            local_subhalo_ids[ptype] = redistribute_data(raw_subhalo_ids[ptype][masks[ptype]], maps[ptype], comm)

    rank_halo_assignments = HaloAssignments(
        halo_ids=local_halo_ids,
        n_total_halos=len(halo_to_rank),
        subhalo_ids=local_subhalo_ids,
    )

    reader.set_maps(slabs=slabs, masks=masks, maps=maps, comm=comm)

    packed_data = execute_pipeline(
        config=config,
        internals=internals,
        reader=reader,
        constants=oc,
        halo_assignments=rank_halo_assignments,  # only runs in serial
        global_subhalo_info=subhalo_info,
    )

    write_catalogue(packed_data=packed_data, catalogue_path=output_path, internals=internals, comm=None)

    assert output_path.exists()

    catalogue = h5py.File(output_path, "r")
    yield catalogue  # use yield not return, otherwise you'll run into issues with closing the catalogue
    catalogue.close()
