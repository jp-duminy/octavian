"""

Configuration file for pytest. Please see https://docs.pytest.org/en/stable/reference/ for more information.

This is for automated CI/CD (did your commit break anything in the analysis?). Please see the validation folder for a more rigorous testing suite (validation_suite.py), which has methods for verifying catalogue correctedness.

"""

from pathlib import Path
import h5py
import pytest
from collections.abc import Generator
from dataclasses import replace

from octavian.run_octavian import execute_pipeline, get_mpi_communicator
from octavian.log import configure_logger
from octavian.data_management.pipeline_management import load_internals
from octavian.data_management.conventions import OctavianConfig

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
        config, simulation_type=sim_type
    )  # use the built-in dataclass method otherwise you need to modify production code in a weird way
    internals = load_internals(internals_filepath=INTERNALS_PATH, config=config)

    execute_pipeline(snapshot_path=snapshot_path, output_path=output_path, config=config, internals=internals)

    assert output_path.exists()

    catalogue = h5py.File(output_path, "r")
    yield catalogue  # use yield not return, otherwise you'll run into issues with closing the catalogue
    catalogue.close()
