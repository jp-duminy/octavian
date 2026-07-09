"""

Configuration file for pytest. Please see https://docs.pytest.org/en/stable/reference/ for more information.

This is for automated CI/CD (did your commit break anything in the analysis?). Please see the validation folder for a more rigorous testing suite (validation_suite.py), which has methods for verifying catalogue correctedness.

"""

from __future__ import annotations

from pathlib import Path
from yaml import safe_load
import h5py
import pytest
from collections.abc import Generator

from octavian.run_octavian import execute_pipeline, get_mpi_communicator
from octavian.data_management.log import configure_logger
from octavian.data_management.pipeline_management import load_internals

GIZMO_TEST_PATH = Path(__file__).parent / "data" / "gizmo_test_snapshot.hdf5"
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
INTERNALS_PATH = Path(__file__).parent.parent / "octavian" / "internals.yaml"

@pytest.fixture(scope="session")
def mock_catalogue(tmp_path_factory: pytest.TempPathFactory) -> Generator[h5py.File, None, None]:
    """
    Generates the mock catalogue for testing (uses a tiny test snapshot of size ~10MB).
    """
    tmp_dir = tmp_path_factory.mktemp("pipeline")
    output_path = tmp_dir / "test_catalogue.hdf5"
    comm = get_mpi_communicator()
    rank = comm.Get_rank() if comm else 0

    configure_logger(rank=rank, output_level="INFO", output_log_directory=tmp_dir)

    with open(CONFIG_PATH, "r") as f:
        config = safe_load(f)

    internals = load_internals(internals_filepath=INTERNALS_PATH, user_config=config)

    execute_pipeline(snapshot_path=GIZMO_TEST_PATH, output_path=output_path, config=config, internals=internals)

    assert output_path.exists()

    catalogue = h5py.File(output_path, "r")
    yield catalogue # use yield not return, otherwise you'll run into issues with closing the catalogue
    catalogue.close()