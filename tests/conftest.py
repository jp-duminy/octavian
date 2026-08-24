"""

Configuration file for pytest. Please see https://docs.pytest.org/en/stable/reference/ for more information.

This is for automated CI/CD (did your commit break anything in the analysis?). Please see the validation folder for a more rigorous testing suite (validation_suite.py), which has methods for verifying catalogue correctedness.

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# default libraries
from pathlib import Path
from collections.abc import Generator
from dataclasses import replace  # for modifying frozen dataclasses

# testing
import pytest

# other packages
import h5py

# internal imports
from octavian import analyse_snapshot, OctavianConfig

GIZMO_TEST_PATH = Path(__file__).parent / "data" / "gizmo_test_snapshot.hdf5"
SWIFT_TEST_PATH = Path(__file__).parent / "data" / "swift_test_snapshot.hdf5"
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
PHOTOMETRY_TABLE_PATH = Path(__file__).parent / "data" / "test_photometry_table.hdf5"
INTERNALS_PATH = Path(__file__).parent.parent / "octavian" / "internals.yaml"


@pytest.fixture(scope="session", params=["GIZMO", "SWIFT"])
def mock_catalogue(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Generator[h5py.File, None, None]:
    """
    Generates the mock catalogue for testing (uses a tiny test snapshot).
    """
    tmp_dir = tmp_path_factory.mktemp("pipeline")

    sim_type = request.param
    snapshot_path = GIZMO_TEST_PATH if sim_type == "GIZMO" else SWIFT_TEST_PATH
    config = OctavianConfig.from_yaml(config_path=CONFIG_PATH)

    config = replace(
        config,
        simulation_type=sim_type,
        snapshot_path=snapshot_path,
        output_dir=tmp_dir,
        halo_id_source="SNAPSHOT",
        photometry_table_filepath=PHOTOMETRY_TABLE_PATH,
        bands=["v"],  # the test table only has the V filter to reduce filesize
        min_dm_per_halo=0,
        min_stars_per_galaxy=2,  # these parameter choices are just so it runs
        b=1.5,
        velocity_factor=5,
    )  # use the built-in dataclass method otherwise you need to modify production code in a weird way

    output_path = analyse_snapshot(config=config)
    assert output_path.exists()

    catalogue = h5py.File(output_path, "r")
    yield catalogue  # use yield not return, otherwise you'll run into issues with closing the catalogue
    catalogue.close()
