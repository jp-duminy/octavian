"""

Configuration file for pytest. Please see https://docs.pytest.org/en/stable/reference/ for more information.

This is for automated CI/CD (did your commit break anything in the analysis?). Please see the validation folder for a more rigorous testing suite (validation_suite.py), which has methods for verifying catalogue correctedness.

"""

# default libraries
from pathlib import Path
from collections.abc import Generator
from dataclasses import replace  # for modifying frozen dataclasses

# testing
import pytest

# other packages
import h5py

# internal imports
from octavius.run_octavius import analyse_snapshot, generate_config
from octavius.data_management.conventions import OctaviusConfig
from octavius.utils.generate_snapshots import generate_gizmo_snapshot, generate_swift_snapshot
from octavius.utils.dynamic_analyser import build_analyser, OctaviusAnalyser
from octavius.utils.loader import load_catalogue, OctaviusCatalogue

CONFIG_PATH = Path(__file__).parent.parent / "octavius" / "config.yaml"
PHOTOMETRY_TABLE_PATH = Path(__file__).parent / "data" / "test_photometry_table.hdf5"
INTERNALS_PATH = Path(__file__).parent.parent / "octavius" / "internals.yaml"

FORMATS = ["GIZMO", "SWIFT-KIARA", "SWIFT-EAGLE", "SWIFT-COLIBRE"]


@pytest.fixture(scope="session", params=FORMATS)
def mock_catalogue(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Generator[h5py.File, None, None]:
    """
    Generates the mock catalogue for testing (uses a tiny test snapshot).
    """
    tmp_dir = tmp_path_factory.mktemp("pipeline")
    tmp_snap = tmp_dir / "test_snapshot.hdf5"
    sim_type = request.param
    generate_config(output_dir=tmp_dir)
    tmp_config = tmp_dir / "octavius_config.yaml"
    assert tmp_config.exists()

    if sim_type == "GIZMO":
        generate_gizmo_snapshot(path=tmp_snap)
    elif "SWIFT" in sim_type:
        model = sim_type.split(sep="-")[1]  # slightly hacky
        generate_swift_snapshot(path=tmp_snap, model=model)

    config = OctaviusConfig.from_yaml(config_path=CONFIG_PATH)

    config = replace(
        config,
        simulation_type=sim_type,
        snapshot_path=tmp_snap,
        output_dir=tmp_dir,
        halo_id_source="SNAPSHOT",
        photometry_table_filepath=PHOTOMETRY_TABLE_PATH,
        bands=["v"],  # the test table only has the V filter to reduce filesize
        min_dm_per_halo=0,
        min_stars_per_galaxy=2,  # these parameter choices are just so it runs
        b=1.5,
        velocity_factor=5,
        compress_catalogue=False,
    )  # use the built-in dataclass method otherwise you need to modify production code in a weird way

    output_path = analyse_snapshot(config=config)
    assert output_path.exists()

    cat = load_catalogue(catalogue_path=output_path)  # check a catalogue object is constructable
    assert isinstance(cat, OctaviusCatalogue)
    analyser = build_analyser(catalogue=cat, config=config)  # check an Analyser object is constructable

    result_c = analyser.compute_core_properties(group_type="haloes", group_indices=[0])
    assert len(result_c.columns) > 0

    result_pt = analyser.compute_ptype_specific_properties(group_type="haloes", group_indices=[0])
    assert len(result_pt.columns) > 0

    result_phot = analyser.compute_photometry(group_indices=[0], keep_spectra=True, orientation="side-on")
    assert len(result_phot.columns) > 0

    cat.close()
    assert isinstance(analyser, OctaviusAnalyser)

    catalogue = h5py.File(output_path, "r")
    yield catalogue  # use yield not return, otherwise you'll run into issues with closing the catalogue
    catalogue.close()
