"""

Generates a synthetic Octavian catalogue from the procedurally-generated snapshots.

"""

# default libraries
from pathlib import Path
from dataclasses import replace

# internal imports
from ..run_octavian import run_octavian
from ..data_management import OctavianConfig
from ..data_management import output_catalogue_path
from .generate_snapshots import generate_gizmo_snapshot, generate_swift_snapshot
from ..log import get_logger

logger = get_logger()
config_path = Path(__file__).parent.parent.parent / "config.yaml"


def generate_test_catalogue(
    output_dir: Path,
    simulation_type: str,
) -> Path:
    """
    Generates a tiny synthetic Octavian catalogue for testing purposes. This catalogue is produced from a procedurally-generated snapshot filled with junk data; this function can therefore be used to assess the catalogue HDF5 file layout and whether the pipeline runs. It produces its own synthetic junk config file too. This must be run under MPI.

    Parameters
    ----------
    output_dir: pathlib.Path
        Where you would like to output the test catalogue.
    simulation_type: str ["GIZMO"/"SWIFT"]
        Which simulation type you would like the test catalogue to be generated from. In practice this does not affect the output catalogue, which is agnostic to the simulation type.

    Returns
    -------
    test_catalogue: pathlib.Path
        A path object pointing to the test catalogue.
    """
    snapshot_path = output_dir / "test_snap.hdf5"
    output_path = output_catalogue_path(snapshot_path=snapshot_path, output_dir=output_dir)

    if simulation_type == "SWIFT":
        generate_swift_snapshot(path=snapshot_path)
    elif simulation_type == "GIZMO":
        generate_gizmo_snapshot(path=snapshot_path)
    else:
        raise ValueError(f"simulation_type {simulation_type} is invalid/unsupported, please provide GIZMO/SWIFT.")

    config = OctavianConfig.from_yaml(config_path=config_path)
    config = replace(
        config,
        simulation_type=simulation_type,
        halo_id_source="SNAPSHOT",
        min_dm_per_halo=0,
        min_stars_per_galaxy=2,
        b=1.5,
        velocity_factor=5,
        keep_logs=False,
        terminal_output_level="DEBUG",
    )

    run_octavian(snapshot_path=snapshot_path, output_dir=output_dir, config=config)

    intermediates_dir = output_dir / "Intermediates"
    snapshot_path.unlink()
    intermediates_dir.rmdir()

    logger.info(f"Successfully created a synthetic catalogue at {output_path}.")

    return output_path
