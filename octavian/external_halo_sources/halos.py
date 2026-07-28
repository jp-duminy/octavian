"""

This file exists to avoid a circular import (I am really bad with imports right now and need to fix this across the codebase!).

"""

from .ahf import AHFHaloSource
from .halo_structures import SnapshotHaloSource, HaloSource
from octavian.data_management import SnapshotReader, OctavianConfig
from octavian.log import get_logger

logger = get_logger()


def build_halo_source(config: OctavianConfig, reader: SnapshotReader) -> HaloSource:
    """
    Builds a halo source depending on what was specified in the config.
    """
    if config.halo_id_source == "SNAPSHOT":
        logger.info("Using snapshot-assigned HaloIDs.")
        return SnapshotHaloSource(reader=reader)
    elif config.halo_id_source == "AHF":
        prefix = config.halo_id_filepath  # renamed for explicitness
        logger.info("Using AHF-assigned HaloIDs.")
        logger.info(f"Finding AHF catalogues at {prefix} .")
        return AHFHaloSource(
            halos_path=prefix.with_suffix(".AHF_halos"),
            particles_path=prefix.with_suffix(".AHF_particles"),
            reader=reader,
        )
    else:
        raise ValueError("Unknown halo ID source, please check config?")
