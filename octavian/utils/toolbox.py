"""

Convenience file to build a config and run Octavian from within the comforts of a Python script.

"""

# default libraries
from pathlib import Path
import subprocess
import shutil

# internal imports
from ..log import get_logger

logger = get_logger()


def repack_catalogue(
    catalogue_path: Path,
    compression_level: int = 4,  #  https://en.wikipedia.org/wiki/Gzip fastest is 4
) -> None:
    """
    Repack an HDF5 catalogue with GZIP compression in-place. The compressed catalogue is named the same as the original.

    Parameters
    ----------
    catalogue_path : Path
        Path to the catalogue you would like to compress.
    compression_level : int
        Desired GZIP compression level (1-9)
    """
    logger.info(f"Compressing catalogue to GZIP-level {compression_level}.")
    h5repack_binary = shutil.which("h5repack")
    if h5repack_binary is None:
        raise FileNotFoundError("h5repack is not available on your system, to recompress files please install it.")

    temp_path = catalogue_path.with_name(f"{catalogue_path.stem}_packed.hdf5")

    # during testing, if cuillin was having a bad day it would stop compressing and leave a corrupted file, motivating this
    try:
        subprocess.run(
            ["h5repack", "-f", f"GZIP={compression_level}", str(catalogue_path), str(temp_path)],
            check=True,
        )
        temp_path.replace(catalogue_path)
    finally:  # if h5repack fails, delete the file
        temp_path.unlink(missing_ok=True)
