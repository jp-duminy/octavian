"""

Logger, much easier than littered than print statements. 

Lets warnings and info go through, debugging is saved for the output file.

"""

import logging
from pathlib import Path

def create_logger(rank: int = 0, output_level: str = "INFO", output_log_directory: Path | None = None):
    """
    Creates a logger object for use throughout stages.

    Specify output_log_directory if you'd like a .txt file with more detailed information.
    """
    root = logging.getLogger(name="OCTAVIAN")

    root.setLevel(level=logging.DEBUG) # allow all calls through, so handlers can suppress lower levels

    if rank == 0:
        console = logging.StreamHandler()
        console.setLevel(getattr(logging, output_level.upper())) # if you have todohighlights on this might look weird
        console.setFormatter(logging.Formatter(
            "%(name)s | %(levelname)s | %(message)s"
        ))
        root.addHandler(console) # from the getattr call above, this means output_level defines what you see
    
    root.propagate = False
