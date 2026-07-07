"""

Logger; much easier than littered than print statements. 

Lets warnings and info go through, debugging is saved for the output file. When running with MPI, only rank 0 will talk to the terminal. The other ranks will only chat to you if some sort of error or warning is flagged. However, the output files will contain everything.

"""

import logging
from pathlib import Path

def configure_logger(rank: int = 0, output_level: str = "INFO", output_log_directory: Path | None = None) -> logging.Logger:
    """
    Creates a logger object for use throughout stages.

    Specify output_log_directory if you'd like a .txt file with more detailed information.
    """
    logger = logging.getLogger(name="OCTAVIAN")

    if logger.handlers:
        return logger

    logger.setLevel(level=logging.DEBUG) # allow all calls through, so handlers can suppress lower levels
    logger.propagate = False

    rank_tag = f"Rank {rank}"
    terminal_format = f"OCTAVIAN [{rank_tag}] | %(levelname)s | %(message)s"
    file_format = f"%(asctime)s | OCTAVIAN [{rank_tag}] | %(levelname)s | %(message)s"

    # if you have todohighlights on this block might look weird (it's the console text)
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, output_level.upper()) if rank == 0 else logging.WARNING) # ranks != 0 only flag any errors
    console.setFormatter(logging.Formatter(terminal_format))
    logger.addHandler(console) # from the getattr call above, this means output_level defines what you see
    
    if output_log_directory is not None:
        output_log_directory.mkdir(parents=True, exist_ok=True)
        log_path = output_log_directory / f"octavian_rank{rank}.log"
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(file_format))
        logger.addHandler(file_handler)

    return logger

def get_logger() -> logging.Logger:
    """
    Helper to retrieve the Octavian logger; please call configure_logger() if not already done so, as early in the pipeline as possible.
    """
    logger = logging.getLogger("OCTAVIAN")

    return logger