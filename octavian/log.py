"""

Logger; much easier than littered than print statements.

Lets warnings and info go through, debugging is saved for the output file. When running with MPI, only rank 0 will talk to the terminal. The other ranks will only chat to you if some sort of error or warning is flagged. However, the output files will contain everything.

"""

import logging
from pathlib import Path


def configure_logger(rank: int = 0, output_level: str = "INFO", log_dir: Path | None = None) -> logging.Logger:
    """
    Creates a logger object for use throughout stages.

    Specify output_log_dir if you'd like a .txt file with more detailed information.
    """
    logger = logging.getLogger(name="OCTAVIAN")

    if logger.handlers:
        return logger

    logger.setLevel(level=logging.DEBUG)  # allow all calls through, so handlers can suppress lower levels
    logger.propagate = False

    rank_tag = f"Rank {rank}"
    terminal_format = f"OCTAVIAN [{rank_tag}] | %(levelname)s | %(message)s"
    file_format = f"%(asctime)s | OCTAVIAN [{rank_tag}] | %(levelname)s | %(message)s"

    # if you have todohighlights on this block might look weird (it's the console text)
    console = logging.StreamHandler()
    console.setLevel(
        getattr(logging, output_level.upper()) if rank == 0 else logging.WARNING
    )  # ranks != 0 only flag any errors
    console.setFormatter(logging.Formatter(terminal_format))
    logger.addHandler(console)  # from the getattr call above, this means output_level defines what you see

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = intermediate_log_path(directory=log_dir, rank=rank)
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


def intermediate_log_path(directory: Path, rank: int) -> Path:
    """
    Returns the Path object pointing to the intermediate log filename for a rank.
    """
    return directory / f"rank_{rank}_log.log"


def merged_log_path(directory: Path) -> Path:
    """
    Returns the Path object pointing to the merged log filename.
    """
    return directory / "output_log.log"


def clean_logs(
    log_dir: Path,
    n_ranks: int,
    keep_logs: bool,
) -> None:
    """
    Cleans the log directory by either merging per-rank logs into one or deleting them entirely.
    """
    logger = get_logger()
    if keep_logs:  # concatenates the per-rank logs rather than time-based zipper merging
        merged_log = log_dir / "octavian.log"
        with open(merged_log, "w") as out:
            for i in range(n_ranks):
                rank_log = intermediate_log_path(directory=log_dir, rank=i)
                if rank_log.exists():
                    out.write(
                        rank_log.read_text()
                    )  # this writes the logs sequentially so it appears in rank order in the merged log
                    rank_log.unlink()
            logger.info(f"Merged then cleaned up {n_ranks} log files.")
    else:
        for i in range(n_ranks):
            (intermediate_log_path(directory=log_dir, rank=i)).unlink(missing_ok=True)  # just remove intermediate logs
        logger.info(f"Removed {n_ranks} log files.")
