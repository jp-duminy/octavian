"""

Executes Octavian analysis pipeline.

NOTE: this is now calling dead code as of v0.3.0, so it will need to be refactored (easy, see validation/testing_suite.py)

"""

# type checking (semantic, do not worry about this)
from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from mpi4py import MPI

# profiling
import logging # runtimes

# data handling
from yaml import safe_load
from pathlib import Path

# octavian pipeline stages

def _get_mpi_communicator() -> MPI.Comm | None:
    """
    Checks whether MPI is enabled; if so, returns the comm object.
    """
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD # mpiexec -n 1 will return an output with _rank_0
    except ImportError:
        pass
    return None
"""
STAGES = [
    # stage name, executor function, dependencies
    ("fof6d", run_fof6d, []),
    ("group_properties", calculate_group_properties, []),
]
"""

def resolve_stages(config: dict[str, Any]) -> set:
    """
    Toggles on any dependencies which were set to false in the config file.
    Returns a set of stages to run.
    """
    enabled = {name for name, _, _ in STAGES if config["stages"].get(name, False)}
    dependencies = {name: d for name, _, d in STAGES}

    for name in enabled:
        enabled.update(dependencies[name])

    return enabled

def _execute_pipeline(snapshot_file: str, config_file: str, output_file: str, comm: MPI.Comm | None) -> None:
    """
    Executes each toggled stage of the Octavian pipeline.
    """
    pass # NOTE: needs updating for new functions.

def run(snapshot_file: str, config_file: str, output_directory: str) -> None:
    """
    Runs an Octavian analysis.
    snapshot_file is the identifier of a filtered snapshot (everything before rank_*)
    Outputs a hdf5 catalogue for each rank.
    """
    comm = _get_mpi_communicator()
    rank = comm.Get_rank() if comm else 0
    size = comm.Get_size() if comm else 1

    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] [%(levelname)s] %(name)s — %(message)s",
    )

    if rank == 0:
        logging.info(f"Running Octavian with {size} nodes.")

    if comm:
        snapshot_file = f"{snapshot_file}_{rank}.hdf5" # note: reassigns the variable name

    snapshot_identifier = Path(snapshot_file).stem # snapshot tag (e.g. snap_m100n1024_151)
    output_file = Path(output_directory) / f"octavian_{snapshot_identifier}_{rank}.hdf5"

    _execute_pipeline(snapshot_file=snapshot_file, config_file=config_file, output_file=output_file, comm=comm)

    if comm:
        comm.Barrier()

    if rank == 0:
        logging.info(f"All ranks complete.")

