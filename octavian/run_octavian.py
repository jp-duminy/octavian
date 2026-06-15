"""

Executes Octavian analysis pipeline.

Author: @jp // June 2026

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
from octavian.data_management import DataManager, save_group_properties, wrap_positions
from octavian.fof6d import run_fof6d
from octavian.aggregate_properties import calculate_group_properties, get_particle_lists

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

STAGES = [
    # stage name, executor function, dependencies
    ("fof6d", run_fof6d, []),
    ("group_properties", calculate_group_properties, []),
]

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
    with open(config_file, 'r') as f:
        config = safe_load(f)
    config['Tlim'] = float(config['Tlim'])

    enabled = resolve_stages(config)

    data_manager = DataManager(snapfile=snapshot_file, config=config, comm=comm)
    data_manager.load_halo_ids()
    data_manager.add_ptype_columns()

    wrap_positions(data_manager=data_manager)

    if "fof6d" in enabled:
        run_fof6d(data_manager=data_manager, nproc=config['nproc'])

    data_manager.initialise_group_data()

    if "group_properties" in enabled:
        calculate_group_properties(data_manager=data_manager)

    get_particle_lists(data_manager=data_manager)
    save_group_properties(data_manager=data_manager, output_directory=output_file)

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

