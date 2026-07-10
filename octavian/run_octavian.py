"""

Executes Octavian analysis pipeline.

NOTE: this is now calling dead code as of v0.3.0, so it will need to be refactored (easy, see validation/testing_suite.py)

"""

# type checking (semantic, do not worry about this)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mpi4py import MPI

from octavian.data_management import (
    filter_snapshot,
    write_analysis_to_output_file,
    construct_particle_csr_lists,
    merge_intermediate_catalogues,
    GizmoReader,
    GroupStore,
    SimulationData,
    Internals,
    OctavianConstants,
    OctavianConfig,
    build_group_store,
    build_particle_stores,
    load_internals,
    resolve_dependencies,
    get_releasable_columns,
    configure_logger,
    get_logger,
)
from octavian.galaxy_finding import find_galaxies
from octavian.aggregate_properties import run_ptype_specific_properties, run_core_properties, run_local_environment

# data handling
from pathlib import Path


def get_mpi_communicator() -> MPI.Comm | None:
    """
    Checks whether MPI is enabled; if so, returns the comm object.
    """
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD  # mpiexec -n 1 will return an output with _rank_0
    except ImportError:
        pass
    return None


def execute_pipeline(snapshot_path: Path, output_path: Path, config: OctavianConfig, internals: Internals) -> None:
    """
    Executes each toggled stage of the Octavian pipeline.
    """
    constants = OctavianConstants(mu=config.MU, frad=config.FRAD)

    reader = GizmoReader(snapshot_path=snapshot_path, constants=constants)
    sim = reader.simulation_attributes
    particles = build_particle_stores(reader=reader, internals=internals, process_ptypes=config.process_ptypes)

    for prop in ["rho", "sfr"]:
        particles["gas"][prop] = reader.read_dataset(ptype="gas", dataset=prop)

    fof6d_result = find_galaxies(particles=particles, simulation=sim, config=config, constants=constants)

    groups: dict[str, GroupStore] = {}
    groups["halos"] = build_group_store(particles=particles, group_type="halos")

    if fof6d_result.n_galaxies > 0:
        groups["galaxies"] = build_group_store(particles=particles, group_type="galaxies")

    for ptype in particles:
        particles[ptype]["potential"] = reader.read_dataset(ptype=ptype, dataset="potential")

    for prop in ["fHI", "fH2", "metallicity"]:
        particles["gas"][prop] = reader.read_dataset(ptype="gas", dataset=prop)

    for prop in ["metallicity", "age"]:
        particles["star"][prop] = reader.read_dataset(ptype="star", dataset=prop)

    particles["bh"]["bhmdot"] = reader.read_dataset(ptype="bh", dataset="bhmdot")

    simulation_data = SimulationData(simulation=sim, constants=constants, particles=particles, groups=groups)

    requested = [name for name, enabled in config.stages.items() if enabled and name != "find_galaxies"]
    ordered_stages = resolve_dependencies(stages=internals.stages, requested=requested)

    stage_dispatch = {
        "properties_core": run_core_properties,
        "properties_ptype_specific": run_ptype_specific_properties,
        "properties_local_environment": run_local_environment,
    }

    for stage_index, stage in enumerate(ordered_stages):
        stage_dispatch[stage.name](simulation_data=simulation_data, config=config)

        releasable = get_releasable_columns(stage_index, ordered_stages)

        for ptype in particles:
            for col in releasable:
                if col in particles[ptype]:
                    particles[ptype].release(col)

    for ptype in particles:
        particles[ptype]["particle_index"] = reader.read_dataset(ptype=ptype, dataset="particle_index")

    particle_lists = construct_particle_csr_lists(data=simulation_data, internals=internals)
    write_analysis_to_output_file(
        data=simulation_data,
        particle_lists=particle_lists,
        internals=internals,
        output_file=output_path,
    )


def run_octavian(
    snapshot_path: Path,
    output_directory: Path,
    config_path: Path,
    internals_path: Path,
    intermediates_exist: bool = False,
) -> None:
    """
    Conduct a full parallel run of Octavian.
    """
    comm = get_mpi_communicator()
    rank = comm.Get_rank() if comm else 0
    size = comm.Get_size() if comm else 1

    intermediate_directory = output_directory / "Intermediates"
    intermediate_directory.mkdir(parents=True, exist_ok=True)

    configure_logger(rank=rank, output_level="INFO", output_log_directory=intermediate_directory)
    logger = get_logger()

    logger.info(f"Analysing {snapshot_path} with {size} ranks.")

    config = OctavianConfig.from_yaml(config_path=config_path)
    internals = load_internals(internals_filepath=internals_path, config=config)

    if rank == 0:
        if not intermediates_exist:
            oc = OctavianConstants()
            reader = GizmoReader(snapshot_path=snapshot_path, constants=oc)

            filter_snapshot(
                snapshot_file=snapshot_path,
                intermediate_directory=intermediate_directory,
                reader=reader,
                config=config,
                n_split=size,
            )

    if comm:
        comm.Barrier()

    intermediate_file = intermediate_directory / f"rank_{rank}.hdf5"
    intermediate_output = intermediate_directory / f"rank_{rank}_intermediate_analysis.hdf5"

    execute_pipeline(
        snapshot_path=intermediate_file, output_path=intermediate_output, config=config, internals=internals
    )

    if comm:
        comm.Barrier()

    output_catalogue = output_directory / "output_catalogue.hdf5"

    if rank == 0:
        files = [intermediate_directory / f"rank_{i}_intermediate_analysis.hdf5" for i in range(size)]
        merge_intermediate_catalogues(files=files, output_path=output_catalogue, internals=internals)
