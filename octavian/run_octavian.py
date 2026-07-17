"""

Executes Octavian analysis pipeline.

NOTE: this is now calling dead code as of v0.3.0, so it will need to be refactored (easy, see validation/testing_suite.py)

"""

# type checking (semantic, do not worry about this)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mpi4py import MPI
    from octavian.data_management import SnapshotReader

from octavian.data_management import (
    write_analysis_to_output_file,
    construct_particle_csr_lists,
    merge_intermediate_catalogues,
    clean_intermediates,
    write_catalogue_metadata,
    GroupStore,
    SimulationData,
    Internals,
    OctavianConstants,
    OctavianConfig,
    build_reader,
    build_group_store,
    build_particle_stores,
    load_internals,
    resolve_dependencies,
    get_releasable_columns,
    compute_rank_assignments,
    output_catalogue_path,
    intermediate_catalogue_path,
)
from octavian.external_halo_sources import (
    SnapshotHaloSource,
    HaloSource,
)
from octavian.galaxy_finding import find_galaxies
from octavian.aggregate_properties import run_ptype_specific_properties, run_core_properties, run_local_environment
from octavian.log import configure_logger, get_logger

# data handling
from pathlib import Path
import numpy as np


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


def execute_pipeline(
    output_path: Path,
    config: OctavianConfig,
    internals: Internals,
    reader: SnapshotReader,
    halo_source: HaloSource,
) -> None:
    """
    Executes each toggled stage of the Octavian pipeline.
    """
    constants = OctavianConstants(mu=config.MU, frad=config.FRAD)
    halo_assignments = halo_source.read_halo_ids(ptypes=reader.available_ptypes())
    sim = reader.simulation_attributes
    particles = build_particle_stores(
        reader=reader, internals=internals, halo_assignments=halo_assignments, process_ptypes=config.process_ptypes
    )

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

    if reader.indices is not None:
        particle_indices = reader.indices
    else:
        particle_indices = {ptype: np.arange(len(particles[ptype]), dtype=np.int64) for ptype in particles}

    particle_lists = construct_particle_csr_lists(data=simulation_data, internals=internals, indices=particle_indices)

    write_analysis_to_output_file(
        data=simulation_data,
        particle_lists=particle_lists,
        internals=internals,
        output_file=output_path,
    )


def run_octavian(
    snapshot_path: Path,
    output_dir: Path,
    config_path: Path,
    internals_path: Path,
) -> None:
    """
    Conduct a full parallel run of Octavian.
    """
    comm = get_mpi_communicator()
    rank = comm.Get_rank() if comm else 0
    size = comm.Get_size() if comm else 1

    intermediate_dir = output_dir / "Intermediates"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    configure_logger(rank=rank, output_level="INFO", log_dir=intermediate_dir)
    logger = get_logger()

    logger.info(f"Analysing {snapshot_path} with {size} ranks.")

    config = OctavianConfig.from_yaml(config_path=config_path)
    internals = load_internals(internals_filepath=internals_path, config=config)
    oc = OctavianConstants()
    reader = build_reader(snapshot_path=snapshot_path, constants=oc, config=config)

    if rank == 0:  # no need for comm.Barrier() here as scatter does it inherently
        all_indices = compute_rank_assignments(
            reader=reader, config=config, n_ranks=size, halo_source=SnapshotHaloSource(reader=reader)
        )
    else:
        all_indices = None

    rank_indices = comm.scatter(all_indices, root=0) if comm else all_indices[0]
    if rank_indices is not None:
        reader.set_indices(
            indices=rank_indices
        )  # make sure this is called before ID assignments otherwise masks won't be applied
    halo_source = SnapshotHaloSource(reader=reader)

    intermediate_output = intermediate_catalogue_path(directory=intermediate_dir, rank=rank)

    execute_pipeline(
        output_path=intermediate_output,
        config=config,
        internals=internals,
        reader=reader,
        halo_source=halo_source,
    )

    if comm:
        comm.Barrier()

    catalogue_path = output_catalogue_path(directory=output_dir)

    if rank == 0:
        files = [intermediate_catalogue_path(directory=intermediate_dir, rank=i) for i in range(size)]
        merge_intermediate_catalogues(files=files, output_path=catalogue_path, internals=internals)
        clean_intermediates(intermediate_dir=intermediate_dir, output_dir=output_dir, n_ranks=size, config=config)
        write_catalogue_metadata(
            catalogue_path=catalogue_path, snapshot_path=snapshot_path, config=config, n_ranks=size
        )
