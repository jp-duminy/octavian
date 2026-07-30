"""

Functions for executing the end-to-end Octavian analysis pipeline.

"""

# type checking (semantic, do not worry about this)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mpi4py import MPI
    from octavian.data_management import SnapshotReader, RankPackedData
    from octavian.external_halo_sources import HaloAssignments, SubhaloInformation
    from octavian.data_management.parallel_reading import RedistributionMap

from octavian.data_management import (
    construct_membership_arrays,
    write_catalogue,
    write_catalogue_metadata,
    GroupStore,
    SimulationData,
    Internals,
    OctavianConstants,
    OctavianConfig,
    build_reader,
    build_galaxy_store,
    build_halo_store,
    build_particle_stores,
    load_internals,
    resolve_dependencies,
    get_releasable_columns,
    output_catalogue_path,
    redistribute_data,
    assign_local_subhalos,
    pack_rank_data,
    generate_rank_halo_assignments,
    generate_slabs,
    build_redistribution_map,
)
from octavian.external_halo_sources import (
    build_halo_source,
)
from octavian.galaxy_finding import find_galaxies
from octavian.aggregate_properties import (
    run_ptype_specific_properties,
    run_core_properties,
    run_local_environment,
    assign_membership,
)
from octavian.log import configure_logger, get_logger, clean_logs

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
    config: OctavianConfig,
    internals: Internals,
    constants: OctavianConstants,
    reader: SnapshotReader,
    halo_assignments: HaloAssignments,
    global_subhalo_info: SubhaloInformation,
) -> RankPackedData:
    """
    Executes each toggled stage of the Octavian pipeline.
    """
    sim = reader.simulation_attributes
    particles = build_particle_stores(
        reader=reader, internals=internals, halo_assignments=halo_assignments, process_ptypes=config.process_ptypes
    )
    subhalo_info = assign_local_subhalos(
        particles=particles, subhalo_info=global_subhalo_info
    )  # this is done locally and is safe, not worth optimising (though an elegant solution is always welcome)

    for prop in ["rho", "sfr"]:
        particles["gas"][prop] = reader.read_dataset(ptype="gas", dataset=prop)

    fof6d_result = find_galaxies(particles=particles, simulation=sim, config=config, constants=constants)

    groups: dict[str, GroupStore] = {}
    groups["halos"] = build_halo_store(
        particles=particles,
        halo_key=internals.group_types["halos"]["key"],
        subhalo_key="SubhaloID",
        group_kind=internals.group_types["halos"]["kind"],
        subhalo_info=subhalo_info,
    )

    if fof6d_result.n_galaxies > 0:
        groups["galaxies"] = build_galaxy_store(
            particles=particles,
            galaxy_key=internals.group_types["galaxies"]["key"],
            group_kind=internals.group_types["galaxies"]["kind"],
        )

    for ptype in particles:
        particles[ptype]["potential"] = reader.read_dataset(ptype=ptype, dataset="potential")

    for prop in ["fHI", "fH2", "metallicity"]:
        particles["gas"][prop] = reader.read_dataset(ptype="gas", dataset=prop)

    for prop in ["metallicity", "age"]:
        particles["star"][prop] = reader.read_dataset(ptype="star", dataset=prop)

    particles["bh"]["bhmdot"] = reader.read_dataset(ptype="bh", dataset="bhmdot")

    simulation_data = SimulationData(simulation=sim, constants=constants, particles=particles, groups=groups)
    assign_membership(simulation_data=simulation_data, subhalo_info=subhalo_info)

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

    if reader.global_indices is not None:
        particle_indices = reader.global_indices
    else:
        particle_indices = {ptype: np.arange(len(particles[ptype]), dtype=np.int64) for ptype in particles}

    membership_arrays = construct_membership_arrays(data=simulation_data, internals=internals, indices=particle_indices)
    packed_data = pack_rank_data(
        data=simulation_data, particle_membership_arrays=membership_arrays, internals=internals
    )

    return packed_data


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

    # initialise logger for console output
    configure_logger(
        rank=rank, output_level="INFO", log_dir=intermediate_dir
    )  # TODO: make this take the config-assigned level
    logger = get_logger()
    logger.info(f"Analysing {snapshot_path} with {size} ranks.")

    # initialise snapshot/halo readers, constants, config, internal metadata
    config = OctavianConfig.from_yaml(config_path=config_path)
    internals = load_internals(internals_filepath=internals_path, config=config)
    oc = OctavianConstants(mu=config.MU, frad=config.FRAD)
    reader = build_reader(snapshot_path=snapshot_path, constants=oc, config=config)
    halo_source = build_halo_source(config=config, reader=reader)

    # parallelism: rank 0 determines which halos need to go to which rank
    if rank == 0:  # no need for comm.Barrier() here as scatter does it inherently
        all_halo_assignments = halo_source.read_halo_ids(ptypes=reader.available_ptypes())
        subhalo_info = halo_source.read_subhalo_info()
        halo_to_rank = generate_rank_halo_assignments(
            halo_assignments=all_halo_assignments, config=config, n_ranks=size
        )
        halo_to_rank_length = len(halo_to_rank)
        assert halo_to_rank.dtype == np.int64, (
            "Bcast is receiving wrong dtype (bit corruption)."
        )  # broadcasting is done on assumption it deals with 64-bit data

    else:  # avoid MPI syntax error
        halo_to_rank = None
        subhalo_info = None
        halo_to_rank_length = 0

    halo_to_rank_length = comm.bcast(halo_to_rank_length, root=0)  # this is just an int so bcast

    if rank != 0:
        halo_to_rank = np.empty(halo_to_rank_length, dtype=np.int64)  # malloc

    comm.Bcast(halo_to_rank, root=0)  # capital B broadcast for halo_to_rank
    subhalo_info = (
        comm.bcast(subhalo_info, root=0) if comm else subhalo_info
    )  # lowercase b broadcast for the subhalo dataclass (subset of halos so smaller)

    # ranks determine which slab of each dataset they will read
    slabs = generate_slabs(rank=rank, n_ranks=size, particle_counts=reader.particle_counts)

    # rank 0 tells other ranks what the (Sub)HaloIDs of the particles on their slabs are
    raw_halo_ids = halo_source.distribute_raw_halo_ids(
        slabs=slabs, comm=comm, global_ids=all_halo_assignments.halo_ids if rank == 0 else None
    )
    raw_subhalo_ids = halo_source.distribute_raw_subhalo_ids(
        slabs=slabs, comm=comm, global_subhalo_ids=all_halo_assignments.subhalo_ids if rank == 0 else None
    )

    # ranks determine the mapping from their slab to other ranks, and the mask for their own allocation of their slab
    masks: dict[str, np.ndarray] = {}
    maps: dict[str, RedistributionMap] = {}
    local_halo_ids: dict[str, np.ndarray] = {}
    local_subhalo_ids: dict[str, np.ndarray] | None = {} if raw_subhalo_ids is not None else None

    for ptype in raw_halo_ids:
        maps[ptype], masks[ptype] = build_redistribution_map(halo_to_rank, raw_halo_ids[ptype], comm)
        local_halo_ids[ptype] = redistribute_data(raw_halo_ids[ptype][masks[ptype]], maps[ptype], comm)
        if raw_subhalo_ids is not None:
            local_subhalo_ids[ptype] = redistribute_data(raw_subhalo_ids[ptype][masks[ptype]], maps[ptype], comm)

    rank_halo_assignments = HaloAssignments(
        halo_ids=local_halo_ids,
        n_total_halos=len(halo_to_rank),
        subhalo_ids=local_subhalo_ids,
    )

    reader.set_maps(
        slabs=slabs, masks=masks, maps=maps, comm=comm
    )  # store this info on the reader for reading datasets

    # analysis pipeline (FOF6D, properties)
    packed_data = execute_pipeline(
        config=config,
        internals=internals,
        constants=oc,
        reader=reader,
        halo_assignments=rank_halo_assignments,
        global_subhalo_info=subhalo_info,
    )

    catalogue_path = output_catalogue_path(directory=output_dir)

    write_catalogue(
        packed_data=packed_data,
        catalogue_path=catalogue_path,
        internals=internals,
        comm=comm,
    )
    if rank == 0:
        write_catalogue_metadata(
            catalogue_path=catalogue_path,
            snapshot_path=snapshot_path,
            config=config,
            n_ranks=size,
        )
        clean_logs(log_dir=intermediate_dir, n_ranks=size, keep_logs=config.keep_logs)
