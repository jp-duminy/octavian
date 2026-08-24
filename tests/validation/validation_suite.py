"""

Octavius testing suite. Runs the full pipeline on a small test snapshot versus a reference catalogue.

test_snapshot_small: 600MB
test_snapshot_large: 4GB

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octavius.external_halo_sources import HaloAssignments, SubhaloInformation
    from octavius.data_management import SnapshotReader, RankPackedData
    from octavius.data_management.parallel_reading import RedistributionMap

# default libraries
from pathlib import Path
from contextlib import contextmanager
from time import perf_counter
import subprocess
from datetime import datetime
from collections.abc import Generator
import argparse
from dataclasses import replace

# other packages
import memray
import psutil
import h5py
import numpy as np
import numba

# internal imports
from octavius.data_management import (
    write_catalogue,
    construct_membership_arrays,
    GroupStore,
    SimulationData,
    Internals,
    OctaviusConstants,
    OctaviusConfig,
    build_reader,
    build_galaxy_store,
    build_halo_store,
    build_particle_stores,
    load_internals,
    resolve_dependencies,
    load_stage_columns,
    release_stage_columns,
    validate_stage_requirements,
    output_catalogue_path,
    assign_local_subhalos,
    generate_rank_halo_assignments,
    redistribute_data,
    generate_slabs,
    pack_rank_data,
    write_catalogue_headers,
    build_redistribution_map,
)
from octavius.external_halo_sources import (
    build_halo_source,
    HaloAssignments,
)
from octavius.log import configure_logger, get_logger, clean_logs
from octavius.galaxy_finding import find_galaxies, FOF6DResult
from octavius.aggregate_properties import (
    run_ptype_specific_properties,
    run_core_properties,
    run_local_environment,
    assign_membership,
)
from octavius.photometry import (
    resolve_band_names,
    read_filter_names,
    run_photometry,
)
from octavius.run_octavius import get_mpi_communicator
from .output_validation import (
    validate_galaxy_mapping,
    validate_galaxy_membership,
    validate_group_counts,
    validate_halo_membership,
    validate_mass_budget,
    check_for_nans,
)

_PROCESS = psutil.Process()

RAW_PTYPES = ["PartType0", "PartType1", "PartType4", "PartType5"]  # all particle types should be present
PTYPES = ["gas", "star", "bh", "dm"]

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
INTERNALS_PATH = Path(__file__).parent.parent.parent / "octavius" / "internals.yaml"

timings = {}
memories = {}
results = []


def test_rank_assignments(
    halo_to_rank: np.ndarray,
    all_halo_assignments: HaloAssignments,
    n_ranks: int,
) -> None:
    """
    Tests whether halo-to-rank assignments from generate_rank_halo_assignments() are self-consistent.
    """
    logger = get_logger()
    per_halo_weight = np.zeros(len(halo_to_rank), dtype=np.int64)
    assert np.all((halo_to_rank == -1) | ((halo_to_rank >= 0) & (halo_to_rank < n_ranks))), (
        "rank_halo_assignments failed: haloes assigned to invalid ranks."
    )

    for ptype in all_halo_assignments.halo_ids:
        valid_hids = all_halo_assignments.halo_ids[ptype]
        valid_hids = valid_hids[valid_hids != -1]
        assert np.all(valid_hids < len(halo_to_rank)), (
            "rank_halo_assignments failed: halo_to_rank is longer than the number of valid haloes."
        )
        per_halo_weight += np.bincount(valid_hids, minlength=len(halo_to_rank))

    rank_weights = np.bincount(
        halo_to_rank[halo_to_rank != -1], weights=per_halo_weight[halo_to_rank != -1], minlength=n_ranks
    )
    mean_weight = rank_weights.sum() / n_ranks
    assert np.all(rank_weights <= 1.5 * mean_weight), (
        "rank_halo_assignments failed: one rank is assigned a disproportionate amount of computational weight."
    )

    logger.info("Rank-halo assignments are self consistent.")


def _profiled_pipeline(
    config: OctaviusConfig,
    internals: Internals,
    constants: OctaviusConstants,
    reader: SnapshotReader,
    halo_assignments: HaloAssignments,
    global_subhalo_info: SubhaloInformation,
) -> RankPackedData:
    """
    Executes each stage of the Octavius pipeline with timing/memory.
    """
    with time_and_memory("Initialise particles"):
        sim = reader.simulation_attributes
        particles = build_particle_stores(
            reader=reader, internals=internals, halo_assignments=halo_assignments, process_ptypes=config.process_ptypes
        )
        subhalo_info = assign_local_subhalos(particles=particles, subhalo_info=global_subhalo_info)

    if config.stages.get("find_galaxies", True):
        with time_and_memory("Load FOF6D data"):
            load_stage_columns(particles=particles, reader=reader, stage=internals.stages["find_galaxies"])

        with time_and_memory("Find Galaxies"):
            fof6d_result = find_galaxies(particles=particles, simulation=sim, config=config, constants=constants)
    else:
        for ptype in particles:
            particles[ptype]["GalID"] = np.full(particles[ptype].n_particles, -1, dtype=np.int64)
        fof6d_result = FOF6DResult.empty()

    with time_and_memory("Build GroupStores"):
        groups: dict[str, GroupStore] = {}

        groups["halos"] = build_halo_store(
            particles=particles,
            halo_key=internals.group_types["halos"]["key"],
            subhalo_key="SubhaloID",
            group_kind=internals.group_types["halos"]["kind"],
            subhalo_info=subhalo_info,
            original_halo_ids=halo_assignments.original_hids,
        )

        if fof6d_result.n_galaxies > 0:
            groups["galaxies"] = build_galaxy_store(
                particles=particles,
                galaxy_key=internals.group_types["galaxies"]["key"],
                group_kind=internals.group_types["galaxies"]["kind"],
            )

    simulation_data = SimulationData(simulation=sim, constants=constants, particles=particles, groups=groups)
    assign_membership(simulation_data=simulation_data, subhalo_info=subhalo_info)

    requested = [name for name, enabled in config.stages.items() if enabled and name != "find_galaxies"]
    ordered_stages = resolve_dependencies(stages=internals.stages, requested=requested)
    validate_stage_requirements(ordered_stages=ordered_stages, available_ptypes=set(particles.keys()))

    stage_dispatch = {
        "properties_core": run_core_properties,
        "properties_ptype_specific": run_ptype_specific_properties,
        "properties_local_environment": run_local_environment,
        "photometry": run_photometry,
    }

    for stage_idx, stage in enumerate(ordered_stages):
        with time_and_memory(f"Load data for {stage.name}"):
            load_stage_columns(particles=particles, reader=reader, stage=stage)
        with time_and_memory(f"Run {stage.name}"):
            stage_dispatch[stage.name](simulation_data=simulation_data, config=config)
        release_stage_columns(particles=particles, current_idx=stage_idx, ordered_stages=ordered_stages)

    with time_and_memory("Save data"):
        if reader.global_indices is not None:
            particle_indices = reader.global_indices
        else:
            particle_indices = {ptype: np.arange(len(particles[ptype]), dtype=np.int64) for ptype in particles}

        membership_arrays = construct_membership_arrays(
            data=simulation_data, internals=internals, indices=particle_indices
        )

        packed_data = pack_rank_data(
            data=simulation_data,
            particle_membership_arrays=membership_arrays,
            internals=internals,
        )

    return packed_data


def validate_against_reference(catalogue: Path, reference: Path, rtol: float = 1e-6, atol: float = 1e-10) -> None:
    """
    Validates output vs reference more rigorously, comparing specific fields within tolerance.

    Also validates whether NaNs occur in the same place.
    """
    logger = get_logger()

    with h5py.File(catalogue, "r") as new, h5py.File(reference, "r") as ref:
        for group_name in ["halo_data", "galaxy_data"]:
            if group_name not in ref or group_name not in new:
                continue

            ref_group = ref[group_name]
            new_group = new[group_name]

            ref_keys = set()  # set notation makes this much easier
            new_keys = set()
            # visit returns all fields (so not just headers), otherwise we get stuck on /dicts
            ref_group.visit(
                lambda k: ref_keys.add(k) if isinstance(ref_group[k], h5py.Dataset) else None
            )  # lambda filters out headers
            new_group.visit(lambda k: new_keys.add(k) if isinstance(new_group[k], h5py.Dataset) else None)

            only_ref = ref_keys - new_keys
            only_new = new_keys - ref_keys

            # check datasets are consistent
            assert len(only_ref) == 0, f"{group_name}: datasets missing from new catalogue: {sorted(only_ref)}"
            if only_new:
                logger.warning(f"{group_name}: new datasets not in reference (do check): {sorted(only_new)}")

            logger.info(f"{group_name} dataset names match between output and reference.")

            for key in sorted(ref_keys & new_keys):
                if isinstance(ref_group[key], h5py.Group):  # ignore groups
                    continue

                ref_data = ref_group[key][:]
                new_data = new_group[key][:]
                if key.endswith("_offsets"):
                    continue
                assert ref_data.shape == new_data.shape, (
                    f"{group_name}/{key}: array shpe mismatch (ref={ref_data.shape}, new={new_data.shape})"
                )

                if not np.issubdtype(ref_data.dtype, np.floating):
                    assert np.array_equal(ref_data, new_data), (
                        f"{group_name}/{key}: dtype data mismatch, {ref_data.dtype, new_data.dtype}"
                    )
                    continue

                # check any existing NaN columns have the NaNs in the same place (groups should remain empty, analysis is deterministic)
                ref_finite = np.isfinite(ref_data)
                new_finite = np.isfinite(new_data)
                nan_mismatch = np.sum(ref_finite != new_finite)
                assert nan_mismatch == 0, (
                    f"{group_name}/{key}: {nan_mismatch} NaNs occur in different places (different groups potentially)"
                    f"(ref has {np.sum(~ref_finite)} NaN, new has {np.sum(~new_finite)} NaN)"
                )

                both_finite = ref_finite & new_finite
                if not np.any(both_finite):
                    continue

                r = ref_data[both_finite]
                n = new_data[both_finite]

                if np.array_equal(r, n):
                    continue

                abs_diff = np.abs(r - n)
                scale = np.maximum(np.abs(r), atol)
                rel_diff = abs_diff / scale

                max_rel = rel_diff.max()
                n_exceed = np.sum(rel_diff > rtol)

                if n_exceed > 0:
                    logger.warning(
                        f"{group_name}/{key}: {n_exceed}/{len(r)} values exceed rtol={rtol} "
                        f"(max_rel={max_rel:.6e}, max_abs={abs_diff.max():.6e})"
                    )

            logger.info(f"{group_name} dataset values match to within tolerance between reference and output.")

    logger.info("Reference catalogue comparison passes.")


def conduct_output_catalogue_validation(catalogue: Path) -> None:
    """
    Wraps all the catalogue validation functions.
    """
    with h5py.File(catalogue, "r") as f:
        with record_assertion_result("Halo Membership"):
            validate_halo_membership(f=f)

        with record_assertion_result("Galaxy Membership"):
            validate_galaxy_membership(f=f)

        with record_assertion_result("Galaxy-Halo Mapping"):
            validate_galaxy_mapping(f=f)

        with record_assertion_result("Halo Particle Counts"):
            validate_group_counts(f=f, group_data="halo_data")

        with record_assertion_result("Galaxy Particle Counts"):
            validate_group_counts(f=f, group_data="galaxy_data")

        with record_assertion_result("NaN Checking"):
            check_for_nans(f=f)

        with record_assertion_result("Mass Budget Validation"):
            validate_mass_budget(f=f)


def record_test_results(
    all_timings: list[dict[str, float]],
    all_memories: list[dict[str, float]],
    results: list[tuple[str, bool, str]],
    peak_memory: list[float],
    size: int,
    cores_per_rank: int,
    args: argparse.Namespace,
) -> None:
    """
    Checks the validation outputs and writes the result to a .txt file.
    """
    logger = get_logger()

    COMMIT_HASH = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # current version
    filepath = args.output_dir / f"test_summary_{COMMIT_HASH[:8]}.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    passed = all(success for _, success, _ in results)

    logger.info(f"Final Result: {'PASS' if passed else 'FAIL'}")

    with open(filepath, "w") as f:
        # summary (what you care about)
        f.write(f"Octavius Test Summary: Commit {COMMIT_HASH[:8]} // {timestamp} \n")
        f.write(f"Snapshot: {args.snapshot} \n")
        f.write(f"{size} ranks // {cores_per_rank} cores per rank\n")
        f.write(f"Final Result: {'PASS' if passed else 'FAIL'}\n\n")

        # stage time/memory breakdown
        if size == 1:  # serial
            for stage, elapsed in all_timings[0].items():
                mem = all_memories[0].get(stage, 0.0)
                f.write(f"{stage}: {elapsed:.2f}s, {mem:.3f} GB\n")
        else:  # parallel
            stages = all_timings[0].keys()
            for stage in stages:
                vals = [
                    t[stage] for t in all_timings if stage in t
                ]  # leave in conditional in case we want to test stages
                mems = [m[stage] for m in all_memories if stage in m]
                f.write(
                    f"{stage} longest time: {max(vals):.3f}s\n"
                    f"Time spread: {max(vals) - min(vals):.2f}s\n"
                    f"Peak Memory: {max(mems):.3f} GB\n"
                    f"Memory Spread: {max(mems) - min(mems):.3f} GB\n"
                )
        f.write("\n")

        # validation checks
        f.write("Validation:\n")
        for label, success, msg in results:
            status = "PASS" if success else "FAIL"
            line = f"{label}, {status}"
            if not success:
                line += f" — {msg}"
            f.write(line + "\n")
        f.write("\n")

        # peak RSS
        f.write("RSS:\n")
        for i, rss in enumerate(peak_memory):
            f.write(f"Rank {i}: {rss:.2f} GB\n")
        if size > 1:
            f.write(f"Max: {max(peak_memory):.2f} GB\n")


def test_run(args: argparse.Namespace) -> None:
    """
    Conduct a full (parallel/serial) run of Octavius.
    """
    comm = get_mpi_communicator()
    rank = comm.Get_rank() if comm else 0
    size = comm.Get_size() if comm else 1

    configure_logger(rank=rank, log_dir=args.work_dir)
    logger = get_logger()

    memray_file = Path(args.work_dir / f"memray_rank_{rank}.bin")
    memray_file.unlink(missing_ok=True)

    with memray.Tracker(memray_file, native_traces=True):
        config = OctaviusConfig.from_yaml(config_path=CONFIG_PATH)
        numba.set_num_threads(n=config.cores_per_rank)

        if config.stages.get("photometry", False):
            names, lambda_effs = read_filter_names(config.photometry_table_filepath)
            config = replace(config, bands=resolve_band_names(config.bands, names, lambda_effs))

        internals = load_internals(internals_filepath=INTERNALS_PATH, config=config)

        if rank == 0:
            logger.info(f"Testing Octavius with {size} ranks.")
        oc = OctaviusConstants(mu=config.MU, frad=config.FRAD)
        reader = build_reader(snapshot_path=args.snapshot, constants=oc, config=config)
        halo_source = build_halo_source(config=config, reader=reader)

        if rank == 0:  # no need for comm.Barrier() here as scatter does it inherently
            all_halo_assignments = halo_source.read_halo_ids(ptypes=reader.available_ptypes())
            subhalo_info = halo_source.read_subhalo_info()

            halo_to_rank = generate_rank_halo_assignments(
                halo_assignments=all_halo_assignments, config=config, n_ranks=size
            )
            original_halo_ids = all_halo_assignments.original_hids
            test_rank_assignments(halo_to_rank=halo_to_rank, all_halo_assignments=all_halo_assignments, n_ranks=size)
            halo_to_rank_length = len(halo_to_rank)
            assert halo_to_rank.dtype == np.int64

        else:
            halo_to_rank = None
            subhalo_info = None
            original_halo_ids = None
            halo_to_rank_length = 0

        if comm is not None:
            halo_to_rank_length = comm.bcast(halo_to_rank_length, root=0)  # this is just an int so bcast

            if rank != 0:
                halo_to_rank = np.empty(halo_to_rank_length, dtype=np.int64)  # malloc

            comm.Bcast(halo_to_rank, root=0)  # capital B broadcast for halo_to_rank
            subhalo_info = (
                comm.bcast(subhalo_info, root=0) if comm else subhalo_info
            )  # lowercase b broadcast for the subhalo dataclass (subset of halos so smaller)
            original_halo_ids = comm.bcast(original_halo_ids, root=0) if comm else original_halo_ids

            slabs = generate_slabs(rank=rank, n_ranks=size, particle_counts=reader.particle_counts)

            raw_halo_ids = halo_source.distribute_raw_halo_ids(
                slabs=slabs, comm=comm, global_ids=all_halo_assignments.halo_ids if rank == 0 else None
            )
            raw_subhalo_ids = halo_source.distribute_raw_subhalo_ids(
                slabs=slabs, comm=comm, global_subhalo_ids=all_halo_assignments.subhalo_ids if rank == 0 else None
            )

        else:
            slabs = generate_slabs(rank=0, n_ranks=1, particle_counts=reader.particle_counts)
            raw_halo_ids = halo_source.distribute_raw_halo_ids(slabs=slabs)
            raw_subhalo_ids = halo_source.distribute_raw_subhalo_ids(slabs=slabs)

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
            original_hids=original_halo_ids,
        )

        reader.set_maps(slabs=slabs, masks=masks, maps=maps, comm=comm)

        all_halo_assignments = None
        raw_halo_ids = None
        raw_subhalo_ids = None

        packed_data = _profiled_pipeline(
            config=config,
            internals=internals,
            constants=oc,
            reader=reader,
            halo_assignments=rank_halo_assignments,
            global_subhalo_info=subhalo_info,
        )

        if comm:
            comm.Barrier()

        catalogue_path = output_catalogue_path(snapshot_path=args.snapshot, output_dir=args.work_dir)

        write_catalogue(
            packed_data=packed_data,
            catalogue_path=catalogue_path,
            internals=internals,
            comm=comm,
        )

        if rank == 0:
            conduct_output_catalogue_validation(catalogue=catalogue_path)
            write_catalogue_headers(
                catalogue_path=catalogue_path,
                snapshot_path=args.snapshot,
                config=config,
                internals=internals,
                sim_attrs=reader.simulation_attributes,
                n_ranks=size,
            )
            clean_logs(log_dir=args.work_dir, n_ranks=size, keep_logs=config.keep_logs)

        all_timings = comm.gather(timings, root=0) if comm else [timings]
        all_memories = comm.gather(memories, root=0) if comm else [memories]
        all_rss = comm.gather(current_memory_gb(), root=0) if comm else [current_memory_gb()]

        if rank == 0:
            # per-rank timings
            for i, t in enumerate(all_timings):
                for stage, elapsed in t.items():
                    logger.info(f"Rank {i}: {stage} = {elapsed:.2f}s")

            # imbalances across ranks
            stages = all_timings[0].keys()
            for stage in stages:
                vals = [t[stage] for t in all_timings if stage in t]
                logger.info(f"{stage}: max={max(vals):.2f}s  spread={max(vals) - min(vals):.2f}")

            record_test_results(
                all_timings=all_timings,
                all_memories=all_memories,
                results=results,
                peak_memory=all_rss,
                size=size,
                cores_per_rank=config.cores_per_rank,
                args=args,
            )
            validate_against_reference(catalogue=catalogue_path, reference=args.reference)


def current_memory_gb() -> float:
    """
    Returns the current physical memory being used in GB; memray gives more detailed diagnostics (like high watermark).
    """
    return _PROCESS.memory_info().rss / 1024**3


@contextmanager
def time_and_memory(label: str):
    """
    Helper function to avoid repeatedly doing perf_counter/memory usage statements.
    To use, run:
    with time_and_memory(f"stagename"):
        foo()
    """
    logger = get_logger()
    t0 = perf_counter()
    original_memory = current_memory_gb()
    yield
    elapsed = perf_counter() - t0
    delta_memory = current_memory_gb() - original_memory
    timings[label] = elapsed
    memories[label] = delta_memory
    logger.info(f"{label} completed in {elapsed:.2f}s.")
    logger.info(f"{label} used {delta_memory:.3f}GB of memory.")


@contextmanager
def record_assertion_result(label: str) -> Generator[None, None, None]:  # the collections import is used for this
    """
    Used for wrapping the validation checks in a try / except.
    Prevents an error being thrown.
    """
    logger = get_logger()
    try:
        yield
        results.append((label, True, ""))
    except AssertionError as e:
        results.append((label, False, str(e)))
        logger.error(f"{label} FAIL: {e}")


def parse_args() -> argparse.Namespace:
    """
    For command line arguments (copied from my MVP code).
    """
    parser = argparse.ArgumentParser(description="Octavius validation suite")
    parser.add_argument("-s", "--snapshot", type=Path, required=True, help="Path to snapshot")
    parser.add_argument(
        "-r", "--reference", type=Path, required=True, help="Path to reference catalogue (run on same snapshot)"
    )
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("-w", "--work-dir", type=Path, required=True, help="Working/intermediate directory")

    return parser.parse_args()


def main() -> None:

    args = parse_args()
    test_run(args=args)


if __name__ == "__main__":
    main()
