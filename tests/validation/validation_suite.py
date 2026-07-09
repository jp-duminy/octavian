"""

Octavian testing suite. Runs the full pipeline on a small test snapshot versus a reference catalogue.

test_snapshot_small: 600MB
test_snapshot_large: 4GB

"""

# default libraries
from pathlib import Path
from contextlib import contextmanager
from time import perf_counter
import subprocess
from datetime import datetime
from collections.abc import Generator
import argparse

# others
import h5py

# memory profiling
import memray
import psutil

# maths
import numpy as np
from matplotlib import pyplot as plt

# octavian pipeline stages
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
from octavian.run_octavian import get_mpi_communicator
from .output_validation import (
    validate_galaxy_mapping,
    validate_galaxy_membership,
    validate_group_counts,
    validate_halo_membership,
    validate_mass_budget,
    check_for_nans,
)


plt.style.use(["science"])
plt.rcParams["text.usetex"] = False
_PROCESS = psutil.Process()

RAW_PTYPES = ["PartType0", "PartType1", "PartType4", "PartType5"]  # all particle types should be present
PTYPES = ["gas", "star", "bh", "dm"]

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
INTERNALS_PATH = Path(__file__).parent.parent.parent / "octavian" / "internals.yaml"

timings = {}
memories = {}
results = []


def test_filter_snapshot(n_ranks: int, args: argparse.Namespace, sentinel_value: int = 0) -> list[Path]:
    """
    Tests whether the snapshot filter keeps all the right particles and distributes load equally.
    The sentinel value is usually 0.
    """
    logger = get_logger()

    for i in range(n_ranks):
        (args.work_dir / f"rank_{i}.hdf5").unlink(missing_ok=True)  # clears previous intermediates

    oc = OctavianConstants()

    with time_and_memory("Filter Snapshot"):
        with h5py.File(args.snapshot, "r") as f:
            header = f["Header"].attrs
            numpart = header["NumPart_Total"]  # particles before non-halo particles get tossed
            logger.info(f"Total particles: {numpart.sum()}")
            logger.info(f"Gas: {numpart[0]}, Dark Matter: {numpart[1]}")
            logger.info(f"Stars: {numpart[4]}, Black Holes: {numpart[5]}")

        reader = GizmoReader(args.snapshot, constants=oc)

        filter_snapshot(
            snapshot_file=args.snapshot,
            intermediate_directory=args.work_dir,
            reader=reader,
            n_split=n_ranks,
        )

    with h5py.File(args.snapshot, "r") as f:
        n_original = sum(
            np.sum(f[pt]["HaloID"][:] != sentinel_value) for pt in RAW_PTYPES
        )  # particles in unfiltered snapshot

    split_files = [args.work_dir / f"rank_{i}.hdf5" for i in range(n_ranks)]
    n_filtered = 0

    for path in split_files:
        with h5py.File(path, "r") as f:
            n_filtered += sum(len(f[pt]["HaloID"]) for pt in RAW_PTYPES)  # number of particles with an assigned HaloID

    logger.info(f"In-halo particles pre-filter: {n_original}")
    logger.info(f"In-halo particles post-filter: {n_filtered}")

    assert n_filtered == n_original, (
        f"Difference: {n_original - n_filtered}"
    )  # no helper function here since that says 'merge'

    logger.info("filter_snapshot passes tests.")


def _profiled_pipeline(
    snapshot_file: Path,
    output_file: Path,
    config: OctavianConfig,
    internals: Internals,
) -> None:
    """
    Executes each stage of the Octavian pipeline with timing/memory.
    """
    constants = OctavianConstants(mu=config.MU, frad=config.FRAD)

    with time_and_memory("Read-in Data"):
        reader = GizmoReader(snapshot_file=snapshot_file, constants=constants)
        sim = reader.simulation_attributes
        particles = build_particle_stores(
            reader=reader, internals=internals, constants=constants, process_ptypes=config.process_ptypes
        )

    with time_and_memory("FOF6D"):
        for prop in ["rho", "sfr"]:
            particles["gas"][prop] = reader.read_dataset(ptype="gas", dataset=prop)

        fof6d_result = find_galaxies(particles=particles, simulation=sim, config=config, constants=constants)

    with time_and_memory("Build GroupStores"):
        groups: dict[str, GroupStore] = {}

        groups["halos"] = build_group_store(particles=particles, group_type="halos")

        if fof6d_result.n_galaxies > 0:
            groups["galaxies"] = build_group_store(particles=particles, group_type="galaxies")

    with time_and_memory("Load Aggregate Columns"):
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
        with time_and_memory(stage.name):
            stage_dispatch[stage.name](simulation_data=simulation_data, config=config)

            releasable = get_releasable_columns(stage_index, ordered_stages)

            for ptype in particles:
                for col in releasable:
                    if col in particles[ptype]:
                        particles[ptype].release(col)

    with time_and_memory("Save data"):
        for ptype in particles:
            particles[ptype]["particle_index"] = reader.read_dataset(ptype=ptype, dataset="particle_index")

        particle_lists = construct_particle_csr_lists(data=simulation_data, internals=internals)
        write_analysis_to_output_file(
            data=simulation_data,
            particle_lists=particle_lists,
            internals=internals,
            output_file=output_file,
        )


def test_remerge(
    files: list[Path],
    output_path: Path,
    internals: Internals,
    sentinel_value: int = 0,
) -> None:
    """
    Tests the remerging of the snapshot.
    """
    logger = get_logger()
    n_halos_original, n_galaxies_original = 0, 0

    for path in files:
        with h5py.File(path, "r") as f:
            n_halos_original += len(f["halo_data"]["HaloID"])
            n_galaxies_original += len(f["galaxy_data"]["GalID"])

    logger.info(f"Halos pre-merge: {n_halos_original}")
    logger.info(f"Galaxies pre-merge: {n_galaxies_original}")

    with time_and_memory("Remerge Catalogues"):
        merge_intermediate_catalogues(files=files, output_path=output_path, internals=internals)

    with h5py.File(output_path, "r") as f:
        n_galaxies_final = len(f["galaxy_data"]["GalID"])
        n_halos_final = len(f["halo_data"]["HaloID"])

    logger.info(f"Halos post-merge: {n_halos_final}")
    logger.info(f"Galaxies post-merge: {n_galaxies_final}")

    _assert_conserved(label="Number of Halos", pre=n_halos_original, post=n_halos_final)
    _assert_conserved(label="Number of Galaxies", pre=n_galaxies_original, post=n_galaxies_final)


def validate_against_reference(catalogue: str, reference: str, rtol: float = 1e-6, atol: float = 1e-10) -> None:
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

                assert ref_data.shape == new_data.shape, (
                    f"{group_name}/{key}: array shpe mismatch (ref={ref_data.shape}, new={new_data.shape})"
                )

                if not np.issubdtype(ref_data.dtype, np.floating):
                    assert np.array_equal(ref_data, new_data), f"{group_name}/{key}: dtype data mismatch"
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


def plot_gsmf(catalogue: Path, boxsize: float, minstars: int = 32) -> None:
    """
    Plots the galactic stellar mass function (useful for FOF6D testing)
    """
    logger = get_logger()

    with h5py.File(catalogue, "r") as f:
        mass = f["galaxy_data"]["properties/core/mass_star"][:]
        mass_gas = f["galaxy_data"]["properties/core/mass_gas"][:]
        n_star = f["galaxy_data"]["properties/core/n_star"][:]
        n_gas = f["galaxy_data"]["properties/core/n_gas"][:]

        star_mask = n_star >= minstars
        mass = mass[star_mask]
        mass_gas = mass_gas[star_mask]
        n_star = n_star[star_mask]
        n_gas = n_gas[star_mask]

        logger.info(f"Resolved star mass: {np.sum(mass):.2e}")
        logger.info(f"Resolved star counts: {np.sum(n_star):.2e}")
        logger.info(f"Resolved gas mass: {np.sum(mass_gas):.2e}")
        logger.info(f"Resolved gas counts: {np.sum(n_gas):.2e}")
        logger.info(f"Resolved galaxies: {star_mask.sum()}")

        positive_mask = mass > 0
        log_mass = np.log10(mass[positive_mask])

        bin_edges = np.arange(8.5, 12.0 + 0.2, 0.2)
        counts, _ = np.histogram(log_mass, bins=bin_edges)

        volume = boxsize**3

        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        phi = counts / (bin_width * volume)
        phi_err = np.sqrt(counts) / (bin_width * volume)

        mask = counts > 0

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(
            x=bin_centres[mask], y=phi[mask], yerr=phi_err[mask], color="red", fmt="o", ls="-", capsize=3, label="GSMF"
        )
        ax.legend()
        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10}(M_\star / M_\odot)$")
        ax.set_ylabel(r"$\Phi$ [dex$^{-1}$ Mpc$^{-3}$ $h^3$]")
        ax.set_title("Galaxy Stellar Mass Function: z=0, all fixes, new algo (no self-pairs)")
        fig.tight_layout()
        fig.savefig(fname="gsmf.png", dpi=300)


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
        f.write(f"Octavian Test Summary: Commit {COMMIT_HASH[:8]} // {timestamp} \n")
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
    Conduct a full (parallel/serial) run of Octavian.
    """
    comm = get_mpi_communicator()
    rank = comm.Get_rank() if comm else 0
    size = comm.Get_size() if comm else 1

    configure_logger(rank=rank, output_log_directory=args.work_dir)
    logger = get_logger()

    memray_file = Path(args.work_dir / f"memray_rank_{rank}.bin")
    memray_file.unlink(missing_ok=True)

    with memray.Tracker(memray_file, native_traces=True):
        config = OctavianConfig.from_yaml(config_path=CONFIG_PATH)
        internals = load_internals(internals_filepath=INTERNALS_PATH, config=config)

        if rank == 0:
            logger.info(f"Testing Octavian with {size} ranks.")
            test_filter_snapshot(n_ranks=size, args=args)
            logger.info("Filtering complete.")

        if comm:
            comm.Barrier()

        snapshot_file = args.work_dir / f"rank_{rank}.hdf5"
        intermediate_file = args.work_dir / f"rank_{rank}_intermediate_analysis.hdf5"
        _profiled_pipeline(
            snapshot_file=snapshot_file, output_file=intermediate_file, config=config, internals=internals
        )

        if comm:
            comm.Barrier()

        output_catalogue = args.work_dir / "output_catalogue.hdf5"

        if rank == 0:
            files = [args.work_dir / f"rank_{i}_intermediate_analysis.hdf5" for i in range(size)]
            test_remerge(files=files, output_path=output_catalogue, internals=internals)
            conduct_output_catalogue_validation(catalogue=output_catalogue)

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
            plot_gsmf(catalogue=output_catalogue, boxsize=25, minstars=32)


def _assert_conserved(label: str, pre: int, post: int):
    """
    Helper function to check whether the counts are preserved.
    """
    logger = get_logger()
    logger.info(f"{label}: pre-merge={pre} / post-merge={post}")
    assert pre == post, f"{label} mismatch: {post - pre:+d}"


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
    parser = argparse.ArgumentParser(description="Octavian validation suite")
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
