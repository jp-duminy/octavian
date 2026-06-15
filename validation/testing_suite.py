"""

Octavian testing suite. Runs the full pipeline on a small test snapshot versus a reference catalogue.

test_snapshot_small: 600MB
test_snapshot_large: 4GB

"""

# type checking (semantic, do not worry about this)
from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from mpi4py import MPI

# these are all default libraries
from pathlib import Path
from contextlib import contextmanager
from time import perf_counter
import logging
import subprocess
from dataclasses import dataclass
from yaml import safe_load
from datetime import datetime
from collections.abc import Generator

# others
import h5py
import memray
import numpy as np
from matplotlib import pyplot as plt

# data
from test_constants import NEVER_NAN, CONDITIONAL_NAN, BARYON_CONDITIONAL_NAN, ZERO_WHEN_EMPTY, SOFT_NAN

# octavian pipeline stages
from octavian.data_management import DataManager, save_group_properties, wrap_positions, filter_snapshot
from octavian.utils import merge_catalogues
from octavian.fof6d import run_fof6d
from octavian.aggregate_properties import calculate_group_properties, get_particle_lists
from octavian.run_octavian import _get_mpi_communicator

@dataclass
class TestConfig:
    test_snapshot: Path
    reference_catalogue: Path
    config_file: Path 
    working_directory: Path  # for the intermediate files
    n_ranks: int = 4 # arbitrary default: 4 mpi ranks each with 6 cores
    n_proc: int = 6 

PTYPES = ['PartType0', 'PartType1', 'PartType4', 'PartType5'] # all particle types should be present
ALL_PARTICLE_LISTS = ["glist", "slist", "dmlist", "bhlist"]
PTYPE_TO_PLIST = {"star": "slist", "gas": "glist", "dm": "dmlist", "bh": "bhlist"}
BARYONIC_PARTICLE_LISTS = ["glist", "slist", "bhlist"]
SUFFIXES = ["lengths", "offsets", "indices"] # for csr indexing

test_config = TestConfig(test_snapshot=Path(f"/home/jpduminy/Octavian/test_snapshot_large.hdf5"),
                            reference_catalogue =Path(f"/home/jpduminy/Octavian/reference_catalogue_small.hdf5"),
                            config_file=Path(f"/home/jpduminy/Repositories/octavian/config.yaml"),
                            working_directory=Path(f"/home/jpduminy/Octavian/Intermediates/"),
                            n_ranks = 2,
                            n_proc = 4
)

logging.basicConfig(
        level=logging.INFO,
        format=f"[%(levelname)s] %(name)s — %(message)s",
    )

logger = logging.getLogger("octavian.tests")

timings = {}
memories = {}
results = []

def peak_memory_gb() -> float:
    """
    Linux-specific: locates the peak memory at a stage
    """
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"): # change to VmHWM (high water mark) if you only want this for overall total.
                return int(line.split()[1]) / 1024**2  # kB to GB

def peak_rss_gb() -> float:
    """
    Peak memory across the run (same as above function but VmRSS -> VmHWM, so resident -> high watermark).
    """
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024**2 # kB to GB
    return -1.0

@contextmanager
def time_and_memory(label: str):
    """
    Helper function to avoid repeatedly doing perf_counter/memory usage statements.
    To use, run:
    with time_and_memory(f"stagename"):
        foo()
    """
    t0 = perf_counter()
    original_memory = peak_memory_gb()
    yield
    elapsed = perf_counter() - t0
    used_memory = peak_memory_gb() - original_memory
    timings[label] = elapsed
    memories[label] = used_memory
    logger.info(f"{label} completed in {elapsed:.2f}s.")
    logger.info(f"{label} used {used_memory:.3f}GB of memory.")

def test_header_info() -> None:
    """
    Checks the header information.
    """
    with h5py.File(test_config.test_snapshot, 'r') as f:

        header = f['Header'].attrs
        logger.info(f"Boxsize: {header['BoxSize']}, h: {header['HubbleParam']}")
        logger.info(f"Omega0: {header['Omega0']}, Omega_lambda: {header['OmegaLambda']}")
        logger.info(f"Redshift: {header['Redshift']:.3f}")        

def test_filter_snapshot(sentinel_value: int = 0) -> list[Path]:
    """
    Tests whether the snapshot filter keeps all the right particles and distributes load equally.
    The sentinel value is usually 0.
    """
    for i in range(test_config.n_ranks):
        Path(f"{test_config.working_directory}_rank_{i}.hdf5").unlink(missing_ok=True) # clears previous intermediates

    with time_and_memory(f"Filter Snapshot"):

        with h5py.File(test_config.test_snapshot, 'r') as f: 

            header = f['Header'].attrs
            numpart = header['NumPart_Total'] # particles before non-halo particles get tossed
            logger.info(f"Total particles: {numpart.sum()}")
            logger.info(f"Gas: {numpart[0]}, Dark Matter: {numpart[1]}")
            logger.info(f"Stars: {numpart[4]}, Black Holes: {numpart[5]}")

        filter_snapshot(snapfile=test_config.test_snapshot, outfile=test_config.working_directory, 
                        nsplit=test_config.n_ranks)

    with h5py.File(test_config.test_snapshot, 'r') as f:

        n_original = sum(np.sum(f[pt]['HaloID'][:] != sentinel_value) for pt in PTYPES) # particles in unfiltered snapshot

    split_files = [f'{test_config.working_directory}_rank_{i}.hdf5' for i in range(test_config.n_ranks)]
    n_filtered = 0

    for path in split_files:

        with h5py.File(path, 'r') as f:

            n_filtered += sum(len(f[pt]['HaloID']) for pt in PTYPES) # number of particles with an assigned HaloID

    logger.info(f"In-halo particles pre-filter: {n_original}")
    logger.info(f"In-halo particles post-filter: {n_filtered}")

    assert n_filtered == n_original, f"Difference: {n_original - n_filtered}" # no helper function here since that says 'merge'

    logger.info(f"filter_snapshot passes tests.")

def _end_to_end_pipeline(snapshot_file: str, output_file: str, comm: MPI.Comm | None) -> None:
    """
    Executes each stage of the Octavian pipeline with timings.
    """
    with open(test_config.config_file, 'r') as f:
        config = safe_load(f)
    config['Tlim'] = float(config['Tlim'])

    with time_and_memory(f"Data Manager Initialisation"):
        data_manager = DataManager(snapfile=snapshot_file, config=config, comm=comm)
        data_manager.load_halo_ids() # keep these stages in since I expect this takes a while
        data_manager.add_ptype_columns()

    with time_and_memory(f"Unwrap Positions"):
        wrap_positions(data_manager=data_manager)

    with time_and_memory(f"FOF6D"):
        run_fof6d(data_manager=data_manager, nproc=test_config.n_proc)

    data_manager.initialise_group_data()

    with time_and_memory(f"Group Properties"):
        calculate_group_properties(data_manager=data_manager)

    with time_and_memory(f"Save data"):
        get_particle_lists(data_manager=data_manager)
        save_group_properties(data_manager=data_manager, filename=output_file)

def _assert_conserved(label: str, pre: int, post: int):
    """
    Helper function to check whether the counts are preserved.
    """
    logger.info(f"{label}: pre-merge={pre} / post-merge={post}")
    assert pre == post, f"{label} mismatch: {post - pre:+d}"

def test_remerge(files: list[str], outfile: str, configfile: str, sentinel_value: int = 0) -> None:
    """
    Tests the remerging of the snapshot.
    """
    n_halos_original, n_galaxies_original = 0, 0
    
    for path in files:

        with h5py.File(path, 'r') as f:

            n_halos_original += len(f['halo_data']['haloID']) 
            n_galaxies_original += len(f['galaxy_data']['galaxyID'])

    logger.info(f"Halos pre-merge: {n_halos_original}")
    logger.info(f"Galaxies pre-merge: {n_galaxies_original}")

    with time_and_memory(f"Remerge Catalogues"):

        merge_catalogues(files=files, outfile=outfile, configfile=configfile)

    with h5py.File(outfile, 'r') as f:

        n_galaxies_final = len(f['galaxy_data']['GalID'])
        n_halos_final = len(f['halo_data']['HaloID'])

    logger.info(f"Halos post-merge: {n_halos_final}")
    logger.info(f"Galaxies post-merge: {n_galaxies_final}")

    _assert_conserved(label=f"Number of Halos", pre=n_halos_original, post=n_halos_final)
    _assert_conserved(label=f"Number of Galaxies", pre=n_galaxies_original, post=n_galaxies_final)

def _validate_csr_integrity(f: h5py.File, group_data: str, particle_list: str) -> None:
    """
    Validates CSR format of particle lists (sanity checks); handles empty groups too.
    """
    lengths = f[group_data][f"{particle_list}_lengths"][:]
    offsets = f[group_data][f"{particle_list}_offsets"][:]
    indices = f[group_data][f"{particle_list}_indices"][:]

    # check offset slicing matches
    expected = np.concatenate([[0], np.cumsum(lengths[:-1])]) # mainly want to verify the prepended 0 is there
    assert np.array_equal(offsets, expected), f"{group_data}/{particle_list} offset array does not match expected format."
    logger.info(f"{group_data}/{particle_list} offset slicing matches.")

    # total count consistency
    assert lengths.sum() == len(indices), f"{group_data}/{particle_list} number of particles does not match number of indices."
    logger.info(f"{group_data}/{particle_list} particle counts are self-consistent.")

    # no duplicates within groups (sometimes different ptypes have the same ID though)
    group_labels = np.repeat(np.arange(len(lengths)), lengths) # full-length array
    order = np.lexsort((indices, group_labels)) 
    sorted_idx = indices[order]
    sorted_grp = group_labels[order]

    same_group = sorted_grp[1:] == sorted_grp[:-1] # false at group boundary
    same_value = sorted_idx[1:] == sorted_idx[:-1] 
    dupes = np.where(same_group & same_value)[0]

    assert len(dupes) == 0, f"{group_data}/{particle_list}: {len(dupes)} duplicate particles in groups."
    logger.info(f"{group_data}/{particle_list} has no intra-group duplicate particles.")

    # no particle appears in two groups
    if len(indices) > 0:

        assert np.all(indices >= 0), f"Invalid indices in {group_data}/{particle_list}."
        assert len(np.unique(indices)) == len(indices), f"In {group_data}/{particle_list}, same particle appears in multiple groups."

    logger.info(f"{group_data}/{particle_list} has no duplicates across groups.")
    logger.info(f"{group_data}/{particle_list} passes tests.")

def _check_keys_exist(f: h5py.File, keys: list[str]) -> None:
    """
    Checks whether the keys in the passed list exist in a h5py file object.
    """
    for key in keys:
        assert key in f, f"Key {key} does not exist in the catalogue."

def validate_halo_membership(f: h5py.File) -> None:
    """
    Tests output catalogue halo membership.
    """

    # check keys exist
    all_keys = [f"{p}_{s}" for p in ALL_PARTICLE_LISTS for s in SUFFIXES] # perhaps a cleaner way to do this
    _check_keys_exist(f=f["halo_data"], keys=all_keys)
    logger.info(f"All keys exist for halos.")

    for plist in ALL_PARTICLE_LISTS:

        _validate_csr_integrity(f=f, group_data="halo_data", particle_list=plist)

    # ensure there are no empty halos
    particles_per_halo = np.sum([f["halo_data"][f"{p}_lengths"][:] for p in ALL_PARTICLE_LISTS], axis=0) 
    assert particles_per_halo.min() > 0, f"Empty halos detected."

    logger.info(f"Halo membership is self-consistent.")

def validate_galaxy_membership(f: h5py.File) -> None:
    """
    Tests output catalogue galaxy membership.
    """
    # check keys exist, same code block as halo function (slightly dubious I know)
    all_keys = [f"{p}_{s}" for p in BARYONIC_PARTICLE_LISTS for s in SUFFIXES] 
    _check_keys_exist(f=f["galaxy_data"], keys=all_keys)
    logger.info(f"All keys exist for galaxies.")

    for plist in BARYONIC_PARTICLE_LISTS:

        _validate_csr_integrity(f=f, group_data="galaxy_data", particle_list=plist)

    # ensure there are no empty galaxies and that particles in galaxies <= particles in halos
    particles_per_halo = np.sum([f["halo_data"][f"{p}_lengths"][:] for p in BARYONIC_PARTICLE_LISTS], axis=0) 
    particles_per_galaxy = np.sum([f["galaxy_data"][f"{p}_lengths"][:] for p in BARYONIC_PARTICLE_LISTS], axis=0) 

    assert particles_per_galaxy.min() > 0, f"Empty galaxies detected."
    assert particles_per_galaxy.sum() <= particles_per_halo.sum(), f"More particles in galaxies than in halos."

    logger.info(f"Galaxy membership is self-consistent.")

def validate_galaxy_mapping(f: h5py.File) -> None:
    """
    Validate galaxy-halo relationships are sensible.
    """
    # check parent halo indices are valid
    parent_halo_indices = f["galaxy_data"]["parent_halo_index"][:]
    n_halos = len(f['halo_data']['HaloID'])
    assert np.all(parent_halo_indices >= 0), f"Invalid parent halo indices."
    assert np.all(parent_halo_indices < n_halos), f"Parent halo index is larger than the number of halos."
    logger.info(f"Parent halo indices are self-consistent.")

    # check the particles have the same halo_id as their host galaxy
    for plist in BARYONIC_PARTICLE_LISTS:

        halo_lengths = f["halo_data"][f"{plist}_lengths"][:]
        halo_indices = f["halo_data"][f"{plist}_indices"][:]
        galaxy_lengths = f["galaxy_data"][f"{plist}_lengths"][:]
        galaxy_indices = f["galaxy_data"][f"{plist}_indices"][:]
        
        # edge case verification for halos that may perhaps genuinely lack a particle list
        if halo_lengths.sum() == 0:
            assert galaxy_lengths.sum() == 0, f"Galaxy {plist} particles exist but no halo {plist} particles."
            continue

        # quick check: the number of particles in galaxies is fewer than the total particles in each halo
        total_galaxy_particles_per_halo = np.bincount(parent_halo_indices,weights=galaxy_lengths, minlength=len(halo_lengths))
        assert np.all(total_galaxy_particles_per_halo <= halo_lengths), f"{plist} particles span multiple galaxies."

        halo_ids = np.repeat(np.arange(len(halo_lengths)), halo_lengths) # unwrapped 
        halo_membership_lookup_array = np.full(fill_value=-1, shape=halo_indices.max()+1, dtype=int) # note: filled with unassigned
        halo_membership_lookup_array[halo_indices] = halo_ids 

        expected_halo_ids = np.repeat(parent_halo_indices, galaxy_lengths)
        actual_halos = halo_membership_lookup_array[galaxy_indices]

        assert np.array_equal(expected_halo_ids, actual_halos), f"{plist} particle halo IDs do not match their galaxy host ID."

    logger.info(f"Particle group membership is self-consistent.")

def validate_group_counts(f: h5py.File, group_data: str) -> None:
    """
    Quick check validating merge_catalogues and CGP agree on group counts.
    """

    if group_data == "halo_data":
        ptypes = PTYPE_TO_PLIST
    elif group_data == "galaxy_data":
        ptypes = {k: v for k, v in PTYPE_TO_PLIST.items() if k != "dm"} # slightly hacky (need to tidy up dm in galaxies)

    for ptype in ptypes:

        n_particles = f[group_data][f"n{ptype}"][:]
        n_particles_csr = f[group_data][f"{PTYPE_TO_PLIST[ptype]}_lengths"][:]
        
        assert np.array_equal(n_particles, n_particles_csr), f"{ptype} total particles disagree between CSR and {group_data}."
        logger.info(f"{ptype} group counts via CSR/total agree.")

def validate_mass_budget(f: h5py.File) -> None:
    """
    Check the mass within galaxies and halos makes sense physically.
    """
    baryonic_mass = {}

    for group in ["halo_data", "galaxy_data"]:

        mass_total = f[group]['dicts/masses.total'][:]
        mass_star = f[group]['dicts/masses.stellar'][:]
        mass_gas = f[group]['dicts/masses.gas'][:]
        mass_bh = f[group]['dicts/masses.bh'][:]
        mass_dm = f[group]['dicts/masses.dm'][:] if group == "halo_data" else np.zeros_like(mass_total) # no dm in galaxies

        baryonic_mass[group] = (mass_star + mass_gas + mass_bh).sum()

        # check no negative masses
        assert np.all(mass_total >= 0), f"{group}: negative total masses"

        # no component exceeds total 
        masses = [mass_star, mass_gas, mass_bh, mass_dm]
        for mass in masses:
            assert np.all(np.where(np.isfinite(mass), mass, 0.0) <= mass_total + mass_total * 1e-9), (
                f"{group}: mass exceeds total"
            )

        # component sum vs total 
        all_masses = np.stack(masses)
        valid = np.all(np.isfinite(all_masses), axis=0)
        n_nan = np.sum(~valid)
        if n_nan > 0:
            logger.info(f"{group} contains unphysical masses.")

        component_sum = mass_star + mass_gas + mass_dm + mass_bh # note: there can be floating point accumulation here

        ratio = component_sum / mass_total
        assert np.all(ratio > 0.99), f"Sum of {group} mass components is significantly less than total."
        assert np.all(ratio <= 1.0 + 1e-6), f"Sum of {group} mass components is significantly more than total."

        # summary
        logger.info(f"{group}: {len(mass_total)} groups")
        logger.info(f"Total Mass {mass_total.sum():.4e}, Stellar Mass {mass_star.sum():.4e}")
        logger.info(f"Gas Mass {mass_gas.sum():.4e}, DM Mass {mass_dm.sum():.4e}, BH Mass {mass_bh.sum():.4e}")
    
    assert baryonic_mass["galaxy_data"] <= baryonic_mass["halo_data"], f"Galaxy baryonic mass exceeds halo baryonic mass."
    logger.info(f"Mass data is self-consistent.")

def check_for_nans(f: h5py.File) -> None:
    """
    Scans the catalogue for any dubious NaN occurences.
    """
    for group in ["halo_data", "galaxy_data"]:

        # datasets which should not have NaN in them
        for dataset in NEVER_NAN:

            if group == "halo_data" and dataset == "dicts/masses.total_30kpc": # HACK: fix this later
                continue

            if group == "galaxy_data" and dataset == "minpotpos" or "minpotvel":
                continue

            assert np.all(np.isfinite(f[group][dataset][:])), f"NaN values detected in {group}/{dataset}"

        # datasets which can have NaN in them in the case where the group is missing a certain particle type
        for lengths, field_keys in CONDITIONAL_NAN[group]: # index into group because halo contains dm

            particles_per_group = f[group][lengths][:] # uses the csr lengths array (n_particles per group)
            has_particles = particles_per_group > 0

            for key in field_keys:
                
                dataset = f[group][key][:]
                assert np.all(np.isfinite(dataset[has_particles])), f"{group}/{key} contains unphysical values."
                assert np.all(np.isnan(dataset[~has_particles])), f"{group}/{key} contains a group with no membership but defined physical values."

        # datasets which should only be 0 if they are empty (e.g. particle mass = 0 if no particles)
        for lengths_key, field_keys in ZERO_WHEN_EMPTY[group]:

            particles_per_group = f[group][lengths_key][:]
            empty = particles_per_group == 0

            for key in field_keys:

                dataset = f[group][key][:]
                assert np.all(np.isfinite(dataset)), f"{group}/{key} should be 0 but contains NaN"
                assert np.all(dataset[empty] == 0.0), f"{group}/{key} is nonzero for empty groups"

        # same as above but for baryonic quantities
        baryonic_particles_per_group = np.sum([f[group][f"{p}list_lengths"][:] for p in ["g", "s", "bh"]], axis=0)
        has_baryonic_particles = baryonic_particles_per_group > 0

        for key in BARYON_CONDITIONAL_NAN:

            dataset = f[group][key][:]
            assert np.all(np.isfinite(dataset[has_baryonic_particles])), f"{group}/{key} contains unphysical values."
            assert np.all(np.isnan(dataset[~has_baryonic_particles])), f"{group}/{key} contains a group with no membership but defined physical values."

        # datasets which can have NaN in them generally (but a high proportion is suspect)
        for key in SOFT_NAN:

            dataset = f[group][key][:]

            if (np.sum(np.isnan(dataset)) /  dataset.size) > 0.5:

                logger.warning(f"{group}/{key} is over 50% NaN.")
            
        logger.info(f"{group} contains no dubious NaN occurences.")

@contextmanager
def record_assertion_result(label: str) -> Generator[None, None, None]: # the collections import is used for this
    """
    Used for wrapping the validation checks in a try / except.
    Prevents an error being thrown.
    """
    try:
        yield
        results.append((label, True, ""))
    except AssertionError as e:
        results.append((label, False, str(e)))
        logger.error(f"{label} FAIL: {e}")

def conduct_output_catalogue_validation(catalogue: str) -> None:
    """
    Wraps all the catalogue validation functions.
    """
    with h5py.File(catalogue, 'r') as f:

        with record_assertion_result(f"Halo Membership"):
            validate_halo_membership(f=f)

        with record_assertion_result(f"Galaxy Membership"):
            validate_galaxy_membership(f=f)

        with record_assertion_result(f"Galaxy-Halo Mapping"):
            validate_galaxy_mapping(f=f)

        with record_assertion_result(f"Halo Particle Counts"):
            validate_group_counts(f=f, group_data="halo_data")

        with record_assertion_result(f"Galaxy Particle Counts"):
            validate_group_counts(f=f, group_data="galaxy_data")

        with record_assertion_result(f"NaN Checking"):
            check_for_nans(f=f) 

        with record_assertion_result(f"Mass Budget Validation"):
            validate_mass_budget(f=f)

def record_test_results(all_timings: list[dict[str, float]], all_memories: list[dict[str, float]],
                        results: list[tuple[str, bool, str]], peak_memory: list[float]):
    """
    Checks the validation outputs and writes the result to a .txt file.
    """
    COMMIT_HASH = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() # current version
    filepath = test_config.working_directory / f"test_summary_{COMMIT_HASH[:8]}.txt"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    passed = all(success for _, success, _ in results)

    logger.info(f"Final Result: {'PASS' if passed else 'FAIL'}")

    with open(filepath, 'w') as f:

        # summary (what you care about)
        f.write(f"Octavian Test Summary: Commit {COMMIT_HASH[:8]} // {timestamp} \n")
        f.write(f"Snapshot: {test_config.test_snapshot} \n")
        f.write(f"{test_config.n_ranks} ranks // {test_config.n_proc} cores per rank\n")
        f.write(f"Final Result: {'PASS' if passed else 'FAIL'}\n\n")

        # stage time/memory breakdown
        if test_config.n_ranks == 1: # serial
            for stage, elapsed in all_timings[0].items():
                mem = all_memories[0].get(stage, 0.0)
                f.write(f"{stage}: {elapsed:.2f}s, {mem:.3f} GB\n")
        else: # parallel
            stages = all_timings[0].keys()
            for stage in stages:
                vals = [t[stage] for t in all_timings if stage in t] # leave in conditional in case we want to test stages
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
        f.write("Peak RSS (High Watermark):\n")
        for i, rss in enumerate(peak_memory):
            f.write(f"Rank {i}: {rss:.2f} GB\n")
        if test_config.n_ranks > 1:
            f.write(f"Max: {max(peak_memory):.2f} GB\n")

def test_full_serial_run() -> None:
    """
    Serial test case.
    """
    memray_file = Path(f"/home/jpduminy/Octavian/Intermediates/memray.bin")
    memray_file.unlink(missing_ok=True)

    with memray.Tracker(memray_file, native_traces=True):

        logging.basicConfig(
            level=logging.INFO,
            format=f"[%(levelname)s] %(name)s — %(message)s",
        )

        logger.info(f"Testing Octavian in serial configuration.")
        test_filter_snapshot()
        logger.info("Filtering complete.")

        snapshot_file = f"{test_config.working_directory}_rank_0.hdf5"
        intermediate_file = f"{test_config.working_directory}_rank_0_analysis.hdf5"
        _end_to_end_pipeline(snapshot_file=snapshot_file, output_file=intermediate_file, comm=None)

        output_catalogue = f"{test_config.working_directory}_output_catalogue.hdf5"

        test_remerge(files=[intermediate_file], outfile=output_catalogue, configfile=test_config.config_file)
        conduct_output_catalogue_validation(catalogue=output_catalogue)

        for stage, elapsed in timings.items():
            logger.info(f"{stage} = {elapsed:.2f}s")

        # wrap function arguments in lists (it expects lists as it is parallelised)
        record_test_results(all_timings=[timings], all_memories=[memories], results=results, peak_memory=[peak_rss_gb()])

def test_full_parallel_run(comm: MPI.Comm) -> None:
    """
    Conduct a full parallel run of Octavian.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    memray_file = Path(f"/home/jpduminy/Octavian/Intermediates/memray_rank_{rank}.bin")
    memray_file.unlink(missing_ok=True)

    with memray.Tracker(memray_file, native_traces=True):

        logging.basicConfig(
            level=logging.INFO,
            format=f"[Rank {rank}] [%(levelname)s] %(name)s — %(message)s",
        )

        if rank == 0:
            logger.info(f"Testing Octavian with {size} ranks.")
            test_filter_snapshot()
            logger.info("Filtering complete.")

        comm.Barrier()

        snapshot_file = f"{test_config.working_directory}_rank_{rank}.hdf5"
        intermediate_file = f"{test_config.working_directory}_rank_{rank}_intermediate_analysis.hdf5"
        _end_to_end_pipeline(snapshot_file=snapshot_file, output_file=intermediate_file, comm=comm)

        comm.Barrier()

        output_catalogue = f"{test_config.working_directory}_output_catalogue.hdf5"

        if rank == 0:

            files = [f"{test_config.working_directory}_rank_{i}_intermediate_analysis.hdf5" for i in range(size)]
            test_remerge(files=files, outfile=output_catalogue, configfile=test_config.config_file)
            conduct_output_catalogue_validation(catalogue=output_catalogue)

        all_timings = comm.gather(timings, root=0)
        all_memories = comm.gather(memories, root=0)
        all_rss = comm.gather(peak_rss_gb(), root=0)

        if rank == 0:
            # per-rank timings
            for i, t in enumerate(all_timings):
                for stage, elapsed in t.items():
                    logger.info(f"Rank {i}: {stage} = {elapsed:.2f}s")

            # imbalances across ranks
            stages = all_timings[0].keys()
            for stage in stages:
                vals = [t[stage] for t in all_timings if stage in t]
                logger.info(f"{stage}: max={max(vals):.2f}s  spread={max(vals)-min(vals):.2f}")

            record_test_results(all_timings=all_timings, all_memories=all_memories, results=results, peak_memory=all_rss)

def main():

    comm = _get_mpi_communicator()

    if comm is not None:
        test_full_parallel_run(comm)
    else:
        test_full_serial_run()

if __name__ == "__main__":
    main()