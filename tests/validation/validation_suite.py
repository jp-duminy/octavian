"""

The validation suite, which allows Octavius to be profiled in memory on larger snapshots, as well as ensure catalogues
are self-consistent and physically sensible; can optionally perform regression testing against reference catalogues.

test_snapshot_small: 600MB
test_snapshot_large: 4GB

"""

# default libraries
from pathlib import Path
from contextlib import contextmanager
import subprocess
from collections.abc import Generator
import argparse
from dataclasses import replace

# other packages
import memray
import h5py
import numpy as np

# internal imports
from octavius.run_octavius import analyse_snapshot, get_mpi_communicator
from octavius.data_management.conventions import OctaviusConfig
from octavius.log import get_logger
from .output_validation import (
    validate_galaxy_mapping,
    validate_subhalo_mapping,
    validate_galaxy_membership,
    validate_halo_membership,
    validate_mass_budget,
    check_for_nans,
)
from octavius.utils import build_analyser, load_catalogue

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
INTERNALS_PATH = Path(__file__).parent.parent.parent / "octavius" / "internals.yaml"
SEED = 2317434
RESULTS = []


def validation_run() -> None:
    """
    Wraps analyse_snapshot() in memray; conducts output catalogue validation; conducts regression tests. Automatically generates memray flamegraphs.
    """
    args = parse_args()
    comm = get_mpi_communicator()
    rank = comm.rank if comm is not None else 0
    config_path = args.config if args.config else CONFIG_PATH

    config = OctaviusConfig.from_yaml(config_path=config_path)
    config = replace(config, terminal_output_level="DEBUG")  # automatically set to DEBUG level for validation runs
    if args.snapshot:
        config = replace(config, snapshot_path=args.snapshot)
    if args.output_dir:
        config = replace(config, output_dir=args.output_dir)
    output_dir = config.output_dir
    memray_file: Path = output_dir / f"memray_rank_{rank}.bin"
    memray_file.unlink(missing_ok=True)

    with memray.Tracker(
        memray_file, native_traces=True
    ):  # python allocations aren't worth it since it can't see what numba does
        catalogue_path = analyse_snapshot(config)

    if rank == 0:  # can take a while on large snapshots
        conduct_output_catalogue_validation(catalogue=catalogue_path)

        if args.reference:
            validate_against_reference(
                catalogue=catalogue_path,
                reference=args.reference,
                rtol=args.rtol,
                atol=args.atol,
            )

        if args.standalone:
            validate_standalone_analysis(
                snapshot_path=config.snapshot_path,
                catalogue=catalogue_path,
                config=config,
            )

    logger = get_logger()

    passed = all(success for _, success, _ in RESULTS)
    logger.info(f"Final Result: {'PASS' if passed else 'FAIL'}")

    logger.info("Generating memray flamegraphs.")
    # auto generate the flamegraphs (can take several minutes on large snaps)
    flamegraph_path = memray_file.with_suffix(".html")
    flamegraph_path.unlink(missing_ok=True)
    subprocess.run(
        ["memray", "flamegraph", str(memray_file), "-o", str(flamegraph_path)],
        check=True,
    )


def validate_against_reference(catalogue: Path, reference: Path, rtol: float = 1e-6, atol: float = 1e-10) -> None:
    """
    Validates output vs reference more rigorously, comparing specific fields within tolerance.

    Also validates whether NaNs occur in the same place.
    """
    logger = get_logger()

    with h5py.File(catalogue, "r") as new, h5py.File(reference, "r") as ref:
        for group_name in ["halo_data", "galaxy_data"]:
            if group_name not in ref:
                logger.warning(f"{group_name} is not in the reference catalogue, so will not be validated.")
                continue
            if group_name not in new:
                logger.warning(f"{group_name} is not in the new catalogue, so will not be validated.")
                continue

            ref_group = ref[group_name]
            new_group = new[group_name]

            # set notation simplifies this process
            ref_keys = set()
            new_keys = set()

            # use visit() in h5py to traverse the datasets within groups
            ref_group.visit(
                lambda k: ref_keys.add(k) if isinstance(ref_group[k], h5py.Dataset) else None
            )  # lambda function to only select datasets

            new_group.visit(lambda k: new_keys.add(k) if isinstance(new_group[k], h5py.Dataset) else None)

            only_ref = ref_keys - new_keys
            only_new = new_keys - ref_keys

            # check whether the same datasets appear in both
            if only_ref:
                logger.warning(
                    f"{group_name}: datasets present in reference catalogue but not in new: {sorted(only_ref)}"
                )
            if only_new:
                logger.warning(
                    f"{group_name}: datasets present in new catalogue but not in reference: {sorted(only_new)}"
                )

            logger.info(f"{group_name}: same datasets appear in both reference and new catalogues.")

            for key in sorted(ref_keys & new_keys):
                with record_assertion_result(f"{group_name}/{key}"):
                    if isinstance(ref_group[key], h5py.Group):  # ignore groups
                        continue

                    ref_data = ref_group[key][:]
                    new_data = new_group[key][:]

                    assert ref_data.shape == new_data.shape, (
                        f"{group_name}/{key}: array shapes are mismatched (reference: {ref_data.shape}, new: {new_data.shape})"
                    )

                    # if dataset is integer type, assert exactly equal
                    if not np.issubdtype(ref_data.dtype, np.floating):
                        mismatches = ref_data != new_data
                        n_mismatches = np.sum(mismatches)
                        assert n_mismatches == 0, (
                            f"{group_name}/{key}: {n_mismatches}/{len(ref_data)} integer values differ"
                        )
                        continue  # following checks are for floats, not integers

                    # check any existing NaN columns have the NaNs in the same place (groups should remain empty, analysis is deterministic)
                    ref_finite = np.isfinite(ref_data)
                    new_finite = np.isfinite(new_data)

                    if not np.any(ref_finite):
                        logger.warning(f"{group_name}/{key}: no finite values in reference catalogue.")

                    if not np.any(new_finite):
                        logger.warning(f"{group_name}/{key}: no finite values in new catalogue.")

                    nan_mismatch = np.sum(ref_finite != new_finite)
                    assert nan_mismatch == 0, (
                        f"{group_name}/{key}: {nan_mismatch} NaN values occur in different elements of reference and new catalogue arrays."
                        f"(reference has {np.sum(~ref_finite)} NaN, new has {np.sum(~new_finite)} NaN)"
                    )

                    both_finite = ref_finite & new_finite

                    if not np.any(both_finite):
                        logger.warning(f"{group_name}/{key}: no finite values in either catalogue.")
                        continue

                    np.testing.assert_allclose(
                        ref_data[both_finite],
                        new_data[both_finite],
                        rtol=rtol,
                        atol=atol,
                        err_msg=f"{group_name}/{key} does not match to within the specified tolerances.",
                    )

            # check whether the group failed by inspecting the RESULTS list for its labels
            group_failures = [label for label, success, _ in RESULTS if label.startswith(group_name) and not success]
            if group_failures:
                logger.warning(f"{group_name}: {len(group_failures)} dataset(s) failed validation.")
            else:
                logger.info(f"{group_name}: all datasets match reference to within tolerance.")

    passed = all(success for _, success, _ in RESULTS)
    if passed:
        logger.info("Reference catalogue comparison passes.")
    else:
        logger.warning("Reference catalogue comparison fails tolerance; see logs please.")


def validate_standalone_analysis(
    snapshot_path: Path, catalogue: Path, config: OctaviusConfig, rtol: float = 1e-6, atol: float = 1e-10
) -> None:
    """
    Validates the on-the-fly analyser returns the same result as the pipeline.
    """
    cat = load_catalogue(catalogue_path=catalogue)
    analyser = build_analyser(snapshot_path=snapshot_path, catalogue=cat, config=config)
    rng = np.random.default_rng(seed=SEED)

    random_gal_idx = np.sort(rng.choice(cat.n_galaxies, size=5, replace=False))
    random_halo_idx = np.sort(rng.choice(cat.n_haloes, size=5, replace=False))

    core_datasets = ["velocity_dispersion_baryon", "radius_half_mass_star", "inertia_tensor_gas"]
    particle_datasets = ["mass_HI", "metallicity_gas_sfr_weighted", "sfr"]
    phot_datasets = ["mag_app_v", "luminosity_fir", "beta_nodust"]

    stage_checks = [
        (analyser.compute_core_properties, "galaxies", random_gal_idx, cat.galaxies, core_datasets + ["BoverT_baryon"]),
        (
            analyser.compute_core_properties,
            "haloes",
            random_halo_idx,
            cat.haloes,
            core_datasets + ["temperature_virial"],
        ),
        (analyser.compute_ptype_specific_properties, "galaxies", random_gal_idx, cat.galaxies, particle_datasets),
        (
            analyser.compute_ptype_specific_properties,
            "haloes",
            random_halo_idx,
            cat.haloes,
            particle_datasets + ["temperature_gas_mass_weighted_cgm"],
        ),
        (analyser.compute_photometry, "galaxies", random_gal_idx, cat.galaxies, phot_datasets),
    ]

    for stage_fn, group_type, group_indices, collection, datasets in stage_checks:
        if stage_fn == analyser.compute_photometry:
            result = stage_fn(group_indices=group_indices)
        else:
            result = stage_fn(group_indices=group_indices, group_type=group_type)

        for dataset in datasets:
            with record_assertion_result(f"Standalone analysis: {group_type}/{dataset}"):
                pipeline_values = collection.get_dataset(dataset, mask=group_indices)
                standalone_values = result[dataset]
                np.testing.assert_allclose(
                    pipeline_values,
                    standalone_values,
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"{dataset} failed for {group_type}.",
                )

    with record_assertion_result(label="Photometry rotation checks"):
        # rotation check for photometry
        face_on = analyser.compute_photometry(group_indices=random_gal_idx, orientation="face-on")
        edge_on = analyser.compute_photometry(group_indices=random_gal_idx, orientation="side-on")

        assert not np.allclose(face_on["mag_app_v"], edge_on["mag_app_v"]), "Rotated photometry has no effect."


@contextmanager
def record_assertion_result(label: str) -> Generator[None, None, None]:  # the collections import is used for this
    """
    Used for wrapping the validation checks in a try / except so an error isn't thrown.

    Appends the results to the RESULTS list.
    """
    logger = get_logger()
    try:
        yield
        RESULTS.append((label, True, ""))
    except AssertionError as e:
        RESULTS.append((label, False, str(e)))
        logger.error(f"{label} FAIL: {e}")


def conduct_output_catalogue_validation(catalogue: Path) -> None:
    """
    Wraps all the catalogue validation functions.
    """
    with h5py.File(catalogue, "r") as f:
        with record_assertion_result("Halo membership arrays"):
            validate_halo_membership(f=f)

        with record_assertion_result("Galaxy membership arrays"):
            validate_galaxy_membership(f=f)

        with record_assertion_result("Galaxy-halo mapping"):
            validate_galaxy_mapping(f=f)

        with record_assertion_result("Subhalo-halo mapping"):
            validate_subhalo_mapping(f=f)

        with record_assertion_result("Dataset NaN validation"):
            check_for_nans(f=f)

        with record_assertion_result("Mass budget validation"):
            validate_mass_budget(f=f)


def parse_args() -> argparse.Namespace:
    """
    Parses the validation suite command line arguments, which are slightly different to those used by analyse_snapshot().
    """
    parser = argparse.ArgumentParser(description="Octavius validation suite")
    parser.add_argument("-s", "--snapshot", type=Path, required=False, default=None, help="Path to the snapshot file.")
    parser.add_argument("-c", "--config", type=Path, help="Path to a config YAML file.")
    parser.add_argument(
        "-r",
        "--reference",
        type=Path,
        required=False,
        default=None,
        help="Path to reference catalogue; should be run on the same snapshot)",
    )
    parser.add_argument("-o", "--output-dir", type=Path, required=False, default=None, help="Output directory.")
    parser.add_argument(
        "--rtol",
        required=False,
        type=float,
        default=1e-6,
        help="Relative tolerance between reference and output catalogue datasets.",
    )
    parser.add_argument(
        "--atol",
        required=False,
        type=float,
        default=1e-10,
        help="Absolute tolerance between reference and output catalogue datasets.",
    )
    parser.add_argument(
        "--standalone",
        required=False,
        action="store_true",
        help="Check whether standalone stages reproduce pipeline results.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    validation_run()
