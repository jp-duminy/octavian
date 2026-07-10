"""

Verifies whether the output catalogue is self-consistentn and does not contain dubious values.

"""

import h5py
import numpy as np

# data
from tests.validation.validation_columns import NEVER_NAN, CONDITIONAL_NAN, ZERO_WHEN_EMPTY, SOFT_NAN
from octavian.log import get_logger

PTYPES = ["gas", "star", "bh", "dm"]
BARYON_PTYPES = ["gas", "star", "bh"]
SUFFIXES = ["lengths", "offsets", "indices"]  # for csr indexing


def _validate_csr_integrity(f: h5py.File, group_data: str, particle_list: str) -> None:
    """
    Validates CSR format of particle lists (sanity checks); handles empty groups too.
    """
    logger = get_logger()
    lengths = f[group_data][f"membership/{particle_list}_lengths"][:]
    offsets = f[group_data][f"membership/{particle_list}_offsets"][:]
    indices = f[group_data][f"membership/{particle_list}_indices"][:]

    # check offset slicing matches
    expected = np.concatenate([[0], np.cumsum(lengths[:-1])])  # mainly want to verify the prepended 0 is there
    assert np.array_equal(offsets, expected), (
        f"{group_data}/{particle_list} offset array does not match expected format."
    )
    logger.info(f"{group_data}/{particle_list} offset slicing matches.")

    # total count consistency
    assert lengths.sum() == len(indices), (
        f"{group_data}/{particle_list} number of particles does not match number of indices."
    )
    logger.info(f"{group_data}/{particle_list} particle counts are self-consistent.")

    # no duplicates within groups (sometimes different ptypes have the same ID though)
    group_labels = np.repeat(np.arange(len(lengths)), lengths)  # full-length array
    order = np.lexsort((indices, group_labels))
    sorted_idx = indices[order]
    sorted_grp = group_labels[order]

    same_group = sorted_grp[1:] == sorted_grp[:-1]  # false at group boundary (only works because of lexsort ^)
    same_value = sorted_idx[1:] == sorted_idx[:-1]
    dupes = np.where(same_group & same_value)[0]

    assert len(dupes) == 0, f"{group_data}/{particle_list}: {len(dupes)} duplicate particles in groups."
    logger.info(f"{group_data}/{particle_list} has no intra-group duplicate particles.")

    # no particle appears in two groups
    if len(indices) > 0:
        assert np.all(indices >= 0), f"Invalid indices in {group_data}/{particle_list}."
        assert len(np.unique(indices)) == len(indices), (
            f"In {group_data}/{particle_list}, same particle appears in multiple groups."
        )

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
    logger = get_logger()
    # check keys exist
    all_keys = [f"membership/{p}_{s}" for p in PTYPES for s in SUFFIXES]  # perhaps a cleaner way to do this
    _check_keys_exist(f=f["halo_data"], keys=all_keys)
    logger.info("All keys exist for halos.")

    for ptype in PTYPES:
        _validate_csr_integrity(f=f, group_data="halo_data", particle_list=ptype)

    # ensure there are no empty halos
    particles_per_halo = np.sum([f["halo_data"][f"membership/{p}_lengths"][:] for p in PTYPES], axis=0)
    assert particles_per_halo.min() > 0, "Empty halos detected."

    logger.info("Halo membership is self-consistent.")


def validate_galaxy_membership(f: h5py.File) -> None:
    """
    Tests output catalogue galaxy membership.
    """
    logger = get_logger()
    # check keys exist, same code block as halo function (slightly dubious I know)
    all_keys = [f"membership/{p}_{s}" for p in BARYON_PTYPES for s in SUFFIXES]
    _check_keys_exist(f=f["galaxy_data"], keys=all_keys)
    logger.info("All keys exist for galaxies.")

    for ptype in BARYON_PTYPES:
        _validate_csr_integrity(f=f, group_data="galaxy_data", particle_list=ptype)

    # ensure there are no empty galaxies and that particles in galaxies <= particles in halos
    particles_per_halo = np.sum([f["halo_data"][f"membership/{p}_lengths"][:] for p in BARYON_PTYPES], axis=0)
    particles_per_galaxy = np.sum([f["galaxy_data"][f"membership/{p}_lengths"][:] for p in BARYON_PTYPES], axis=0)

    assert particles_per_galaxy.min() > 0, "Empty galaxies detected."
    assert particles_per_galaxy.sum() <= particles_per_halo.sum(), "More particles in galaxies than in halos."

    logger.info("Galaxy membership is self-consistent.")


def validate_galaxy_mapping(f: h5py.File) -> None:
    """
    Validate galaxy-halo relationships are sensible.
    """
    logger = get_logger()
    # check parent halo indices are valid
    parent_halo_indices = f["galaxy_data"]["properties/core/parent_halo_index"][:]
    n_halos = len(f["halo_data"]["HaloID"])
    assert np.all(parent_halo_indices >= 0), "Invalid parent halo indices."
    assert np.all(parent_halo_indices < n_halos), "Parent halo index is larger than the number of halos."
    logger.info("Parent halo indices are self-consistent.")

    # check the particles have the same halo_id as their host galaxy
    for ptype in BARYON_PTYPES:
        halo_lengths = f["halo_data"][f"membership/{ptype}_lengths"][:]
        halo_indices = f["halo_data"][f"membership/{ptype}_indices"][:]
        galaxy_lengths = f["galaxy_data"][f"membership/{ptype}_lengths"][:]
        galaxy_indices = f["galaxy_data"][f"membership/{ptype}_indices"][:]

        # edge case verification for halos that may perhaps genuinely lack a particle list
        if halo_lengths.sum() == 0:
            assert galaxy_lengths.sum() == 0, f"Galaxy {ptype} particles exist but no halo {ptype} particles."
            continue

        # quick check: the number of particles in galaxies is fewer than the total particles in each halo
        total_galaxy_particles_per_halo = np.bincount(
            parent_halo_indices, weights=galaxy_lengths, minlength=len(halo_lengths)
        )
        assert np.all(total_galaxy_particles_per_halo <= halo_lengths), f"{ptype} particles span multiple galaxies."

        halo_ids = np.repeat(np.arange(len(halo_lengths)), halo_lengths)  # unwrapped
        halo_membership_lookup_array = np.full(
            fill_value=-1, shape=halo_indices.max() + 1, dtype=np.int64
        )  # note: filled with unassigned
        halo_membership_lookup_array[halo_indices] = halo_ids

        expected_halo_ids = np.repeat(parent_halo_indices, galaxy_lengths)
        actual_halos = halo_membership_lookup_array[galaxy_indices]

        assert np.array_equal(expected_halo_ids, actual_halos), (
            f"{ptype} particle halo IDs do not match their galaxy host ID."
        )

    logger.info("Particle group membership is self-consistent.")


def validate_group_counts(f: h5py.File, group_data: str) -> None:
    """
    Quick check validating merge_catalogues and CGP agree on group counts.
    """
    logger = get_logger()

    if group_data == "halo_data":
        ptypes = PTYPES
    elif group_data == "galaxy_data":
        ptypes = BARYON_PTYPES

    for ptype in ptypes:
        n_particles = f[group_data][f"properties/core/n_{ptype}"][:]
        n_particles_csr = f[group_data][f"membership/{ptype}_lengths"][:]

        assert np.array_equal(n_particles, n_particles_csr), (
            f"{ptype} total particles disagree between CSR and {group_data}."
        )
        logger.info(f"{ptype} group counts via CSR/total agree.")


def validate_mass_budget(f: h5py.File) -> None:
    """
    Check the mass within galaxies and halos makes sense physically.
    """
    logger = get_logger()

    baryonic_mass = {}

    for group in ["halo_data", "galaxy_data"]:
        mass_total = (
            f[group]["properties/core/mass_total"][:]
            if group == "halo_data"
            else f[group]["properties/core/mass_baryon"][:]
        )
        mass_star = f[group]["properties/core/mass_star"][:]
        mass_gas = f[group]["properties/core/mass_gas"][:]
        mass_bh = f[group]["properties/core/mass_bh"][:]
        mass_dm = f[group]["properties/core/mass_dm"][:] if group == "halo_data" else np.zeros_like(mass_total)

        baryonic_mass[group] = (mass_star + mass_gas + mass_bh).sum()

        # check no negative masses
        assert np.all(mass_total > 0), f"{group}: negative/zero total masses"

        # no component exceeds total
        masses = [mass_star, mass_gas, mass_bh, mass_dm]
        for mass in masses:
            assert np.all(np.where(np.isfinite(mass), mass, 0.0) <= mass_total + mass_total * 1e-9), (
                f"{group}: mass exceeds total"
            )

        component_sum = mass_star + mass_gas + mass_dm + mass_bh  # NOTE: there can be floating point accumulation here

        ratio = component_sum / mass_total
        assert np.all(ratio > 0.99), f"Sum of {group} mass components is significantly less than total."
        assert np.all(ratio <= 1.0 + 1e-6), f"Sum of {group} mass components is significantly more than total."

        # summary
        logger.info(f"{group}: {len(mass_total)} groups")
        logger.info(f"Total Mass {mass_total.sum():.4e}, Stellar Mass {mass_star.sum():.4e}")
        logger.info(f"Gas Mass {mass_gas.sum():.4e}, DM Mass {mass_dm.sum():.4e}, BH Mass {mass_bh.sum():.4e}")

    assert baryonic_mass["galaxy_data"] <= baryonic_mass["halo_data"], (
        "Galaxy baryonic mass exceeds halo baryonic mass."
    )
    logger.info("Mass data is self-consistent.")


def check_for_nans(f: h5py.File) -> None:
    """
    Scans the catalogue for any dubious NaN occurences.
    """
    logger = get_logger()

    for group in ["halo_data", "galaxy_data"]:
        # datasets which should not have NaN in them
        for dataset in NEVER_NAN[group]:
            assert np.all(np.isfinite(f[group][dataset][:])), f"NaN values detected in {group}/{dataset}"

        csr_keys_total = [
            "membership/gas_lengths",
            "membership/star_lengths",
            "membership/dm_lengths",
            "membership/bh_lengths",
        ]
        csr_keys_baryon = ["membership/gas_lengths", "membership/star_lengths", "membership/bh_lengths"]
        lengths = {k: f[group][k][:] for k in csr_keys_total if k in f[group]}
        lengths["n_total"] = sum(lengths[k] for k in csr_keys_total if k in lengths)
        lengths["n_baryon"] = sum(lengths[k] for k in csr_keys_baryon if k in lengths)

        # datasets which can have NaN in them in the case where the group is missing a certain particle type
        for lengths_key, field_keys in CONDITIONAL_NAN[group]:
            particles_per_group = lengths[lengths_key]
            has_particles = particles_per_group > 0

            for key in field_keys:
                dataset = f[group][key][:]
                assert np.all(np.isfinite(dataset[has_particles])), f"{group}/{key} contains unphysical values."
                assert np.all(np.isnan(dataset[~has_particles])), (
                    f"{group}/{key} contains a group with no membership but defined physical values."
                )

        # datasets which should only be 0 if they are empty (e.g. particle mass = 0 if no particles)
        for lengths_key, field_keys in ZERO_WHEN_EMPTY[group]:
            particles_per_group = f[group][lengths_key][:]
            empty = particles_per_group == 0

            for key in field_keys:
                dataset = f[group][key][:]
                assert np.all(np.isfinite(dataset)), f"{group}/{key} should be 0 but contains NaN"
                assert np.all(dataset[empty] == 0.0), f"{group}/{key} is nonzero for empty groups"

        # datasets which can have NaN in them generally (but a high proportion is suspect)
        for key in SOFT_NAN:
            if key not in f[group]:
                continue
            dataset = f[group][key][:]

            if (np.sum(np.isnan(dataset)) / dataset.size) > 0.5:
                logger.warning(f"{group}/{key} is over 50% NaN.")

        logger.info(f"{group} contains no dubious NaN occurences.")
