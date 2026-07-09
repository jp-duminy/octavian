"""

Tests whether the catalogue output is self-consistent and physical. This is to catch corrupted/obviously incorrect fields; to check whether the numbers are reasonable and behave as expected, please refer to validation/validation_suite.py which has methods for inspecting the fields and contrasting with a reference catalogue. 

"""

import h5py

from tests.validation.output_validation import (
    check_for_nans, 
    validate_galaxy_mapping, 
    validate_galaxy_membership, 
    validate_group_counts, 
    validate_halo_membership, 
    validate_mass_budget,
)

def test_nans(mock_catalogue: h5py.File) -> None:
    """
    Verifies NaN fields specified in validation/validation_columns.py
    """
    check_for_nans(f=mock_catalogue)

def test_galaxies(mock_catalogue: h5py.File) -> None:
    """
    Verifies galaxy-halo mapping and galaxy membership consistency.
    """
    validate_galaxy_mapping(f=mock_catalogue)
    validate_galaxy_membership(f=mock_catalogue)
    validate_group_counts(f=mock_catalogue, group_data="galaxy_data")

def test_halos(mock_catalogue: h5py.File) -> None:
    """
    Verifies halo membership consistency.
    """
    validate_halo_membership(f=mock_catalogue)
    validate_group_counts(f=mock_catalogue, group_data="halo_data")

def test_masses(mock_catalogue: h5py.File) -> None:
    """
    Verifies individual & combined halo/galaxy mass budgets.
    """    
    validate_mass_budget(f=mock_catalogue)
    