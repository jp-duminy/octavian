"""

Tests whether the pipeline produces the expected top-level fields.

"""

import h5py

def test_pipeline(mock_catalogue: h5py.File) -> None:
    """
    Verifies the pipeline produces output.
    """
    assert "halo_data" in mock_catalogue
    assert "membership" in mock_catalogue["halo_data"]
    assert "properties" in mock_catalogue["halo_data"]
    assert "core" in mock_catalogue["halo_data/properties"]
    assert "particle_specific" in mock_catalogue["halo_data/properties"]

    assert "galaxy_data" in mock_catalogue
    assert "membership" in mock_catalogue["galaxy_data"]
    assert "properties" in mock_catalogue["galaxy_data"]
    assert "core" in mock_catalogue["galaxy_data/properties"]
    assert "particle_specific" in mock_catalogue["galaxy_data/properties"]
    assert "environment" in mock_catalogue["galaxy_data/properties"]