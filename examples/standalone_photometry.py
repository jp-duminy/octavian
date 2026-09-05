"""

Post-catalogue standalone photometry with rotations and extinction laws.

"""

from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np
import octavius as oc

catalogue_path = Path("/path/to/catalogue.hdf5")

# instantiate necessary data structures
config = oc.OctaviusConfig(...)  # type parameters or load from YAML
catalogue = oc.load_catalogue(catalogue_path)

# get galaxies of most massive halo
halo_masses = catalogue.haloes.get_dataset("mass_total")
most_massive_halo = np.argmax(halo_masses)  # get catalogue index of most massive halo
gal_indices = catalogue.haloes.get_galaxies(halo_index=most_massive_halo)

# build analyser and re-run photometry on those galaxies
analyser = oc.build_analyser(catalogue=catalogue, config=config)
face_on_result = analyser.compute_photometry(group_indices=gal_indices, orientation="face-on", keep_spectra=True)

wavelengths = face_on_result["wavelengths"]  # in angstrom
face_on_spectra = face_on_result["spectra"]  # in Lsun / Hz

# change to use cardelli
analyser.update_config(extinction_law="CARDELLI")  # change extinction law
face_on_result_cardelli = analyser.compute_photometry(
    group_indices=gal_indices, orientation="face-on", keep_spectra=True
)
face_on_spectra_cardelli = face_on_result_cardelli["spectra"]

# rotate the galaxies 90 degrees about the x-axis
analyser.update_config(extinction_law="COMPOSITE")  # default law
rotation_matrix = Rotation.from_rotvec(np.pi / 2 * np.array([1, 0, 0])).as_matrix()

rotated_result = analyser.compute_photometry(group_indices=gal_indices, orientation=rotation_matrix, keep_spectra=True)
rotated_spectra = rotated_result["spectra"]
