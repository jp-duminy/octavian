"""

Running Octavius end-to-end in a Python script with no command line, YAML file, or MPI.

"""

from pathlib import Path
import octavius as oc

snap_path = Path("/path/to/snapshot.hdf5")
output_dir = Path("/path/to/output_dir/")

# generate a photometry table
photometry_table_path = Path("/path/to/photometry_table.hdf5")
oc.generate_photometry_table(output_path=photometry_table_path)

# instantiate a config dataclass
config = oc.OctaviusConfig(
    snapshot_path=snap_path,
    output_dir=output_dir,
    simulation_type="SWIFT",
    halo_id_source="SNAPSHOT",
    cores_per_rank=-1,  # if not using MPI, set to -1 to use all available cores
    photometry_table_path=photometry_table_path,
)

# analyse snapshot
catalogue_path = oc.analyse_snapshot(config=config)

# load output catalogue
cat = oc.load_catalogue(catalogue_path=catalogue_path)

# access datasets
absolute_v_mag = cat.galaxies.get_dataset("mag_abs_v")
