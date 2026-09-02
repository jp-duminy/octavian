"""

Post-catalogue analysis of halo gass mass above temperature thresholds.

"""

from pathlib import Path
import numpy as np
import octavius as oc


config = oc.OctaviusConfig.from_yaml(Path("/path/to/config.yaml"))
catalogue_path = oc.analyse_snapshot(config)

# load the catalogue and check hot gas mass
catalogue = oc.load_catalogue(catalogue_path)
hot_gas_mass = catalogue.haloes.get_dataset("mass_gas_hot")

# now recompute with a different T_lim for massive haloes
halo_masses = catalogue.haloes.get_dataset("mass_total")
massive_haloes = np.where(halo_masses > 1e13)[0]

analyser = oc.build_analyser(catalogue=catalogue, config=config)
analyser.update_config(T_lim=1e6)  # raise threshold from default

result = analyser.compute_ptype_specific_properties(group_indices=massive_haloes, group_type="haloes")
hot_gas_mass_new_tlim = result["mass_gas_hot"]
