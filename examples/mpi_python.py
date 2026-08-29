"""

Working with Octavius in a Python script under MPI and using a configuration YAML file.

"""

from pathlib import Path
from mpi4py import MPI
from matplotlib import pyplot as plt
import numpy as np
import octavius as oc

comm = MPI.COMM_WORLD

config_path = Path("/path/to/config.yaml")
config = oc.OctaviusConfig.from_yaml(config_path=config_path)

catalogue_path = oc.analyse_snapshot(config=config)  # analyse_snapshot() is natively MPI-aware

# we will now plot the galactic stellar mass function
if comm.rank == 0:
    cat = oc.load_catalogue(catalogue_path=catalogue_path)

    all_star_counts = cat.galaxies.get_dataset("n_star")
    resolved_mask = all_star_counts >= 32  # filter resolution floor

    stellar_mass = cat.galaxies.get_dataset(
        "mass_star",
        to_units="Msun",
        mask=resolved_mask,
    )
    boxsize = cat.sim_info("boxsize", to_units="Mpc")  # no h factors in catalogue datasets

    cat.close()  # we have everything we need, so close the catalogue

    # quantities needed for GSMF
    volume = boxsize**3
    log_mass = np.log10(stellar_mass)

    # binning
    counts, bin_edges = np.histogram(log_mass, bins=20, range=[8.5, 12.5])
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # number densities
    phi = counts / (bin_width * volume)
    phi_err = np.sqrt(counts) / (bin_width * volume)  # poisson errors
    mask = counts > 0

    # plot result
    fig, ax = plt.subplots()
    ax.errorbar(
        x=bin_centres[mask],
        y=phi[mask],
        yerr=phi_err[mask],
        color="red",
        fmt="o",
        ls="-",
        capsize=3,
        label="GSMF",
    )
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(M_\star / M_\odot)$")
    ax.set_ylabel(r"$\Phi$ [dex$^{-1}$ Mpc$^{-3}$]")
    ax.set_title(f"Galactic Stellar Mass Function for {catalogue_path.name}")
    fig.savefig("gsmf.png", dpi=300)
