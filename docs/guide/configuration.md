# Configuration

## Usage

The pipeline is configured through a .yaml file, which grants flexible control over many aspects of the analysis. You can generate a file with default options from the command line via:

```bash
octavius init
```

When running from the command line, you should specify the path to the configuration file with the -c argument, like so:

```bash
mpiexec -n 2 octavius analyse -c /path/to/config.yaml
```

When running from a Python script, you must import the `OctaviusConfig` dataclass to pass to `analyse_snapshot()`. You can either use the `from_yaml()` class method with a `pathlib.Path` object pointing to the filepath of your configuration file, or type the parameters manually.

```python
from pathlib import Path
from octavius import OctaviusConfig

config_path = Path("/path/to/config.yaml")
config = OctaviusConfig.from_yaml(config_path)  # from .yaml
config = OctaviusConfig(...)  # type parameters manually
```

:::{note}
The catalogue will inherit the name of the snapshot prefixed with `octavius_`, as will the log file (if enabled).
:::

## General Options

`simulation_type`: the format of the snapshot (GIZMO/SWIFT).

`halo_id_source`: where the halo catalogue should be parsed from (SNAPSHOT/AHF).

`snapshot_path`: the filepath of the snapshot (can be overridden at runtime via command-line arguments).

`output_dir`: the directory to which you would like outputs routed.

`halo_id_filepath`: the filepath of the halo catalogue (can be left blank if using snapshot IDs). AHF users should specify the stem of the filename before the `.AHF_` extension.

`compress_catalogue`: whether to apply lossless GZIP compression to the catalogue (default: `True`).

## Stages

`find_galaxies`: run the [6D friends-of-friends algorithm](../features/galaxy_finding.md) to locate galaxies in the snapshot (default: `True`).

`properties_core`: compute [core properties](../features/aggregate_properties.md#core-properties), which includes most basic physical properties such as kinematics (default: `True`).

`properties_ptype_specific`: compute [particle-type specific properties](../features/aggregate_properties.md#particle-specific-properties), such as supermassive black hole Eddington fractions (default: `True`).

`properties_local_environment`: compute [properties of galaxies' local environment](../features/aggregate_properties.md#local-environment-properties): aperture masses, local densities/masses (default: `True`).

`photometry`: compute [photometric properties](../features/photometry.md) for galaxies (default: `True`).

Dependency resolution is performed at runtime; for example, photometry requires gas SFR-weighted metallicities to exist, so if `photometry` is enabled but `properties_ptype_specific` is disabled in the config, `properties_ptype_specific` will be re-enabled. The exception to this is galaxy finding, which will disable its dependent stages as it is considered a pre-processing step.

## Particles

`gas`: process gas particles (default: `true`).

`star`: process star particles (default: `true`).

`dm`: process dark matter particles (default: `true`).

`bh`: process black hole particles (default: `true`).

Please note disabling a particle entirely may have unintended consequences: for example, disabling gas particles will prevent galaxy finding from running altogether, and aggregate properties may become inaccurate. 

## Thresholds

`min_stars_per_galaxy`: the minimum number of stars a cluster of baryonic particles must contain to be considered a galaxy (default: `16`).

`min_dm_per_halo`: the minimum number of dark matter particles a halo should contain. Haloes below this threshold are disregarded during analysis and will not appear in the catalogue (default: `24`).

`nH_lim`: the density, in $n_{H} \ cm^{-3}$, above which gas is considered dense. Used to locate dense gas in the FOF6D algorithm, and to define CGM quantities for haloes (default: `0.13`).

`T_lim`: the temperature, in $K$, below which gas is considered cold. Used to locate cold gas in the FOF6D algorithm (default: `1.0e5`).

## Physics Parameters

`FRAD`: the radiative efficiency of accretion, used for Eddington fractions (default: `0.1`).

`MU`: the mean molecular weight, used for the virial temperature scaling relation (default: `0.6`).

## FOF6D Parameters

`b`: the scaling factor used to determine the position-space linking length, which is defined as $b \times \lambda$, where $\lambda$ is the mean interparticle separation (default: `0.02`).

`velocity_factor`: the velocity factor used to determine linking in velocity space. The velocity-space analogy of linking length is defined as the velocity factor multiplied by the local velocity dispersion (default: `1.0`).

## Aggregate Property Parameters

`radial_quantiles`: a dictionary of quantiles (keyed by name) for enclosed mass radial profiles (default: `{"r20": 0.2, "half_mass": 0.5, "r80": 0.8}`).

`aperture_size`: a list of the radius (or radii), in kiloparsecs, of the apertures wherein aperture masses are computed. You can specify multiple aperture sizes (default: `[30]`).

`virial_factors`: a list of the overdensity threshold(s) for which halo virial quantities are computed (default: `[200, 500, 2500]`).

`density_radii`: a list of the radii, in kiloparsecs, at which local environment properties are computed. (default: `[300, 1000, 3000]`)

## Photometry Parameters

`bands`: the FSPS-supported bands in which to compute magnitudes. Please run `fsps.list_filters()` to see the list of available options, or inspect the datasets in the Octavius photometry data file. For filters with multiple bands, e.g. `sdss_u`, `sdss_v` you can simply specify `sdss` and all of its bands will be run. The `v` filter is always enabled for $A_v$ computation. Furthermore, you can specify `all` for magnitudes in all available bands, or `uvoir` for magnitudes in all bands bluewards of five microns. (default: `["all"]`)

`photometry_table_filepath`: the filepath to the Octavius photometry HDF5 datafile (can be left blank if not running photometry).

`extinction_law`: Specify the extinction law to use. Octavius currently supports `power_law`, `calzetti`, `conroy`, `cardelli`, `smc`, `lmc`. In addition, two composite extinction laws are provided: `mix_calz_mw` uses Cardelli for galaxies with $\log_{10}(sSFR) < 0.1 Gyr^-1$, Calzetti for $\log_{10}(sSFR) > 1.0 Gyr^-1$ and a linear mix in between; `composite` adds a further metallicity dependence, using `mix_calz_mw` for $Z > Z_{\odot}$, `smc` for $Z < 0.1 Z_{\odot}$, and a linear combination in between. (default: `composite`).

`viewing_axis`: Specifies the Cartesian axis along which galaxies are viewed (options: `x`, `y`, `z`).

`use_dust`: Specifies whether to use dust masses from the snapshot. (default: `True`)

`use_cosmic_extinction`: Specifies whether to apply a redshift-dependent IGM attenuation as described in Madau (1995) `doi: 10.1086/175332` (default: `True`).

`interpolation_bins`: Specifies the granularity of the numerical approximation of the line-of-sight kernel weight integral used for gas metal column densities (default: `5000`).

`kernel_type`: Specifies the type of kernel used for gas metal column densities (options: `cubic`, `quintic`) (default: `cubic`).

`power_law_alpha`: the exponent $\alpha$ to use on `power_law` attenuation (if using), which goes as $(\frac{\lambda}{\lambda_0})^{-\alpha}$ (default: `1.0`)

`split_age`: the threshold age in $Gyr$ below which the ages of star particles are divided into sub-bins to improve the accuracy of their spectra (default: `0.01`)

## Parallelism

`n_io_chunks`: the number of chunks used to load datasets into the pipeline when conducting parallel file reads off disc (default: `10`).

`cores_per_rank`: the number of cores to associate with each MPI rank. 

## Logging

`terminal_output_level`: the level of log output you would like to display in the terminal. Set to `INFO` for default runtime notifications, `DEBUG` for detailed diagnostics, and `WARNING` for critical notifications only. This does not affect the `.log` file at the end, which contains all output (default: `"INFO"`).

`keep_logs`: whether to keep the runtime logs. If `true`, the logs from multiple ranks are concatenated into a single file where they appear in rank-ascending order (default: `False`).

