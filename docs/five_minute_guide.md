# Five Minute Guide

## What is Octavius?

Octavius is a galaxy simulation analysis toolkit designed to build analysis catalogues of haloes and galaxies in simulation snapshots. As simulations continue to grow in size and complexity, the analysis of their outputs has become a big data computational challenge in and of itself; snapshots can be hundreds of gigabytes large. The parallel, high-performance pipeline is a post-processing step which produces a lightweight (<1% the size of the original snapshot) catalogue of properties, thus bridging the gap between simulations and analysis. The package is designed to be lean and lightweight to slot cleanly into analysis workflows.

## How do I get it?

Octavius is available on pip:

```bash
pip install octavius
```

By design, the package has minimal dependencies; it is written entirely in Python.

:::{tip}
The package is currently pinned to relatively recent versions of its dependencies, and as such, a package manager such as `uv` might be useful.
:::

## What snapshots are supported?

Currently, [SWIFT](https://swift.strw.leidenuniv.nl/) and [GIZMO](http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html) snapshots are supported. AHF halo catalogues are also supported, and [HBT-HERONS](https://hbt-herons.strw.leidenuniv.nl/) support is planned soon. The catalogue is agnostic to the inputs it receives.

## How does it work?

The pipeline is designed around considering haloes as a unit of work. To this end, it starts by determining which particles in the snapshot are in haloes; then, using a simple binning algorithm, it divides haloes amongst ranks such that the computational load is balanced. Then, a simple topological sorting algorithm is run to determine the most efficient way to run the user-requested pipeline stages. Assuming all stages are enabled, the analysis proceeds as follows:

- Run a 6D friends-of-friends algorithm to locate galaxies within haloes
- Construct membership mapping between haloes and galaxies, and their constituent particles
- Compute an assortment of aggregate properties for all groups
- Compute photometric properties for galaxies
- Synthesise the outputs from the pipeline into a lightweight HDF5 catalogue

:::{note}
Data is dynamically loaded and released between stages to minimise the memory footprint. Stable sorting algorithms are used in conjunction with a global sort order to ensure the outputs of a catalogue are invariant and deterministic.
:::

## How fast is it?

The pipeline is primarily written in [numba](https://numba.readthedocs.io/en/stable/user/overview.html), a [just-in-time](https://en.wikipedia.org/wiki/Just-in-time_compilation) (JIT) compiler for Python code which achieves machine-level speed. The computations are therefore competitive with low-level languages. The performance is proven to scale well: for example, the flagship SIMBA snapshot ($100 cMpc {h^-1}$ at $z = 0$, $2 \times 10^10$ particles) can be fully analysed in around an hour.

## How much memory does it need?

In addition to the raw speed advantage, numba avoids the intermediate array allocations which the same code in idiomatic numpy would produce. The memory profile is therefore lean relative to the size of the snapshots being analysed; for example, on the aforementioned 241GB flagship SIMBA snapshot, the memory usage with 4 ranks and full parallellism was about 150GB. 

## How do I know it works?

Octavius contains unit tests for its functions to ensure the obtained results are accurate. Furthermore, it contains a full validation suite with reference catalogues to catch any bugs or regressions in the code, and verify outputs are both valid and physically sensible. The codebase contains an authoritative astropy-backed internal source for its units and constants to ensure dimensional self-consistency. The validation suite includes detailed runtime and memory diagnostic profiling which is run on large snapshots to ensure the code is performant and scaleable.

Commits and pull requests must pass a continuous-integration (CI) workflow, which runs automated tests and codebase linting to prevent any bugs from reaching production code.
