# Analysing Snapshots

Snapshots can be analysed entirely through Python scripts by manually typing the `OctaviusConfig` dataclass. This is less flexible, so is not strictly recommended, but is still possible. Octavius provides a high degree of flexibility with how you analyse snapshots, to support quick straightforward analyses on small boxes and intensive runs at scale. 

## Example 1: Pure Python analysis

```{literalinclude} ../../examples/pure_python.py
:language: python
```

## Example 2: Python analysis with MPI and a config YAML file

```{literalinclude} ../../examples/mpi_python.py
:language: python
```

## Command line analysis examples

Generating a default config YAML file in a desired output directory:

```bash
octavius init -o /path/to/output/dir
```

Once filled in, this can be used with the `-o` flag:

```bash
octavius analyse -c /path/to/config.yaml
```

Running under MPI:

```bash
mpiexec -n 4 octavius analyse -c /path/to/config.yaml
```

Using the command line overrides to loop over snapshots and produce catalogues:

```bash
for snap in /path/to/snapshots/snap_*.hdf5; do
    mpiexec -n 4 octavius analyse -c /path/to/config.yaml -s "$snap"
done
```

This can also be done with halo catalogue paths for each snapshot, if using external halo catalogues.