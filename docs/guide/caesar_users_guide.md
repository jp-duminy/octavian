# Caesar Users Guide

The development of Octavius was necessitated by the growing big data challenge of analysing ever-larger simulations, which Caesar struggled to meet at scale. The code was developed from the ground up reimplementing, rather than directly porting, Caesar routines; this opened up opportunities for optimisation and architectural improvements; Octavius has been designed with performance at scale as an _ab initio_ concern.

This has resulted in divergences in the API and the usage patterns of the codes. Codes must please be adapted to accommodate these differences. To help ensure a smooth transition, the usage guides and examples for Octavius are intended to be illustrative.

The leading advantage of Octavius over its predecessor is in performance. In raw speed, memory profile, and scalability, Octavius offers significant, order-of-magnitude improvements over Caesar with fewer dependencies.

During development of Octavius, many existing bugs were discovered in Caesar and fixed in the reimplementations of routines. This is particularly relevant to galaxy finding, where the two codes can be expected to produce noticeably different results from their FOF6D algorithms owing to bugs in the Caesar code. Not only are the galaxy assignments different, but there have been numerous bug fixes and precision improvements across aggregate properties and photometry, meaning Octavius catalogues will not replicate the same results found by Caesar.

If you have identified a feature present in Caesar but absent in Octavius, or have identified implausible differences in the physical output, please open an issue on the [source repository](https://github.com/jp-duminy/octavius/issues) or [contact the developer](mailto:jp@duminy.org).

## API Differences

- Octavius entirely drops the use of [yt](https://github.com/yt-project) throughout the codebase.
- Octavius has no Cython dependency; performance-critical code is written in numba.
- The data manager and object access patterns in Caesar are replaced by modularised dataclasses and aligned arrays in Octavius.
- Idiomatic usage of Caesar catalogues uses list comprehension; Octavius relies on vectorised numpy.
- Dataset names follow a standardised convention and may have changed.

## General Differences

- Caesar works in 32-bit data, whereas Octavius generally uses 64-bit to avoid overflow bugs.
- Octavius verifies dimensional consistency with astropy.
- Octavius includes more guards and checks to prevent unintended behaviour.
- Functions in Octavius are unit tested and regression tests exist for catalogues.
- Several hardcoded parameters in Caesar are modifiable in Octavius through the configuration file.
- Octavius fixes many bugs in Caesar, some quite significant.

## Standalone Analysis

Octavius is designed around building catalogues with an end-to-end pipeline, and currently does not support standalone analysis of specified groups as Caesar did through its API. It is hoped this functionality will come in the near future, though there is no current estimate for when this can be delivered.