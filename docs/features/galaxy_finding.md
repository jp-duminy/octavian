# Galaxy Finding

One of the flagship features of Octavius is the ability to locate galaxies within haloes. Identifying and classifying structure in simulations is a deceptively difficult task; Octavius uses a high-performance implementation of the friends-of-friends algorithm to identify its galaxies.

## Friends of Friends

The friends-of-friends (FOF) algorithm is a simple yet effective and widely-used method of identifying structure originally described by Davis et al. 1985 (`doi: 10.1086/163168`). The idea is simple: we define a characteristic distance $\ell_{fof}$, known as the _linking length_, and consider particles within $\ell_{fof}$ of each other to be _linked_ (friends). Particles are linked recursively to the particles within $\ell_{fof}$ of their friends (the friends of their friends). We define the linking length by the following formula:

$\ell_{fof} = b \times \bar{\lambda}$

Where $b$ is a dimensionless scaling parameter, and $\bar{\lambda}$ is the mean interparticle separation. By defining the linking length as a fraction of the mean interparticle separation, the algorithm can capture particles which are positionally-clustered relative to the background density. 

These clusters are mathematically known as _connected components_. It is customary to apply a minimum size filter to avoid small, spurious clumps of particles being considered galaxies. 

## 6D Extension

While the FOF algorithm can easily identify clusters of particles, it is not without limitations. Consider two galaxies undergoing a merger event: while they are two distinct structures, their overlapping spatial coordinates will cause the FOF algorithm to absorb them into one single structure. In 2006, Diemand et al. (`doi: 10.1086/506377`) proposed a simple solution to this problem: to use velocity information to extend the FOF algorithm into phase space. This partially remedies the aforementioned merger problem, but also helps distinguish particles which belong to the galaxy from interlopers which happen to be within close proximity. 

The 'best' method of identifying structure is a topic of debate, as is the ideal choice of $b$. However, the 6D FOF algorithm remains dependable and widely-used.

## Octavius FOF6D

### Configurable Parameters

- `b`: the FOF dimensionless scaling parameter

- `velocity_factor`: the number of standard deviations from the local velocity dispersion within which a particle considers a neighbour to be linked in phase space.

- `min_stars_per_galaxy`: the minimum number of stars which define a galaxy.

- `gas_criterion`: which criterion to apply to gas entering the algorithm.

- `subhalo_override`: enforces the substructure boundaries defined by the external finder, if subhaloes are present.

- `T_lim`: the temperature below which gas is considered _cold._

- `nH_lim`: the density in $n_{H} \ cm^{-3}$ below which gas is considered _dense._

### Algorithm Description

Octavius locates galaxies within haloes using the FOF6D algorithm. The implementation goes as follows:

Firstly, position-space links must be identified. A naïve implementation may check the criterion for every particle against every other; however, the time of this approach goes as $\mathcal{O}(N^2)$ and is thus untenable at scale. A spatial hashing structure known as a cell linked-list is therefore used. In this structure, a halo is divided into spatial cells of size $\ell_{fof}$. By partitioning the halo this way, if we want to find the links of a particle, the query radius of $\ell_{fof}$ will only extend into adjacent cells. Thus we must only query the 27 neighbouring cells instead of the entire halo, avoiding a significant amount of wasted computation. 

We must also consider the geometry of the problem at hand. If we partition haloes into these $\ell_{fof}$-sized cells, we will still be wasting time in the outskirt regions, where the density is low and thus we will have empty cells. For this reason, the cell linked list is represented in compressed-sparse row (CSR) format, meaning zero entries are skipped. This brings the memory-complexity down to $\mathcal{O}(N)$. The time-complexity will be of order $\mathcal{O}(N \log{N})$, but in the worst-case scenario of highly dense regions where all particles are within $\ell_{fof}$ of one another, the $\mathcal{O}(N^2)$ scaling becomes unavoidable.

The second step is to apply the velocity criterion. To do this, for each particle, we compute its local velocity dispersion $\sigma$. Then, for linked particles, we compare the difference in their velocity against their local $\sigma$ values: if the difference is within a certain factor of $\sigma$, the particles are considered friends in phase space. This criterion is symmetric, meaning both particles must satisfy it.

A textbook [path-compressed union find](https://en.wikipedia.org/wiki/Disjoint-set_data_structure#Union_by_rank) is then applied to create the connected components, which are then masked to the user-configured minimum star criterion before being propagated into the analysis pipeline.

:::{note}
Only gas which passes the cold, dense filter will be assigned to galaxies.
:::
