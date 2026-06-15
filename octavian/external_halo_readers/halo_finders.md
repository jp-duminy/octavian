# Halo Finders

Octavian has compatibility with the AHF halo finder. For the purposes of both keeping code clean and modular, and for future-compatibility, the core infrastructure of how the halo catalogues are read in is done in an agnostic format. This can be found in halo_utils.py.

### halo_utils

This contains three classes:
- HaloReader
- HaloTree
- HaloMembership

HaloReader handles I/O: mapping particle IDs and types and cross-referencing against the original snapshot. HaloTree captures the hierarchy of the output. It can be set to either only deal with field (the top-level parent) halos, or to detail the entire substructure. HaloMembership characterises each halo i.e. which particles belong to it, and then sorts out disentangling and post-processing membership.

Readers then need to follow a set convention to be compatible with this infrastructure. The agnostic classes expect:

- Halo IDs 
- Halo Parent IDs
- Particle (member) Halo IDs
- Halo IDs must be sorted in 0, 1, 2... order
- Halos with no parent (field halos) or particles not in a halo have their parent ID as -1
- Encode particle types as int8 numbers (see PTYPE_ENCODE in halo_utils.py)

## AHF: Amiga's Halo Finder

AHF is slightly finicky because its format is stored in the following way:

Number of Particles | Halo ID 
--- | --- 
Particle ID | Particle Type 
Particle ID | Particle Type 
etc.

This is not convenient to work with because halos have different numbers of particles. This means you cannot easily vectorise the loop as you need information on where the boundaries are, information you can only extract from iterating on the dataset. This is a slow and costly process that can take minutes in Python. To this end AHF comes with an ahf_parser.c file. This is a state machine which goes through the entire dataset and writes in hids, pids and ptypes. Since it is written in C it must be compiled with the following command:

gcc -O2 -shared -fPIC -o ahf_parser.so ahf_parser.c

And then it is called as read_ahf_particles_c in ahf.py. 

The major timesink with AHF is in assigning the particles their halo IDs.

## HBT+

HBT+ is a bit more user-friendly. It outputs a SubSnap_* file which is in hdf5 format. Each subhalo has its own length array of corresponding particle IDs, but this can be quickly vectorised. HBT also assigns unique IDs across a snapshot, meaning they need not be cross-referenced: this is done in the form of TrackID. There are subhalos which can wither away across a simulation. These become known as orphan halos and are tracked by their most-bound particle. As is convention, these are assigned a parent ID of -1.

Where AHF captures its substructure in a tree-like format, HBT+ instead assigns halos to a FoF group where there is a designated central halo and satellite halos. Satellite halos point to their central halo's TrackID. 

HBT+ does not assign particle IDs, only particle types. Thus, we must cross-reference against the original snapshot to assign particles their IDs.
